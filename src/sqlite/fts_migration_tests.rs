//! Contention tests for the FTS5 drop-and-recreate migration (issue #255).
//!
//! Sibling of `fts_tests.rs` (both wired from `sqlite/tests.rs`), split out to
//! keep each file under the project's 500-line cap. These are the cases that
//! drive `Database::initialize_fts` against a real `BEGIN EXCLUSIVE` writer,
//! plus the classifier/context contract of the error it returns — a coherent
//! seam separate from the BM25/desync coverage next door.

use crate::sqlite::Database;
use tempfile::TempDir;

/// Bring a store into the state the drop-and-recreate branch handles: the
/// FTS table exists but was built without the `project_id` column, and
/// `user_version` is at the latest so `Database::open` runs no migration of
/// its own (a pending migration takes `BEGIN EXCLUSIVE` on open and would
/// fail earlier, testing the open rather than the rebuild).
fn put_store_in_need_of_fts_rebuild(path: &std::path::Path, wal: bool) {
    let conn = rusqlite::Connection::open(path).unwrap();
    if wal {
        // vipune never sets `journal_mode`, so the default is `delete`; a
        // WAL store is what an operator who set it externally (or a future
        // change) produces. The two modes differ under contention — see
        // `a_lock_before_the_fts_rebuild_fails_earlier_in_delete_journal_mode`.
        conn.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
    }
    conn.execute_batch("DROP TABLE IF EXISTS memories_fts;")
        .unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE memories_fts USING fts5(
            content,
            tokenize='porter unicode61',
            content_rowid='rowid',
            content='memories'
         );",
    )
    .unwrap();
    conn.pragma_update(None, "user_version", 6).unwrap();
}

#[test]
fn a_lock_during_the_fts_schema_migration_stays_the_locked_verdict() {
    // The rebuild's `execute_batch` used to be flattened with
    // `format!("…: {e}")`, which erased the SQLite result code — so a
    // `SQLITE_BUSY` came back as `Error::Database` carrying busy text with
    // `is_locked() == false`, the one case `errors.rs` says must never
    // happen. This drives a real `BEGIN EXCLUSIVE` writer (the shape a
    // running MCP server produces) rather than a hand-built variant.
    //
    // WAL is required for the rebuild to be the statement that notices the
    // lock: in the default `delete` journal mode the read that precedes it
    // fails first (pinned separately by
    // `a_lock_before_the_fts_rebuild_fails_earlier_in_delete_journal_mode`),
    // which would test the precheck rather than the classifier.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("locked-migration.db");

    // Open once to create the schema, then drop the handle so the only
    // handle into the FTS rebuild is the contended one below.
    {
        let db = Database::open(&path).unwrap();
        db.insert(
            "proj1",
            "needs the rebuild",
            &vec![0.1f32; 384],
            None,
            "fact",
            "active",
        )
        .unwrap();
    }
    put_store_in_need_of_fts_rebuild(&path, true);

    let db = Database::open(&path).unwrap();
    let lock_conn = rusqlite::Connection::open(&path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();
    // A `BEGIN EXCLUSIVE` alone takes no lock until the writer's first
    // statement; the write is what makes the lock real.
    lock_conn
        .execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (9999, 'hold')",
            [],
        )
        .unwrap();

    // The rebuild must be reached through the *contended* handle with a
    // zero timeout — otherwise SQLite waits out the default 5s busy handler
    // instead of failing, and the test measures latency rather than the
    // verdict.
    db.set_busy_timeout(std::time::Duration::ZERO).unwrap();

    let err = db
        .initialize_fts()
        .expect_err("the rebuild cannot take the write lock");
    assert!(
        matches!(err, crate::sqlite::Error::Locked),
        "a busy rebuild must keep the storage Locked verdict, got {err:?}"
    );
    let public: crate::errors::Error = err.into();
    assert!(
        public.is_locked(),
        "the verdict must survive the public conversion, got {public:?}"
    );

    lock_conn.execute("ROLLBACK", []).unwrap();
    std::mem::forget(dir);
}

#[test]
fn a_lock_before_the_fts_rebuild_is_swallowed_in_delete_journal_mode() {
    // The other journal mode, and the reason the test above needs WAL.
    //
    // In vipune's default `delete` mode a `BEGIN EXCLUSIVE` writer blocks
    // the `sqlite_master` read that opens `initialize_fts`, and that read
    // is `unwrap_or(false)` — so a *lock* is silently reported as "no FTS
    // table at all". `initialize_fts` then decides the table does not
    // exist, skips the rebuild branch entirely, and returns `Ok(())`: no
    // error at all, and therefore nothing classified. That is a pre-existing
    // swallowing bug, not something this change introduces or is in scope
    // to fix — but it is exactly why a real-writer test for the rebuild
    // classifier has to run in WAL mode, and pinning it here stops the next
    // reader from "simplifying" the WAL setup away and quietly turning the
    // test above into one that passes on a store that was never contended.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("locked-migration-delete.db");
    {
        let db = Database::open(&path).unwrap();
        db.insert(
            "proj1",
            "needs the rebuild",
            &vec![0.1f32; 384],
            None,
            "fact",
            "active",
        )
        .unwrap();
    }
    put_store_in_need_of_fts_rebuild(&path, false);
    let mode: String = Database::open(&path)
        .unwrap()
        .conn()
        .query_row("PRAGMA journal_mode;", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        mode, "delete",
        "the premise of this test is vipune's default journal mode"
    );

    let db = Database::open(&path).unwrap();
    let lock_conn = rusqlite::Connection::open(&path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();
    lock_conn
        .execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (9999, 'hold')",
            [],
        )
        .unwrap();
    db.set_busy_timeout(std::time::Duration::ZERO).unwrap();

    // The observable bug: the store is still in need of the rebuild, yet
    // `initialize_fts` reports success.
    db.initialize_fts().expect(
        "KNOWN SWALLOWING BUG: in delete-journal mode the locked \
         sqlite_master read is unwrap_or(false)'d, so a lock looks like \
         'no FTS table' and the rebuild is skipped without any error",
    );

    // Release the writer before gathering the evidence: the same lock that
    // blocked the precheck would block this read too, and an error here
    // would be the lock rather than the schema shape.
    lock_conn.execute("ROLLBACK", []).unwrap();

    let still_legacy: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories_fts') WHERE name = 'project_id'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        still_legacy, 0,
        "the rebuild never ran, so the store is still on the legacy schema — \
         confirming the lock was swallowed rather than reported"
    );

    std::mem::forget(dir);
}

#[test]
fn an_unclassified_fts_migration_failure_keeps_its_context_and_its_message() {
    // The other half of `with_context`: the context is prepended after
    // classification, so a genuine (non-lock) failure still reads as the
    // operator saw it before, and a non-`Sqlite` verdict is never
    // flattened to make room for prose.
    let plain = crate::sqlite::Error::Sqlite("no such table: nope".to_string())
        .with_context("FTS5 schema migration failed");
    match plain {
        crate::sqlite::Error::Sqlite(msg) => {
            assert_eq!(msg, "FTS5 schema migration failed: no such table: nope");
        }
        other => panic!("expected Sqlite with the context prepended, got {other:?}"),
    }

    // A verdict has no message to grow, so the context must not swallow it.
    let locked = crate::sqlite::Error::Locked.with_context("FTS5 schema migration failed");
    assert!(
        matches!(locked, crate::sqlite::Error::Locked),
        "context must never flatten the locked verdict, got {locked:?}"
    );
}
