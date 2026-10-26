//! Contention tests for the FTS5 drop-and-recreate migration (issue #255).
//!
//! Sibling of `fts_tests.rs` (both wired from `sqlite/tests.rs`), split out to
//! keep each file under the project's 500-line cap. These are the cases that
//! drive `Database::initialize_fts` against a real `BEGIN EXCLUSIVE` writer,
//! plus the classifier/context contract of the error it returns — a coherent
//! seam separate from the BM25/desync coverage next door.

use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::sqlite::Database;
use tempfile::TempDir;

/// Bring a store into the state the drop-and-recreate branch handles: the
/// FTS table exists but was built without the `project_id` column, and
/// `user_version` is at the latest so `Database::open` runs no migration of
/// its own (a pending migration takes `BEGIN EXCLUSIVE` on open and would
/// fail earlier, testing the open rather than the rebuild). The version comes
/// from the migration roster rather than a literal, so appending a migration
/// cannot quietly turn these contention tests into migration tests.
fn put_store_in_need_of_fts_rebuild(conn: &rusqlite::Connection, wal: bool) {
    if wal {
        // vipune never sets `journal_mode`, so the default is `delete`; a
        // WAL store is what an operator who set it externally (or a future
        // change) produces. The two modes differ under contention — see
        // `a_lock_on_the_precheck_read_errors_instead_of_looking_like_no_fts_table`,
        // which drives the precheck in the default mode.
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
    conn.pragma_update(
        None,
        "user_version",
        crate::sqlite::migrations::latest_schema_version(),
    )
    .unwrap();
}

/// Seed a store that is in need of the FTS rebuild, then put it under a real
/// `BEGIN EXCLUSIVE` writer that holds the write lock (the shape a running MCP
/// server produces), and return everything the caller needs to contend
/// `initialize_fts` against it and to gather evidence afterwards.
///
/// The lock is *live* on return: the contended statement is what the caller
/// runs, and `lock_conn` must be rolled back before anything reads the file (a
/// read through the lock reports the lock, not the schema shape). Each caller
/// drops `dir` after releasing it.
///
/// `a_lock_before_the_fts_rebuild_now_errors_in_delete_journal_mode` and
/// `a_lock_on_the_precheck_read_errors_instead_of_looking_like_no_fts_table`
/// shared this setup byte for byte; what differs between them is why each one
/// exists (see their own comments), so the mechanics live here and the history
/// stays with the tests.
struct Contended {
    /// Owned so the directory outlives both handles and is removed when the
    /// test returns — the property #278 is about, and the reason this is a field
    /// rather than a local the helper drops on the way out.
    #[allow(dead_code)] // read by drop, not by code: the field exists for its Drop impl
    dir: TempDir,
    /// Handle over the store, ready for the contended statement.
    db: Database,
    /// The connection holding the write lock; roll it back when done.
    lock_conn: rusqlite::Connection,
    /// Journal mode of the store, read before the lock was taken (see
    /// `seed_store_under_exclusive_writer`).
    journal_mode: String,
}

/// How many `memories_fts` columns are named `project_id` — zero means the
/// drop-and-recreate never ran, which is the positive evidence a precheck
/// failure has to be told apart from a rebuild that hit the lock partway.
fn project_id_columns(db: &Database) -> i64 {
    db.conn()
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories_fts') WHERE name = 'project_id'",
            [],
            |r| r.get(0),
        )
        .unwrap()
}

/// The `delete` journal mode these tests take as their premise: vipune never
/// sets `journal_mode`, so this is what a real store is in. The value is read
/// by the seeding helper *before* the writer steps, and carried on `Contended`
/// — a read after the lock would report the lock, not the mode.
fn seed_store_under_exclusive_writer(filename: &str, wal: bool) -> Contended {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    // Open once to create the schema, then drop the handle so the only handle
    // into the store is the contended one below.
    {
        let db = Database::open(&path).unwrap();
        db.insert(
            "proj1",
            "needs the rebuild",
            &vec![0.1f32; 384],
            None,
            MemoryType::Fact.as_str(),
            MemoryStatus::Active.as_str(),
        )
        .unwrap();
    }
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        put_store_in_need_of_fts_rebuild(&conn, wal);
    }

    let db = Database::open(&path).unwrap();
    let lock_conn = rusqlite::Connection::open(&path).unwrap();
    // The journal-mode premise is read *here*, while the file is still
    // uncontended: once the writer below has stepped, every read is the lock.
    let mode: String = db
        .conn()
        .query_row("PRAGMA journal_mode;", [], |r| r.get(0))
        .unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();
    // A bare `BEGIN EXCLUSIVE` takes no lock until the writer's first statement;
    // the write is what makes the lock real.
    lock_conn
        .execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (9999, 'hold')",
            [],
        )
        .unwrap();
    // Zero timeout: without it SQLite waits out the default 5s busy handler and
    // the test measures latency rather than the verdict.
    db.set_busy_timeout(std::time::Duration::ZERO).unwrap();

    Contended {
        dir,
        db,
        lock_conn,
        journal_mode: mode,
    }
}

/// Assert both halves of a locked-`initialize_fts` verdict: the error *is* the
/// lock (checked through the public conversion, which is where #255 lost it),
/// and — once the writer is released — the rebuild did *not* run.
fn assert_locked_verdict_and_unrun_rebuild(err: crate::sqlite::Error, contended: &Contended) {
    assert!(
        matches!(err, crate::sqlite::Error::Locked),
        "a busy precheck must keep the storage Locked verdict, got {err:?}",
    );
    let public: crate::errors::Error = err.into();
    assert!(
        public.is_locked(),
        "the verdict must survive the public conversion, got {public:?}",
    );

    // Release the writer before gathering the evidence: the same lock that
    // blocked the precheck would block this read too, and an error here would be
    // the lock rather than the schema shape.
    contended
        .lock_conn
        .execute("ROLLBACK", [])
        .expect("release the writer");

    assert_eq!(
        project_id_columns(&contended.db),
        0,
        "the rebuild never ran — the error came from the precheck, not from a \
         half-completed migration",
    );
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
    // fails first (pinned by
    // `a_lock_on_the_precheck_read_errors_instead_of_looking_like_no_fts_table`),
    // which would test the precheck rather than the classifier. After #278
    // that earlier read *errors* rather than returning Ok, so this test still
    // needs WAL to reach the rebuild at all — do not "simplify" it away.
    let contended = seed_store_under_exclusive_writer("locked-migration.db", true);

    let err = contended
        .db
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

    contended.lock_conn.execute("ROLLBACK", []).unwrap();
    // `contended.dir` is dropped here rather than forgotten: the lock has been
    // released and the two handles closed, so the temp dir can go with them
    // (issue #278).
}

#[test]
fn a_lock_before_the_fts_rebuild_now_errors_in_delete_journal_mode() {
    // This is the test PR #273 left behind to *pin the swallow*: in vipune's
    // default `delete` mode a `BEGIN EXCLUSIVE` writer blocked the
    // `sqlite_master` read that opens `initialize_fts`, that read was
    // `unwrap_or(false)`, and so a lock looked like "no FTS table at all" —
    // `initialize_fts` returned `Ok(())` on a store that still needed the
    // migration. Issue #278 propagates the precheck, so the assertion is
    // inverted: the same setup must now produce the locked verdict. Kept as its
    // own test (rather than folded into
    // `a_lock_on_the_precheck_read_errors_…`) so the history of the pinned
    // behaviour stays in the file, and so the WAL test above keeps its
    // motivation visible: without WAL the rebuild is never the contended
    // statement, and that test would pass on a store that was never contended.
    //
    // The shared mechanics (seed, drop-to-legacy-schema, take the lock, zero the
    // busy timeout) are in `seed_store_under_exclusive_writer`.
    let contended = seed_store_under_exclusive_writer("locked-migration-delete.db", false);
    assert_eq!(
        contended.journal_mode, "delete",
        "the premise of this test is vipune's default journal mode"
    );

    // The observable fix: where the pinned version of this test expected
    // `Ok(())`, the locked precheck now returns the locked verdict.
    let err = contended.db.initialize_fts().expect_err(
        "#278: a locked sqlite_master read must surface as an error, not 'no FTS table'",
    );
    assert_locked_verdict_and_unrun_rebuild(err, &contended);
}

#[test]
fn a_lock_on_the_precheck_read_errors_instead_of_looking_like_no_fts_table() {
    // The counterpart to `a_lock_during_the_fts_schema_migration_stays_the_locked_verdict`:
    // the WAL test drives the *rebuild*; this one drives the read that precedes
    // it, which is the statement that actually fails first in vipune's default
    // `delete` journal mode.
    //
    // In vipune's default `delete` mode a `BEGIN EXCLUSIVE` writer blocks the
    // `sqlite_master` read that opens `initialize_fts`. That read used to be
    // `unwrap_or(false)`, so a *lock* was reported as "no FTS table at all":
    // the rebuild branch was skipped and `initialize_fts` returned `Ok(())` on
    // a store that still needed it — no error, nothing classified, no log line.
    // Issue #278 propagates the precheck instead, so the same contention is now
    // the locked verdict. Two things are asserted, because "an error came back"
    // alone could be true for the wrong reason (a lock noticed later, or a
    // rebuild that half-ran): the verdict *is* the lock, and the rebuild did
    // *not* run. Both live in `assert_locked_verdict_and_unrun_rebuild`, which
    // is shared with `a_lock_before_the_fts_rebuild_now_errors_…` — the two
    // differ only in why they exist, not in what they measure.
    let contended = seed_store_under_exclusive_writer("locked-precheck.db", false);
    assert_eq!(
        contended.journal_mode, "delete",
        "the premise of this test is vipune's default journal mode"
    );

    let err = contended
        .db
        .initialize_fts()
        .expect_err("a locked precheck must no longer look like 'no FTS table'");
    assert_locked_verdict_and_unrun_rebuild(err, &contended);
}

#[test]
fn initialize_fts_reports_ok_when_the_fts_table_legitimately_does_not_exist() {
    // The counterpart to the fix: propagating the precheck must not turn the
    // ordinary "no `memories_fts` table" case into an error. `Database::open`
    // always creates the table, so the absence is built by hand here — the
    // table *and* its triggers dropped, and `user_version` pinned to the latest
    // so nothing is pending: the reopen below would otherwise run a migration,
    // which takes `BEGIN EXCLUSIVE` and would make this measure the open rather
    // than the precheck. Same premise `put_store_in_need_of_fts_rebuild`
    // establishes, and for the same reason — derived from the migration roster,
    // not a literal, so appending a migration moves it automatically.
    //
    // With nothing to migrate, `initialize_fts` takes the `fts_exists == false`
    // branch and returns `Ok(())` without touching the rebuild.
    //
    // The table is dropped through a raw connection rather than `Database::open`:
    // `open` re-runs `create_schema`, which would re-create with `IF NOT EXISTS`
    // whatever this test just removed (and `put_store_in_need_of_fts_rebuild`
    // above relies on the same asymmetry). The triggers go with the table — an
    // INSERT trigger pointing at a dropped FTS table makes the next write to
    // `memories` fail outright — and `memories` keeps one row so the empty
    // precheck is reached with real data behind it.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no-fts.db");
    {
        let db = Database::open(&path).unwrap();
        db.insert(
            "proj1",
            "a row with no fts index",
            &vec![0.1f32; 384],
            None,
            MemoryType::Fact.as_str(),
            MemoryStatus::Active.as_str(),
        )
        .unwrap();
    }
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "DROP TABLE IF EXISTS memories_fts;
         DROP TRIGGER IF EXISTS memories_fts_insert;
         DROP TRIGGER IF EXISTS memories_fts_delete;
         DROP TRIGGER IF EXISTS memories_fts_update;",
    )
    .unwrap();
    conn.pragma_update(
        None,
        "user_version",
        crate::sqlite::migrations::latest_schema_version(),
    )
    .unwrap();
    // The precheck has to run on a handle that did not re-bootstrap the schema:
    // every `Database::open` re-runs `create_schema`, whose
    // `CREATE VIRTUAL TABLE IF NOT EXISTS` resurrects the table this test just
    // removed. `Database::from_conn` wraps the very connection that did
    // the drop, so the assertion is about `initialize_fts` and not about which
    // handle type the test happened to use.
    let db = Database::from_conn(conn);
    let absent: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(absent, 0, "the premise: the FTS table is genuinely absent");

    db.initialize_fts()
        .expect("a genuinely absent FTS table is Ok, not an error");
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

#[test]
fn with_context_leaves_every_non_sqlite_variant_untouched() {
    // The arms `with_context` must NOT touch. An earlier revision mapped the
    // three structured decoding variants to `Error::Locked` here, which turned
    // a corrupt BLOB into a retryable lock and dropped the `id`/`expected`/
    // `actual` a caller acts on — so each is pinned individually, plus the
    // verdict arm above, rather than covered by a blanket wildcard.
    // Each case pairs a storage error with what its public form must be; the
    // `fn` pointer is factored out because the pair type trips
    // `clippy::type_complexity` when written inline.
    type Kept = (crate::sqlite::Error, fn(&crate::sqlite::Error) -> bool);
    let cases: Vec<Kept> = vec![
        (
            crate::sqlite::Error::CorruptEmbedding {
                id: "mem-1".to_string(),
                reason: "not enough bytes".to_string(),
            },
            |e| {
                matches!(e, crate::sqlite::Error::CorruptEmbedding { id, reason }
                if id == "mem-1" && reason == "not enough bytes")
            },
        ),
        (
            crate::sqlite::Error::InvalidBlobSize {
                expected: crate::embedding::EMBEDDING_DIMS * 4,
                actual: 4,
            },
            |e| {
                matches!(
                    e,
                    crate::sqlite::Error::InvalidBlobSize {
                        expected,
                        actual: 4
                    } if *expected == crate::embedding::EMBEDDING_DIMS * 4
                )
            },
        ),
        (
            crate::sqlite::Error::MismatchedDimensions {
                expected: 384,
                actual: 768,
            },
            |e| {
                matches!(
                    e,
                    crate::sqlite::Error::MismatchedDimensions {
                        expected: 384,
                        actual: 768
                    }
                )
            },
        ),
        (crate::sqlite::Error::Locked, |e| {
            matches!(e, crate::sqlite::Error::Locked)
        }),
    ];
    for (err, is_kept) in cases {
        let expected = err.to_string();
        let after = err.with_context("FTS5 schema migration failed");
        assert!(
            is_kept(&after),
            "a non-Sqlite variant must survive with_context with its payload, got {after:?}"
        );
        assert_eq!(
            after.to_string(),
            expected,
            "with_context must not rewrite a non-Sqlite variant's rendering"
        );
    }
}
