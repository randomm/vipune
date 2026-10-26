//! Contention tests for the FTS5 drop-and-recreate migration (issue #255).
//!
//! Sibling of `fts_tests.rs` (both wired from `sqlite/tests.rs`), split out to
//! keep each file under the project's 500-line cap. These are the cases that
//! drive `Database::initialize_fts` against a real `BEGIN EXCLUSIVE` writer,
//! plus the classifier/context contract of the error it returns — a coherent
//! seam separate from the BM25/desync coverage next door.
//!
//! Issue #278 added the two `delete`-mode precheck cases and the
//! absent-FTS-table case, and stopped these tests leaking their `TempDir`
//! (`std::mem::forget(dir)` was what filled the operator's temp quota).

/// The context string the `with_context` cases below prepend — the same one
/// `initialize_fts`'s rebuild arm uses, so those cases exercise the real call
/// shape rather than a paraphrase of it.
const MIGRATION_CONTEXT: &str = "FTS5 schema migration failed";

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
        // `a_lock_before_the_fts_rebuild_now_errors_in_delete_journal_mode`.
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

/// Everything a contended test needs: the store handle, the connection holding
/// the write lock, and the temp dir that owns the file both handles point at.
struct Contended {
    /// Owned so the directory outlives both handles and is removed when the
    /// test returns — the property #278 is about, and the reason this is a
    /// field rather than a local the seeding helper drops on the way out.
    /// Read by `Drop`, never by code, hence the underscore: the lib target
    /// denies warnings and would otherwise reject a field nothing reads.
    _dir: TempDir,
    /// Handle over the store, ready for the contended statement.
    db: Database,
    /// The connection holding the write lock; release it with `release_writer`.
    lock_conn: rusqlite::Connection,
}

/// Release the `BEGIN EXCLUSIVE` writer. Deliberately *not* folded into the
/// assertion helper: when an assertion there fires the panic unwinds, and a
/// temp dir dropped under a live writer is a second, unrelated failure that
/// buries the real one. Every caller calls this before it asserts.
fn release_writer(contended: &Contended) {
    contended
        .lock_conn
        .execute("ROLLBACK", [])
        .expect("release the writer");
}

/// How many `memories_fts` columns are named `project_id` — zero means the
/// drop-and-recreate never completed, which is the evidence a precheck failure
/// has to be told apart from a rebuild that hit the lock partway.
fn project_id_columns(db: &Database) -> i64 {
    db.conn()
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories_fts') WHERE name = 'project_id'",
            [],
            |r| r.get(0),
        )
        .expect("read the memories_fts column roster after releasing the writer")
}

/// Seed a store that needs the FTS rebuild, then put it under a real
/// `BEGIN EXCLUSIVE` writer that holds the write lock (the shape a running MCP
/// server produces), and return everything the caller needs to contend
/// `initialize_fts` against it and to gather evidence afterwards.
///
/// The lock is *live* on return: the contended statement is what the caller
/// runs, and `release_writer` has to run before anything reads the file (a read
/// through the lock reports the lock, not the schema shape).
///
/// The WAL test and the two `delete`-mode tests shared this setup byte for
/// byte; what differs between them is why each one exists (see their own
/// comments), so the mechanics live here and the history stays with the tests.
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
            "fact",
            "active",
        )
        .unwrap();
    }
    put_store_in_need_of_fts_rebuild(&path, wal);

    let db = Database::open(&path).unwrap();
    let lock_conn = rusqlite::Connection::open(&path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();
    // A bare `BEGIN EXCLUSIVE` takes no lock until the writer's first
    // statement; the write is what makes the lock real.
    lock_conn
        .execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (9999, 'hold')",
            [],
        )
        .unwrap();
    // Zero timeout: without it SQLite waits out the default 5s busy handler
    // and the test measures latency rather than the verdict.
    db.set_busy_timeout(std::time::Duration::ZERO).unwrap();

    Contended {
        _dir: dir,
        db,
        lock_conn,
    }
}

/// Run the contended `initialize_fts`, then release the writer, and hand back
/// the error. Split into release-then-assert (rather than one helper doing
/// both) so a failing assertion can never leave the dir dropped under an
/// unreleased writer.
fn contend_and_release(contended: &Contended, why: &str) -> crate::sqlite::Error {
    let err = contended.db.initialize_fts().expect_err(why);
    release_writer(contended);
    err
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
    // `a_lock_before_the_fts_rebuild_now_errors_in_delete_journal_mode`),
    // which would test the precheck rather than the classifier. After #278
    // that earlier read *errors* rather than returning Ok, so this test still
    // needs WAL to reach the rebuild at all — do not "simplify" it away.
    let contended = seed_store_under_exclusive_writer("locked-migration.db", true);

    let err = contend_and_release(&contended, "the rebuild cannot take the write lock");
    assert!(
        matches!(err, crate::sqlite::Error::Locked),
        "a busy rebuild must keep the storage Locked verdict, got {err:?}",
    );
    let public: crate::errors::Error = err.into();
    assert!(
        public.is_locked(),
        "the verdict must survive the public conversion, got {public:?}",
    );
    // The rebuild, not the precheck, is the contended statement here, so the
    // same zero from the precheck tests carries the opposite meaning: the
    // migration was reached and the writer kept it from completing.
    assert_eq!(
        project_id_columns(&contended.db),
        0,
        "the rebuild was reached and blocked — it did not complete",
    );
    // `contended._dir` is dropped here rather than forgotten (issue #278): the lock
    // has been released and both handles close with the test.
}

#[test]
fn a_lock_before_the_fts_rebuild_now_errors_in_delete_journal_mode() {
    // This is the test PR #273 left behind to *pin the swallow*: in vipune's
    // default `delete` mode a `BEGIN EXCLUSIVE` writer blocked the
    // `sqlite_master` read that opens `initialize_fts`, that read was
    // `unwrap_or(false)`, and so a lock looked like "no FTS table at all" —
    // `initialize_fts` returned `Ok(())` on a store that still needed the
    // migration. Issue #278 propagates the precheck, so the assertion is
    // inverted: the same setup must now produce the locked verdict. Kept as
    // its own test rather than folded into the WAL case above, so the history
    // of the pinned behaviour stays in the file and the WAL test keeps its
    // motivation visible — without WAL the rebuild is never the contended
    // statement, and that test would pass on a store that was never contended.
    //
    // The premise this test rests on is vipune's *default* journal mode, so it
    // is read from the store rather than taken on faith from the `wal: false`
    // argument (and read before contending, since after the writer steps every
    // read reports the lock instead).
    let contended = seed_store_under_exclusive_writer("locked-migration-delete.db", false);
    // vipune never sets `journal_mode`, so `delete` is both SQLite's default and
    // the premise of this test; read it from the store rather than trusting the
    // seeding helper's `wal: false` argument, and read it *before* contending
    // (after the writer steps, every read reports the lock instead of the mode).
    let mode: String = contended
        .db
        .conn()
        .query_row("PRAGMA journal_mode;", [], |r| r.get(0))
        .expect("read the journal-mode premise");
    assert_eq!(
        mode, "delete",
        "the premise of this test is vipune's default journal mode",
    );

    // The observable fix: where the pinned version of this test expected
    // `Ok(())`, the locked precheck now returns the locked verdict.
    let err = contend_and_release(
        &contended,
        "#278: a locked sqlite_master read must surface as an error, not 'no FTS table'",
    );
    // Two things are asserted, because "an error came back" alone could be true
    // for the wrong reason (a lock noticed later, or a rebuild that half-ran):
    // the verdict *is* the lock, and the rebuild did *not* run.
    assert!(
        matches!(err, crate::sqlite::Error::Locked),
        "a busy precheck must keep the storage Locked verdict, got {err:?}",
    );
    let public: crate::errors::Error = err.into();
    assert!(
        public.is_locked(),
        "the verdict must survive the public conversion, got {public:?}",
    );
    assert_eq!(
        project_id_columns(&contended.db),
        0,
        "the rebuild never ran — the error came from the precheck, not from a \
         half-completed migration",
    );
}

#[test]
fn initialize_fts_reports_ok_when_the_fts_table_legitimately_does_not_exist() {
    // The counterpart to the fix: propagating the precheck must not turn the
    // ordinary "no `memories_fts` table" case into an error. `Database::open`
    // always creates the table, so the absence is built by hand here — the
    // table *and* its triggers dropped, and `user_version` pinned to the
    // latest so nothing is pending: a reopen would otherwise run a migration,
    // which takes `BEGIN EXCLUSIVE` and would make this measure the open
    // rather than the precheck.
    //
    // With nothing to migrate, `initialize_fts` takes the `fts_exists == false`
    // branch and returns `Ok(())` without touching the rebuild.
    //
    // The table is dropped through a raw connection rather than
    // `Database::open`: `open` re-runs `create_schema`, whose
    // `CREATE VIRTUAL TABLE IF NOT EXISTS` resurrects whatever this test just
    // removed (`put_store_in_need_of_fts_rebuild` relies on the same
    // asymmetry). The triggers have to go with the table — an INSERT trigger
    // pointing at a dropped FTS table makes the next write to `memories` fail
    // outright — so they are enumerated from `sqlite_master` rather than named
    // by hand: a trigger added to `create_schema` or to the rebuild batch is
    // dropped here too, which keeps the completeness claim mechanical rather
    // than prose. `memories` keeps one row so the empty precheck is reached
    // with real data behind it.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no-fts.db");
    {
        let db = Database::open(&path).unwrap();
        db.insert(
            "proj1",
            "a row with no fts index",
            &vec![0.1f32; 384],
            None,
            "fact",
            "active",
        )
        .unwrap();
    }
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch("DROP TABLE IF EXISTS memories_fts;")
        .unwrap();
    let fts_triggers: Vec<String> = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'memories_fts_%'",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<String>>>()
        .unwrap();
    assert!(
        !fts_triggers.is_empty(),
        "the premise: the store has FTS triggers to drop, got {fts_triggers:?}",
    );
    for name in &fts_triggers {
        // `name` came from `sqlite_master`, so it is an identifier this store
        // itself created; the interpolation cannot widen the statement.
        conn.execute_batch(&format!("DROP TRIGGER IF EXISTS {name};"))
            .unwrap();
    }
    conn.pragma_update(None, "user_version", 6).unwrap();

    // The precheck has to run on a handle that did not re-bootstrap the
    // schema: every `Database::open` re-runs `create_schema`, whose
    // `CREATE VIRTUAL TABLE IF NOT EXISTS` resurrects the table this test just
    // removed. `Database::from_conn` wraps the very connection that did the
    // drop, so the assertion is about `initialize_fts` and not about which
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

    let leftover_triggers: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'memories_fts_%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        leftover_triggers, 0,
        "no trigger may survive pointing at the dropped FTS table — the next write \
         to `memories` would fail outright rather than reach the precheck",
    );

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
        .with_context(MIGRATION_CONTEXT);
    match plain {
        crate::sqlite::Error::Sqlite(msg) => {
            assert_eq!(msg, "FTS5 schema migration failed: no such table: nope");
        }
        other => panic!("expected Sqlite with the context prepended, got {other:?}"),
    }

    // A verdict has no message to grow, so the context must not swallow it.
    let locked = crate::sqlite::Error::Locked.with_context(MIGRATION_CONTEXT);
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
    // verdict arm in the test above, rather than covered by a blanket wildcard.
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
        let after = err.with_context(MIGRATION_CONTEXT);
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
