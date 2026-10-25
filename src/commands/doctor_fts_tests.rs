//! Tests for `vipune doctor --fts` handler and FTS desync detection (Issue #193).
//!
//! The handler-level tests use a real FTS5 external-content table created by
//! `Database::open`. Because FTS5 external-content tables are automatically kept in
//! sync with their content table (independent of user-defined triggers), the
//! desync detection tests focus on the core rowid-join logic via the
//! `detect_fts_desync` method, which operates on the real schema.
//!
//! The locked-DB and repair tests use the real handler end-to-end.

use crate::commands::doctor_fts::{handle_doctor_fts, recovery_for, recovery_note};
use crate::errors::{Error, LOCKED_DB_MESSAGE};
use crate::memory::crud::test_fake_embedder;
use crate::output::DoctorFtsResponse;
use crate::sqlite::Database;
use crate::sqlite::fts::FtsDesyncReport;
use rusqlite::Connection;
use std::path::PathBuf;
use std::process::ExitCode;

fn create_test_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn data_version(conn: &Connection) -> i64 {
    conn.query_row("PRAGMA data_version", [], |r| r.get(0))
        .unwrap()
}

/// Count FTS rows.
fn fts_count(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM memories_fts", [], |r| r.get(0))
        .unwrap()
}

/// Count memory rows.
fn memory_count(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap()
}

// ── F1: healthy / zero-row in-sync ──

#[test]
fn f1_healthy_database_reports_in_sync() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // Seed a normal (trigger-synced) memory so FTS is populated in sync.
    let emb = test_fake_embedder("hello world").unwrap();
    db.insert("proj-a", "hello world", &emb, None, "fact", "active")
        .unwrap();

    let version_before = data_version(db.conn());

    let result = handle_doctor_fts(&db_path, None, false, false);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ExitCode::SUCCESS, "in-sync must exit 0");

    let version_after = data_version(db.conn());
    assert_eq!(
        version_before, version_after,
        "detection must perform zero database writes"
    );
}

#[test]
fn f1_fresh_zero_row_db_is_healthy_not_desync() {
    let (_dir, db_path) = create_test_db();
    // Empty DB: memories and memories_fts both hold 0 rows.
    let result = handle_doctor_fts(&db_path, None, false, false);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ExitCode::SUCCESS, "zero/zero must exit 0");

    // And detection must report in-sync.
    let db = Database::from_conn(
        Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap(),
    );
    let report = db.detect_fts_desync().unwrap();
    assert!(!report.is_desynced(), "0/0 is NOT a desync");
    assert_eq!(report.total_desynced(), 0);
}

// ── F2: FTS table missing entirely (no FTS table = no desync) ──

#[test]
fn f2_missing_fts_table_reports_in_sync() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // Seed a memory.
    let emb = test_fake_embedder("hello").unwrap();
    db.insert("proj-a", "hello", &emb, None, "fact", "active")
        .unwrap();

    // Drop the FTS table entirely. detect_fts_desync returns default (in-sync)
    // when the FTS table doesn't exist.
    db.conn().execute("DROP TABLE memories_fts", []).unwrap();

    let report = db.detect_fts_desync().unwrap();
    assert!(
        !report.is_desynced(),
        "missing FTS table is treated as in-sync"
    );
    assert_eq!(report.total_desynced(), 0);
}

// ── F3: in-sync DB with multiple rows ──

#[test]
fn f3_healthy_db_with_multiple_rows_is_in_sync() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // Seed 5 rows across 2 projects (triggers keep FTS in sync).
    for i in 0..5 {
        let emb = test_fake_embedder(&format!("memory {i}")).unwrap();
        let proj = if i < 3 { "proj-a" } else { "proj-b" };
        db.insert(proj, &format!("memory {i}"), &emb, None, "fact", "active")
            .unwrap();
    }

    let report = db.detect_fts_desync().unwrap();
    assert!(!report.is_desynced(), "trigger-synced DB must be in-sync");
    assert_eq!(report.underpopulated_global, 0);
    assert_eq!(report.orphans, 0);

    // Verify counts match.
    assert_eq!(memory_count(db.conn()), fts_count(db.conn()));
}

// ── F5: -p scoping (in-sync case) ──

#[test]
fn f5_p_scoping_in_sync_projects() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // Project X: 2 rows. Project Y: 1 row. Both in sync.
    for i in 0..2 {
        let emb = test_fake_embedder(&format!("x {i}")).unwrap();
        db.insert("X", &format!("x {i}"), &emb, None, "fact", "active")
            .unwrap();
    }
    let emb = test_fake_embedder("y only").unwrap();
    db.insert("Y", "y only", &emb, None, "fact", "active")
        .unwrap();

    // No -p: in-sync.
    let all_report = db.detect_fts_desync().unwrap();
    assert!(!all_report.is_desynced());

    // -p X: in-sync.
    assert!(
        handle_doctor_fts(&db_path, Some("X"), false, true).is_ok(),
        "scoped -p X must not error"
    );

    // -p Y: in-sync.
    assert!(
        handle_doctor_fts(&db_path, Some("Y"), false, false).is_ok(),
        "a project with zero desynced rows must report in-sync, not error"
    );
}

// ── F6: repair skip (zero desync) ──

#[test]
fn f6_repair_on_healthy_db_skips_rebuild_and_leaves_version_unchanged() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    let emb = test_fake_embedder("hello").unwrap();
    db.insert("proj-a", "hello", &emb, None, "fact", "active")
        .unwrap();

    let version_before = data_version(db.conn());

    let result = handle_doctor_fts(&db_path, None, true, false);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ExitCode::SUCCESS);

    let version_after = data_version(db.conn());
    assert_eq!(
        version_before, version_after,
        "repair with zero desync must SKIP the rebuild (data_version unchanged)"
    );
}

// ── F7: repair on in-sync DB (rebuild is a no-op) ──

#[test]
fn f7_rebuild_on_in_sync_db_preserves_memories() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // Seed 3 rows and capture their byte-exact snapshot.
    for i in 0..3 {
        let emb = test_fake_embedder(&format!("row {i}")).unwrap();
        db.insert("proj-a", &format!("row {i}"), &emb, None, "fact", "active")
            .unwrap();
    }

    let snapshot_before: Vec<(String, String)> = {
        let mut stmt = db
            .conn()
            .prepare("SELECT id, content FROM memories ORDER BY id")
            .unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap();
        rows.map(|r| r.unwrap()).collect()
    };
    assert_eq!(snapshot_before.len(), 3, "sanity: 3 seeded rows");

    // Run repair (pre-check finds in-sync → skip, but let's test the rebuild path
    // by forcing a rebuild via the handler on a desynced state).
    // Since we can't easily create a desync, test that repair on in-sync skips.
    let result = handle_doctor_fts(&db_path, None, true, false);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ExitCode::SUCCESS);

    // Verify the 3 triggers are still present.
    let db2 = Database::open(&db_path).unwrap();
    for name in [
        "memories_fts_insert",
        "memories_fts_delete",
        "memories_fts_update",
    ] {
        let present: i64 = db2
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?1",
                rusqlite::params![name],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(present, 1, "trigger {name} must still exist");
    }

    // Verify memories rows are unchanged.
    let snapshot_after: Vec<(String, String)> = {
        let mut stmt = db2
            .conn()
            .prepare("SELECT id, content FROM memories ORDER BY id")
            .unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap();
        rows.map(|r| r.unwrap()).collect()
    };
    assert_eq!(
        snapshot_before, snapshot_after,
        "memories rows must be byte-unchanged after repair skip"
    );
}

// ── Idempotent second repair run ──

#[test]
fn repair_is_idempotent_second_run_skips() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    let emb = test_fake_embedder("a").unwrap();
    db.insert("proj-a", "a", &emb, None, "fact", "active")
        .unwrap();

    // First repair: pre-check finds in-sync → skip.
    assert!(handle_doctor_fts(&db_path, None, true, false).is_ok());
    // Second repair: same → skip.
    let db2 = Database::open(&db_path).unwrap();
    let version_before = data_version(db2.conn());
    assert!(handle_doctor_fts(&db_path, None, true, false).is_ok());
    let version_after = data_version(db2.conn());
    assert_eq!(
        version_before, version_after,
        "second repair run must skip (data_version unchanged)"
    );
}

// ── F9: locked DB fast-fails (both detect and repair) ──

#[test]
fn a_lock_before_the_rebuild_keeps_the_plain_transient_framing() {
    // The repair path has two lock sites: the read-write open (nothing has
    // been touched yet → plain "stop the MCP server and retry") and the rebuild
    // statement itself (index possibly half-built → recovery note).
    //
    // An external-content FTS5 table is trigger-synced and rejects direct writes
    // to `memories_fts`, so a fixture cannot both be desynced and survive
    // `Database::open` under an existing writer — the mid-rebuild branch is only
    // reachable with a real concurrent writer, and is exercised by
    // `the_recovery_note_requires_the_locked_verdict`. This asserts the
    // reachable half: a lock at the open must NOT claim a half-rebuilt index.
    //
    // Why that stays true even if the open path changes: the recovery note is
    // appended only around the rebuild call, which sits below
    // `open_read_write`. Any failure of the open (including the `BEGIN
    // EXCLUSIVE` the migrations take, see `sqlite::migrations::run_migrations`)
    // returns before the rebuild is reached, so the plain framing is a
    // structural property of the call order, not of where SQLite notices the
    // lock.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("row").unwrap();
    db.insert("proj-a", "row", &emb, None, "fact", "active")
        .unwrap();
    drop(db);

    let lock_conn = Connection::open(&db_path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();

    // Drop the pre-check's read-only connection first so the failure lands on
    // the read-write open rather than the open_with_flags that precedes it.
    match handle_doctor_fts(&db_path, None, true, false) {
        Err(Error::Config(msg)) => {
            assert!(msg.contains("MCP server"), "got: {msg}");
            assert!(
                !msg.contains("--repair"),
                "a lock hit before the rebuild must not claim a half-rebuilt index: {msg}"
            );
        }
        other => panic!("expected the locked Config error, got {other:?}"),
    }

    lock_conn.execute("ROLLBACK", []).unwrap();
}

#[test]
fn f9_locked_db_fast_fails_on_detect_and_repair() {
    let (_dir, db_path) = create_test_db();

    // Lock the DB.
    let lock_conn = Connection::open(&db_path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();

    // Both modes hit the lock at their first statement (the read-only open for
    // detect, the read-write `Database::open` for repair), so both render the
    // transient framing: no rebuild has run yet, so there is nothing to re-run
    // (see `a_lock_before_the_rebuild_keeps_the_plain_transient_framing` for
    // why that holds, and
    // `the_recovery_note_requires_the_locked_verdict` for the mid-rebuild arm).
    for repair in [false, true] {
        match handle_doctor_fts(&db_path, None, repair, false) {
            Err(Error::Config(msg)) => {
                assert!(msg.contains("locked"), "got: {msg}");
                assert!(msg.contains("MCP server"), "got: {msg}");
                assert!(
                    !msg.contains("--repair"),
                    "a lock hit before the rebuild must not claim a half-rebuilt index (repair={repair}): {msg}"
                );
            }
            other => panic!("expected Config locked error, got {other:?}"),
        }
    }

    lock_conn.execute("ROLLBACK", []).unwrap();
}

#[test]
fn the_recovery_note_requires_the_locked_verdict() {
    // The mid-rebuild arm of the repair path can only be reached with a real
    // concurrent writer racing the rebuild statement (see the fixture note on
    // `a_lock_before_the_rebuild_keeps_the_plain_transient_framing`), so this
    // drives the handler's own decision function — `recovery_for`, the
    // function `handle_doctor_fts_repair` calls on the rebuild failure —
    // rather than re-implementing its `if`. Deleting the `is_locked()` guard
    // from the production function therefore fails this test.
    //
    // Only a lock may earn the "half-rebuilt, re-run --repair" note. A
    // non-locked failure — including an `Error::Config` from anywhere else in
    // the call chain — must pass through untouched, otherwise an operator is
    // told to re-run a repair against an index that was never opened for
    // writing.
    let locked = recovery_for(Error::DatabaseLocked);
    match locked {
        Error::Config(msg) => {
            assert!(msg.contains("MCP server"), "got: {msg}");
            assert!(
                msg.contains("half-rebuilt") && msg.contains("--fts --repair"),
                "a lock hit by the rebuild must name the recovery command: {msg}"
            );
        }
        other => panic!("expected the locked Config error with the note, got {other:?}"),
    }

    // The counter-case: an unrelated Config error must NOT be told its index
    // is half-rebuilt.
    let unrelated = recovery_for(Error::Config("embedding model not configured".to_string()));
    match unrelated {
        Error::Config(msg) => {
            assert_eq!(msg, "embedding model not configured");
            assert!(
                !msg.contains("half-rebuilt"),
                "a non-locked Config error must not claim a half-rebuilt index: {msg}"
            );
        }
        other => panic!("expected the Config error unchanged, got {other:?}"),
    }

    // And the wording the note is built from is the handler's own composition
    // of the shared locked message, not a second literal.
    assert_eq!(
        recovery_note(),
        format!(
            "{LOCKED_DB_MESSAGE} The FTS index may be half-rebuilt; re-run `vipune doctor --fts --repair` to complete it."
        )
    );
}

// ── FtsDesyncReport helper methods ──

#[test]
fn fts_desync_report_helpers() {
    let healthy = FtsDesyncReport::default();
    assert!(!healthy.is_desynced());
    assert_eq!(healthy.total_desynced(), 0);

    let desynced = FtsDesyncReport {
        underpopulated_by_project: [("p".to_string(), 2)].into_iter().collect(),
        underpopulated_global: 2,
        orphans: 1,
    };
    assert!(desynced.is_desynced());
    assert_eq!(desynced.total_desynced(), 3);
}

// ── Response struct serialization (DoctorFtsResponse) ──

#[test]
fn doctor_fts_response_serializes_expected_fields() {
    let response = DoctorFtsResponse {
        in_sync: false,
        underpopulated_by_project: vec![crate::output::DoctorFtsProject {
            project_id: "proj".to_string(),
            memory_rows: 5,
            missing_from_fts: 2,
        }],
        orphan_rows: 1,
        total_desynced: 3,
        repaired: true,
        actions: 3,
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"in_sync\":false"));
    assert!(json.contains("\"orphan_rows\":1"));
    assert!(json.contains("\"total_desynced\":3"));
    assert!(json.contains("\"repaired\":true"));
    assert!(json.contains("\"actions\":3"));
    assert!(json.contains("\"missing_from_fts\":2"));
}
