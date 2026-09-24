//! Unit tests for `crate::sqlite::identity` (model identity, marker
//! lifecycle, mismatch refusal, `reindex --force` migration).
//!
//! Wired from `identity.rs` via `#[path = "identity_tests.rs"]`.

use super::*;
use crate::sqlite::Database;

fn migrated_conn() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE model_identity (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_id TEXT,
            model_revision TEXT,
            migration_marker TEXT
        );",
    )
    .unwrap();
    conn
}

fn e5_identity() -> ModelIdentity {
    ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    }
}

// --- read/write round-trips ---

#[test]
fn test_no_row_means_none() {
    let conn = migrated_conn();
    assert_eq!(read_identity(&conn).unwrap(), None);
    assert_eq!(read_marker(&conn).unwrap(), None);
}

#[test]
fn test_read_identity_roundtrip() {
    let conn = migrated_conn();
    let identity = e5_identity();
    record_identity_and_clear_marker(&conn, &identity).unwrap();
    assert_eq!(read_identity(&conn).unwrap(), Some(identity));
    assert_eq!(read_marker(&conn).unwrap(), None);
}

#[test]
fn test_write_marker_persists_marker_and_target() {
    let conn = migrated_conn();
    let target = e5_identity();
    write_marker(&conn, &target).unwrap();
    let marker = read_marker(&conn).unwrap();
    assert_eq!(
        marker.as_deref(),
        Some(
            "migrating to intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
        )
    );
    // The marker row stages the interrupted target identity.
    let staged = read_identity(&conn).unwrap();
    assert_eq!(staged, Some(target));
}

#[test]
fn test_record_identity_clears_marker() {
    let conn = migrated_conn();
    let target = e5_identity();
    // Crash-recovery: a mid-run marker is cleared by the final
    // record-and-clear step.
    write_marker(&conn, &target).unwrap();
    record_identity_and_clear_marker(&conn, &target).unwrap();
    assert_eq!(read_identity(&conn).unwrap(), Some(target));
    assert_eq!(read_marker(&conn).unwrap(), None);
}

#[test]
fn test_marker_overwrites_existing_marker() {
    let conn = migrated_conn();
    let first = ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "rev-a".to_string(),
    };
    let second = ModelIdentity {
        model_id: "BAAI/bge-small-en-v1.5".to_string(),
        revision: "rev-b".to_string(),
    };
    write_marker(&conn, &first).unwrap();
    write_marker(&conn, &second).unwrap();
    let marker = read_marker(&conn).unwrap();
    assert_eq!(
        marker.as_deref(),
        Some("migrating to BAAI/bge-small-en-v1.5@rev-b")
    );
    // Still a single row.
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM model_identity", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

// --- identity resolution ---

#[test]
fn test_default_identity_matches_embed_constants() {
    let default = ModelIdentity::default_identity();
    assert_eq!(default.model_id, EMBED_MODEL_ID);
    assert_eq!(default.revision, EMBED_MODEL_REVISION);
    assert_eq!(
        default.display(),
        "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    );
}

#[test]
fn test_configured_identity_resolves_builtin_profiles() {
    let bge = configured_identity(EMBED_MODEL_ID);
    assert_eq!(bge, ModelIdentity::default_identity());
    let e5 = configured_identity("intfloat/multilingual-e5-small");
    assert_eq!(e5.model_id, "intfloat/multilingual-e5-small");
    assert_eq!(e5.revision, "614241f622f53c4eeff9890bdc4f31cfecc418b3");
    // Unknown id: id kept, revision empty (config validation rejects
    // unknown ids before this is ever reached in practice).
    let unknown = configured_identity("no/such-model");
    assert_eq!(unknown.model_id, "no/such-model");
    assert_eq!(unknown.revision, "");
}

#[test]
fn test_record_identity_overwrites_previous_identity() {
    let conn = migrated_conn();
    let first = ModelIdentity::default_identity();
    record_identity_and_clear_marker(&conn, &first).unwrap();
    let second = e5_identity();
    record_identity_and_clear_marker(&conn, &second).unwrap();
    assert_eq!(read_identity(&conn).unwrap(), Some(second));
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM model_identity", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

// --- assert_identity_ok (mismatch / marker refusal) ---

#[test]
fn test_identity_ok_on_fresh_store_with_default_model() {
    // No recorded row + default (bge) configured → Ok (zero-change
    // contract: existing stores behave exactly as before).
    let conn = migrated_conn();
    assert!(assert_identity_ok(&conn, EMBED_MODEL_ID).is_ok());
}

#[test]
fn test_identity_refuses_unrecorded_store_with_nondefault_model() {
    // A not-yet-recorded (bge-default) store with e5 configured is a
    // mismatch: effective identity bge ≠ configured e5.
    let conn = migrated_conn();
    let err = assert_identity_ok(&conn, "intfloat/multilingual-e5-small").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("model identity mismatch"), "{msg}");
    assert!(msg.contains("intfloat/multilingual-e5-small"), "{msg}");
    assert!(msg.contains("vipune reindex --force"), "{msg}");
}

#[test]
fn test_identity_refuses_on_recorded_mismatch() {
    let conn = migrated_conn();
    let recorded = ModelIdentity {
        model_id: "other-model".to_string(),
        revision: "rev-1".to_string(),
    };
    record_identity_and_clear_marker(&conn, &recorded).unwrap();
    let err = assert_identity_ok(&conn, EMBED_MODEL_ID).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("other-model@rev-1"), "{msg}");
    assert!(msg.contains("BAAI/bge-small-en-v1.5"), "{msg}");
}

#[test]
fn test_identity_refuses_revision_change_same_model_id() {
    // Name-only identity is insufficient: same id, different revision.
    let conn = migrated_conn();
    let recorded = ModelIdentity {
        model_id: EMBED_MODEL_ID.to_string(),
        revision: "some-other-revision".to_string(),
    };
    record_identity_and_clear_marker(&conn, &recorded).unwrap();
    assert!(assert_identity_ok(&conn, EMBED_MODEL_ID).is_err());
}

#[test]
fn test_identity_refuses_while_marker_present() {
    let conn = migrated_conn();
    let target = ModelIdentity {
        model_id: "e5-model".to_string(),
        revision: "rev-2".to_string(),
    };
    write_marker(&conn, &target).unwrap();
    let err = assert_identity_ok(&conn, EMBED_MODEL_ID).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("migrating"), "{msg}");
    assert!(msg.contains("e5-model@rev-2"), "{msg}");
    assert!(msg.contains("vipune reindex --force"), "{msg}");
}

#[test]
fn test_identity_ok_after_matching_identity_recorded() {
    let conn = migrated_conn();
    let id = ModelIdentity::default_identity();
    record_identity_and_clear_marker(&conn, &id).unwrap();
    assert!(assert_identity_ok(&conn, EMBED_MODEL_ID).is_ok());
}

// --- force_migrate_database (marker/identity lifecycle) ---

/// Temp-path database with one project and one row, plus a target identity.
fn migrate_db_with_row() -> (tempfile::TempDir, Database, ModelIdentity) {
    let dir = tempfile::TempDir::new().unwrap();
    let db = Database::open(&dir.path().join("test.db")).unwrap();
    db.insert(
        "proj",
        "test content",
        &vec![0.5; 384],
        None,
        "fact",
        "active",
    )
    .unwrap();
    let target = ModelIdentity {
        model_id: "test-model".to_string(),
        revision: "test-rev".to_string(),
    };
    (dir, db, target)
}

/// The full success path; also keeps every intermediate function visible
/// to the lib target's dead-code analysis.
#[test]
fn test_force_migrate_database_full_cycle() {
    let (_dir, db, target) = migrate_db_with_row();
    let (reindexed, skipped, failed) =
        force_migrate_database(&db, &target, &["proj".to_string()], |_| Ok(vec![0.1; 384]))
            .unwrap();
    assert_eq!(reindexed, 1);
    assert_eq!(skipped, 0);
    assert!(failed.is_empty());

    // Identity recorded, marker cleared.
    assert_eq!(current_identity(db.conn()).unwrap(), target);
    assert!(!is_migrating(db.conn()).unwrap());
}

/// A failed pass leaves the marker in place and never records the new
/// identity through `record_identity_and_clear_marker` — so the store can
/// never be left with a cleanly recorded identity while some project still
/// holds old-model vectors (the silently mixed store the feature prevents).
#[test]
fn test_force_migrate_database_keeps_marker_records_nothing_on_failure() {
    let (_dir, db, target) = migrate_db_with_row();
    let result = force_migrate_database(&db, &target, &["proj".to_string()], |_content| {
        Err(Error::Sqlite("boom".to_string()))
    });
    assert!(result.is_err(), "a failed pass must not return Ok");
    // Marker stays, so operations refuse.
    assert!(is_migrating(db.conn()).unwrap());
    // The marker row stages the interrupted TARGET identity (written once by
    // write_marker before the pass); the final identity is written only by
    // record_identity_and_clear_marker, which a failed pass never reaches —
    // and a marker-present store always refuses, so the state is
    // unambiguous.
    let staged = read_identity(db.conn()).unwrap();
    assert_eq!(staged, Some(target.clone()), "marker row stages the target");
}
