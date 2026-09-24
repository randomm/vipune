//! Unit tests for `crate::sqlite::identity` (model identity, marker
//! semantics, mismatch refusal, NULL-identity reads).
//!
//! Wired from `identity.rs` via `#[path = "identity_tests.rs"]`.

use super::*;

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

/// A row with a NULL model_id (what the marker-first step of `reindex
/// --force` writes on a store that had no recorded identity) must read as
/// "no recorded identity" — the bge default — NOT as a bogus
/// ""@<revision> identity.
#[test]
fn test_null_model_id_row_reads_as_no_identity() {
    let conn = migrated_conn();
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, NULL, NULL, 'migrating to x@y')",
        [],
    )
    .unwrap();
    assert_eq!(
        read_identity(&conn).unwrap(),
        None,
        "NULL model_id means no recorded identity (bge default)"
    );
    assert_eq!(
        read_marker(&conn).unwrap().as_deref(),
        Some("migrating to x@y")
    );
    // The combined read agrees with the two single reads.
    let (identity, marker) = read_identity_and_marker(&conn).unwrap();
    assert_eq!(identity, None);
    assert_eq!(marker.as_deref(), Some("migrating to x@y"));
    // ...and the effective identity is the bge default.
    assert_eq!(
        current_identity(&conn).unwrap(),
        ModelIdentity::default_identity()
    );
}

/// Same NULL rule via the combined read on a fresh, row-less table.
#[test]
fn test_combined_read_no_row_is_none_none() {
    let conn = migrated_conn();
    assert_eq!(read_identity_and_marker(&conn).unwrap(), (None, None));
}

/// A row written by the marker-first step on a store that HAS a recorded
/// identity leaves that identity untouched (only the marker is staged): the
/// recorded identity stays old, the marker names the target.
#[test]
fn test_marker_only_write_leaves_recorded_identity_untouched() {
    let conn = migrated_conn();
    let old = ModelIdentity {
        model_id: "old-model".to_string(),
        revision: "old-rev".to_string(),
    };
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)",
        (&old.model_id, &old.revision),
    )
    .unwrap();
    // Simulate the marker-first write (only the marker column is touched).
    conn.execute(
        "UPDATE model_identity SET migration_marker = ?1 WHERE id = 1",
        ["migrating to new-model@new-rev"],
    )
    .unwrap();
    assert_eq!(
        read_identity(&conn).unwrap(),
        Some(old.clone()),
        "an interrupted migration must keep reporting the OLD identity"
    );
    assert_eq!(
        read_marker(&conn).unwrap().as_deref(),
        Some("migrating to new-model@new-rev"),
        "the marker must name the interrupted TARGET"
    );
    // Refusal while the marker is present names the target, not the old id.
    let err = assert_identity_ok(&conn, "old-model").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("migrating"), "{msg}");
    assert!(msg.contains("new-model@new-rev"), "{msg}");
}

/// The final record-and-clear step is the ONLY place the recorded identity
/// is replaced: marker present + old identity → new identity + no marker.
#[test]
fn test_record_identity_replaces_old_and_clears_marker() {
    let conn = migrated_conn();
    let old = ModelIdentity {
        model_id: "old-model".to_string(),
        revision: "old-rev".to_string(),
    };
    let new = e5_identity();
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, 'migrating to x@y')",
        (&old.model_id, &old.revision),
    )
    .unwrap();
    conn.execute(
        "UPDATE model_identity SET model_id = ?1, model_revision = ?2, migration_marker = NULL
         WHERE id = 1",
        (&new.model_id, &new.revision),
    )
    .unwrap();
    assert_eq!(read_identity(&conn).unwrap(), Some(new.clone()));
    assert_eq!(read_marker(&conn).unwrap(), None);
    assert_eq!(current_identity(&conn).unwrap(), new);
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

// --- assert_identity_ok (mismatch / marker refusal) ---

#[test]
fn test_identity_ok_on_fresh_store_with_default_model() {
    // No recorded row + default (bge) configured → Ok (zero-change
    // contract: existing stores behave exactly as before).
    let conn = migrated_conn();
    assert!(assert_identity_ok(&conn, EMBED_MODEL_ID).is_ok());
}

#[test]
fn test_identity_ok_when_only_a_null_identity_marker_row_exists_with_matching_model() {
    // A marker-only row (NULL identity) with the bge model configured: the
    // identity is the bge default and matches, but the marker still refuses.
    let conn = migrated_conn();
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, NULL, NULL, 'migrating to x@y')",
        [],
    )
    .unwrap();
    let err = assert_identity_ok(&conn, EMBED_MODEL_ID).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("migrating"), "{msg}");
    assert!(msg.contains("x@y"), "{msg}");
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
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)",
        (&recorded.model_id, &recorded.revision),
    )
    .unwrap();
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
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)",
        (&recorded.model_id, &recorded.revision),
    )
    .unwrap();
    assert!(assert_identity_ok(&conn, EMBED_MODEL_ID).is_err());
}

#[test]
fn test_identity_refuses_while_marker_present() {
    let conn = migrated_conn();
    let target = ModelIdentity {
        model_id: "e5-model".to_string(),
        revision: "rev-2".to_string(),
    };
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, NULL, NULL, ?1)",
        [format!("migrating to {}", target.display())],
    )
    .unwrap();
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
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)",
        (&id.model_id, &id.revision),
    )
    .unwrap();
    assert!(assert_identity_ok(&conn, EMBED_MODEL_ID).is_ok());
}

/// A failed (interrupted) migration: the marker is in place, the recorded
/// identity is still the OLD one, and the combined read reports both — the
/// store can never report the new model while most vectors are old.
#[test]
fn test_interrupted_migration_reports_old_identity_and_marker_target() {
    let conn = migrated_conn();
    let old = ModelIdentity::default_identity();
    let target = e5_identity();
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, ?3)",
        (
            &old.model_id,
            &old.revision,
            format!("migrating to {}", target.display()),
        ),
    )
    .unwrap();
    let (identity, marker) = read_identity_and_marker(&conn).unwrap();
    assert_eq!(identity, Some(old.clone()), "identity is still the old one");
    assert!(
        marker.as_deref().is_some_and(|m| m.contains(&target.display())),
        "marker names the target: {marker:?}"
    );
    assert_eq!(read_identity(&conn).unwrap(), Some(old));
    assert!(is_migrating(&conn).unwrap());
}
