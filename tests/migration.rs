//! Library integration tests for `vipune::migrate_model` (issue #221).
//!
//! These tests exercise the **public** API types and error conversions.
//! The full end-to-end migration path is tested in the in-crate tests in
//! `src/migration.rs` (which have access to the crate-private
//! `migrate_model_with_embedder` and can use fake embedders without a
//! model download).

use vipune::{
    Config, Database, Error, MigrationReport, MigrationRowFailure, ModelIdentity, SqliteError,
    is_migrating, read_identity, read_marker,
};

#[test]
fn migration_report_fields_match_documented_shape() {
    let report = MigrationReport {
        reindexed: 42,
        skipped: 1,
        failures: vec![MigrationRowFailure {
            id: "mem-abc".to_string(),
            error: "embed failed".to_string(),
        }],
    };
    assert_eq!(report.reindexed, 42);
    assert_eq!(report.skipped, 1);
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].id, "mem-abc");
    assert_eq!(report.failures[0].error, "embed failed");
}

#[test]
fn migration_refused_error_is_public_and_carries_offending() {
    let offending = vec!["row-1".to_string(), "row-2".to_string()];
    let err: Error = SqliteError::MigrationRefused {
        offending: offending.clone(),
    }
    .into();
    match err {
        Error::MigrationRefused { offending: ids } => {
            assert_eq!(ids, offending);
        }
        other => panic!("expected MigrationRefused, got {other:?}"),
    }
}

#[test]
fn migration_incomplete_error_is_public_and_carries_report() {
    let report = MigrationReport {
        reindexed: 10,
        skipped: 2,
        failures: vec![MigrationRowFailure {
            id: "row-9".to_string(),
            error: "token limit exceeded".to_string(),
        }],
    };
    let err: Error = SqliteError::MigrationIncomplete {
        report: report.clone(),
    }
    .into();
    match err {
        Error::MigrationIncomplete { report: r } => {
            assert_eq!(r, report);
        }
        other => panic!("expected MigrationIncomplete, got {other:?}"),
    }
}

#[test]
fn migrate_model_function_is_publicly_accessible() {
    // Verify the function signature compiles — we don't call it (no model
    // download in integration tests), but the path must resolve.
    let _f: fn(&std::path::Path, &Config) -> Result<MigrationReport, Error> = vipune::migrate_model;
}

#[test]
fn migration_marker_and_identity_are_readable_after_marker_write() {
    // Verify the public identity API: after a marker-only write (simulated
    // here with raw SQL), the marker is visible and the identity is None
    // (bge default applies).
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let db = Database::open(&db_path).unwrap();

    // Insert a memory row so the DB is non-empty.
    db.insert("proj", "content", &vec![0.1; 384], None, "fact", "active")
        .unwrap();

    // Write a marker (as the library's write_migration_marker does).
    let target = ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };
    let marker_text = format!("migrating to {}", target.display());
    db.conn()
        .execute(
            "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
             VALUES (1, NULL, NULL, ?1)
             ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
            [&marker_text],
        )
        .unwrap();

    assert_eq!(read_marker(db.conn()).unwrap(), Some(marker_text));
    assert!(
        is_migrating(db.conn()).unwrap(),
        "is_migrating must be true while the marker is present"
    );
    assert!(
        read_identity(db.conn()).unwrap().is_none(),
        "marker-only row must not report an identity"
    );
}

#[test]
fn bge_default_profile_has_no_prefix() {
    let profile = vipune::profile_for("BAAI/bge-small-en-v1.5").unwrap();
    assert_eq!(profile.passage_prefix, "");
    assert_eq!(profile.query_prefix, "");
}

#[test]
fn e5_profile_has_passage_prefix() {
    let profile = vipune::profile_for("intfloat/multilingual-e5-small").unwrap();
    assert_eq!(profile.passage_prefix, "passage: ");
    assert_eq!(profile.query_prefix, "query: ");
}
