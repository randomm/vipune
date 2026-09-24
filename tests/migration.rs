//! Library integration tests for the `vipune::migrate_model` public API
//! (issue #221).
//!
//! Thin public-API tests only: type shape, error-variant conversion, and
//! function-path resolution. The full end-to-end migration lifecycle
//! (marker, identity, pre-flight, re-run) is covered in-crate by the
//! fake-embedder tests in `src/migration_tests.rs`.

use vipune::{Config, Database, Error, MigrationReport, MigrationRowFailure, ModelIdentity};

#[test]
fn migrate_model_function_is_publicly_accessible() {
    // Verify the public entry point resolves with the documented signature:
    // `fn(&Path, &Config) -> Result<MigrationReport, Error>`.
    let _f: fn(&std::path::Path, &Config) -> std::result::Result<MigrationReport, Error> =
        vipune::migrate_model;
}

#[test]
fn migration_report_and_row_failure_match_documented_shape() {
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

    assert_eq!(
        db.conn()
            .query_row::<Option<String>, _, _>(
                "SELECT migration_marker FROM model_identity WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap(),
        Some(marker_text)
    );
    assert!(
        vipune::is_migrating(db.conn()).unwrap(),
        "is_migrating must be true while the marker is present"
    );
    assert!(
        vipune::read_identity(db.conn()).unwrap().is_none(),
        "marker-only row must not report an identity"
    );
}
