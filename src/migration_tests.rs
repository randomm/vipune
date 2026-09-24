//! In-crate fake-embedder tests for the migration core (issue #221).
//!
//! All tests use fake closures (no model download, no real engine). The
//! embedder is deterministic: it returns a 384-dim L2-normalised vector
//! derived from the content, so re-running the pass on the same content is
//! idempotent at the vector level. It fails on content starting with "FAIL:"
//! to simulate per-row failures. The token counter is a word counter, so any
//! row with >512 words is flagged by the pre-flight (a lower bound on the
//! real token count).

use super::*;
use crate::sqlite::identity;
use tempfile::TempDir;

/// A 384-dim L2-normalized vector (norm ≈ 1.0, classified as Real).
fn real_vec() -> Vec<f32> {
    let v = 1.0f32 / (384.0f32).sqrt();
    vec![v; 384]
}

/// Deterministic fake embedder: returns a 384-dim f32 vector derived from
/// the content so re-embedding the same content is idempotent. Fails on
/// content starting with "FAIL:" to simulate per-row failures.
fn fake_embed(content: &str) -> Result<Vec<f32>, SqliteError> {
    if content.starts_with("FAIL:") {
        return Err(SqliteError::Sqlite("simulated embed failure".to_string()));
    }
    let mut vec = vec![0.0f32; 384];
    for (i, b) in content.bytes().enumerate() {
        vec[i % 384] += b as f32 / 255.0;
    }
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    Ok(vec)
}

/// Stub token counter: counts whitespace-separated tokens (a lower bound on
/// the real token count, so the >512 threshold is conservative).
fn word_count(content: &str) -> Result<usize, SqliteError> {
    Ok(content.split_whitespace().count())
}

fn test_config(db_path: &Path, model: &str) -> crate::config::Config {
    // `Config::load()` resolves the full default config (including the
    // home-dir database path). The migration only reads `embedding_model`
    // and `database_path` here; the other fields are carried through as-is.
    let mut config = crate::config::Config::load().unwrap();
    config.database_path = db_path.to_path_buf();
    config.embedding_model = model.to_string();
    config
}

fn make_db_dir() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn get_embedding_blob(db: &Database, id: &str) -> Vec<u8> {
    db.conn()
        .query_row(
            "SELECT embedding FROM memories WHERE id = ?1",
            [id],
            |row| row.get(0),
        )
        .unwrap()
}

#[test]
fn test_migrate_success_records_identity_clears_marker() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // Insert rows across two projects.
    db.insert(
        "proj_a",
        "alpha content",
        &real_vec(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert(
        "proj_b",
        "beta content",
        &real_vec(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count)
        .expect("migration must succeed");

    assert_eq!(report.reindexed, 2, "both rows should be reindexed");
    assert_eq!(report.skipped, 0);
    assert!(report.failures.is_empty());

    // Marker must be cleared, new identity must be recorded.
    assert_eq!(
        identity::read_marker(db.conn()).unwrap(),
        None,
        "marker must be cleared after a clean pass"
    );
    let target = ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };
    assert_eq!(
        identity::read_identity(db.conn()).unwrap(),
        Some(target),
        "new identity must be recorded"
    );
}

#[test]
fn test_migrate_unknown_model_fails_before_any_write() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();
    db.insert("proj", "content", &real_vec(), None, "fact", "active")
        .unwrap();

    let config = test_config(&db_path, "not/a-real-model");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let result = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count);
    assert!(result.is_err(), "unknown model must fail");

    // No marker written, no identity recorded.
    assert_eq!(
        identity::read_marker(db.conn()).unwrap(),
        None,
        "no marker on unknown model"
    );
    assert!(
        identity::read_identity(db.conn()).unwrap().is_none(),
        "no identity on unknown model"
    );
}

/// Pre-flight refusal (issue #221): a row over the token limit yields
/// `MigrationRefused` listing the offending id, and the call writes nothing
/// — no marker, no row changes (embedding BLOBs byte-identical).
#[test]
fn test_migrate_preflight_refusal_lists_offending_id_and_writes_nothing() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // A row well over 512 words (hence over the stub counter's 512 limit),
    // plus a short row.
    let long_content: String = (0..700)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let long_id = db
        .insert("proj", &long_content, &real_vec(), None, "fact", "active")
        .unwrap();
    let short_id = db
        .insert("proj", "short content", &real_vec(), None, "fact", "active")
        .unwrap();

    let long_blob_before = get_embedding_blob(&db, &long_id);
    let short_blob_before = get_embedding_blob(&db, &short_id);

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let err = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count)
        .expect_err("pre-flight must refuse when a row is over the limit");
    match err {
        SqliteError::MigrationRefused { offending } => {
            assert_eq!(
                offending,
                vec![long_id.clone()],
                "only the long row should be reported"
            );
        }
        other => panic!("expected MigrationRefused, got {other:?}"),
    }

    // No marker written; BLOBs byte-identical (nothing was written).
    assert_eq!(
        identity::read_marker(db.conn()).unwrap(),
        None,
        "no marker on pre-flight refusal"
    );
    assert_eq!(get_embedding_blob(&db, &long_id), long_blob_before);
    assert_eq!(get_embedding_blob(&db, &short_id), short_blob_before);
}

#[test]
fn test_migrate_per_row_failure_keeps_marker_and_old_identity() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // One good row, one row that will fail to embed.
    db.insert("proj", "good content", &real_vec(), None, "fact", "active")
        .unwrap();
    let fail_id = db
        .insert(
            "proj",
            "FAIL:bad content",
            &real_vec(),
            None,
            "fact",
            "active",
        )
        .unwrap();

    // Record an old identity first (to verify it is preserved).
    let old_identity = ModelIdentity {
        model_id: "BAAI/bge-small-en-v1.5".to_string(),
        revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a".to_string(),
    };
    db.conn()
        .execute(
            "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
             VALUES (1, ?1, ?2, NULL)",
            (&old_identity.model_id, &old_identity.revision),
        )
        .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let result = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count);
    let err = result.expect_err("must fail when a row fails to embed");

    match err {
        SqliteError::MigrationIncomplete { report } => {
            assert_eq!(report.reindexed, 1, "good row was reindexed");
            assert_eq!(report.failures.len(), 1, "one failure expected");
            assert_eq!(report.failures[0].id, fail_id);
        }
        other => panic!("expected MigrationIncomplete, got {other:?}"),
    }

    // Marker must be present (migration left in-flight).
    assert!(
        identity::read_marker(db.conn()).unwrap().is_some(),
        "marker must be kept after a failed pass"
    );
    // Old identity must be unchanged.
    assert_eq!(
        identity::read_identity(db.conn()).unwrap(),
        Some(old_identity),
        "old identity must be preserved on failure"
    );
}

#[test]
fn test_migrate_rerun_after_interruption_performs_full_pass() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // Three rows.
    for i in 0..3 {
        db.insert(
            "proj",
            &format!("content {i}"),
            &real_vec(),
            None,
            "fact",
            "active",
        )
        .unwrap();
    }

    // Simulate an interrupted migration: marker present.
    let target = ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };
    db.conn()
        .execute(
            "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
             VALUES (1, NULL, NULL, ?1)
             ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
            [migration_marker_for(&target)],
        )
        .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count)
        .expect("re-run must succeed");

    // Full pass: all 3 rows re-embedded (not just the unprocessed remainder).
    assert_eq!(report.reindexed, 3, "full pass must re-embed all rows");
    assert_eq!(report.skipped, 0);
    assert!(report.failures.is_empty());

    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
}

#[test]
fn test_migrate_skips_corrupted_rows_and_still_clears_marker() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // One good row + one corrupted (zero-vector) row.
    db.insert("proj", "good content", &real_vec(), None, "fact", "active")
        .unwrap();
    let corrupted_id = db
        .insert("proj", "corrupted", &vec![0.0; 384], None, "fact", "active")
        .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count)
        .expect("migration must succeed even with corrupted rows");

    assert_eq!(report.reindexed, 1);
    assert_eq!(
        report.skipped, 1,
        "corrupted row must be counted in skipped"
    );
    assert!(report.failures.is_empty());

    // Marker cleared, identity recorded (despite the corrupted row).
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
    // The corrupted row's BLOB must be untouched.
    let blob = get_embedding_blob(&db, &corrupted_id);
    assert_eq!(
        classify_embedding(&sqlite::blob_to_vec(&blob).unwrap()),
        EmbeddingClass::Unknown,
        "corrupted row must be left untouched"
    );
}

#[test]
fn test_migrate_fresh_store_with_no_identity_row() {
    // A fresh store (no model_identity row at all) must be migratable: the
    // marker write inserts a row with NULL model_id + marker, and the final
    // record replaces it.
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();
    db.insert("proj", "content", &real_vec(), None, "fact", "active")
        .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count)
        .expect("migration on a fresh store must succeed");
    assert_eq!(report.reindexed, 1);

    let target = ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
}

#[test]
fn test_migrate_bge_default_profile_empty_prefix() {
    // The bge profile has an empty passage prefix; the pass must still
    // succeed and record the bge identity.
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "normal content here",
        &real_vec(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let config = test_config(&db_path, "BAAI/bge-small-en-v1.5");
    let mut embed = |content: &str| fake_embed(content);
    let mut count = |content: &str| word_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count)
        .expect("bge migration must succeed");
    assert_eq!(report.reindexed, 1);

    let target = ModelIdentity {
        model_id: "BAAI/bge-small-en-v1.5".to_string(),
        revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a".to_string(),
    };
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
}

#[test]
fn test_migration_refused_error_variant_carries_offending_ids() {
    let offending = vec!["id-1".to_string(), "id-2".to_string()];
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
fn test_migration_incomplete_error_variant_carries_report() {
    let report = MigrationReport {
        reindexed: 5,
        skipped: 1,
        failures: vec![MigrationRowFailure {
            id: "row-1".to_string(),
            error: "simulated".to_string(),
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
fn test_migration_marker_names_target() {
    let target = ModelIdentity {
        model_id: "m".to_string(),
        revision: "r".to_string(),
    };
    assert_eq!(migration_marker_for(&target), "migrating to m@r");
}

#[test]
fn test_migrate_busy_database_fails_fast() {
    // Open a connection that holds a write lock, then attempt to open the
    // database for migration — with the busy timeout at zero, the open must
    // fail immediately ("database is locked") rather than waiting.
    let (_dir, db_path) = make_db_dir();

    // Hold a write lock on the database via a raw connection.
    let locker = std::rc::Rc::new(std::cell::RefCell::new({
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("BEGIN EXCLUSIVE", []).unwrap();
        conn
    }));

    // The open must fail because the file is locked and busy_timeout=0.
    let db = Database::open(&db_path);
    assert!(
        db.is_err(),
        "opening a locked database with busy_timeout=0 must fail fast"
    );

    // Drop the locker to release the lock (cleanup).
    drop(locker);
}
