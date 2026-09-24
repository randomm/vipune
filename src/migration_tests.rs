use super::*;
use crate::sqlite::identity;
use tempfile::TempDir;

/// A 384-dim L2-normalized vector (norm ≈ 1.0, classified as Real).
fn real_vec() -> Vec<f32> {
    let v = 1.0f32 / (384.0f32).sqrt();
    vec![v; 384]
}

/// Deterministic fake embedder: returns a 384-dim f32 vector derived
/// from the content so re-embedding the same content is idempotent.
/// Fails on content starting with "FAIL:" to simulate per-row failures.
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

/// Fake token counter: whitespace word count (conservative lower bound).
fn fake_token_count(content: &str) -> Result<usize, SqliteError> {
    Ok(content.split_whitespace().count())
}

fn test_config(db_path: &Path, model: &str) -> crate::config::Config {
    crate::config::Config {
        database_path: db_path.to_path_buf(),
        embedding_model: model.to_string(),
        ..Default::default()
    }
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
    let mut tc = |content: &str| fake_token_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc)
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
    let mut tc = |content: &str| fake_token_count(content);

    let result = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc);
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

#[test]
fn test_migrate_preflight_refusal_lists_offending_ids_and_writes_nothing() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // A row well over 512 words (hence well over 512 tokens).
    let long_content: String = (0..600)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let long_id = db
        .insert("proj", &long_content, &real_vec(), None, "fact", "active")
        .unwrap();
    let short_id = db
        .insert("proj", "short content", &real_vec(), None, "fact", "active")
        .unwrap();

    let blob_before_long = get_embedding_blob(&db, &long_id);
    let blob_before_short = get_embedding_blob(&db, &short_id);

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut tc = |content: &str| fake_token_count(content);

    let result = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc);
    let err = result.expect_err("pre-flight must refuse");

    match err {
        SqliteError::MigrationRefused { offending } => {
            assert_eq!(offending.len(), 1, "only the long row should be reported");
            assert_eq!(offending[0], long_id);
        }
        other => panic!("expected MigrationRefused, got {other:?}"),
    }

    // No marker written.
    assert_eq!(
        identity::read_marker(db.conn()).unwrap(),
        None,
        "no marker on pre-flight refusal"
    );
    // BLOBs byte-identical.
    assert_eq!(get_embedding_blob(&db, &long_id), blob_before_long);
    assert_eq!(get_embedding_blob(&db, &short_id), blob_before_short);
}

#[test]
fn test_migrate_per_row_failure_keeps_marker_and_old_identity() {
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // One good row, one row that will fail.
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

    // Record an old identity first (to verify it's preserved).
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
    let mut tc = |content: &str| fake_token_count(content);

    let result = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc);
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
    let mut tc = |content: &str| fake_token_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc)
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
    let mut tc = |content: &str| fake_token_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc)
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
    // A fresh store (no model_identity row at all) must be migratable.
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();
    db.insert("proj", "content", &real_vec(), None, "fact", "active")
        .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut tc = |content: &str| fake_token_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc)
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
    // The bge profile has no prefix; the fake token count (word count)
    // should still work correctly.
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
    let mut tc = |content: &str| fake_token_count(content);

    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc)
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
fn test_preflight_exactly_at_limit_is_allowed() {
    // A row with exactly MAX_EMBEDDING_TOKENS "tokens" (word count)
    // must be allowed (the check is strict >, not >=).
    let (_dir, db_path) = make_db_dir();
    let mut db = Database::open(&db_path).unwrap();

    // Exactly 512 words.
    let exact_content: String = (0..512)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    db.insert("proj", &exact_content, &real_vec(), None, "fact", "active")
        .unwrap();

    let config = test_config(&db_path, "intfloat/multilingual-e5-small");
    let mut embed = |content: &str| fake_embed(content);
    let mut tc = |content: &str| fake_token_count(content);

    // Must succeed (512 is not > 512).
    let report = migrate_model_with_embedder(&mut db, &config, &mut embed, &mut tc)
        .expect("exactly-at-limit row must be allowed");
    assert_eq!(report.reindexed, 1);
}

#[test]
fn test_migrate_busy_database_fails_fast() {
    // Open a second connection that holds a write lock, then attempt
    // to migrate — the busy_timeout is 0 so it must fail immediately.
    let (_dir, db_path) = make_db_dir();

    // Hold a write lock on the database via a raw connection.
    let locker = std::rc::Rc::new(std::cell::RefCell::new({
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("BEGIN EXCLUSIVE", []).unwrap();
        conn
    }));

    // Database::open will fail because the file is locked by the
    // exclusive transaction above.
    let db = Database::open(&db_path);
    assert!(
        db.is_err(),
        "opening a locked database with busy_timeout=0 must fail"
    );

    // Drop the locker to release the lock (cleanup).
    drop(locker);
}
