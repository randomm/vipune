//! Integration tests for batch ingest API.

use std::env;

use vipune::{BatchIngestItemResult, Config, IngestPolicy, MemoryStore};

/// Test batch_ingest with Force policy.
#[test]
fn test_batch_ingest_with_force_policy_adds_all_items() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let items = vec![
        ("First memory", None),
        ("Second memory", Some(r#"{"tag": "important"}"#)),
        ("Third memory", None),
    ];

    let result = store
        .batch_ingest("test-project", items, IngestPolicy::Force)
        .expect("Batch ingest should succeed");

    assert_eq!(result.results.len(), 3);
    assert!(matches!(
        &result.results[0],
        BatchIngestItemResult::Added { .. }
    ));
    assert!(matches!(
        &result.results[1],
        BatchIngestItemResult::Added { .. }
    ));
    assert!(matches!(
        &result.results[2],
        BatchIngestItemResult::Added { .. }
    ));

    std::fs::remove_file(db_path).ok();
}
