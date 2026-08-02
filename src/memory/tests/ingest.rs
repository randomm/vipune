//! Tests for ingest operations.

use crate::config::Config;
use crate::memory::MemoryStore;
use crate::memory_types::{AddResult, IngestPolicy};

#[test]
fn test_ingest_invalid_input_empty() {
    let mut store = MemoryStore::test_store();

    // Empty content should error
    let result = store.ingest("test-project", "", None, IngestPolicy::Force);
    assert!(result.is_err());
}

#[ignore]
#[test]
fn test_ingest_happy_path_conflict_aware() {
    // Full integration test with real model
    // Requires: cargo test -- --ignored
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).unwrap();

    // Ingest with ConflictAware policy when no conflicts exist
    let result = store
        .ingest(
            "test-project",
            "new memory",
            None,
            IngestPolicy::ConflictAware,
        )
        .unwrap();

    match result {
        AddResult::Added { id } => {
            assert!(!id.is_empty());
            // Verify memory was actually stored
            let memory = store.get(&id, "test-project").unwrap().unwrap();
            assert_eq!(memory.content, "new memory");
        }
        AddResult::Conflicts { .. } => panic!("Expected AddResult::Added"),
    }
}

#[ignore]
#[test]
fn test_ingest_conflic_aware_rejects_duplicates() {
    // Full integration test with real model
    // Requires: cargo test -- --ignored
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");

    let config = Config {
        similarity_threshold: 0.9,
        ..Config::default()
    };

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).unwrap();

    let content = "Alice works at Microsoft";

    // Add first memory
    let result1 = store
        .ingest("test-project", content, None, IngestPolicy::ConflictAware)
        .unwrap();
    assert!(matches!(result1, AddResult::Added { .. }));

    // Try to add nearly identical content - should detect conflict
    let result2 = store
        .ingest("test-project", content, None, IngestPolicy::ConflictAware)
        .unwrap();

    match result2 {
        AddResult::Conflicts {
            proposed,
            conflicts,
        } => {
            assert_eq!(proposed, content);
            assert!(!conflicts.is_empty());
        }
        AddResult::Added { .. } => panic!("Expected AddResult::Conflicts"),
    }
}

#[ignore]
#[test]
fn test_ingest_force_bypasses_conflicts() {
    // Full integration test with real model
    // Requires: cargo test -- --ignored
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).unwrap();

    let content = "duplicate content";

    // Add first memory
    let result1 = store
        .ingest("test-project", content, None, IngestPolicy::ConflictAware)
        .unwrap();
    assert!(matches!(result1, AddResult::Added { .. }));

    // Force add duplicate - should succeed despite conflict
    let result2 = store
        .ingest("test-project", content, None, IngestPolicy::Force)
        .unwrap();

    match result2 {
        AddResult::Added { id } => {
            assert!(!id.is_empty());
            // Verify both memories exist
            let memories = store.list("test-project", 10, None, None).unwrap();
            assert_eq!(memories.len(), 2);
        }
        AddResult::Conflicts { .. } => panic!("Expected AddResult::Added with Force policy"),
    }
}

#[ignore]
#[test]
fn test_ingest_with_metadata() {
    // Full integration test with real model
    // Requires: cargo test -- --ignored
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).unwrap();

    let metadata = "{\"source\": \"test\"}";

    let result = store
        .ingest(
            "test-project",
            "memory with metadata",
            Some(metadata),
            IngestPolicy::Force,
        )
        .unwrap();

    match result {
        AddResult::Added { id } => {
            let memory = store.get(&id, "test-project").unwrap().unwrap();
            assert_eq!(memory.metadata, Some(metadata.to_string()));
        }
        AddResult::Conflicts { .. } => panic!("Expected AddResult::Added"),
    }
}
