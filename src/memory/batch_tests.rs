//! Batch ingest tests for the memory store.

use super::*;
use crate::config::Config;
use crate::memory_types::{BatchIngestItemResult, IngestPolicy};
use crate::sqlite::Database;

#[test]
fn test_batch_ingest_empty_batch() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    let result = store
        .batch_ingest("test-project", vec![], IngestPolicy::ConflictAware)
        .unwrap();

    assert_eq!(result.results.len(), 0);
}

#[test]
fn test_batch_ingest_mixed_outcomes() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    // First, add a memory that will cause conflicts
    // Use the same mock embedding function that add_with_conflict uses
    // so exact duplicates produce identical embeddings (deterministic)
    let embedding = super::crud::mock_embedding_for_content("Alice works at Microsoft");
    db.insert("test-project", "Alice works at Microsoft", &embedding, None)
        .unwrap();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    // Create a batch with mixed outcomes:
    // 0: empty -> Error
    // 1: valid unique -> Added
    // 2: conflict -> Conflicts
    // 3: valid unique -> Added
    let items = vec![
        ("", None),                         // Empty input -> Error
        ("Bob works at Google", None),      // Should add (unique)
        ("Alice works at Microsoft", None), // Should conflict with existing
        ("Charlie works at Amazon", None),  // Should add (unique)
    ];

    let result = store
        .batch_ingest("test-project", items.clone(), IngestPolicy::ConflictAware)
        .unwrap();

    // Verify we have 4 results (one per input)
    assert_eq!(result.results.len(), 4);

    // Check item 0: Error (empty input)
    match &result.results[0] {
        BatchIngestItemResult::Error { message } => {
            assert!(message.contains("empty") || message.contains("EmptyInput"));
        }
        _ => panic!("Expected Error for item 0, got {:?}", result.results[0]),
    }

    // Check item 1: Added (unique content)
    match &result.results[1] {
        BatchIngestItemResult::Added { id } => {
            assert!(!id.is_empty());
            // Verify it was actually stored
            let memory = store.get(id).unwrap().unwrap();
            assert_eq!(memory.content, "Bob works at Google");
        }
        _ => panic!("Expected Added for item 1, got {:?}", result.results[1]),
    }

    // Check item 2: Conflicts (similar to existing memory)
    match &result.results[2] {
        BatchIngestItemResult::Conflicts {
            proposed,
            conflicts,
        } => {
            assert_eq!(proposed, "Alice works at Microsoft");
            assert!(!conflicts.is_empty());
            assert!(conflicts.iter().any(|c| c.content.contains("Alice")));
        }
        _ => panic!("Expected Conflicts for item 2, got {:?}", result.results[2]),
    }

    // Check item 3: Added (unique content)
    match &result.results[3] {
        BatchIngestItemResult::Added { id } => {
            assert!(!id.is_empty());
            // Verify it was actually stored
            let memory = store.get(id).unwrap().unwrap();
            assert_eq!(memory.content, "Charlie works at Amazon");
        }
        _ => panic!("Expected Added for item 3, got {:?}", result.results[3]),
    }
}

#[test]
fn test_batch_ingest_deterministic_index_mapping() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    // Pre-populate with conflicting content
    // Use the same mock embedding function that add_with_conflict uses
    // so exact duplicates produce identical embeddings (deterministic)
    let embedding = super::crud::mock_embedding_for_content("conflict with item 1");
    db.insert("test-project", "conflict with item 1", &embedding, None)
        .unwrap();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    // Create batch with specific ordering
    let items = vec![
        ("unique item 0", None),
        ("conflict with item 1", None), // Will conflict
        ("", None),                     // Will error
        ("unique item 3", Some("metadata")),
    ];

    let result = store
        .batch_ingest("test-project", items.clone(), IngestPolicy::ConflictAware)
        .unwrap();

    assert_eq!(result.results.len(), 4);

    // Verify index 0 is Added (unique)
    assert!(matches!(
        &result.results[0],
        BatchIngestItemResult::Added { .. }
    ));

    // Verify index 1 is Conflicts (we pre-populated a conflict)
    assert!(matches!(
        &result.results[1],
        BatchIngestItemResult::Conflicts { .. }
    ));

    // Verify index 2 is Error (empty input)
    assert!(matches!(
        &result.results[2],
        BatchIngestItemResult::Error { .. }
    ));

    // Verify index 3 is Added (unique with metadata)
    if let BatchIngestItemResult::Added { id } = &result.results[3] {
        let memory = store.get(id).unwrap().unwrap();
        assert_eq!(memory.content, "unique item 3");
        assert_eq!(memory.metadata, Some("metadata".to_string()));
    } else {
        panic!("Expected Added for index 3");
    }
}

#[test]
fn test_batch_ingest_policy_force() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    // Pre-populate with conflicting content
    // Use the same mock embedding function for consistency
    let embedding = super::crud::mock_embedding_for_content("Alice works at Microsoft");
    db.insert("test-project", "Alice works at Microsoft", &embedding, None)
        .unwrap();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    // Create batch with conflicts
    let items = vec![
        ("Alice works at Microsoft", None), // Would conflict under ConflictAware
        ("Bob works at Google", None),      // Unique
    ];

    // With Force policy, should add everything (except empty/invalid)
    let result = store
        .batch_ingest("test-project", items.clone(), IngestPolicy::Force)
        .unwrap();

    assert_eq!(result.results.len(), 2);

    // Both should be Added (no conflicts with Force policy)
    match &result.results[0] {
        BatchIngestItemResult::Added { id } => {
            assert!(!id.is_empty());
            // Verify it was actually stored despite potential similarity
            let memory = store.get(id).unwrap().unwrap();
            assert_eq!(memory.content, "Alice works at Microsoft");
        }
        _ => panic!("Expected Added for item 0 with Force policy"),
    }

    match &result.results[1] {
        BatchIngestItemResult::Added { id } => {
            assert!(!id.is_empty());
            let memory = store.get(id).unwrap().unwrap();
            assert_eq!(memory.content, "Bob works at Google");
        }
        _ => panic!("Expected Added for item 1 with Force policy"),
    }
}

#[test]
fn test_batch_ingest_invalid_inputs() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    // Create oversized input (100,001 characters)
    let too_long = "a".repeat(100_001);

    // Create whitespace-only input
    let whitespace = "   \t\n  ";

    let items = vec![
        ("", None),            // Empty
        ("   ", None),         // Whitespace-only
        (&too_long, None),     // Too long
        ("valid input", None), // Valid
        (whitespace, None),    // Whitespace
    ];

    let result = store
        .batch_ingest("test-project", items.clone(), IngestPolicy::ConflictAware)
        .unwrap();

    assert_eq!(result.results.len(), 5);

    // All invalid inputs should return Error
    for &idx in &[0_usize, 1, 2, 4] {
        match &result.results[idx] {
            BatchIngestItemResult::Error { .. } => {
                // Expected error
            }
            other => panic!(
                "Expected Error for invalid input at index {}, got {:?}",
                idx, other
            ),
        }
    }

    // Only the valid input should succeed
    match &result.results[3] {
        BatchIngestItemResult::Added { id } => {
            assert!(!id.is_empty());
            let memory = store.get(id).unwrap().unwrap();
            assert_eq!(memory.content, "valid input");
        }
        _ => panic!("Expected Added for valid input at index 3"),
    }
}

#[test]
fn test_batch_ingest_all_errors() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    // Create a batch where every item fails validation
    let items = vec![
        ("", None),    // Empty
        ("   ", None), // Whitespace-only
    ];

    let result = store
        .batch_ingest("test-project", items.clone(), IngestPolicy::ConflictAware)
        .unwrap();

    assert_eq!(result.results.len(), 2);

    for (idx, item_result) in result.results.iter().enumerate() {
        match item_result {
            BatchIngestItemResult::Error { .. } => {
                // Expected error
            }
            _ => panic!("Expected Error for all items at index {}", idx),
        }
    }
}

#[test]
fn test_batch_ingest_metadata_preservation() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let config = Config::default();

    let mut store = MemoryStore::new_with_db(db, config).unwrap();

    let items = vec![
        ("content with metadata", Some(r#"{"tag": "important"}"#)),
        ("content without metadata", None),
        (
            "json metadata",
            Some(r#"{"source": "user", "priority": 1}"#),
        ),
    ];

    let result = store
        .batch_ingest("test-project", items.clone(), IngestPolicy::ConflictAware)
        .unwrap();

    assert_eq!(result.results.len(), 3);

    // Verify metadata is preserved for items 0 and 2
    if let BatchIngestItemResult::Added { id } = &result.results[0] {
        let memory = store.get(id).unwrap().unwrap();
        assert_eq!(memory.metadata, Some(r#"{"tag": "important"}"#.to_string()));
    } else {
        panic!("Expected Added for item 0");
    }

    if let BatchIngestItemResult::Added { id } = &result.results[1] {
        let memory = store.get(id).unwrap().unwrap();
        assert_eq!(memory.metadata, None);
    } else {
        panic!("Expected Added for item 1");
    }

    if let BatchIngestItemResult::Added { id } = &result.results[2] {
        let memory = store.get(id).unwrap().unwrap();
        assert_eq!(
            memory.metadata,
            Some(r#"{"source": "user", "priority": 1}"#.to_string())
        );
    } else {
        panic!("Expected Added for item 2");
    }
}
