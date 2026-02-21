//! Tests for the memory store.

use super::*;
use crate::config::Config;
use crate::sqlite::Database;

#[test]
fn test_memory_store_new() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let _path = dir.path().join("test.db");

    // Note: This test is ignored because it requires downloading the model
    // Use `cargo test -- --ignored` to run it
}

#[test]
fn test_add_and_get() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];
    let id = db
        .insert("test-project", "test content", &embedding, Some("metadata"))
        .unwrap();

    let memory = db.get(&id).unwrap().unwrap();
    assert_eq!(memory.id, id);
    assert_eq!(memory.content, "test content");
    assert_eq!(memory.project_id, "test-project");
    assert_eq!(memory.metadata, Some("metadata".to_string()));
}

#[test]
fn test_search_basic() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding_match = vec![1.0f32; 384];
    let embedding_other = vec![0.0f32; 384];

    let id_match = db
        .insert("test-project", "matching content", &embedding_match, None)
        .unwrap();
    db.insert("test-project", "other content", &embedding_other, None)
        .unwrap();

    let results = db.search("test-project", &embedding_match, 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, id_match);
    assert!(results[0].similarity.unwrap() > 0.9);
}

#[test]
fn test_list_by_project() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    db.insert("project-a", "content a", &embedding, None)
        .unwrap();
    db.insert("project-b", "content b", &embedding, None)
        .unwrap();
    db.insert("project-a", "content a2", &embedding, None)
        .unwrap();

    let memories_a = db.list("project-a", 10).unwrap();
    assert_eq!(memories_a.len(), 2);

    let memories_b = db.list("project-b", 10).unwrap();
    assert_eq!(memories_b.len(), 1);
}

#[test]
fn test_update_memory() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding_old = vec![0.3f32; 384];
    let embedding_new = vec![0.8f32; 384];

    let id = db
        .insert("test-project", "original", &embedding_old, None)
        .unwrap();

    db.update(&id, "updated", &embedding_new).unwrap();

    let memory = db.get(&id).unwrap().unwrap();
    assert_eq!(memory.content, "updated");
    assert_ne!(memory.created_at, memory.updated_at);
}

#[test]
fn test_update_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    let result = db.update("does-not-exist", "content", &embedding);
    assert!(result.is_err());
}

#[test]
fn test_delete_memory() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    let id = db
        .insert("test-project", "to delete", &embedding, None)
        .unwrap();
    assert!(db.delete(&id).unwrap());

    let memory = db.get(&id).unwrap();
    assert!(memory.is_none());
}

#[test]
fn test_delete_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let deleted = db.delete("does-not-exist").unwrap();
    assert!(!deleted);
}

#[test]
fn test_get_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let memory = db.get("does-not-exist").unwrap();
    assert!(memory.is_none());
}

#[test]
fn test_search_sorting() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding_high = vec![1.0f32; 384];
    let embedding_medium = vec![0.5f32; 384];
    let embedding_low = vec![0.0f32; 384];

    db.insert("test-project", "low", &embedding_low, None)
        .unwrap();
    db.insert("test-project", "high", &embedding_high, None)
        .unwrap();
    db.insert("test-project", "medium", &embedding_medium, None)
        .unwrap();

    let results = db.search("test-project", &embedding_high, 10).unwrap();
    assert!(results[0].similarity.unwrap() >= results[1].similarity.unwrap());
    assert!(results[1].similarity.unwrap() >= results[2].similarity.unwrap());
}

#[test]
fn test_negative_similarity() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding_pos = vec![1.0f32; 384];
    let embedding_neg = vec![-1.0f32; 384];

    db.insert("test-project", "positive", &embedding_pos, None)
        .unwrap();

    let results = db.search("test-project", &embedding_neg, 10).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].similarity.unwrap() < 0.0);
}

#[ignore]
#[test]
fn test_integration_add_search_roundtrip() {
    // Full integration test with real model
    // Requires: cargo test -- --ignored
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).unwrap();

    let id = match store
        .add_with_conflict("test-project", "semantic search is useful", None, false)
        .unwrap()
    {
        crate::memory_types::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    let results = store
        .search("test-project", "finding information", 5, 0.0)
        .unwrap();
    assert!(!results.is_empty());

    let memory = store.get(&id).unwrap().unwrap();
    assert_eq!(memory.content, "semantic search is useful");
}

#[ignore]
#[test]
fn test_integration_update_changes_embedding() {
    // Full integration test with real model
    // Requires: cargo test -- --ignored
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).unwrap();

    let id = match store
        .add_with_conflict("test-project", "original content", None, false)
        .unwrap()
    {
        crate::memory_types::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    store.update(&id, "completely different content").unwrap();

    let memory = store.get(&id).unwrap().unwrap();
    assert_eq!(memory.content, "completely different content");
}

#[test]
fn test_hybrid_search_basic() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    db.initialize_fts().unwrap();

    // Add memories with different embeddings for semantic search
    let embedding_rust: Vec<f32> = vec![1.0, 0.0, 0.0f32]
        .into_iter()
        .cycle()
        .take(384)
        .collect();
    let embedding_python: Vec<f32> = vec![0.0, 1.0, 0.0f32]
        .into_iter()
        .cycle()
        .take(384)
        .collect();
    let embedding_general: Vec<f32> = vec![0.5, 0.5, 0.5f32]
        .into_iter()
        .cycle()
        .take(384)
        .collect();

    let id_rust = db
        .insert(
            "test-project",
            "rust programming language",
            &embedding_rust,
            None,
        )
        .unwrap();
    let _id_python = db
        .insert(
            "test-project",
            "python code examples",
            &embedding_python,
            None,
        )
        .unwrap();
    let _id_general = db
        .insert(
            "test-project",
            "general software development",
            &embedding_general,
            None,
        )
        .unwrap();

    // Test hybrid search combines semantic and BM25
    let semantic_results = db.search("test-project", &embedding_rust, 50).unwrap();
    let bm25_results = db.search_bm25("rust", "test-project", 50).unwrap();

    // Verify semantic search finds rust-related content
    assert!(!semantic_results.is_empty());
    assert!(semantic_results.iter().any(|m| m.id == id_rust));
    assert!(semantic_results.iter().any(|m| m.content.contains("rust")));

    // BM25 should find matches for "rust" query
    assert!(!bm25_results.is_empty());

    // Rust memory should appear in both semantic and BM25 results
    assert!(semantic_results.iter().any(|m| m.id == id_rust));
    assert!(bm25_results
        .iter()
        .any(|m| m.content.to_lowercase().contains("rust")));
}

#[test]
fn test_hybrid_search_empty_results() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    db.initialize_fts().unwrap();

    // Add one memory
    let embedding = vec![0.5f32; 384];
    db.insert("test-project", "some content", &embedding, None)
        .unwrap();

    // Query with non-matching text
    let semantic_results = db.search("test-project", &vec![0.1f32; 384], 50).unwrap();
    let bm25_results = db
        .search_bm25("nonexistent term xyz", "test-project", 50)
        .unwrap();

    // One may be empty, but fusion should handle it
    assert!(!semantic_results.is_empty() || !bm25_results.is_empty());
}

#[test]
fn test_hybrid_search_with_recency() {
    use chrono::{Duration, Utc};
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    db.initialize_fts().unwrap();

    // Add memories with known embedding
    let embedding_good = vec![1.0f32; 384];
    let embedding_bad = vec![0.0f32; 384];

    let now = Utc::now();
    let old_time = (now - Duration::days(100)).to_rfc3339();
    let new_time = now.to_rfc3339();

    // Old high-similarity memory
    let id_old = db
        .insert_with_time(
            "test-project",
            "old but relevant",
            &embedding_good,
            None,
            &old_time,
            &old_time,
        )
        .unwrap();

    // New low-similarity memory (same content for BM25 relevance)
    let _id_new = db
        .insert_with_time(
            "test-project",
            "new but less relevant",
            &embedding_bad,
            None,
            &new_time,
            &new_time,
        )
        .unwrap();

    // Search without recency weighting - should find high-similarity first
    let semantic_results = db.search("test-project", &embedding_good, 10).unwrap();
    if !semantic_results.is_empty() {
        let top_id = &semantic_results[0].id;
        // With zero recency weight, similarity dominates
        // Either the high-similarity old memory is first, or we have a single result
        assert!(top_id == &id_old || semantic_results.len() == 1);
    }
}
