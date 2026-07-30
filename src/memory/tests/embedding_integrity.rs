//! Embedding integrity tests — verify stored vectors match the embedder's output.
//!
//! These tests require the real ONNX model, so they are `#[ignore]`d and run
//! via `cargo test -- --ignored` in the nightly CI job.

use crate::config::Config;
use crate::memory::MemoryStore;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory_types::AddResult;
use tempfile::TempDir;

#[ignore]
#[test]
fn test_golden_fixture_roundtrip() {
    // Golden fixture regression test: store two texts via the real embedder,
    // then search for the first. Assert both:
    //   - top hit is the stored copy of the query text with similarity > 0.99
    //   - the paraphrase is returned with similarity > 0.65
    //
    // Under the old bug the stored vectors were mock and the query was real,
    // so even the exact-text match scored as noise rather than ~1.0.

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store =
        MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).expect("create store");

    // Add two memories using the real embedder
    let query_text = "the cat sat on the mat";
    let paraphrase = "a feline rested on the rug";

    let result1 = store.add_with_conflict(
        "test-project",
        query_text,
        None,
        false,
        MemoryType::Fact,
        MemoryStatus::Active,
    );
    assert!(
        matches!(result1, Ok(AddResult::Added { .. })),
        "Expected first add to succeed, got {:?}",
        result1
    );

    let result2 = store.add_with_conflict(
        "test-project",
        paraphrase,
        None,
        false,
        MemoryType::Fact,
        MemoryStatus::Active,
    );
    assert!(
        matches!(result2, Ok(AddResult::Added { .. })),
        "Expected second add to succeed, got {:?}",
        result2
    );

    // Search for the query text
    let results = store
        .search(
            "test-project",
            query_text,
            5,
            0.0,
            crate::memory::SearchOptions::default(),
        )
        .expect("search");

    assert!(!results.is_empty(), "Expected at least one search result");

    // Top hit must be the stored copy of the query text with similarity > 0.99
    let top = &results[0];
    let self_match_similarity = top.similarity.unwrap_or(0.0);
    assert!(
        self_match_similarity > 0.99,
        "Self-match similarity too low: {} (expected > 0.99). \
         This suggests stored embeddings are mock while query uses real model.",
        self_match_similarity
    );

    // The paraphrase should be returned with similarity > 0.65
    let paraphrase_similarity = results
        .iter()
        .find(|m| m.content == paraphrase)
        .map(|m| m.similarity.unwrap_or(0.0))
        .unwrap_or(0.0);
    assert!(
        paraphrase_similarity > 0.65,
        "Paraphrase similarity too low: {} (expected > 0.65). \
         Top hit self-match was: {}",
        paraphrase_similarity,
        self_match_similarity
    );
}

/// Assert a stored embedding equals what the embedder produces for the same content.
/// This is the coverage gap that let the mock-vector bug ship.
#[ignore]
#[test]
fn test_stored_embedding_matches_embedder() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();

    let mut store =
        MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", config).expect("create store");

    let content = "embedding integrity verification";

    // Add the memory (uses real embedder)
    let result = store.add_with_conflict(
        "test-project",
        content,
        None,
        false,
        MemoryType::Fact,
        MemoryStatus::Active,
    );
    let id = match result {
        Ok(AddResult::Added { id }) => id,
        _ => panic!("Expected AddResult::Added, got {:?}", result),
    };

    // Retrieve the stored memory
    let memory = store
        .get(&id, "test-project")
        .expect("get")
        .expect("memory exists");

    // Generate fresh embedding for same content
    let fresh_embedding = store
        .embedder()
        .expect("embedder available")
        .embed(content)
        .expect("embed");

    // Compare byte-for-byte (same model, same input => identical output)
    assert_eq!(
        memory.embedding.len(),
        fresh_embedding.len(),
        "Embedding length mismatch"
    );
    for (i, (stored, fresh)) in memory
        .embedding
        .iter()
        .zip(fresh_embedding.iter())
        .enumerate()
    {
        assert!(
            (stored - fresh).abs() < 1e-6,
            "Dimension {} differs: stored = {}, fresh = {}",
            i,
            stored,
            fresh
        );
    }
}
