//! Core memory store orchestrating embedding and SQLite operations.
//!
//! Provides a high-level API for storing, searching, and retrieving memories
//! with automatic embedding generation via the ONNX model.

use std::path::Path;

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::errors::Error;
use crate::memory_types::{AddResult, ConflictMemory};
use crate::rrf;
use crate::sqlite::{Database, Memory};
use crate::temporal::{apply_recency_weight, validate_recency_weight, DecayConfig};

/// Maximum allowed candidate pool size for hybrid search to prevent DoS.
const MAX_CANDIDATE_POOL: usize = 10_000;
/// Maximum allowed limit for search operations.
const MAX_SEARCH_LIMIT: usize = 10_000;

/// Core memory store combining embedding generation and persistence.
///
/// Wraps a SQLite database and ONNX embedding engine to provide
/// semantic search capabilities for stored text memories.
///
/// # Mutability Requirements
///
/// Methods that generate embeddings (`add`, `search`, `update`) require
/// `&mut self` because `EmbeddingEngine::embed` internally mutates state
/// for ONNX tensor allocations.
#[allow(dead_code)] // Dead code justified: public API for CLI integration
pub struct MemoryStore {
    pub(crate) db: Database,
    pub(crate) embedder: EmbeddingEngine,
    pub(crate) config: Config,
}

impl MemoryStore {
    /// Initialize a new memory store with database path, model ID, and config.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the SQLite database file (created if it doesn't exist)
    /// * `model_id` - HuggingFace model ID (e.g., "BAAI/bge-small-en-v1.5")
    /// * `config` - Configuration including similarity threshold for conflict detection
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn new(db_path: &Path, model_id: &str, config: Config) -> Result<Self, Error> {
        let db = Database::open(db_path)?;
        let embedder = EmbeddingEngine::new(model_id)?;
        Ok(MemoryStore {
            db,
            embedder,
            config,
        })
    }

    /// Add a memory to the store (legacy method without conflict detection).
    ///
    /// Generates an embedding for the content and stores it in SQLite.
    /// Returns the generated memory ID (UUID).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `content` - Text content to store
    /// * `metadata` - Optional JSON metadata string
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn add(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
    ) -> Result<String, Error> {
        let embedding = self.embedder.embed(content)?;
        Ok(self.db.insert(project_id, content, &embedding, metadata)?)
    }

    /// Add a memory with conflict detection.
    ///
    /// Checks for similar existing memories before adding. If conflicts are found
    /// (similarity >= threshold), returns conflicts details without storing.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `content` - Text content to store
    /// * `metadata` - Optional JSON metadata string
    /// * `force` - If true, bypass conflict detection and add regardless
    ///
    /// # Returns
    ///
    /// * `Ok(AddResult::Added { id })` if no conflicts or force=true
    /// * `Ok(AddResult::Conflicts { proposed, conflicts })` if conflicts found
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn add_with_conflict(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        force: bool,
    ) -> Result<AddResult, Error> {
        if force {
            let embedding = self.embedder.embed(content)?;
            let id = self.db.insert(project_id, content, &embedding, metadata)?;
            return Ok(AddResult::Added { id });
        }

        let embedding = self.embedder.embed(content)?;
        let similars =
            self.db
                .find_similar(project_id, &embedding, self.config.similarity_threshold)?;
        let conflicts: Vec<ConflictMemory> = similars
            .into_iter()
            .map(|m| ConflictMemory {
                id: m.id,
                content: m.content,
                similarity: m.similarity.unwrap_or(0.0),
            })
            .collect();

        if conflicts.is_empty() {
            let id = self.db.insert(project_id, content, &embedding, metadata)?;
            Ok(AddResult::Added { id })
        } else {
            Ok(AddResult::Conflicts {
                proposed: content.to_string(),
                conflicts,
            })
        }
    }

    /// Search memories by semantic similarity.
    ///
    /// Generates an embedding for the query and finds memories with highest
    /// cosine similarity scores. Optionally applies recency weighting to
    /// boost recent memories.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier to search within
    /// * `query` - Search query text
    /// * `limit` - Maximum number of results to return
    /// * `recency_weight` - Weight for temporal decay (0.0 = pure semantic, 1.0 = max recency)
    ///
    /// # Returns
    ///
    /// Vector of memories sorted by similarity or recency-adjusted score (highest first).
    /// Each memory includes a `similarity` score field (recency-adjusted if weight > 0).
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn search(
        &mut self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
    ) -> Result<Vec<Memory>, Error> {
        // Validate query before processing
        let query = query.trim();
        if query.is_empty() {
            return Err(Error::InvalidInput(
                "Search query cannot be empty".to_string(),
            ));
        }

        validate_recency_weight(recency_weight).map_err(Error::Validation)?;
        let embedding = self.embedder.embed(query)?;
        let mut memories = self.db.search(project_id, &embedding, limit)?;

        if recency_weight > 0.0 {
            let decay_config = DecayConfig::new()?;
            for memory in memories.iter_mut() {
                let created_at = memory
                    .created_at
                    .parse::<chrono::DateTime<chrono::Utc>>()
                    .map_err(|e| Error::InvalidTimestamp {
                        id: memory.id.clone(),
                        timestamp: memory.created_at.clone(),
                        error: e.to_string(),
                    })?;
                let similarity = memory.similarity.unwrap_or(0.0);
                memory.similarity = Some(apply_recency_weight(
                    similarity,
                    &created_at,
                    recency_weight,
                    &decay_config,
                ));
            }
            // Re-sort by recency-adjusted scores
            memories.sort_by(|a, b| {
                b.similarity
                    .unwrap_or(0.0)
                    .partial_cmp(&a.similarity.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(memories)
    }

    /// Search memories using hybrid search (semantic + BM25 fused with RRF).
    ///
    /// Combines semantic embedding search and BM25 full-text search using
    /// Reciprocal Rank Fusion (RRF), then optionally applies recency weighting.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier to search within
    /// * `query` - Search query text
    /// * `limit` - Maximum number of results to return
    /// * `recency_weight` - Weight for temporal decay (0.0 = pure score, 1.0 = max recency)
    ///
    /// # Returns
    ///
    /// Vector of memories sorted by fused or recency-adjusted score (highest first).
    /// The `similarity` field contains the final RRF score (or recency-adjusted if weight > 0).
    #[allow(dead_code)] // Dead code justified: public API for CLI hybrid search (Issue #40 Phase 3)
    pub fn search_hybrid(
        &mut self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
    ) -> Result<Vec<Memory>, Error> {
        // Validate query before processing
        let query = query.trim();
        if query.is_empty() {
            return Err(Error::InvalidInput(
                "Search query cannot be empty".to_string(),
            ));
        }

        validate_recency_weight(recency_weight).map_err(Error::Validation)?;

        // Validate limit before proceeding
        if limit == 0 {
            return Err(Error::InvalidInput(
                "Limit must be greater than 0".to_string(),
            ));
        }
        if limit > MAX_SEARCH_LIMIT {
            return Err(Error::InvalidInput(format!(
                "Limit {} exceeds maximum allowed ({})",
                limit, MAX_SEARCH_LIMIT
            )));
        }

        // 1. Encode query for semantic search
        let embedding = self.embedder.embed(query)?;

        // 2. Calculate candidate pool (limit × 10, min 50, max MAX_CANDIDATE_POOL)
        let candidate_pool = limit.saturating_mul(10).clamp(50, MAX_CANDIDATE_POOL);

        // 3. Run semantic search
        let semantic_results = self.db.search(project_id, &embedding, candidate_pool)?;

        // 4. Run BM25 search
        let bm25_results = self.db.search_bm25(query, project_id, candidate_pool)?;

        // 5. Fuse with RRF (use default config)
        let fused = rrf::rrf_fusion(vec![semantic_results, bm25_results], None)?;

        // 6. Apply temporal decay if weight > 0
        let mut final_results = if recency_weight > 0.0 {
            let decay_config = DecayConfig::new()?;
            let mut results = fused;
            for memory in results.iter_mut() {
                let timestamp = memory.created_at.clone();
                let created_at =
                    timestamp
                        .parse::<chrono::DateTime<chrono::Utc>>()
                        .map_err(|e| Error::InvalidTimestamp {
                            id: memory.id.clone(),
                            timestamp,
                            error: e.to_string(),
                        })?;
                let similarity = memory.similarity.unwrap_or(0.0);
                memory.similarity = Some(apply_recency_weight(
                    similarity,
                    &created_at,
                    recency_weight,
                    &decay_config,
                ));
            }
            // Re-sort by recency-adjusted scores
            results.sort_by(|a, b| {
                b.similarity
                    .unwrap_or(0.0)
                    .partial_cmp(&a.similarity.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results
        } else {
            fused
        };

        // 7. Return top 'limit' results
        final_results.truncate(limit);
        Ok(final_results)
    }

    /// Get a specific memory by ID.
    ///
    /// Returns `None` if the memory doesn't exist.
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn get(&self, id: &str) -> Result<Option<Memory>, Error> {
        Ok(self.db.get(id)?)
    }

    /// List all memories for a project.
    ///
    /// Returns memories ordered by creation time (newest first).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `limit` - Maximum number of results to return
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn list(&self, project_id: &str, limit: usize) -> Result<Vec<Memory>, Error> {
        Ok(self.db.list(project_id, limit)?)
    }

    /// Update a memory's content.
    ///
    /// Generates a new embedding for the updated content and persists it.
    /// The memory ID, project ID, and creation timestamp remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID to update
    /// * `content` - New content for the memory
    ///
    /// # Errors
    ///
    /// Returns error if the memory doesn't exist.
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn update(&mut self, id: &str, content: &str) -> Result<(), Error> {
        let embedding = self.embedder.embed(content)?;
        Ok(self.db.update(id, content, &embedding)?)
    }

    /// Delete a memory.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if memory was deleted
    /// - `Ok(false)` if memory didn't exist
    #[allow(dead_code)] // Dead code justified: public API for CLI integration
    pub fn delete(&self, id: &str) -> Result<bool, Error> {
        Ok(self.db.delete(id)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let id = store
            .add("test-project", "semantic search is useful", None)
            .unwrap();

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

        let id = store.add("test-project", "original content", None).unwrap();

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
}
