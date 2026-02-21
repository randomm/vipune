//! Search operations for the memory store (semantic and hybrid search).

use crate::errors::Error;
use crate::rrf;
use crate::sqlite::Memory;
use crate::temporal::{apply_recency_weight, validate_recency_weight, DecayConfig};

use super::store::{MemoryStore, validate_limit};

/// Maximum allowed candidate pool size for hybrid search to prevent DoS.
const MAX_CANDIDATE_POOL: usize = 10_000;

impl MemoryStore {
    #[must_use = "handle the error or results may be lost"]
    /// Search memories by semantic similarity.
    ///
    /// Generates an embedding for the query and finds memories with highest
    /// cosine similarity scores. Optionally applies recency weighting to
    /// boost recent memories.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier to search within
    /// * `query` - Search query text (1 to 100,000 characters)
    /// * `limit` - Maximum number of results to return
    /// * `recency_weight` - Weight for temporal decay (0.0 = pure semantic, 1.0 = max recency)
    ///
    /// # Returns
    ///
    /// Vector of memories sorted by similarity or recency-adjusted score (highest first).
    /// Each memory includes a `similarity` score field (recency-adjusted if weight > 0).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Query is empty
    /// - Query exceeds 100,000 characters
    /// - Recency weight is invalid
    /// - Embedding generation fails
    /// - Database operations fail
    pub fn search(
        &mut self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
    ) -> Result<Vec<Memory>, Error> {
        // Validate limit to prevent resource exhaustion
        validate_limit(limit)?;

        // Validate query before processing
        let query = query.trim();
        Self::validate_input_length(query)?;

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

    #[must_use = "handle the error or results may be lost"]
    /// Search memories using hybrid search (semantic + BM25 fused with RRF).
    ///
    /// Combines semantic embedding search and BM25 full-text search using
    /// Reciprocal Rank Fusion (RRF), then optionally applies recency weighting.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier to search within
    /// * `query` - Search query text (1 to 100,000 characters)
    /// * `limit` - Maximum number of results to return
    /// * `recency_weight` - Weight for temporal decay (0.0 = pure score, 1.0 = max recency)
    ///
    /// # Returns
    ///
    /// Vector of memories sorted by fused or recency-adjusted score (highest first).
    /// The `similarity` field contains the final RRF score (or recency-adjusted if weight > 0).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Query is empty
    /// - Query exceeds 100,000 characters
    /// - Recency weight is invalid
    /// - Embedding generation fails
    /// - Database operations fail
    pub fn search_hybrid(
        &mut self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
    ) -> Result<Vec<Memory>, Error> {
        // Validate query before processing
        let query = query.trim();
        Self::validate_input_length(query)?;

        validate_recency_weight(recency_weight).map_err(Error::Validation)?;

        // Validate limit before proceeding
        validate_limit(limit)?;

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
}
