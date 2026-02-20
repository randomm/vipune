//! Reciprocal Rank Fusion (RRF) algorithm for hybrid search
//!
//! Merges multiple ranked result lists without score normalization.
//! Formula: score = Σ 1 / (k + rank) for each ranking list
//!
//! Documents appearing in both BM25 and semantic rankings get boosted scores.

use crate::errors::Error;
use crate::sqlite::Memory;
use std::collections::HashMap;

/// RRF fusion configuration
#[derive(Debug, Clone, Copy)]
pub struct RrfConfig {
    /// The k parameter for RRF formula (default: 25.0)
    /// Prevents division by very small numbers and gives diminishing returns for top ranks
    pub k: f64,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 25.0 }
    }
}

/// Fuses multiple search result lists using Reciprocal Rank Fusion (RRF)
///
/// RRF combines rankings from different retrieval methods without requiring score normalization.
/// Formula: score = Σ (1 / (k + rank)) for each result across all result sets
///
/// # Arguments
///
/// * `result_lists` - Vector of search result lists from different retrieval methods.
///   Each list should be pre-sorted by relevance (best results first).
/// * `config` - Optional RRF configuration. Uses default (k=25.0) if None.
///
/// # Returns
///
/// Fused and ranked list of unique search results, sorted by accumulated RRF score descending.
/// The `similarity` field in each Memory contains the fused RRF score.
///
/// # Example
///
/// ```ignore
/// // Semantic search results (sorted by cosine similarity)
/// let semantic_results = vec![memory_a, memory_b, memory_c];
///
/// // BM25 search results (sorted by BM25 score)
/// let bm25_results = vec![memory_b, memory_d, memory_e];
///
/// // Fuse with RRF
/// let fused = rrf_fusion(vec![semantic_results, bm25_results], None)?;
///
/// // memory_b appears in both lists → gets highest RRF score
/// assert_eq!(fused[0].id, memory_b.id);
/// ```
pub fn rrf_fusion(
    result_lists: Vec<Vec<Memory>>,
    config: Option<RrfConfig>,
) -> Result<Vec<Memory>, Error> {
    let config = config.unwrap_or_default();

    if result_lists.is_empty() {
        return Ok(vec![]);
    }

    // Map memory ID to accumulated Memory and RRF score
    let mut fused_results: HashMap<String, (Memory, f64)> = HashMap::new();

    // Process each result list
    for result_list in result_lists {
        for (rank, mut result) in result_list.into_iter().enumerate() {
            let rank = rank + 1; // 1-based ranking for RRF formula
            let rrf_score = 1.0f64 / (config.k + rank as f64);

            // Additive scoring for duplicate documents across different retrieval methods
            let id = result.id.clone();
            match fused_results.get_mut(&id) {
                Some((_, accumulated_score)) => {
                    *accumulated_score += rrf_score;
                }
                None => {
                    // Store RRF score in similarity field temporarily
                    result.similarity = Some(rrf_score);
                    fused_results.insert(id, (result, rrf_score));
                }
            }
        }
    }

    // Convert to vector and sort by accumulated RRF score (higher is better)
    let mut fused_vec: Vec<(Memory, f64)> = fused_results.into_values().collect();
    fused_vec.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.id.cmp(&b.0.id))
    });

    // Extract Memory objects with final RRF scores
    let final_results = fused_vec
        .into_iter()
        .map(|(mut result, score)| {
            result.similarity = Some(score);
            result
        })
        .collect();

    Ok(final_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_memory(
        id: &str,
        content: &str,
        project_id: &str,
        similarity: Option<f64>,
    ) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: project_id.to_string(),
            content: content.to_string(),
            metadata: None,
            similarity,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_rrf_fusion_basic() {
        // Simulate semantic search results
        let semantic_results = vec![
            create_test_memory("mem-1", "rust programming", "proj-a", Some(0.9)),
            create_test_memory("mem-2", "python code", "proj-a", Some(0.7)),
        ];

        // Simulate BM25 results (mem-2 appears in both lists)
        let bm25_results = vec![
            create_test_memory("mem-2", "python code", "proj-a", Some(0.5)),
            create_test_memory("mem-3", "database query", "proj-a", Some(0.3)),
        ];

        let fused = rrf_fusion(vec![semantic_results, bm25_results], None).unwrap();

        assert_eq!(fused.len(), 3);

        // mem-2 appears in both lists, should have highest RRF score
        assert_eq!(fused[0].id, "mem-2");
        assert!(fused[0].similarity.unwrap() > fused[1].similarity.unwrap());
    }

    #[test]
    fn test_rrf_fusion_empty_lists() {
        let fused = rrf_fusion(vec![], None).unwrap();
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_fusion_single_list() {
        let results = vec![
            create_test_memory("mem-1", "content 1", "proj-a", Some(0.9)),
            create_test_memory("mem-2", "content 2", "proj-a", Some(0.8)),
        ];

        let fused = rrf_fusion(vec![results], None).unwrap();

        assert_eq!(fused.len(), 2);
        // With single list, RRF preserves order
        assert_eq!(fused[0].id, "mem-1");
    }

    #[test]
    fn test_rrf_fusion_duplicate_documents() {
        // Same memory appears in all 3 result lists at rank 1
        let memory = create_test_memory("mem-1", "shared content", "proj-a", Some(0.8));

        let list1 = vec![memory.clone()];
        let list2 = vec![memory.clone()];
        let list3 = vec![memory];

        let default_config = RrfConfig::default();
        let fused = rrf_fusion(vec![list1, list2, list3], None).unwrap();

        assert_eq!(fused.len(), 1);

        // Should have accumulated RRF score from all 3 lists at rank=1
        // score = 3 * (1 / (k + 1))
        let expected_score = 3.0 * (1.0 / (default_config.k + 1.0));
        assert!((fused[0].similarity.unwrap() - expected_score).abs() < 0.001);
    }

    #[test]
    fn test_rrf_fusion_different_k_values() {
        let configs = [RrfConfig { k: 10.0 }, RrfConfig { k: 100.0 }];

        let semantic_results = vec![create_test_memory("mem-1", "doc1", "proj-a", Some(0.9))];
        let bm25_results = vec![create_test_memory("mem-1", "doc1", "proj-a", Some(0.9))];

        for &config in &[configs[0], configs[1]] {
            let fused = rrf_fusion(
                vec![semantic_results.clone(), bm25_results.clone()],
                Some(config),
            )
            .unwrap();

            // Memory appears at rank 1 in both lists
            // score = 2 * (1 / (k + 1))
            let expected_score = 2.0 * (1.0 / (config.k + 1.0));
            assert!((fused[0].similarity.unwrap() - expected_score).abs() < 0.001);
        }
    }

    #[test]
    fn test_rrf_fusion_ranking_priority() {
        // RRF scoring verification:
        // With k=25, rank 1&3 gives higher score than rank 2&2
        // mem-1: 1/(25+1) + 1/(25+3) = 0.03846 + 0.03571 = 0.07417
        // mem-2: 1/(25+1) + 1/(25+3) = 0.03846 + 0.03571 = 0.07417 (same ranks)
        // mem-3: 1/(25+2) + 1/(25+2) = 0.03704 + 0.03704 = 0.07408 (slightly lower)

        let semantic_results = vec![
            create_test_memory("mem-1", "doc_a", "proj-a", Some(0.9)), // rank 1
            create_test_memory("mem-3", "doc_c", "proj-a", Some(0.8)), // rank 2
            create_test_memory("mem-2", "doc_b", "proj-a", Some(0.7)), // rank 3
        ];

        let bm25_results = vec![
            create_test_memory("mem-2", "doc_b", "proj-a", Some(0.9)), // rank 1
            create_test_memory("mem-3", "doc_c", "proj-a", Some(0.8)), // rank 2
            create_test_memory("mem-1", "doc_a", "proj-a", Some(0.7)), // rank 3
        ];

        let fused = rrf_fusion(vec![semantic_results, bm25_results], None).unwrap();

        assert_eq!(fused.len(), 3);

        // mem-1 and mem-2 (rank 1&3, 3&1) should tie for highest RRF score
        // Both beat mem-3 (rank 2&2)
        assert_eq!(fused[0].similarity.unwrap(), fused[1].similarity.unwrap());
        assert!(fused[2].similarity.unwrap() < fused[0].similarity.unwrap());
    }

    #[test]
    fn test_rrf_fusion_preserves_metadata() {
        let memory = Memory {
            id: "mem-1".to_string(),
            project_id: "proj-a".to_string(),
            content: "test content".to_string(),
            metadata: Some("metadata".to_string()),
            similarity: Some(0.9),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let fused = rrf_fusion(vec![vec![memory]], None).unwrap();

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].metadata, Some("metadata".to_string()));
        assert_eq!(fused[0].project_id, "proj-a");
    }

    #[test]
    fn test_rrf_fusion_empty_lists_in_vector() {
        let results = vec![create_test_memory("mem-1", "content", "proj-a", Some(0.9))];

        // Empty list should not affect fusion
        let fused = rrf_fusion(vec![vec![], results.clone()], None).unwrap();

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "mem-1");
    }

    #[test]
    fn test_rrf_fusion_many_results() {
        let list1: Vec<Memory> = (1..=10)
            .map(|i| {
                create_test_memory(
                    &format!("mem-{}", i),
                    &format!("content {}", i),
                    "proj-a",
                    None,
                )
            })
            .collect();

        let list2: Vec<Memory> = (5..=15)
            .map(|i| {
                create_test_memory(
                    &format!("mem-{}", i),
                    &format!("content {}", i),
                    "proj-a",
                    None,
                )
            })
            .collect();

        let fused = rrf_fusion(vec![list1, list2], None).unwrap();

        // mem-5 through mem-10 appear in both lists → should have highest scores
        assert_eq!(fused.len(), 15);
        verify_top_results_are_overlap(&fused, 5, 10);
    }

    fn verify_top_results_are_overlap(results: &[Memory], start: u32, end: u32) {
        let mut overlap_count = 0;
        for (_idx, result) in results.iter().take((end - start + 1) as usize).enumerate() {
            let id_num = result
                .id
                .strip_prefix("mem-")
                .unwrap()
                .parse::<u32>()
                .unwrap();
            if id_num >= start && id_num <= end {
                overlap_count += 1;
            }
        }
        assert!(
            overlap_count >= 3,
            "Expected at least 3 overlapping documents in top results, got {}",
            overlap_count
        );
    }

    #[test]
    fn test_rrf_fusion_order_consistency() {
        // Same input should produce same output order
        let list1 = vec![
            create_test_memory("mem-1", "a", "proj-a", None),
            create_test_memory("mem-2", "b", "proj-a", None),
        ];
        let list2 = vec![
            create_test_memory("mem-2", "b", "proj-a", None),
            create_test_memory("mem-1", "a", "proj-a", None),
        ];

        let fused1 = rrf_fusion(vec![list1.clone(), list2.clone()], None).unwrap();
        let fused2 = rrf_fusion(vec![list1, list2], None).unwrap();

        assert_eq!(fused1.len(), fused2.len());
        for (r1, r2) in fused1.iter().zip(fused2.iter()) {
            assert_eq!(r1.id, r2.id);
        }
    }
}
