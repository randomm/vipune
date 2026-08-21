//! Semantic search and similarity operations.

use super::{Database, EMBEDDING_COLUMN, Error, Memory, embedding};
use crate::memory::store::MAX_SEARCH_LIMIT;

pub type Result<T> = std::result::Result<T, Error>;

/// Validate search limit is within acceptable bounds.
pub fn validate_limit(limit: usize) -> Result<()> {
    if limit == 0 {
        return Err(Error::InvalidLimit(
            "Limit must be greater than 0".to_string(),
        ));
    }
    if limit > i64::MAX as usize || limit > MAX_SEARCH_LIMIT {
        return Err(Error::InvalidLimit(format!(
            "Limit {} exceeds maximum allowed ({})",
            limit, MAX_SEARCH_LIMIT
        )));
    }
    Ok(())
}

impl Database {
    /// Search for similar memories using semantic (cosine) similarity.
    ///
    /// Retrieves all memories for a project, computes cosine similarity with the query
    /// embedding, sorts by similarity (highest first), and returns the top `limit` results.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `query_embedding` - The embedding vector to compare against
    /// * `limit` - Maximum number of results to return
    /// * `memory_types` - Optional filter for memory types (None = no filter by type)
    /// * `statuses` - Optional filter for lifecycle statuses (None = default to 'active')
    ///
    /// # Errors
    ///
    /// Returns error if the query embedding has invalid dimensions or if the database
    /// query fails.
    pub fn search(
        &self,
        project_id: &str,
        query_embedding: &[f32],
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<Vec<Memory>> {
        validate_limit(limit)?;

        let mut where_clauses = vec!["project_id = ?1".to_string()];
        let mut param_index = 2usize;

        // Status filter (default to active if None)
        if let Some(statuses) = statuses {
            if !statuses.is_empty() {
                let placeholders: Vec<String> = (0..statuses.len())
                    .map(|i| format!("?{}", param_index + i))
                    .collect();
                where_clauses.push(format!("status IN ({})", placeholders.join(", ")));
                param_index += statuses.len();
            }
        } else {
            where_clauses.push(format!("status = ?{}", param_index));
            param_index += 1;
        }

        // Type filter (only if explicitly provided)
        if let Some(types) = memory_types {
            if !types.is_empty() {
                let placeholders: Vec<String> = (0..types.len())
                    .map(|i| format!("?{}", param_index + i))
                    .collect();
                where_clauses.push(format!("type IN ({})", placeholders.join(", ")));
            }
        }

        let where_clause = where_clauses.join(" AND ");
        let query = format!(
            "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
             FROM memories WHERE {} ORDER BY created_at DESC",
            where_clause
        );

        let mut stmt = self.conn.prepare(&query)?;

        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&project_id];
        if let Some(statuses) = statuses {
            if statuses.is_empty() {
                // explicit empty = no status filter, but we didn't add a clause
            } else {
                for s in statuses {
                    params.push(s);
                }
            }
        } else {
            params.push(&"active");
        }
        if let Some(types) = memory_types {
            for t in types {
                params.push(t);
            }
        }

        let mut memories: Vec<Memory> = Vec::new();

        // The query vector is identical for every row in the scan, so its L2
        // norm is computed exactly once here (f64 accumulation, matching
        // `cosine_similarity_with_norm`) instead of once per row. This is
        // safe because `query_norm` is read only at the final division, which
        // is unreachable for the degenerate cases: an empty corpus leaves the
        // per-row loop unrun; a NaN, empty, or dimension-mismatched query
        // errors before the division; and a zero-norm operand (all-zero
        // query or stored vector) hits the `Ok(0.0)` guard first. (A NaN
        // query yields a NaN hoisted norm — NaN squared is NaN — but that
        // path errors as well.)
        let query_norm: f64 = query_embedding
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        let rows = stmt.query_map(params.as_slice(), |row| {
            // Positions match the SELECT above: column 4 = `embedding`
            // (see EMBEDDING_COLUMN).
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, Option<String>>(9)?,
                row.get::<_, i64>(10)?,
                row.get::<_, Option<String>>(11)?,
            ))
        })?;

        for row_result in rows {
            let (
                id,
                pid,
                content,
                metadata,
                blob,
                created_at,
                updated_at,
                type_val,
                status_val,
                superseded_by,
                retrieval_count,
                last_retrieved_at,
            ) = row_result?;
            let stored_embedding = embedding::blob_to_vec(&blob).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    EMBEDDING_COLUMN,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;
            let similarity = Some(embedding::cosine_similarity_with_norm(
                query_embedding,
                query_norm,
                &stored_embedding,
            )?);

            memories.push(Memory {
                id,
                project_id: pid,
                content,
                metadata,
                embedding: stored_embedding,
                similarity,
                created_at,
                updated_at,
                memory_type: type_val,
                status: status_val,
                superseded_by,
                retrieval_count,
                last_retrieved_at,
            });
        }

        memories.sort_by(|a, b| {
            b.similarity
                .unwrap_or(0.0)
                .partial_cmp(&a.similarity.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Find memories similar to the given embedding above a threshold.
    ///
    /// Uses semantic search to find all memories with cosine similarity >= threshold.
    ///
    /// # Errors
    ///
    /// Returns error if the search fails.
    pub fn find_similar(
        &self,
        project_id: &str,
        embedding: &[f32],
        threshold: f64,
    ) -> Result<Vec<Memory>> {
        let all_results = self.search(project_id, embedding, MAX_SEARCH_LIMIT, None, None)?;
        Ok(all_results
            .into_iter()
            .filter(|m| m.similarity.unwrap_or(0.0) >= threshold)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db() -> Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    #[test]
    fn test_validate_limit_zero() {
        assert!(validate_limit(0).is_err());
    }

    #[test]
    fn test_validate_limit_too_large() {
        assert!(validate_limit(100_000).is_err());
    }

    #[test]
    fn test_validate_limit_valid() {
        assert!(validate_limit(10).is_ok());
        assert!(validate_limit(5000).is_ok());
    }

    #[test]
    fn test_search_basic() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "rust programming",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert(
            "proj1",
            "python data science",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();

        let results = db.search("proj1", &embedding, 10, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity.unwrap() >= 0.9);
    }

    #[test]
    fn test_search_limit() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        for i in 0..5 {
            db.insert(
                "proj1",
                &format!("content {}", i),
                &embedding,
                None,
                "fact",
                "active",
            )
            .unwrap();
        }

        let results = db.search("proj1", &embedding, 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_project_isolation() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "project 1 memory",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert(
            "proj2",
            "project 2 memory",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();

        let results = db.search("proj1", &embedding, 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project_id, "proj1");
    }

    #[test]
    fn test_search_empty_corpus_degenerate_queries() {
        let db = create_test_db();

        // Zero rows for "proj1": the per-row validation inside cosine_similarity
        // never runs, so degenerate query vectors must return Ok(empty), not Err.
        // This locks the empty-corpus contract: hoisting validation out of the
        // loop (a follow-up) would break this test and force an explicit
        // contract decision.

        // Empty query vector
        let results = db.search("proj1", &[], 10, None, None).unwrap();
        assert!(results.is_empty());

        // NaN-containing query vector
        let nan_query: Vec<f32> = vec![f32::NAN; 384];
        let results = db.search("proj1", &nan_query, 10, None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_corrupt_embedding_names_embedding_column() {
        let db = create_test_db();
        let conn = db.conn();
        // Insert a row whose embedding BLOB is not a valid f32 buffer:
        // 3 bytes cannot be reinterpreted as f32 values.
        conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, retrieval_count, last_retrieved_at)
            VALUES ('corrupt-id', 'proj1', 'corrupt', X'000102', NULL, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active', 0, NULL)
            "#,
            [],
        )
        .unwrap();

        let query = vec![0.1f32; 384];
        let err = db.search("proj1", &query, 10, None, None).unwrap_err();

        // The error surfaces as Error::Sqlite(err.to_string()); the
        // FromSqlConversionFailure display includes "Conversion error from type Blob at index: 4".
        match err {
            Error::Sqlite(msg) => {
                assert!(
                    msg.contains("at index: 4"),
                    "expected embedding column index 4 in error, got: {}",
                    msg
                );
            }
            other => panic!("expected Sqlite error, got: {:?}", other),
        }
    }

    #[test]
    fn test_find_similar_with_threshold() {
        let db = create_test_db();
        let embedding1 = vec![1.0f32; 384];
        let mut embedding2 = vec![1.0f32; 384];
        embedding2[0] = 0.0; // Slightly different

        db.insert("proj1", "memory 1", &embedding1, None, "fact", "active")
            .unwrap();
        db.insert("proj1", "memory 2", &embedding2, None, "fact", "active")
            .unwrap();

        let results = db.find_similar("proj1", &embedding1, 0.99).unwrap();
        assert!(!results.is_empty());
    }
}
