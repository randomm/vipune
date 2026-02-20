//! Semantic search and similarity operations.

use super::{embedding, Database, Error, Memory};

pub type Result<T> = std::result::Result<T, Error>;

const MAX_SEARCH_LIMIT: usize = 10_000;

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
    /// # Errors
    ///
    /// Returns error if the query embedding has invalid dimensions or if the database
    /// query fails.
    pub fn search(
        &self,
        project_id: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Memory>> {
        validate_limit(limit)?;

        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, created_at, updated_at, embedding
            FROM memories
            WHERE project_id = ?1
            "#,
        )?;

        let mut memories: Vec<Memory> = Vec::new();

        let rows = stmt.query_map([project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Vec<u8>>(6)?,
            ))
        })?;

        for row_result in rows {
            let (id, pid, content, metadata, created_at, updated_at, blob) = row_result?;
            let stored_embedding = embedding::blob_to_vec(&blob)?;
            let similarity = Some(embedding::cosine_similarity(
                query_embedding,
                &stored_embedding,
            )?);

            memories.push(Memory {
                id,
                project_id: pid,
                content,
                metadata,
                similarity,
                created_at,
                updated_at,
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
        let all_results = self.search(project_id, embedding, MAX_SEARCH_LIMIT)?;
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
        db.insert("proj1", "rust programming", &embedding, None)
            .unwrap();
        db.insert("proj1", "python data science", &embedding, None)
            .unwrap();

        let results = db.search("proj1", &embedding, 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity.unwrap() >= 0.9);
    }

    #[test]
    fn test_search_limit() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        for i in 0..5 {
            db.insert("proj1", &format!("content {}", i), &embedding, None)
                .unwrap();
        }

        let results = db.search("proj1", &embedding, 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_project_isolation() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "project 1 memory", &embedding, None)
            .unwrap();
        db.insert("proj2", "project 2 memory", &embedding, None)
            .unwrap();

        let results = db.search("proj1", &embedding, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project_id, "proj1");
    }

    #[test]
    fn test_find_similar_with_threshold() {
        let db = create_test_db();
        let embedding1 = vec![1.0f32; 384];
        let mut embedding2 = vec![1.0f32; 384];
        embedding2[0] = 0.0; // Slightly different

        db.insert("proj1", "memory 1", &embedding1, None).unwrap();
        db.insert("proj1", "memory 2", &embedding2, None).unwrap();

        let results = db.find_similar("proj1", &embedding1, 0.99).unwrap();
        assert!(results.len() >= 1);
    }
}
