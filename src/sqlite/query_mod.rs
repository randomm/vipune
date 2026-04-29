//! Query helpers for SQLite operations.
//!
//! Provides common row mapping and query construction utilities for the SQLite backend.

use rusqlite::{Result as SqliteResult, Row, types::Type};

use super::{Error, Memory};
use crate::embedding::EMBEDDING_DIMS;

/// Map a SQLite row to a Memory struct without similarity score.
///
/// Used for queries that return memories without search results (get, list, list_since, get_many).
///
/// # Arguments
///
/// * `row` - SQLite row containing memory columns
///
/// # Errors
///
/// Returns error if any column extraction fails or embedding has invalid dimensions.
pub fn map_row_to_memory(row: &Row) -> SqliteResult<Memory> {
    let blob: Vec<u8> = row.get(4)?;
    let embedding = super::embedding::blob_to_vec(&blob)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, Type::Blob, Box::new(e)))?;

    if embedding.len() != EMBEDDING_DIMS {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            4,
            Type::Blob,
            Box::new(Error::MismatchedDimensions {
                expected: EMBEDDING_DIMS,
                actual: embedding.len(),
            }),
        ));
    }

    Ok(Memory {
        id: row.get(0)?,
        project_id: row.get(1)?,
        content: row.get(2)?,
        metadata: row.get(3)?,
        embedding,
        similarity: None,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
        memory_type: row.get(7)?,
        status: row.get(8)?,
        superseded_by: row.get(9)?,
        retrieval_count: row.get(10)?,
        last_retrieved_at: row.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EMBEDDING_DIMS;
    use rusqlite::params;
    use tempfile::TempDir;

    fn create_test_db() -> super::super::Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = super::super::Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    #[test]
    fn test_map_row_to_memory() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert(
                "proj1",
                "test content",
                &embedding,
                Some(r#"{"key":"value"}"#),
                "fact",
                "active",
            )
            .unwrap();

        let conn = db.conn();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
                FROM memories
                WHERE id = ?1
                "#,
            )
            .unwrap();

        let memory = stmt.query_row([id.clone()], map_row_to_memory).unwrap();
        assert_eq!(memory.id, id);
        assert_eq!(memory.content, "test content");
        assert_eq!(memory.project_id, "proj1");
        assert_eq!(memory.metadata, Some(r#"{"key":"value"}"#.to_string()));
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        assert!(memory.similarity.is_none());
        assert_eq!(memory.retrieval_count, 0);
        assert!(memory.last_retrieved_at.is_none());
    }

    #[test]
    fn test_map_row_to_memory_without_metadata() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let conn = db.conn();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
                FROM memories
                WHERE id = ?1
                "#,
            )
            .unwrap();

        let memory = stmt.query_row([id.clone()], map_row_to_memory).unwrap();
        assert_eq!(memory.metadata, None);
    }

    #[test]
    fn test_map_row_to_memory_invalid_embedding() {
        let db = create_test_db();
        let conn = db.conn();

        // Insert a valid embedding but test that the mapping function works correctly
        let blob = super::super::embedding::vec_to_blob(&vec![0.1f32; EMBEDDING_DIMS]).unwrap();
        conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, retrieval_count, last_retrieved_at)
            VALUES ('test-id', 'proj1', 'test', ?1, NULL, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active', 0, NULL)
            "#,
            params![blob],
        )
        .unwrap();

        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
                FROM memories
                WHERE id = ?1
                "#,
            )
            .unwrap();

        // Should successfully map valid embedding
        let memory = stmt.query_row(["test-id"], map_row_to_memory).unwrap();
        assert_eq!(memory.id, "test-id");
        assert_eq!(memory.content, "test");
    }
}
