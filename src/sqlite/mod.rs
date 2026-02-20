//! SQLite backend for vipune memory storage.
//!
//! This module provides:
//! - `Database`: Core SQLite connection and schema management
//! - `Memory`: Data structure for stored memories
//! - `embedding`: BLOB conversion and cosine similarity
//! - `search`: Semantic search operations
//! - `fts`: FTS5 full-text search (Issue #40)

pub mod embedding;
pub mod fts;
pub mod search;

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Result as SqliteResult};
use std::path::Path;
use uuid::Uuid;

pub use self::embedding::vec_to_blob;

/// A single memory record with metadata and optional similarity score.
#[derive(Clone)]
pub struct Memory {
    pub id: String,
    pub project_id: String,
    pub content: String,
    pub metadata: Option<String>,

    /// Similarity score:
    /// - Semantic search: Cosine similarity (0.0-1.0, higher = better match)
    /// - FTS5 search: BM25 score (lower = better match, typically negative to positive)
    pub similarity: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
}

/// Error types for SQLite operations.
#[derive(Debug)]
pub enum Error {
    Sqlite(String),
    InvalidBlobSize { expected: usize, actual: usize },
    MismatchedDimensions { expected: usize, actual: usize },
    EmptyVector,
    InvalidEmbedding(String),
    InvalidLimit(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Sqlite(msg) => write!(f, "Database error: {}", msg),
            Error::InvalidBlobSize { expected, actual } => {
                write!(
                    f,
                    "Invalid BLOB size: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            Error::MismatchedDimensions { expected, actual } => {
                write!(
                    f,
                    "Mismatched dimensions: expected {} dimensions, got {} dimensions",
                    expected, actual
                )
            }
            Error::EmptyVector => write!(f, "Cannot compute similarity with empty vector"),
            Error::InvalidEmbedding(msg) => write!(f, "Invalid embedding: {}", msg),
            Error::InvalidLimit(msg) => write!(f, "Invalid limit: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<rusqlite::Error> for Error {
    fn from(err: rusqlite::Error) -> Self {
        Error::Sqlite(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// SQLite database backend for vipune.
pub struct Database {
    conn: Connection,
}

/// Initialize database schema and create necessary tables and triggers.
fn create_schema(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            project_id UNINDEXED,
            tokenize='porter unicode61',
            content_rowid='rowid',
            content='memories'
        );

        CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, project_id)
            VALUES (new.rowid, new.content, new.project_id);
        END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, project_id)
            VALUES('delete', old.rowid, old.content, old.project_id);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, project_id)
            VALUES('delete', old.rowid, old.content, old.project_id);
            INSERT INTO memories_fts(rowid, content, project_id)
            VALUES (new.rowid, new.content, new.project_id);
        END;
        "#,
    )?;
    Ok(())
}

impl Database {
    /// Open or create a SQLite database at the given path.
    ///
    /// Initializes the schema if the database is new.
    ///
    /// # Errors
    ///
    /// Returns error if the database cannot be opened or schema initialization fails.
    pub fn open(path: &Path) -> Result<Self> {
        let mut conn = Connection::open(path)?;
        create_schema(&mut conn)?;
        Ok(Self { conn })
    }

    /// Insert a new memory with embedding.
    ///
    /// # Errors
    ///
    /// Returns error if the embedding has invalid dimensions or database write fails.
    pub fn insert(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![&id, project_id, content, &blob, metadata, &now, &now],
        )?;

        Ok(id)
    }

    /// Insert a memory with explicit timestamps (for testing).
    ///
    /// This is used in tests to control the created_at and updated_at timestamps.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn insert_with_time(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        created_at: &str,
        updated_at: &str,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![&id, project_id, content, &blob, metadata, created_at, updated_at],
        )?;

        Ok(id)
    }

    /// Retrieve a single memory by ID.
    ///
    /// Returns None if the memory does not exist.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn get(&self, id: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, created_at, updated_at
            FROM memories
            WHERE id = ?1
            "#,
        )?;

        let result = stmt
            .query_row([id], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    metadata: row.get(3)?,
                    similarity: None,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                })
            })
            .optional()?;

        Ok(result)
    }

    /// List memories for a project, ordered by creation time (newest first).
    ///
    /// # Errors
    ///
    /// Returns error if the limit is invalid or the query fails.
    pub fn list(&self, project_id: &str, limit: usize) -> Result<Vec<Memory>> {
        search::validate_limit(limit)?;

        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, created_at, updated_at
            FROM memories
            WHERE project_id = ?1
            ORDER BY created_at DESC
            LIMIT ?2
            "#,
        )?;

        let memories: SqliteResult<Vec<Memory>> = stmt
            .query_map(params![project_id, limit as i64], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    metadata: row.get(3)?,
                    similarity: None,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                })
            })?
            .collect();

        Ok(memories?)
    }

    /// Update a memory's content and embedding.
    ///
    /// Returns an error if the memory does not exist.
    ///
    /// # Errors
    ///
    /// Returns error if the embedding has invalid dimensions, memory not found, or query fails.
    pub fn update(&self, id: &str, content: &str, embedding: &[f32]) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        let rows = self.conn.execute(
            r#"
            UPDATE memories
            SET content = ?1, embedding = ?2, updated_at = ?3
            WHERE id = ?4
            "#,
            params![content, &blob, &now, id],
        )?;

        if rows == 0 {
            return Err(Error::Sqlite(format!("No memory found with id: {}", id)));
        }

        Ok(())
    }

    /// Delete a memory by ID.
    ///
    /// Returns true if a memory was deleted, false if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", [id])?;
        Ok(rows > 0)
    }

    /// Get internal connection (for internal use, e.g., tests).
    #[allow(dead_code)] // Used in fts.rs tests
    pub(crate) fn conn(&self) -> &Connection {
        &self.conn
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
    fn test_insert_and_get() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "test content", &embedding, None)
            .unwrap();

        let memory = db.get(&id).unwrap();
        assert!(memory.is_some());
        let m = memory.unwrap();
        assert_eq!(m.content, "test content");
        assert_eq!(m.project_id, "proj1");
    }

    #[test]
    fn test_insert_with_metadata() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert(
                "proj1",
                "test content",
                &embedding,
                Some(r#"{"key": "value"}"#),
            )
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.metadata, Some(r#"{"key": "value"}"#.to_string()));
    }

    #[test]
    fn test_insert_invalid_embedding() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 256];
        let result = db.insert("proj1", "test", &embedding, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_nonexistent() {
        let db = create_test_db();
        let memory = db.get("nonexistent").unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_list_ordering() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id1 = db
            .insert_with_time(
                "proj1",
                "first",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            )
            .unwrap();
        let id2 = db
            .insert_with_time(
                "proj1",
                "second",
                &embedding,
                None,
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
            )
            .unwrap();

        let memories = db.list("proj1", 10).unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].id, id2); // Newest first
        assert_eq!(memories[1].id, id1);
    }

    #[test]
    fn test_list_limit() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        for i in 0..5 {
            db.insert("proj1", &format!("content {}", i), &embedding, None)
                .unwrap();
        }

        let memories = db.list("proj1", 2).unwrap();
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_update() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db.insert("proj1", "original", &embedding, None).unwrap();

        db.update(&id, "updated", &embedding).unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.content, "updated");
    }

    #[test]
    fn test_update_nonexistent() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let result = db.update("nonexistent", "content", &embedding);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db.insert("proj1", "content", &embedding, None).unwrap();

        let deleted = db.delete(&id).unwrap();
        assert!(deleted);

        let memory = db.get(&id).unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let db = create_test_db();
        let deleted = db.delete("nonexistent").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_project_isolation() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "proj1 content", &embedding, None)
            .unwrap();
        db.insert("proj2", "proj2 content", &embedding, None)
            .unwrap();

        let list1 = db.list("proj1", 10).unwrap();
        let list2 = db.list("proj2", 10).unwrap();

        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].project_id, "proj1");
        assert_eq!(list2[0].project_id, "proj2");
    }
}
