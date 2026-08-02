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
pub mod list;
pub mod migrations;
pub mod query_mod;
pub mod search;
pub mod supersede;
pub mod update;

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::time::Duration;
use uuid::Uuid;

pub use self::embedding::{blob_to_vec, vec_to_blob};
pub use self::query_mod::map_row_to_memory;

/// A single memory record with metadata, embedding vector, and optional similarity score.
///
/// Contains the stored memory content, metadata, embedding, and timestamps. The similarity
/// field is populated only during search operations.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Memory {
    /// Unique identifier for this memory.
    pub id: String,
    /// Project identifier that owns this memory.
    pub project_id: String,
    /// The memory content (text to be embedded and searched).
    pub content: String,
    /// Optional user-provided metadata (JSON string).
    pub metadata: Option<String>,
    /// The embedding vector (384-dimensional f32 values).
    pub embedding: Vec<f32>,

    /// Similarity score (search-dependent):
    /// - Semantic search: Cosine similarity (0.0-1.0, higher = better match)
    /// - FTS5 search: BM25 score (lower = better match, typically negative to positive)
    pub similarity: Option<f64>,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Last update timestamp in RFC3339 format.
    pub updated_at: String,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    pub status: String,
    /// ID of the memory that superseded this one (if any).
    pub superseded_by: Option<String>,
    /// Number of times this memory was retrieved via search or get.
    pub retrieval_count: i64,
    /// RFC3339 timestamp of last retrieval (None if never retrieved).
    pub last_retrieved_at: Option<String>,
}

/// Error types for SQLite operations.
#[derive(Debug)]
pub enum Error {
    /// SQLite database error with message.
    Sqlite(String),
    /// Embedding BLOB has unexpected size.
    InvalidBlobSize { expected: usize, actual: usize },
    /// Embedding vector dimensions do not match model dimensions.
    MismatchedDimensions { expected: usize, actual: usize },
    /// Cannot embed an empty vector.
    EmptyVector,
    /// Invalid embedding data or format.
    InvalidEmbedding(String),
    /// Invalid search limit value.
    InvalidLimit(String),
    /// Entity not found.
    NotFound(String),
    /// Invalid input provided.
    InvalidInput(String),
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
            Error::NotFound(msg) => write!(f, "Not found: {}", msg),
            Error::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
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
    /// Active SQLite connection to the database.
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
    /// Initializes the schema if the database is new, then runs any pending migrations.
    ///
    /// **Schema creation vs. migrations**:
    /// - Schema creation handles the initial table setup (CREATE TABLE IF NOT EXISTS).
    /// - Migrations handle incremental changes from version to version.
    /// - For fresh DBs, both run: `create_schema` sets up tables, migration 1 is a no-op baseline.
    /// - For existing DBs: `create_schema` is a no-op (IF NOT EXISTS), migrations apply incrementally.
    ///
    /// # Errors
    ///
    /// Returns error if the database cannot be opened, schema initialization fails,
    /// or migration fails.
    pub fn open(path: &Path) -> Result<Self> {
        let mut conn = Connection::open(path)?;
        create_schema(&mut conn)?;
        migrations::run_migrations(&conn)?;
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
        memory_type: &str,
        status: &str,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![&id, project_id, content, &blob, metadata, &now, &now, memory_type, status],
        )?;

        Ok(id)
    }

    /// Insert a memory with explicit timestamps (for testing).
    ///
    /// This is used in tests to control the created_at and updated_at timestamps.
    #[cfg(test)]
    pub(crate) fn insert_with_time(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        created_at: &str,
        updated_at: &str,
        memory_type: &str,
        status: &str,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![&id, project_id, content, &blob, metadata, created_at, updated_at, memory_type, status],
        )?;

        Ok(id)
    }

    /// Retrieve a single memory by ID scoped to a project.
    ///
    /// Returns None if the memory does not exist or belongs to a different project.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn get(&self, id: &str, project_id: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
            FROM memories
            WHERE id = ?1 AND project_id = ?2
            "#,
        )?;

        let result = stmt
            .query_row([id, project_id], map_row_to_memory)
            .optional()?;
        Ok(result)
    }

    /// Delete a memory by ID scoped to a project.
    ///
    /// Returns true if a memory was deleted, false if it didn't exist or belongs to a different project.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn delete(&self, id: &str, project_id: &str) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM memories WHERE id = ?1 AND project_id = ?2",
            [id, project_id],
        )?;
        Ok(rows > 0)
    }

    /// Get internal connection (for test use only).
    #[cfg(test)]
    pub(crate) fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Set the SQLite busy timeout.
    ///
    /// Used by the reindex command to enforce fast-fail behavior on database locks.
    ///
    /// # Errors
    ///
    /// Returns error if the busy timeout cannot be set.
    pub fn set_busy_timeout(&self, timeout: Duration) -> Result<()> {
        self.conn.busy_timeout(timeout)?;
        Ok(())
    }

    /// List ALL rows for a project, including superseded and deprecated.
    ///
    /// Unlike `list()`, this does not filter by status and has no limit.
    /// Returns id, content, and embedding for each row.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn list_all_rows_for_project(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, String, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, embedding FROM memories WHERE project_id = ?1")?;

        let rows = stmt.query_map([project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row_result in rows {
            let (id, content, blob) = row_result?;
            let embedding = blob_to_vec(&blob)?;
            results.push((id, content, embedding));
        }
        Ok(results)
    }

    /// Update only the embedding BLOB for a memory, leaving all other fields intact.
    ///
    /// This does NOT touch `updated_at`, `retrieval_count`, or `last_retrieved_at`.
    /// Required for reindex operations where only the vector changes.
    ///
    /// # Errors
    ///
    /// Returns error if the embedding has invalid dimensions or the database write fails.
    pub fn update_embedding(&self, id: &str, embedding: &[f32]) -> Result<()> {
        let blob = vec_to_blob(embedding)?;
        let rows = self.conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            rusqlite::params![&blob, id],
        )?;
        if rows == 0 {
            return Err(Error::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// List all distinct project_ids in the database.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn list_all_project_ids(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT project_id FROM memories ORDER BY project_id")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut results = Vec::new();
        for row_result in rows {
            results.push(row_result?);
        }
        Ok(results)
    }
}
