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
pub mod migrations;
pub mod query_mod;
pub mod search;

pub use self::embedding::{blob_to_vec, vec_to_blob};
pub use self::query_mod::map_row_to_memory;

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Result as SqliteResult, params};
use std::path::Path;
use uuid::Uuid;

/// A single memory record with metadata, embedding vector, and optional similarity score.
///
/// Contains the stored memory content, metadata, embedding, and timestamps. The similarity
/// field is populated only during search operations.
#[derive(Clone, Debug)]
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
    #[allow(dead_code)] // Library API: exposed for consumers, unused in CLI
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
    #[allow(dead_code)] // Library API: exposed for consumers
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    #[allow(dead_code)] // Library API: exposed for consumers
    pub status: String,
    /// ID of the memory that superseded this one (if any).
    #[allow(dead_code)] // Library API: exposed for consumers
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
            SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
            FROM memories
            WHERE id = ?1
            "#,
        )?;

        let result = stmt.query_row([id], map_row_to_memory).optional()?;
        Ok(result)
    }

    /// List memories for a project, ordered by creation time (newest first).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `limit` - Maximum number of results to return
    /// * `memory_types` - Optional filter for memory types (None = no filter by type)
    /// * `statuses` - Optional filter for lifecycle statuses (None = default to 'active')
    ///
    /// # Errors
    ///
    /// Returns error if the limit is invalid or the query fails.
    pub fn list(
        &self,
        project_id: &str,
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<Vec<Memory>> {
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
                param_index += types.len();
            }
        }

        let where_clause = where_clauses.join(" AND ");
        let query = format!(
            "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by
             FROM memories WHERE {} ORDER BY created_at DESC LIMIT ?{}",
            where_clause, param_index
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
        let limit_param = limit as i64;
        params.push(&limit_param);

        let memories: SqliteResult<Vec<Memory>> = stmt
            .query_map(params.as_slice(), map_row_to_memory)?
            .collect();

        Ok(memories?)
    }

    /// Update a memory's content and/or metadata.
    ///
    /// - If content is provided: updates content and embedding
    /// - If metadata is provided: updates metadata (full replacement, not merge)
    /// - If both provided: updates both
    ///
    /// Returns an error if the memory does not exist.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Embedding has invalid dimensions (when content is provided)
    /// - Memory not found
    /// - Query fails
    pub fn update(
        &self,
        id: &str,
        content: Option<&str>,
        embedding: Option<&[f32]>,
        metadata: Option<&str>,
        memory_type: Option<&str>,
        status: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        // Build dynamic UPDATE query based on what's being updated
        let mut set_clauses: Vec<&str> = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(text) = content {
            set_clauses.push("content = ?");
            let blob =
                vec_to_blob(embedding.ok_or_else(|| {
                    Error::Sqlite("Content update requires embedding".to_string())
                })?)?;
            params.push(Box::new(text.to_string()));
            set_clauses.push("embedding = ?");
            params.push(Box::new(blob));
        }

        if let Some(meta) = metadata {
            set_clauses.push("metadata = ?");
            params.push(Box::new(meta.to_string()));
        }

        if let Some(t) = memory_type {
            set_clauses.push("type = ?");
            params.push(Box::new(t.to_string()));
        }

        if let Some(s) = status {
            set_clauses.push("status = ?");
            params.push(Box::new(s.to_string()));
        }

        if set_clauses.is_empty() {
            return Err(Error::Sqlite(
                "At least one of content, metadata, memory_type, or status must be provided"
                    .to_string(),
            ));
        }

        set_clauses.push("updated_at = ?");
        params.push(Box::new(now));

        let sql = format!(
            "UPDATE memories SET {} WHERE id = ?",
            set_clauses.join(", ")
        );

        // Add id as last parameter
        params.push(Box::new(id.to_string()));

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = self.conn.execute(&sql, param_refs.as_slice())?;

        if rows == 0 {
            return Err(Error::Sqlite("No memory found".to_string()));
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

    /// Atomically supersede an old memory with a new one.
    ///
    /// In a single transaction:
    /// 1. Inserts the new memory
    /// 2. Sets old memory's status to 'superseded' and superseded_by to new ID
    #[allow(dead_code)] // Public API: used by CLI --supersedes flag and library consumers
    pub fn supersede(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        memory_type: &str,
        old_id: &str,
    ) -> Result<String> {
        // Verify old memory exists
        let old = self
            .get(old_id)?
            .ok_or_else(|| Error::NotFound(format!("memory to supersede not found: {}", old_id)))?;
        if old.project_id != project_id {
            return Err(Error::InvalidInput(
                "cannot supersede memory from different project".to_string(),
            ));
        }

        let new_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &new_id,
                project_id,
                content,
                &blob,
                metadata,
                &now,
                &now,
                memory_type,
                "active"
            ],
        )?;
        tx.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
            params![&new_id, &now, old_id],
        )?;
        tx.commit()?;

        Ok(new_id)
    }

    /// List memories for a project created since a given timestamp.
    ///
    /// Returns memories with `created_at > since_timestamp`, ordered by creation time (newest first).
    /// The timestamp comparison is exclusive (does not include memories created exactly at the timestamp).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `since_timestamp` - RFC3339-formatted timestamp (exclusive lower bound)
    /// * `limit` - Maximum number of results to return
    /// * `memory_types` - Optional filter for memory types (None = no filter by type)
    /// * `statuses` - Optional filter for lifecycle statuses (None = default to 'active')
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The timestamp is not valid RFC3339
    /// - The limit is invalid
    /// - The database query fails
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chrono::Utc;
    /// let one_hour_ago = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
    /// let recent = db.list_since("project", &one_hour_ago, 10, None, None)?;
    /// ```
    #[allow(dead_code)] // Public API for library consumers (e.g., kide)
    pub fn list_since(
        &self,
        project_id: &str,
        since_timestamp: &str,
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<Vec<Memory>> {
        // Validate timestamp format by parsing it
        let _parsed = chrono::DateTime::parse_from_rfc3339(since_timestamp)
            .map_err(|e| Error::Sqlite(format!("Invalid RFC3339 timestamp: {}", e)))?;

        let mut where_clauses = vec!["project_id = ?1".to_string(), "created_at > ?2".to_string()];
        let mut param_index = 3usize;

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
                param_index += types.len();
            }
        }

        let where_clause = where_clauses.join(" AND ");
        let query = format!(
            "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
             FROM memories WHERE {} ORDER BY created_at DESC LIMIT ?{}",
            where_clause, param_index
        );

        let mut stmt = self.conn.prepare(&query)?;

        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&project_id, &since_timestamp];
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
        let limit_param = limit as i64;
        params.push(&limit_param);

        let memories: SqliteResult<Vec<Memory>> = stmt
            .query_map(params.as_slice(), map_row_to_memory)?
            .collect();

        Ok(memories?)
    }

    /// Get multiple memories by their IDs.
    ///
    /// Returns results in the same order as the input IDs. Missing IDs are represented as `None`.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of memory IDs to retrieve
    ///
    /// # Returns
    ///
    /// Vector of `Option<Memory>` with the same length as `ids`. Each position corresponds
    /// to the ID at the same index in the input. `Some(memory)` if found, `None` if not found.
    ///
    /// # Errors
    ///
    /// Returns error if any database query fails (individual not-found cases are handled via `None`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let results = db.get_many(&["id1", "id2", "missing-id"])?;
    /// assert_eq!(results.len(), 3);
    /// assert!(results[0].is_some()); // Found id1
    /// assert!(results[1].is_some()); // Found id2
    /// assert!(results[2].is_none()); // Missing ID
    /// ```
    #[allow(dead_code)] // Public API for library consumers (e.g., kide)
    pub fn get_many(&self, ids: &[&str]) -> Result<Vec<Option<Memory>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!(
            r#"
            SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
            FROM memories
            WHERE id IN ({})
            "#,
            placeholders
        );

        let mut stmt = self.conn.prepare(&query)?;

        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

        let rows: SqliteResult<Vec<(String, Memory)>> = stmt
            .query_map(params.as_slice(), |row| {
                let blob: Vec<u8> = row.get(4)?;
                let embedding = blob_to_vec(&blob).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Blob,
                        Box::new(e),
                    )
                })?;
                Ok((
                    row.get::<_, String>(0)?,
                    Memory {
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
                    },
                ))
            })?
            .collect();

        let found_memories: std::collections::HashMap<String, Memory> = rows?.into_iter().collect();

        // Preserve input ordering
        let results: Vec<Option<Memory>> = ids
            .iter()
            .map(|id| found_memories.get(*id).cloned())
            .collect();

        Ok(results)
    }

    /// Get internal connection (for test use).
    #[cfg(test)]
    pub(crate) fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Increment retrieval_count and set last_retrieved_at for given memory IDs.
    ///
    /// Called by CLI handlers and MCP handlers after retrieving memories via search/get.
    /// This method is NOT called internally by supersede or other operations that don't
    /// represent user-initiated retrieval.
    pub fn touch_memories(&self, ids: &[&str]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let now = chrono::Utc::now().to_rfc3339();
        let placeholders: Vec<String> = (0..ids.len()).map(|i| format!("?{}", i + 2)).collect();
        let sql = format!(
            "UPDATE memories SET retrieval_count = retrieval_count + 1, last_retrieved_at = ?1 WHERE id IN ({})",
            placeholders.join(", ")
        );
        let params: Vec<&dyn rusqlite::types::ToSql> =
            std::iter::once(&now as &dyn rusqlite::types::ToSql)
                .chain(ids.iter().map(|id| id as &dyn rusqlite::types::ToSql))
                .collect();
        self.conn.execute(&sql, params.as_slice())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EMBEDDING_DIMS;
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
            .insert("proj1", "test content", &embedding, None, "fact", "active")
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
                "fact",
                "active",
            )
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.metadata, Some(r#"{"key": "value"}"#.to_string()));
    }

    #[test]
    fn test_insert_invalid_embedding() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 256];
        let result = db.insert("proj1", "test", &embedding, None, "fact", "active");
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
                "fact",
                "active",
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
                "fact",
                "active",
            )
            .unwrap();

        let memories = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].id, id2); // Newest first
        assert_eq!(memories[1].id, id1);
    }

    #[test]
    fn test_list_limit() {
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

        let memories = db.list("proj1", 2, None, None).unwrap();
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_update() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original", &embedding, None, "fact", "active")
            .unwrap();

        db.update(&id, Some("updated"), Some(&embedding), None, None, None)
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.content, "updated");
    }

    #[test]
    fn test_update_text_only_preserves_metadata() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert(
                "proj1",
                "original",
                &embedding,
                Some(r#"{"tag": "old"}"#),
                "fact",
                "active",
            )
            .unwrap();

        let new_embedding = vec![0.2f32; 384];
        db.update(&id, Some("updated"), Some(&new_embedding), None, None, None)
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.content, "updated");
        assert_eq!(m.metadata, Some(r#"{"tag": "old"}"#.to_string()));
    }

    #[test]
    fn test_update_metadata_only() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original", &embedding, None, "fact", "active")
            .unwrap();

        db.update(&id, None, None, Some(r#"{"tag": "new"}"#), None, None)
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.content, "original"); // Content unchanged
        assert_eq!(m.metadata, Some(r#"{"tag": "new"}"#.to_string()));
    }

    #[test]
    fn test_update_both_text_and_metadata() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert(
                "proj1",
                "original",
                &embedding,
                Some(r#"{"tag": "old"}"#),
                "fact",
                "active",
            )
            .unwrap();

        let new_embedding = vec![0.2f32; 384];
        db.update(
            &id,
            Some("updated"),
            Some(&new_embedding),
            Some(r#"{"tag": "new"}"#),
            None,
            None,
        )
        .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.content, "updated");
        assert_eq!(m.metadata, Some(r#"{"tag": "new"}"#.to_string()));
    }

    #[test]
    fn test_update_with_none_both() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original", &embedding, None, "fact", "active")
            .unwrap();

        let result = db.update(&id, None, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_nonexistent() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let result = db.update(
            "nonexistent",
            Some("content"),
            Some(&embedding),
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_update_memory_type_only() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        // Update memory type only
        db.update(&id, None, None, None, Some("preference"), None)
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.memory_type, "preference");
        // Other fields unchanged
        assert_eq!(m.content, "test content");
        assert_eq!(m.status, "active");
    }

    #[test]
    fn test_update_status_only() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert(
                "proj1",
                "test content",
                &embedding,
                None,
                "fact",
                "candidate",
            )
            .unwrap();

        // Update status only
        db.update(&id, None, None, None, None, Some("active"))
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.status, "active");
        // Other fields unchanged
        assert_eq!(m.content, "test content");
        assert_eq!(m.memory_type, "fact");
    }

    #[test]
    fn test_update_type_and_status_together() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert(
                "proj1",
                "test content",
                &embedding,
                None,
                "fact",
                "candidate",
            )
            .unwrap();

        // Update both type and status in one call
        db.update(&id, None, None, None, Some("preference"), Some("active"))
            .unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.memory_type, "preference");
        assert_eq!(m.status, "active");
        // Content unchanged
        assert_eq!(m.content, "test content");
    }

    #[test]
    fn test_update_rejects_invalid_type() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        // Database layer accepts any type string - validation happens at store layer
        let result = db.update(&id, None, None, None, Some("invalid_type"), None);
        assert!(result.is_ok()); // DB allows it
    }

    #[test]
    fn test_update_rejects_empty_args() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let result = db.update(&id, None, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "content", &embedding, None, "fact", "active")
            .unwrap();

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
        db.insert("proj1", "proj1 content", &embedding, None, "fact", "active")
            .unwrap();
        db.insert("proj2", "proj2 content", &embedding, None, "fact", "active")
            .unwrap();

        let list1 = db.list("proj1", 10, None, None).unwrap();
        let list2 = db.list("proj2", 10, None, None).unwrap();

        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].project_id, "proj1");
        assert_eq!(list2[0].project_id, "proj2");
    }

    #[test]
    fn test_get_includes_embedding() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id).unwrap().unwrap();
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        for (i, &val) in embedding.iter().enumerate() {
            assert!((memory.embedding[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_list_includes_embeddings() {
        let db = create_test_db();
        let embedding1 = vec![0.1f32; EMBEDDING_DIMS];
        let embedding2 = vec![0.2f32; EMBEDDING_DIMS];

        db.insert("proj1", "first", &embedding1, None, "fact", "active")
            .unwrap();
        db.insert("proj1", "second", &embedding2, None, "fact", "active")
            .unwrap();

        let memories = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(memories.len(), 2);

        for memory in &memories {
            assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        }
    }

    #[test]
    fn test_embedding_roundtrip() {
        let db = create_test_db();
        let original = [0.123f32, 0.456f32, 0.789f32];
        let mut full_embedding = vec![0.1f32; EMBEDDING_DIMS];
        full_embedding[0] = original[0];
        full_embedding[1] = original[1];
        full_embedding[EMBEDDING_DIMS - 1] = original[2];

        let id = db
            .insert("proj1", "test", &full_embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id).unwrap().unwrap();
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        assert!((memory.embedding[0] - original[0]).abs() < 1e-6);
        assert!((memory.embedding[1] - original[1]).abs() < 1e-6);
        assert!((memory.embedding[EMBEDDING_DIMS - 1] - original[2]).abs() < 1e-6);
    }
}
