//! SQLite backend for vipune memory storage.
//!
//! Provides: `Database`, `Memory`, and the `db_error`, `embedding`, `search`,
//! `fts`, `hash`, `identity`, `project_queries` submodules. The last two were
//! split out of this file to keep it near the project's 500-line cap; the
//! module map in `docs/architecture.md` lists where each surface now lives.

pub(crate) mod db_error;
pub mod embedding;
pub mod export_scan;
pub mod fts;
pub mod hash;
pub mod identity;
pub mod import;
pub mod insert;
pub mod list;
pub mod migration_types;
pub mod migrations;
pub(crate) mod project_queries;
pub mod query_mod;
pub mod search;
pub mod supersede;
pub mod update;

pub use migration_types::MigrationReport;

#[cfg(test)]
pub(crate) mod test_fixtures;

#[cfg(test)]
mod tests;

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::time::Duration;
use uuid::Uuid;

pub use self::embedding::{blob_to_vec, vec_to_blob};
pub(crate) use self::insert::NewRow;
pub use self::query_mod::{build_filters, map_row_to_memory};
pub use self::update::UpdateOptions;

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
    /// Operator-assigned importance (low, medium, high, critical; default medium).
    pub importance: String,
}

/// Error types for SQLite operations.
#[derive(Debug, Clone)]
pub enum Error {
    /// SQLite database error with message.
    Sqlite(String),
    /// The database is locked or busy (`SQLITE_BUSY` / `SQLITE_LOCKED`).
    ///
    /// The verdict is decided once from the SQLite result code by
    /// [`db_error::classify_db_error`] and carried as this variant, so no
    /// layer above has to re-derive it from message text (issue #255).
    Locked,
    /// Embedding BLOB for a specific memory failed to decode.
    ///
    /// Carries the affected memory `id` so the corrupt row can be located and
    /// repaired (see issue #186) instead of surfacing as a generic per-column
    /// conversion failure.
    CorruptEmbedding { id: String, reason: String },
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
    /// Pre-flight token scan found rows exceeding the embedding token limit.
    /// Carries the offending memory ids structurally so callers can act
    /// without parsing a message.
    MigrationRefused { offending: Vec<String> },
    /// The re-embed pass ran but at least one row failed to embed.
    /// The migration marker is left in place and the old identity is kept;
    /// the `report` carries the per-row failure details.
    MigrationIncomplete { report: MigrationReport },
}

impl Error {
    /// True for the structured embedding-decoding variants — the ones that
    /// carry fields a caller acts on (`id`/`reason`, `expected`/`actual`) and
    /// are therefore allowed to cross the public boundary as typed payloads
    /// and to be trusted as the source of a row-conversion failure.
    ///
    /// The single enumeration of that set: `db_error::is_trusted_domain_error`,
    /// `commands::db_busy::open_failure`, `From<crate::sqlite::Error> for
    /// vipune::Error` and the MCP `data` arms all key off it, so adding a
    /// structured variant is one edit here rather than four coordinated lists.
    #[must_use]
    pub(crate) fn is_structured_embedding_error(&self) -> bool {
        matches!(
            self,
            Error::CorruptEmbedding { .. }
                | Error::InvalidBlobSize { .. }
                | Error::MismatchedDimensions { .. }
        )
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Sqlite(msg) => write!(f, "Database error: {}", msg),
            // Byte-compatible with the old `Sqlite("database is locked")`
            // rendering: the pi-rukas `ctx7` shell wrapper (outside this repo)
            // matches this text in the CLI's stderr. `hook/run.rs` does NOT
            // match it — its only text classifier is the UNIQUE-constraint
            // check, and a locked insert is `Error::Locked`, which that
            // matcher never sees.
            Error::Locked => write!(f, "{}", crate::sqlite::db_error::LOCKED_DISPLAY),
            Error::CorruptEmbedding { id, reason } => {
                write!(f, "Corrupt embedding for memory id: {} ({})", id, reason)
            }
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
            Error::MigrationRefused { offending } => write!(
                f,
                "Migration refused: {} row(s) exceed the token limit: {}",
                offending.len(),
                offending.join(", ")
            ),
            Error::MigrationIncomplete { report } => write!(
                f,
                "Migration incomplete: {} row(s) failed ({} reindexed, {} skipped)",
                report.failures.len(),
                report.reindexed,
                report.skipped
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<rusqlite::Error> for Error {
    fn from(err: rusqlite::Error) -> Self {
        match db_error::classify_db_error(&err) {
            // The typed verdict survives the boundary: the classifier read the
            // SQLite result code, so nothing above has to match the busy text.
            db_error::DbErrorKind::Locked => Error::Locked,
            db_error::DbErrorKind::Domain(domain) => domain.clone(),
            db_error::DbErrorKind::Other => Error::Sqlite(err.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// 0-indexed column position of `embedding` in the canonical 12-column memory row shape
/// (see issue #186). NOT applicable to projections that omit leading columns.
pub(crate) const EMBEDDING_COLUMN: usize = 4;

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
    /// Initializes the schema if new, then runs any pending migrations.
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

    /// Insert a memory with explicit timestamps.
    ///
    /// Production counterpart to [`insert`]: the lifecycle commands (prune,
    /// promote) need deterministic `created_at` values so eligibility rules
    /// such as "age > T" can be tested and reasoned about without sleeping.
    #[allow(clippy::too_many_arguments)] // signature mirrors insert(); 7/8 data fields map 1:1 to columns
    #[allow(dead_code)] // used by prune/promote lifecycle handlers in the binary target
    pub fn insert_with_time(
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
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, importance)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                &id,
                project_id,
                content,
                &blob,
                metadata,
                created_at,
                updated_at,
                memory_type,
                status,
                "medium",
            ],
        )?;

        Ok(id)
    }

    /// Insert a new memory with embedding and a precomputed dedup hash (issue #191).
    ///
    /// Used by the agent-lifecycle hook path so `content_hash` is populated at
    /// insert time and the `idx_memories_dedup` unique index enforces dedup.
    ///
    /// # Errors
    ///
    /// Returns error on invalid embedding dimensions, write failure, or unique
    /// index rejection of the `(project_id, content_hash)` pair.
    #[allow(clippy::too_many_arguments)] // mirrors insert(); 7/8 data fields map 1:1 to columns
    #[allow(dead_code)] // used by hook insert path (task-b) once src/hook/ lands
    pub fn insert_with_hash(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        memory_type: &str,
        status: &str,
        content_hash: &str,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata,
                created_at, updated_at, type, status, content_hash)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                &id,
                project_id,
                content,
                &blob,
                metadata,
                &now,
                &now,
                memory_type,
                status,
                content_hash
            ],
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
            SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
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

    /// Construct a `Database` from an already-opened `Connection`.
    /// Bypasses schema creation and migration; the connection must point
    /// to a fully initialised database.
    #[allow(dead_code)] // lib target compiles src/sqlite/ but not src/commands/; only caller is in binary target
    pub(crate) fn from_conn(conn: Connection) -> Self {
        Self { conn }
    }

    /// Borrow the connection — the escape hatch for *connection-level* work
    /// like `rusqlite::backup::Backup` (`commands/backup.rs`) or a `PRAGMA`, not
    /// for queries (`set_busy_timeout` is the `Database` method for the busy
    /// timeout, for instance). `&Connection` is write-capable: rusqlite's
    /// `execute` takes `&self`, so a statement through it bypasses `Database`'s
    /// error mapping and transaction discipline. New queries belong on a
    /// `Database` method.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Open a write transaction on the connection.
    ///
    /// `&mut self` because rusqlite's `Connection::transaction` takes `&mut
    /// self`: it uses that borrow to exclude every other statement for the
    /// transaction's duration. A constraint on other handles, not on writes —
    /// [`Self::conn`] hands out a write-capable `&Connection`, and a statement
    /// through it is atomic only if its caller opened a transaction first.
    ///
    /// # Errors
    ///
    /// Returns error if the transaction cannot be started (e.g. database
    /// locked beyond the busy timeout).
    pub fn begin_transaction(&mut self) -> rusqlite::Result<rusqlite::Transaction<'_>> {
        self.conn.transaction()
    }

    /// Set the SQLite busy timeout. Used by reindex for fast-fail on locks.
    ///
    /// # Errors
    ///
    /// Returns error if the busy timeout cannot be set.
    pub fn set_busy_timeout(&self, timeout: Duration) -> Result<()> {
        self.conn.busy_timeout(timeout)?;
        Ok(())
    }

    /// List ALL rows for a project (no status filter, no limit).
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
            let embedding = blob_to_vec(&blob)
                .map_err(|e| query_mod::corrupt_embedding_error(id.clone(), e))?;
            results.push((id, content, embedding));
        }
        Ok(results)
    }

    /// Update only the embedding BLOB for a memory (used by reindex).
    /// Does NOT touch `updated_at`, `retrieval_count`, or `last_retrieved_at`.
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

    /// Merge all rows from one project_id into another within a single transaction.
    /// `from == to` short-circuits (returns 0, no writes). Only `project_id` changes.
    ///
    /// # Returns
    ///
    /// Number of rows moved.
    ///
    /// # Errors
    ///
    /// Returns error if the database query or transaction fails.
    pub fn merge_project_ids(
        &mut self,
        from_project_id: &str,
        to_project_id: &str,
    ) -> Result<usize> {
        // Short-circuit: merging into self does nothing.
        if from_project_id == to_project_id {
            return Ok(0);
        }

        let tx = self.conn.transaction()?;

        // Move them and count rows affected.
        let count: usize = tx.execute(
            "UPDATE memories SET project_id = ?1 WHERE project_id = ?2",
            [to_project_id, from_project_id],
        )?;

        tx.commit()?;
        Ok(count)
    }
}
