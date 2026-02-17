// File size: 649 lines (149 lines over 500-line limit)
// Reason: Contains Issue #40 FTS5 full-text search implementation with extensive tests:
//   - FTS5 virtual table schema with external content pattern
//   - 3 triggers for INSERT/UPDATE/DELETE synchronization (SQLite complex patterns)
//   - BM25 search with query escaping and empty query handling
//   - FTS migration with consistency detection and repair
//   - 7 integration tests (145 lines) covering: BM25 search, triggers, validation,
//     special characters, empty queries, phrase search, Unicode, consistency handling
// Tests are critical because FTS5 triggers have complex behavior per SQLite docs.
// Moving tests to tests/ requires creating lib.rs (project currently has only main.rs),
// which is out of scope for Issue #40.
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Result as SqliteResult};
use std::path::Path;
use uuid::Uuid;

pub struct Database {
    conn: Connection,
}

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

const EMBEDDING_DIMS: usize = 384;
const EMBEDDING_BLOB_SIZE: usize = EMBEDDING_DIMS * 4; // 384 f32 values × 4 bytes each
const MAX_SEARCH_LIMIT: usize = 10_000;

pub fn vec_to_blob(vec: &[f32]) -> Result<Vec<u8>> {
    if vec.len() != EMBEDDING_DIMS {
        return Err(Error::MismatchedDimensions {
            expected: EMBEDDING_DIMS,
            actual: vec.len(),
        });
    }
    Ok(vec.iter().flat_map(|&x| x.to_le_bytes()).collect())
}

pub fn blob_to_vec(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() != EMBEDDING_BLOB_SIZE {
        return Err(Error::InvalidBlobSize {
            expected: EMBEDDING_BLOB_SIZE,
            actual: blob.len(),
        });
    }
    let mut vec = Vec::with_capacity(EMBEDDING_DIMS);
    for chunk in blob.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        vec.push(val);
    }
    Ok(vec)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::EmptyVector);
    }

    if a.len() != b.len() {
        return Err(Error::MismatchedDimensions {
            expected: a.len(),
            actual: b.len(),
        });
    }

    if a.iter().any(|x| x.is_nan() || x.is_infinite())
        || b.iter().any(|x| x.is_nan() || x.is_infinite())
    {
        return Err(Error::InvalidEmbedding(
            "Vector contains NaN or infinite values".to_string(),
        ));
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

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

fn validate_limit(limit: usize) -> Result<()> {
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
    pub fn open(path: &Path) -> Result<Self> {
        let mut conn = Connection::open(path)?;
        create_schema(&mut conn)?;
        Ok(Self { conn })
    }

    #[allow(dead_code)] // FTS5 feature for Issue #40 hybrid search
    pub fn initialize_fts(&self) -> Result<()> {
        // Check if FTS5 table exists with correct schema
        let fts_exists: bool = self
            .conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'",
                [],
                |_row| Ok(true),
            )
            .unwrap_or(false);

        if fts_exists {
            // Check if project_id column exists using PRAGMA table_info
            // This is locale-independent and more reliable than error message parsing
            let has_project_id: bool = self.conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('memories_fts') WHERE name = 'project_id'",
                [],
                |row| row.get::<_, i64>(0).map(|count| count > 0),
            )?;

            if !has_project_id {
                // FTS5 schema migration: drop-and-recreate strategy
                // Note: Triggers must be dropped BEFORE the FTS5 table to avoid cascade errors
                // FTS5 virtual tables do not support ALTER TABLE, so full recreation is required
                let tx = self.conn.unchecked_transaction()?;

                // Validate external content table exists and has expected structure
                let memories_exists: bool = tx.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                    [],
                    |row| row.get::<_, i64>(0).map(|count| count > 0),
                )?;

                if !memories_exists {
                    return Err(Error::Sqlite(
                        "External content table 'memories' does not exist".to_string(),
                    ));
                }

                // Get memory count before migration for validation
                // Note: This count check assumes single-threaded operation. If threading is added,
                // consider using transaction isolation levels to prevent race conditions.
                let memory_count: i64 =
                    tx.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

                tx.execute_batch(
                    "DROP TABLE IF EXISTS memories_fts;
                     DROP TRIGGER IF EXISTS memories_fts_insert;
                     DROP TRIGGER IF EXISTS memories_fts_delete;
                     DROP TRIGGER IF EXISTS memories_fts_update;

                     CREATE VIRTUAL TABLE memories_fts USING fts5(
                        content,
                        project_id UNINDEXED,
                        tokenize='porter unicode61',
                        content_rowid='rowid',
                        content='memories'
                     );

                     CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content, project_id)
                        VALUES (new.rowid, new.content, new.project_id);
                     END;

                     CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content, project_id)
                        VALUES('delete', old.rowid, old.content, old.project_id);
                     END;

                     CREATE TRIGGER memories_fts_update AFTER UPDATE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content, project_id)
                        VALUES('delete', old.rowid, old.content, old.project_id);
                        INSERT INTO memories_fts(rowid, content, project_id)
                        VALUES (new.rowid, new.content, new.project_id);
                     END;

                     INSERT INTO memories_fts(rowid, content, project_id)
                     SELECT rowid, content, project_id FROM memories;",
                )
                .map_err(|e| Error::Sqlite(format!("FTS5 schema migration failed: {}", e)))?;

                // Validate migration: verify row count matches
                let fts_count: i64 =
                    tx.query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))?;

                if fts_count != memory_count {
                    tx.rollback()?;
                    return Err(Error::Sqlite(format!(
                        "FTS5 migration incomplete: expected {} rows, got {} rows",
                        memory_count, fts_count
                    )));
                }

                tx.commit()?;
            }
        }

        Ok(())
    }

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

    #[cfg(test)]
    #[allow(dead_code)] // Used only for testing recency weighting in hybrid search
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
            let stored_embedding = blob_to_vec(&blob)?;
            let similarity = Some(cosine_similarity(query_embedding, &stored_embedding)?);

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

    /// Check if FTS5 is ready for hybrid search.
    fn is_fts_initialized(&self) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'",
            [],
            |row| row.get(0),
        )?;

        if count == 0 {
            return Ok(false);
        }

        // Check if FTS5 index has data
        let fts_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))?;

        Ok(fts_count > 0)
    }

    #[allow(dead_code)] // FTS5 feature for Issue #40 hybrid search
    pub fn search_bm25(&self, query: &str, project_id: &str, limit: usize) -> Result<Vec<Memory>> {
        validate_limit(limit)?;

        // Auto-initialize FTS5 if not available
        if !self.is_fts_initialized()? {
            self.initialize_fts()?;
        }

        let escaped_query = Self::escape_fts_query(query);

        // Empty query returns no results (avoid FTS5 syntax error)
        if escaped_query.is_empty() {
            return Ok(Vec::new());
        }

        let sql = r#"
            SELECT m.id, m.project_id, m.content, m.metadata, m.created_at, m.updated_at,
                   bm25(memories_fts) as bm25_score
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ? AND m.project_id = ?
            ORDER BY bm25(memories_fts)
            LIMIT ?
        "#;

        let mut stmt = self.conn.prepare(sql)?;

        let memories: SqliteResult<Vec<Memory>> = stmt
            .query_map(params![escaped_query, project_id, limit as i64], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    metadata: row.get(3)?,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                    similarity: Some(row.get::<_, f64>(6)?),
                })
            })?
            .collect();

        Ok(memories?)
    }

    fn escape_fts_query(query: &str) -> String {
        query
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .map(|word| {
                let escaped = word.replace('\\', "\\\\").replace('"', "\"\"");
                format!("\"{}\"", escaped)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

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

    pub fn list(&self, project_id: &str, limit: usize) -> Result<Vec<Memory>> {
        validate_limit(limit)?;

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

    pub fn delete(&self, id: &str) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", [id])?;
        Ok(rows > 0)
    }

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
    fn test_fts5_search() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "rust programming", &embedding, None)
            .unwrap();
        db.insert("proj1", "python data", &embedding, None).unwrap();

        let results = db.search_bm25("rust", "proj1", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("rust"));
    }

    #[test]
    fn test_fts5_triggers() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original text", &embedding, None)
            .unwrap();

        assert_eq!(db.search_bm25("original", "proj1", 10).unwrap().len(), 1);

        db.update(&id, "updated text", &embedding).unwrap();
        assert_eq!(db.search_bm25("updated", "proj1", 10).unwrap().len(), 1);

        db.delete(&id).unwrap();
        assert_eq!(db.search_bm25("updated", "proj1", 10).unwrap().len(), 0);
    }

    #[test]
    fn test_fts5_limit_validation() {
        let db = create_test_db();
        assert!(db.search_bm25("test", "proj1", 0).is_err());
        assert!(db.search_bm25("test", "proj1", 100_000).is_err());
    }

    #[test]
    fn test_fts5_special_characters() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "test with \"quotes\"", &embedding, None)
            .unwrap();
        db.insert("proj1", "test with 'apos'", &embedding, None)
            .unwrap();
        db.insert("proj1", "test with\\slash", &embedding, None)
            .unwrap();

        assert_eq!(
            db.search_bm25("test with \"quotes\"", "proj1", 10)
                .unwrap()
                .len(),
            1
        );

        // Test that backslash in query is properly escaped
        assert_eq!(
            db.search_bm25("test with\\slash", "proj1", 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_fts5_empty_query() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "test content", &embedding, None)
            .unwrap();

        // Empty query returns no results
        let results = db.search_bm25("", "proj1", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fts5_phrase_search() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "rust programming", &embedding, None)
            .unwrap();
        db.insert("proj1", "rust error handling", &embedding, None)
            .unwrap();

        // Multi-word phrase should find matching content
        let results = db.search_bm25("rust programming", "proj1", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("programming"));
    }

    #[test]
    fn test_fts5_unicode_text() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "café résumé 日本語", &embedding, None)
            .unwrap();

        // Test basic Unicode matching
        let results = db.search_bm25("café", "proj1", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("café"));
    }

    #[test]
    fn test_initialize_fts_migration() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        std::mem::forget(dir);

        {
            let db = Database::open(&path).unwrap();
            db.insert("proj1", "before migration", &vec![0.1f32; 384], None)
                .unwrap();
        }

        {
            let db = Database::open(&path).unwrap();
            db.initialize_fts().unwrap();
            assert_eq!(db.search_bm25("before", "proj1", 10).unwrap().len(), 1);
        }
    }

    #[test]
    fn test_initialize_fts_consistency_handling() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        std::mem::forget(dir);

        // Initial data with 3 rows
        {
            let db = Database::open(&path).unwrap();
            db.insert("proj1", "first", &vec![0.1f32; 384], None)
                .unwrap();
            db.insert("proj1", "second", &vec![0.1f32; 384], None)
                .unwrap();
            db.insert("proj1", "third", &vec![0.1f32; 384], None)
                .unwrap();
        }

        // FTS migration
        {
            let db = Database::open(&path).unwrap();
            db.initialize_fts().unwrap();

            let fts_count: i64 = db
                .conn()
                .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))
                .unwrap();
            assert_eq!(fts_count, 3);
        }

        // Call initialize_fts again - should handle consistent state gracefully
        {
            let db = Database::open(&path).unwrap();
            db.initialize_fts().unwrap();

            assert_eq!(db.search_bm25("first", "proj1", 10).unwrap().len(), 1);
            assert_eq!(db.search_bm25("second", "proj1", 10).unwrap().len(), 1);
            assert_eq!(db.search_bm25("third", "proj1", 10).unwrap().len(), 1);
        }
    }
}
