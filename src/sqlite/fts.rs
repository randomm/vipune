//! FTS5 full-text search and BM25 ranking (Issue #40).

use super::{Database, Error, Memory};
use rusqlite::params;

pub type Result<T> = std::result::Result<T, Error>;

impl Database {
    /// Initialize FTS5 table if needed and validate/migrate schema.
    ///
    /// This method:
    /// 1. Checks if memories_fts table exists with correct schema
    /// 2. If schema is outdated, performs drop-and-recreate migration
    /// 3. Validates consistency by comparing row counts
    ///
    /// # Errors
    ///
    /// Returns error if migration fails or consistency check detects data loss.
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

    /// Search memories using FTS5 BM25 ranking.
    ///
    /// # Errors
    ///
    /// Returns error if the FTS5 search fails.
    #[allow(dead_code)] // FTS5 feature for Issue #40 hybrid search
    pub fn search_bm25(&self, query: &str, project_id: &str, limit: usize) -> Result<Vec<Memory>> {
        super::search::validate_limit(limit)?;

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

        let memories: rusqlite::Result<Vec<Memory>> = stmt
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

    /// Escape and normalize FTS5 query string.
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
