//! FTS5 full-text search and BM25 ranking (Issue #40).

use super::{Database, Error, Memory};

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
    /// # Arguments
    ///
    /// * `query` - Search query text
    /// * `project_id` - Project identifier
    /// * `limit` - Maximum number of results
    /// * `memory_types` - Optional filter by memory types (None = no filter)
    /// * `statuses` - Optional filter by statuses (None = default to 'active')
    ///
    /// # Errors
    ///
    /// Returns error if the FTS5 search fails.
    pub fn search_bm25(
        &self,
        query: &str,
        project_id: &str,
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<Vec<Memory>> {
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

        let mut where_clauses = vec![
            "memories_fts MATCH ?1".to_string(),
            "m.project_id = ?2".to_string(),
        ];
        let mut param_index = 3usize;

        // Status filter (default to active if None)
        if let Some(statuses) = statuses {
            if !statuses.is_empty() {
                let placeholders: Vec<String> = (0..statuses.len())
                    .map(|i| format!("?{}", param_index + i))
                    .collect();
                where_clauses.push(format!("m.status IN ({})", placeholders.join(", ")));
                param_index += statuses.len();
            }
        } else {
            where_clauses.push(format!("m.status = ?{}", param_index));
            param_index += 1;
        }

        // Type filter (only if explicitly provided)
        if let Some(types) = memory_types {
            if !types.is_empty() {
                let placeholders: Vec<String> = (0..types.len())
                    .map(|i| format!("?{}", param_index + i))
                    .collect();
                where_clauses.push(format!("m.type IN ({})", placeholders.join(", ")));
                param_index += types.len();
            }
        }

        let where_clause = where_clauses.join(" AND ");
        let sql = format!(
            r#"
            SELECT m.id, m.project_id, m.content, m.metadata, m.embedding, m.created_at, m.updated_at, m.type, m.status, m.superseded_by, m.retrieval_count, m.last_retrieved_at,
                   bm25(memories_fts) as bm25_score
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE {}
            ORDER BY bm25(memories_fts)
            LIMIT ?{}
            "#,
            where_clause, param_index
        );

        let mut stmt = self.conn.prepare(&sql)?;

        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&escaped_query, &project_id];
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
        let limit_i64 = limit as i64;
        params.push(&limit_i64);

        let memories: rusqlite::Result<Vec<Memory>> = stmt
            .query_map(params.as_slice(), |row| {
                let blob: Vec<u8> = row.get(4)?;
                let embedding = super::embedding::blob_to_vec(&blob).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Blob,
                        Box::new(e),
                    )
                })?;
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    metadata: row.get(3)?,
                    embedding,
                    created_at: row.get(5)?,
                    updated_at: row.get(6)?,
                    memory_type: row.get(7)?,
                    status: row.get(8)?,
                    superseded_by: row.get(9)?,
                    retrieval_count: row.get(10)?,
                    last_retrieved_at: row.get(11)?,
                    similarity: Some(row.get::<_, f64>(12)?),
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
