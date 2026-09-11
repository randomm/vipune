//! FTS5 full-text search and BM25 ranking (Issue #40).

use super::{Database, EMBEDDING_COLUMN, Error, Memory};

pub type Result<T> = std::result::Result<T, Error>;

/// Result of the bidirectional rowid-join FTS desync detection.
///
/// Detection is a bidirectional rowid join, NOT a count comparison (a count-only
/// check misses partial desync where counts are equal but rowid sets differ):
///
/// * `underpopulated_global` — number of `memories` rows whose `rowid` is missing
///   from `memories_fts` (the index is under-populated). Global; no project scope.
/// * `underpopulated_by_project` — the same rows attributed per `project_id` so the
///   doctor can name the affected project. Summing this map equals
///   `underpopulated_global`.
/// * `orphans` — number of `memories_fts` rows whose `rowid` has no `memories` row
///   (the index holds rows whose source memory is gone). Always a global count with
///   no project attribution; orphan content in an external-content table is undefined,
///   so orphan enumeration must never join on the content column.
///
/// A healthy database — including a brand-new one where both tables hold zero rows —
/// reports `is_desynced() == false`. Zero is not a desync.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FtsDesyncReport {
    /// Per-project count of `memories` rowids missing from `memories_fts`.
    pub underpopulated_by_project: std::collections::HashMap<String, usize>,
    /// Global count of `memories` rowids missing from `memories_fts`.
    pub underpopulated_global: usize,
    /// Global count of `memories_fts` rowids with no matching `memories` row.
    pub orphans: usize,
}

impl FtsDesyncReport {
    /// True when either direction of the join reports a mismatch.
    pub fn is_desynced(&self) -> bool {
        self.underpopulated_global > 0 || self.orphans > 0
    }

    /// Total number of actions a rebuild would take (rows to re-index + orphans to drop).
    pub fn total_desynced(&self) -> usize {
        self.underpopulated_global + self.orphans
    }
}

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
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&escaped_query, &project_id];
        let param_index = super::build_filters(
            &mut where_clauses,
            &mut params,
            3,
            statuses,
            memory_types,
            "m.",
        );

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

        let limit_i64 = limit as i64;
        params.push(&limit_i64);

        let mut stmt = self.conn.prepare(&sql)?;

        let memories: rusqlite::Result<Vec<Memory>> = stmt
            .query_map(params.as_slice(), |row| {
                // Positions match the SELECT above (see EMBEDDING_COLUMN).
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(EMBEDDING_COLUMN)?;
                let embedding = super::embedding::blob_to_vec(&blob)
                    .map_err(|e| super::query_mod::corrupt_embedding_error(id.clone(), e))?;
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

    /// Detect FTS desync via a bidirectional rowid join.
    ///
    /// This is a READ-ONLY query: it only SELECTs rowids, never writes, and never
    /// reads the (undefined) content of an orphan FTS row. It returns a
    /// [`FtsDesyncReport`] with per-project under-population counts and a global
    /// orphan count. A zero/zero database is healthy, not desynced.
    ///
    /// # Errors
    ///
    /// Returns an error if the FTS table does not exist or a query fails.
    pub fn detect_fts_desync(&self) -> Result<FtsDesyncReport> {
        let fts_exists: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'",
            [],
            |row| row.get(0),
        )?;
        if fts_exists == 0 {
            // No FTS table at all — nothing to desync against.
            return Ok(FtsDesyncReport::default());
        }

        // Direction 1 (under-population): memories rowids missing from memories_fts,
        // attributed per project. Uses `IS NOT` (not `NOT IN`) to be NULL-safe.
        let mut by_project: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut underpopulated_global: usize = 0;
        let mut stmt = self.conn.prepare(
            "SELECT m.project_id, COUNT(*) AS c FROM memories m WHERE m.rowid IS NOT (SELECT fts.rowid FROM memories_fts fts WHERE fts.rowid = m.rowid) GROUP BY m.project_id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row_res in rows {
            let (pid, c) = row_res?;
            let count = c as usize;
            underpopulated_global += count;
            *by_project.entry(pid).or_insert(0) += count;
        }

        // Direction 2 (orphans): memories_fts rowids with no memories row. Enumerated
        // via rowid IS NOT (SELECT ...) — never a join that reads the content column,
        // because content for an orphan row in an external-content table is undefined.
        let orphan_rowids: Vec<i64> = {
            let mut ostmt = self.conn.prepare(
                "SELECT fts.rowid FROM memories_fts fts WHERE fts.rowid IS NOT (SELECT m.rowid FROM memories m WHERE m.rowid = fts.rowid)",
            )?;
            let orows = ostmt.query_map([], |row| row.get::<_, i64>(0))?;
            let mut out = Vec::new();
            for row_res in orows {
                out.push(row_res?);
            }
            out
        };
        let orphans = orphan_rowids.len();

        Ok(FtsDesyncReport {
            underpopulated_by_project: by_project,
            underpopulated_global,
            orphans,
        })
    }

    /// Rebuild the FTS index in place using the FTS5 `rebuild` special command.
    ///
    /// This is the only rebuild path the repo uses and it must be called on a
    /// read-write connection (the caller is responsible for opening one and setting
    /// a bounded busy timeout). It re-reads every `memories` row into `memories_fts`
    /// and does not touch `memories` data.
    ///
    /// # Errors
    ///
    /// Returns an error if the FTS table does not exist or the rebuild statement fails.
    pub fn rebuild_fts(&self) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')",
            [],
        )?;
        Ok(())
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
