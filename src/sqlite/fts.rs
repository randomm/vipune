//! FTS5 full-text search and BM25 ranking (Issue #40).

use super::{Database, EMBEDDING_COLUMN, Error, Memory};
pub type Result<T> = std::result::Result<T, Error>;

/// Per-project under-population result from an FTS5 desync scan.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // lib target compiles src/sqlite/ but not src/commands/; only caller is in binary target
pub struct ProjectFtsCounts {
    /// Project identifier the counts belong to.
    pub project_id: String,
    /// Number of `memories` rows for this project.
    pub memory_count: i64,
    /// Number of `memories_fts` rows for this project.
    pub fts_count: i64,
    /// Rowids present in `memories` but missing from `memories_fts`
    /// (the FTS index is under-populated for this project).
    pub missing_from_fts: i64,
}

/// Read-only report of FTS5 desync, produced by a bidirectional rowid join
/// (issue #193). Never reads the content column of `memories_fts`: for an
/// orphan FTS rowid the external-content backing row does not exist, so
/// `content` is undefined and only `rowid` may be selected.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // lib target compiles src/sqlite/ but not src/commands/; only caller is in binary target
pub struct FtsDesyncReport {
    /// Under-population counts, one entry per project found in `memories`.
    /// The global orphan count is intentionally NOT attributed to a project.
    pub projects: Vec<ProjectFtsCounts>,
    /// Total number of `memories` rows across all projects.
    pub total_memories: i64,
    /// Total number of `memories_fts` rows across all projects.
    pub total_fts: i64,
    /// Rowids present in `memories_fts` but with no `memories` row. Always
    /// a global count — orphan content is undefined so no project attribution
    /// is possible.
    pub orphan_fts_rows: i64,
}

impl FtsDesyncReport {
    /// True when either direction of the rowid join disagrees: some project
    /// has `memories` rowids missing from `memories_fts`, or `memories_fts`
    /// holds rowids with no backing `memories` row.
    ///
    /// A 0/0 database (brand-new or fully empty) is NOT a desync.
    #[allow(dead_code)] // lib target compiles src/sqlite/ but not src/commands/; only caller is in binary target
    pub fn is_desynced(&self) -> bool {
        self.projects.iter().any(|p| p.missing_from_fts > 0) || self.orphan_fts_rows > 0
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

    /// Detect FTS5 desync with a bidirectional rowid join (issue #193).
    ///
    /// Reports BOTH directions of the rowid mismatch, not a count comparison
    /// (equal-but-different rowid sets would defeat a count check):
    /// - per project: `memories` rowids missing from `memories_fts`
    ///   (under-population), scoped to `project_id` when given, all projects
    ///   otherwise;
    /// - globally: `memories_fts` rowids with no `memories` row (orphans),
    ///   enumerated via `SELECT rowid ... NOT IN (SELECT rowid FROM memories)`
    ///   — never a join pulling the content column, which is undefined for an
    ///   orphan row in an external-content table.
    ///
    /// Read-only: performs no writes, so the result is the same whether the
    /// connection is read-only or read-write. The caller (the `doctor --fts`
    /// handler) is responsible for opening with `SQLITE_OPEN_READ_ONLY`.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Optional project scope for the under-population side.
    ///   `None` scans every project present in `memories`.
    ///
    /// # Errors
    ///
    /// Returns error if any of the detection queries fails.
    #[allow(dead_code)] // lib target compiles src/sqlite/ but not src/commands/; only caller is in binary target
    pub fn detect_fts_desync(&self, project_id: Option<&str>) -> Result<FtsDesyncReport> {
        // Per-project under-population: memories rowids NOT IN memories_fts,
        // plus both row counts, joined in one pass over memories so the
        // per-project counts and the missing count cannot diverge.
        const UNDERPOP_SQL: &str = concat!(
            "SELECT m.project_id, m.missing, m.cnt, COALESCE(f.cnt, 0) AS fcnt FROM\n",
            "(SELECT project_id,\n",
            "        COUNT(*) AS cnt,\n",
            "        SUM(rowid NOT IN (SELECT rowid FROM memories_fts)) AS missing\n",
            " FROM memories GROUP BY project_id) m\n",
            "LEFT JOIN (SELECT project_id, COUNT(*) AS cnt FROM memories_fts GROUP BY project_id) f\n",
            " ON m.project_id = f.project_id"
        );

        let mut stmt = self.conn.prepare(UNDERPOP_SQL)?;

        let rows: Vec<(String, i64, i64, i64)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut projects: Vec<ProjectFtsCounts> = rows
            .into_iter()
            .map(
                |(project_id, missing, memory_count, fts_count)| ProjectFtsCounts {
                    project_id,
                    memory_count,
                    fts_count,
                    missing_from_fts: missing,
                },
            )
            .collect();

        let total_memories: i64 = projects.iter().map(|p| p.memory_count).sum();
        let total_fts: i64 = projects.iter().map(|p| p.fts_count).sum();

        if let Some(scope) = project_id {
            projects.retain(|p| p.project_id == scope);
        } else {
            projects.sort_by(|a, b| a.project_id.cmp(&b.project_id));
        }

        // Global orphan count: memories_fts rowids with no memories row.
        // SELECT rowid only — content for an orphan row is undefined.
        let orphan_fts_rows: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM (SELECT rowid FROM memories_fts WHERE rowid NOT IN (SELECT rowid FROM memories))",
            [],
            |row| row.get(0),
        )?;

        Ok(FtsDesyncReport {
            projects,
            total_memories,
            total_fts,
            orphan_fts_rows,
        })
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
