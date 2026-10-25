//! Project-level aggregate queries on `Database` (list / count).
//!
//! Split out of `mod.rs` to keep that file under the project 500-line cap;
//! these are the project-scan surface the doctor and prune handlers drive,
//! and need nothing but the read-only [`Database::conn`] handle.

use super::{Database, Result};

impl Database {
    /// List all distinct project_ids in the database.
    ///
    /// Callers that need counts as well should drive
    /// [`Database::row_counts_by_project`] and read the keys off its result:
    /// one `GROUP BY` pass covers both, whereas this query plus that one is
    /// two passes over `memories`.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn list_all_project_ids(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT DISTINCT project_id FROM memories ORDER BY project_id")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut results = Vec::new();
        for row_result in rows {
            results.push(row_result?);
        }
        Ok(results)
    }

    /// Row count for EVERY project, in one `GROUP BY` pass.
    ///
    /// The project-scan helper the doctor handlers use. A `COUNT(*)` per
    /// project id costs one query per project; the `doctor --projects` and
    /// `doctor --fts` scans drive this instead of looping (the same
    /// don't-drive-it-from-a-loop shape `export_scan::scan_all_rows` uses).
    ///
    /// **Contract: a project with zero `memories` rows is absent from the
    /// result** — `GROUP BY` only emits groups that exist. Callers that must
    /// report such a project read it from elsewhere (`list_all_project_ids`)
    /// or treat a missing key as zero; `build_response` in
    /// `commands/doctor_fts.rs` and `detect_split_pairs` in
    /// `commands/doctor.rs` both do the latter. Pinned by
    /// `sqlite::tests::row_counts_by_project_omits_zero_row_projects`.
    ///
    /// `INDEXED BY idx_memories_project` pins the covering-index path: the
    /// aggregate reads only `project_id`, so the scan of that index is all
    /// SQLite needs, and the hint stops the planner from choosing a full-table
    /// scan instead. `idx_memories_project` is created by `create_schema`, so
    /// it exists on every database this method can be called on.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn row_counts_by_project(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn().prepare(
            "SELECT project_id, COUNT(*) FROM memories \
                 INDEXED BY idx_memories_project GROUP BY project_id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut results = Vec::new();
        for row_result in rows {
            results.push(row_result?);
        }
        Ok(results)
    }
}
