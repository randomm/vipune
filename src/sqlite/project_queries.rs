//! Project-level aggregate queries on `Database` (list / count).
//!
//! Split out of `mod.rs` to keep that file under the project 500-line cap;
//! these two are the project-scan surface the doctor and prune handlers drive,
//! and need nothing but the read-only [`Database::conn`] handle.

use super::{Database, Result};

impl Database {
    /// List all distinct project_ids in the database.
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

    /// Count rows for a specific project_id.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn count_rows_for_project(&self, project_id: &str) -> Result<usize> {
        let count: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?",
            [project_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}
