//! Database list and retrieval operations.

use rusqlite::Result as SqliteResult;

use super::{Error, Result, map_row_to_memory};

impl super::Database {
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
    ) -> Result<Vec<super::Memory>> {
        let mut where_clauses = vec!["project_id = ?1".to_string()];
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&project_id];
        let param_index = super::build_filters(
            &mut where_clauses,
            &mut params,
            2,
            statuses,
            memory_types,
            "",
        );
        let limit_param = limit as i64;
        params.push(&limit_param);

        let where_clause = where_clauses.join(" AND ");
        let query = format!(
            "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
             FROM memories WHERE {} ORDER BY created_at DESC LIMIT ?{}",
            where_clause, param_index
        );

        let mut stmt = self.conn.prepare(&query)?;

        let memories: SqliteResult<Vec<super::Memory>> = stmt
            .query_map(params.as_slice(), map_row_to_memory)?
            .collect();

        Ok(memories?)
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
    ) -> Result<Vec<super::Memory>> {
        // Validate timestamp format by parsing it
        let _parsed = chrono::DateTime::parse_from_rfc3339(since_timestamp)
            .map_err(|e| Error::Sqlite(format!("Invalid RFC3339 timestamp: {}", e)))?;

        let mut where_clauses = vec!["project_id = ?1".to_string(), "created_at > ?2".to_string()];
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&project_id, &since_timestamp];
        let param_index = super::build_filters(
            &mut where_clauses,
            &mut params,
            3,
            statuses,
            memory_types,
            "",
        );
        let limit_param = limit as i64;
        params.push(&limit_param);

        let where_clause = where_clauses.join(" AND ");
        let query = format!(
            "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at
             FROM memories WHERE {} ORDER BY created_at DESC LIMIT ?{}",
            where_clause, param_index
        );

        let mut stmt = self.conn.prepare(&query)?;

        let memories: SqliteResult<Vec<super::Memory>> = stmt
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
    pub fn get_many(&self, ids: &[&str]) -> Result<Vec<Option<super::Memory>>> {
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

        let rows: SqliteResult<Vec<(String, super::Memory)>> = stmt
            .query_map(params.as_slice(), |row| {
                let id: String = row.get(0)?;
                let memory = map_row_to_memory(row)?;
                Ok((id, memory))
            })?
            .collect();

        let found_memories: std::collections::HashMap<String, super::Memory> =
            rows?.into_iter().collect();

        // Preserve input ordering
        let results: Vec<Option<super::Memory>> = ids
            .iter()
            .map(|id| found_memories.get(*id).cloned())
            .collect();

        Ok(results)
    }

    /// Increment retrieval_count and set last_retrieved_at for given memory IDs.
    ///
    /// Called by CLI handlers and MCP handlers after retrieving memories via search/get.
    /// This method is NOT called internally by supersede or other operations that don't
    /// represent user-initiated retrieval.
    #[allow(dead_code)] // Library API: unused when MCP feature is disabled
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
