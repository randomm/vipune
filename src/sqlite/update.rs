//! Database update operations.

use chrono::Utc;

use super::{Error, Result, vec_to_blob};

impl super::Database {
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
}
