//! Database update operations.

use chrono::Utc;

use super::{Error, Result, vec_to_blob};

/// Options for updating a memory in the database.
///
/// Groups the optional fields that may be updated, reducing the `Database::update`
/// function signature and making call sites self-documenting.
pub struct UpdateOptions<'a> {
    /// New content for the memory.
    pub content: Option<&'a str>,
    /// Pre-computed embedding (required when content is provided).
    pub embedding: Option<&'a [f32]>,
    /// New metadata JSON string (full replacement).
    pub metadata: Option<&'a str>,
    /// New memory type.
    pub memory_type: Option<&'a str>,
    /// New lifecycle status.
    pub status: Option<&'a str>,
}

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
    pub fn update(&self, id: &str, project_id: &str, options: UpdateOptions<'_>) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        // Build dynamic UPDATE query based on what's being updated
        let mut set_clauses: Vec<&str> = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(text) = options.content {
            set_clauses.push("content = ?");
            let blob =
                vec_to_blob(options.embedding.ok_or_else(|| {
                    Error::Sqlite("Content update requires embedding".to_string())
                })?)?;
            params.push(Box::new(text.to_string()));
            set_clauses.push("embedding = ?");
            params.push(Box::new(blob));
        }

        if let Some(meta) = options.metadata {
            set_clauses.push("metadata = ?");
            params.push(Box::new(meta.to_string()));
        }

        if let Some(t) = options.memory_type {
            set_clauses.push("type = ?");
            params.push(Box::new(t.to_string()));
        }

        if let Some(s) = options.status {
            set_clauses.push("status = ?");
            params.push(Box::new(s.to_string()));
        }

        set_clauses.push("updated_at = ?");
        params.push(Box::new(now));

        if set_clauses.len() == 1 {
            return Err(Error::InvalidInput(
                "At least one field must be provided for update".to_string(),
            ));
        }

        let sql = format!(
            "UPDATE memories SET {} WHERE id = ? AND project_id = ?",
            set_clauses.join(", ")
        );

        // Add id and project_id as last parameters
        params.push(Box::new(id.to_string()));
        params.push(Box::new(project_id.to_string()));

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = self.conn.execute(&sql, param_refs.as_slice())?;

        if rows == 0 {
            return Err(Error::NotFound(
                "No memory found for the given id".to_string(),
            ));
        }

        Ok(())
    }
}
