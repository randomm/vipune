//! Database supersede operations.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use super::{Error, Result, vec_to_blob};

impl super::Database {
    /// Atomically supersede an old memory with a new one.
    ///
    /// In a single transaction:
    /// 1. Inserts the new memory
    /// 2. Sets old memory's status to 'superseded' and superseded_by to new ID
    pub fn supersede(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        memory_type: &str,
        old_id: &str,
    ) -> Result<String> {
        // Verify old memory exists and belongs to this project (scoped get enforces it)
        let old = self
            .get(old_id, project_id)?
            .ok_or_else(|| Error::NotFound(format!("memory to supersede not found: {}", old_id)))?;
        if old.project_id != project_id {
            return Err(Error::InvalidInput(
                "cannot supersede memory from different project".to_string(),
            ));
        }

        let new_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &new_id,
                project_id,
                content,
                &blob,
                metadata,
                &now,
                &now,
                memory_type,
                "active"
            ],
        )?;
        tx.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = ?1, updated_at = ?2 WHERE id = ?3",
            params![&new_id, &now, old_id],
        )?;
        tx.commit()?;

        Ok(new_id)
    }
}
