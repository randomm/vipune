//! Database supersede operations.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use super::{Error, Result, vec_to_blob};

impl super::Database {
    /// Atomically supersede an old memory with a new one.
    ///
    /// Delegates to [`Self::supersede_with_importance`] with the `medium`
    /// default; both share one INSERT so the two cannot drift.
    ///
    /// In a single transaction:
    /// 1. Inserts the new memory
    /// 2. Sets old memory's status to 'superseded' and superseded_by to new ID
    #[allow(dead_code)] // Library API for consumers; the store's supersede path goes through supersede_with_importance so the column is never defaulted (issue #238)
    pub fn supersede(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        memory_type: &str,
        old_id: &str,
    ) -> Result<String> {
        self.supersede_with_importance(
            project_id,
            content,
            embedding,
            metadata,
            memory_type,
            old_id,
            "medium",
        )
    }

    /// Atomically supersede an old memory with a new one carrying an explicit
    /// importance level.
    ///
    /// The new row's `importance` is bound, not left to the column DEFAULT:
    /// `vipune add --supersedes <id> --importance critical` creates that new
    /// row, and an omitted column silently reset the level the operator just
    /// asked for — and with it the prune protection that keys off it
    /// (issue #238).
    ///
    /// # Errors
    ///
    /// Returns an error if the memory to supersede does not exist, belongs to
    /// another project, or the transaction fails.
    #[allow(clippy::too_many_arguments)] // signature mirrors supersede(); data fields map 1:1 to columns
    pub fn supersede_with_importance(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        memory_type: &str,
        old_id: &str,
        importance: &str,
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
            "INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, importance)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                &new_id,
                project_id,
                content,
                &blob,
                metadata,
                &now,
                &now,
                memory_type,
                "active",
                importance
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
