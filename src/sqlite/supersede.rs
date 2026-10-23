//! Database supersede operations.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use super::{Error, NewRow, Result, vec_to_blob};

impl super::Database {
    /// Atomically supersede an old memory with a new one at the `medium`
    /// importance default.
    ///
    /// In a single transaction:
    /// 1. Inserts the new memory
    /// 2. Sets old memory's status to 'superseded' and superseded_by to new ID
    ///
    /// Delegates to [`Self::supersede_with_importance`], the one supersede
    /// INSERT, so the two paths cannot drift.
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
            &NewRow {
                project_id,
                content,
                metadata,
                memory_type,
                status: "active",
                importance: "medium",
            },
            embedding,
            old_id,
        )
    }

    /// Atomically supersede an old memory with a new one carrying an explicit
    /// importance level.
    ///
    /// The new row's `importance` is bound, not left to the column DEFAULT:
    /// `vipune add --supersedes <id> --importance critical` creates that row
    /// through here, and an omitted column silently reset the level the
    /// operator had just asked for — with it the prune protection that keys
    /// off `high`/`critical` (issue #238). The caller supplies the canonical
    /// spelling, and `active` for `row.status`: the replacement row is what
    /// searches return.
    pub(crate) fn supersede_with_importance(
        &self,
        row: &NewRow<'_>,
        embedding: &[f32],
        old_id: &str,
    ) -> Result<String> {
        // Verify old memory exists and belongs to this project (scoped get enforces it)
        let old = self
            .get(old_id, row.project_id)?
            .ok_or_else(|| Error::NotFound(format!("memory to supersede not found: {}", old_id)))?;
        if old.project_id != row.project_id {
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
                row.project_id,
                row.content,
                &blob,
                row.metadata,
                &now,
                &now,
                row.memory_type,
                row.status,
                row.importance
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
