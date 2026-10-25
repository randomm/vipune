//! Database supersede operations.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use crate::memory::lifecycle::MemoryImportance;

use super::{Error, NewRow, Result, vec_to_blob};

impl super::Database {
    /// Atomically supersede an old memory with a new one at the `medium`
    /// importance and `active` status defaults.
    ///
    /// In a single transaction:
    /// 1. Inserts the new memory
    /// 2. Sets old memory's status to 'superseded' and superseded_by to new ID
    ///
    /// Delegates to [`Self::supersede_with_importance`], the one supersede
    /// INSERT, so the two paths cannot drift — and binds both lifecycle columns
    /// by name rather than leaving either to the column DEFAULT. This is the
    /// entry point for a caller that names neither level nor status; a caller
    /// that does name them goes through [`crate::MemoryStore::supersede_with_lifecycle`],
    /// which is how the CLI and MCP supersede paths reach the INSERT.
    #[allow(dead_code)] // Public API for library consumers (e.g., kide)
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
                importance: MemoryImportance::default().as_str(),
            },
            embedding,
            old_id,
        )
    }

    /// Atomically supersede an old memory with a new row whose lifecycle
    /// columns are bound explicitly.
    ///
    /// Both `importance` and `status` come from `row`, never from the column
    /// DEFAULT: `vipune add --supersedes <id> --importance critical --status
    /// candidate` creates that row through here, and an omitted column silently
    /// reset what the operator had just asked for — the level costs the row the
    /// `high`/`critical` prune protection that keys off it (issue #238), the
    /// status silently hid a `candidate` replacement under `active` (issue #258).
    /// The caller supplies both in the canonical spelling, and the INSERT binds
    /// them; a `candidate` replacement is deliberately hidden from default
    /// search/list until it is activated (`vipune update <id> --status active`).
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
