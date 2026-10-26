//! Database supersede operations.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use crate::memory::lifecycle::{MemoryImportance, MemoryStatus};

use super::{Error, NewRow, Result, vec_to_blob};

/// The rejected spellings, read off the enum rather than restated as literals.
const RETIRED_STATUSES: [MemoryStatus; 2] = [MemoryStatus::Superseded, MemoryStatus::Deprecated];

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
    /// that does name them goes through [`crate::MemoryStore::supersede_with`],
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
    ///
    /// Only `active` and `candidate` rows may be superseded. The status column
    /// has no CHECK constraint and this body — the one INSERT every supersede
    /// crosses, including the public [`Self::supersede`] wrapper above it — is the
    /// last thing that can refuse: re-superseding an already-`superseded` row
    /// overwrites its `superseded_by` and forks the chain into a row nothing
    /// points at, and superseding a `deprecated` row resurrects one `prune`
    /// retired. The rejected statuses are [`MemoryStatus::Superseded`] and
    /// [`MemoryStatus::Deprecated`] read off the enum rather than restated as
    /// literals, so a rename moves the guard and the column together.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when `old_id` is absent from `row.project_id`,
    /// [`Error::InvalidInput`] when it belongs to another project or is already
    /// `superseded`/`deprecated` (naming its current status, and its
    /// `superseded_by` when set), and a storage error if the write itself fails.
    /// The guard runs before the transaction opens, so a refusal writes nothing.
    pub(crate) fn supersede_with_importance(
        &self,
        row: &NewRow<'_>,
        embedding: &[f32],
        old_id: &str,
    ) -> Result<String> {
        // Opened before the read below and held to the commit: SQLite's
        // deferred transaction promotes on the first write, so reading the old
        // row's status and then writing inside one snapshot is what makes the
        // guard atomic rather than a check another process can slip between.
        let tx = self.conn.unchecked_transaction()?;

        // Verify old memory exists and belongs to this project (scoped get enforces it)
        let old = {
            let mut stmt = tx.prepare(
                "SELECT status, superseded_by FROM memories WHERE id = ?1 AND project_id = ?2",
            )?;
            let mut rows = stmt.query_map([old_id, row.project_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
            })?;
            rows.next().transpose()?.ok_or_else(|| {
                Error::NotFound(format!("memory to supersede not found: {}", old_id))
            })?
        };
        // The re-supersede guard. The status column has no CHECK constraint, and
        // the store's validating edges are the only thing that normally keeps a
        // retired spelling out of it, so the comparison is against the enum's
        // canonical spellings rather than a parse: a row whose status came from
        // anywhere else is not one of the two retired states and stays
        // supersedeable (failing closed would make such a row unreplaceable).
        if RETIRED_STATUSES
            .iter()
            .any(|retired| old.0 == retired.as_str())
        {
            return Err(Error::InvalidInput(match old.1 {
                Some(by) => format!(
                    "cannot supersede memory {old_id}: status is already '{}' (superseded_by {by})",
                    old.0
                ),
                None => {
                    format!(
                        "cannot supersede memory {old_id}: status is already '{}'",
                        old.0
                    )
                }
            }));
        }

        let new_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

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
