//! Database supersede operations.

use chrono::Utc;
use rusqlite::{Transaction, TransactionBehavior, params};
use uuid::Uuid;

use crate::memory::lifecycle::{MemoryImportance, MemoryStatus};

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
    /// retired. The verdict comes from [`MemoryStatus::stored_is_retired`], the
    /// one predicate the insert rule shares, so a new variant moves the guard and
    /// the column together and a retired status in any casing or padding cannot
    /// slip past a raw string comparison. The refusal always quotes the canonical
    /// [`MemoryStatus::as_str`], never the raw column, so the message the operator
    /// reads is one spelling rather than whatever `import` left there — the
    /// row's own `superseded_by` is still reported verbatim, since that is an id.
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] when `old_id` is absent from `row.project_id` — which
    /// includes a row that exists under another project, as the scoped `WHERE`
    /// cannot tell the two apart — [`Error::InvalidInput`] when the row is already
    /// `superseded` or `deprecated` (naming its current status, and its
    /// `superseded_by` when set), and a storage error if the write itself fails.
    /// The guard runs inside the write transaction, which is opened
    /// [`TransactionBehavior::Immediate`] so the write lock is taken *before* the
    /// read it checks: a refusal writes nothing, and no other process can retire
    /// the row between that read and the UPDATE.
    pub(crate) fn supersede_with_importance(
        &self,
        row: &NewRow<'_>,
        embedding: &[f32],
        old_id: &str,
    ) -> Result<String> {
        // `Immediate`, not the deferred default: `BEGIN IMMEDIATE` takes the write
        // lock at BEGIN and holds it to the commit, so the guard's read below and
        // the two writes after it are one unit no other writer interleaves with.
        // A deferred transaction takes no lock for a plain read, so the guard
        // would check a snapshot and only then discover it cannot write.
        // `Transaction::new_unchecked` is the `&Connection` form of rusqlite's
        // `unchecked_transaction`, which this `&self` method needs; the no-nesting
        // guarantee those two trade `&mut self` for is held here by this being the
        // only transaction any path opens on this connection.
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;

        // Verify old memory exists, belongs to this project (the scoped WHERE
        // enforces it), and has not already been retired. The status is decoded
        // as `Option<String>`: a NULL column is not a retired one, and decoding it
        // as `String` would turn that row into a storage error instead.
        let old = {
            let mut stmt = tx.prepare(
                "SELECT status, superseded_by FROM memories WHERE id = ?1 AND project_id = ?2",
            )?;
            let mut rows = stmt.query_map([old_id, row.project_id], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<String>>(1)?,
                ))
            })?;
            rows.next().transpose()?.ok_or_else(|| {
                Error::NotFound(format!("memory to supersede not found: {}", old_id))
            })?
        };
        // The re-supersede guard. Parsed through the enum rather than compared to
        // string literals, so a retired row written below the store's validating
        // edges (`import`, `Database::update`) is refused whatever its casing or
        // padding. A status outside the vocabulary is not a retired one and stays
        // supersedeable — failing closed would make such a row impossible to
        // replace — and a NULL status is not retired either.
        if let Some(stored) = old.0.as_deref()
            && MemoryStatus::stored_is_retired(stored)
        {
            // Parsed once and rendered through the enum, so the message quotes
            // the canonical spelling rather than `stored`: the column is
            // unvalidated text and the operator's error should read like the
            // vocabulary the CLI documents.
            let mut msg = match MemoryStatus::from_str(stored.trim()) {
                Ok(canonical) => format!(
                    "cannot supersede memory {old_id}: status is already '{}'",
                    canonical.as_str()
                ),
                // Unreachable while `stored_is_retired` parses the same way; if
                // the two ever disagreed the message degrades to the raw column
                // rather than inventing a status name.
                Err(_) => format!("cannot supersede memory {old_id}: status is already '{stored}'"),
            };
            if let Some(by) = old.1 {
                msg.push_str(&format!(" (superseded_by {by})"));
            }
            return Err(Error::InvalidInput(msg));
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
