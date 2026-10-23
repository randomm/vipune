//! The store's add path: the shape of a new row and the one INSERT that writes it.
//!
//! Split out of `crud.rs` (which keeps the read and update operations) for
//! issue #238: `vipune add --importance` shipped as a silent no-op because the
//! validated level was dropped one line above the call that inserted the row.
//! Every row the add path creates funnels through [`MemoryStore::add_impl`]
//! and [`MemoryStore::insert_row`], so no add entry point can learn to name a
//! different column set.

use crate::embedding_profiles::EmbeddingRole;
use crate::errors::Error;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, ConflictMemory, IngestPolicy};
use crate::sqlite::NewRow;

use super::store::MemoryStore;

/// The caller-supplied fields of a memory the store is about to create on the
/// add path.
///
/// The supersede path is a separate entry point that takes its fields as
/// plain parameters, so this shape belongs to the add entry points only (the
/// store-layer counterpart of the storage layer's `UpdateOptions`, and the
/// sibling of [`UpdateParams`](crate::memory::UpdateParams) on the write
/// side). Grouping the fields keeps the internal helpers short without a
/// `too_many_arguments` suppression, and keeps the importance level a named
/// value at every call site — the trailing argument `--importance` dropped on
/// its way to the column in issue #238.
#[derive(Clone, Copy, Debug)]
pub struct NewMemory<'a> {
    /// Project identifier (e.g., git repo URL or user-defined).
    pub project_id: &'a str,
    /// Text content to store (1 to 100,000 characters).
    pub content: &'a str,
    /// Optional JSON metadata string.
    pub metadata: Option<&'a str>,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: MemoryType,
    /// Lifecycle status of the new row (active, candidate).
    pub status: MemoryStatus,
}

impl MemoryStore {
    /// The conflict-aware add body behind [`Self::add_with_conflict`] and
    /// [`Self::ingest_with_importance`], so the two cannot learn to write
    /// different column sets — the shape `--importance` shipped as a no-op in
    /// issue #238.
    ///
    /// `force` bypasses the similarity gate. A caller that took `--importance`
    /// passes a parsed [`MemoryImportance`], so the canonical `as_str` spelling
    /// is what reaches the column (the level decides the `high`/`critical`
    /// prune exemption). A caller that expresses no level passes `None` and
    /// gets [`MemoryImportance::default`], which [`Self::insert_row`] binds
    /// explicitly — never the column's `NOT NULL DEFAULT`.
    pub(crate) fn add_impl(
        &mut self,
        row: &NewMemory<'_>,
        force: bool,
        importance: Option<MemoryImportance>,
    ) -> Result<AddResult, Error> {
        Self::validate_input_length(row.content)?;

        let embedding = self.get_embedding(row.content, EmbeddingRole::Passage)?;

        if force {
            let id = self.insert_row(row, &embedding, importance)?;
            return Ok(AddResult::Added { id });
        }

        let similars =
            self.db
                .find_similar(row.project_id, &embedding, self.config.similarity_threshold)?;
        let conflicts: Vec<ConflictMemory> = similars
            .into_iter()
            .map(|m| ConflictMemory {
                id: m.id,
                content: m.content,
                similarity: m.similarity.unwrap_or(0.0),
            })
            .collect();

        if conflicts.is_empty() {
            let id = self.insert_row(row, &embedding, importance)?;
            Ok(AddResult::Added { id })
        } else {
            Ok(AddResult::Conflicts {
                proposed: row.content.to_string(),
                conflicts,
            })
        }
    }

    /// Ingest with explicit memory type, status, and operator-assigned
    /// importance — the add path `vipune add` drives. The level is a parsed
    /// [`MemoryImportance`], so SQLite gets the canonical `as_str` spelling
    /// rather than the raw flag string (issue #238).
    #[must_use = "handle the error or results may be lost"]
    pub fn ingest_with_importance(
        &mut self,
        row: &NewMemory<'_>,
        policy: IngestPolicy,
        importance: MemoryImportance,
    ) -> Result<AddResult, Error> {
        self.add_impl(row, policy == IngestPolicy::Force, Some(importance))
    }

    /// Insert one new row, binding `importance` explicitly.
    ///
    /// One call, one INSERT: a caller that named no level gets
    /// [`MemoryImportance::default`], bound by name like any explicit level —
    /// the column's `NOT NULL DEFAULT` never supplies it (issue #238).
    fn insert_row(
        &self,
        row: &NewMemory<'_>,
        embedding: &[f32],
        importance: Option<MemoryImportance>,
    ) -> Result<String, Error> {
        let level = importance.unwrap_or_default();
        Ok(self.db.insert_with_importance(
            &NewRow {
                project_id: row.project_id,
                content: row.content,
                metadata: row.metadata,
                memory_type: row.memory_type.as_str(),
                status: row.status.as_str(),
                importance: level.as_str(),
            },
            embedding,
        )?)
    }
}
