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
/// The supersede path is a separate entry point that builds its own
/// [`NewSupersedingMemory`](crate::memory::NewSupersedingMemory), which embeds
/// this shape and adds the replaced-id and level fields, so the shared columns
/// have one definition (the store-layer counterpart of the storage layer's
/// `UpdateOptions`, and the sibling of
/// [`UpdateParams`](crate::memory::UpdateParams) on the write side). Grouping
/// the fields keeps the internal helpers short without a `too_many_arguments`
/// suppression, and keeps the importance level a named value at every call
/// site — the trailing argument `--importance` dropped on its way to the column
/// in issue #238.
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
    /// gets [`MemoryImportance::default`], bound by name by the public
    /// `Database::insert` wrapper — never the column's `NOT NULL DEFAULT`.
    ///
    /// The insert-status rule has exactly one call site for the add side, in
    /// the first statement of this body, which [`Self::add_with_conflict`] (and
    /// through it `ingest`, `ingest_with_type_status` and `batch_ingest`) and
    /// [`Self::ingest_with_importance`] both forward to — so no add entry point
    /// can insert a `superseded`/`deprecated` row whatever default it binds. See
    /// [`MemoryStatus::ensure_valid_for_insert`] for the map of call sites.
    ///
    /// Metadata is validated here too ([`Self::validate_metadata`]) — the rule
    /// the supersede and update paths already ran — so before #271 the add path
    /// stored a blank or non-JSON string verbatim while the other two refused it.
    /// `batch_ingest` inherits the rule and reports the refusal per item.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a non-insertable `row.status` or for blank /
    /// non-JSON `row.metadata`; [`Error::EmptyInput`] and [`Error::InputTooLong`]
    /// for the content. All are answered before the embedding call, so a refused
    /// row costs no inference.
    pub(crate) fn add_impl(
        &mut self,
        row: &NewMemory<'_>,
        force: bool,
        importance: Option<MemoryImportance>,
    ) -> Result<AddResult, Error> {
        // The insert-status rule, at the one body every add path shares:
        // `add_with_conflict` (and through it `ingest`,
        // `ingest_with_type_status` and `batch_ingest`) and
        // `ingest_with_importance` both call this, so no add entry point can
        // insert a `superseded`/`deprecated` row whatever default it binds.
        row.status.ensure_valid_for_insert()?;

        Self::validate_input_length(row.content)?;
        Self::validate_metadata(row.metadata)?;

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

    /// Insert one new row: a named level goes to `insert_with_importance`, no
    /// level to the public `Database::insert`, which binds the default by name.
    fn insert_row(
        &self,
        row: &NewMemory<'_>,
        embedding: &[f32],
        importance: Option<MemoryImportance>,
    ) -> Result<String, Error> {
        let (memory_type, status) = (row.memory_type.as_str(), row.status.as_str());
        match importance {
            Some(level) => Ok(self.db.insert_with_importance(
                &NewRow {
                    project_id: row.project_id,
                    content: row.content,
                    metadata: row.metadata,
                    memory_type,
                    status,
                    importance: level.as_str(),
                },
                embedding,
            )?),
            None => Ok(self.db.insert(
                row.project_id,
                row.content,
                embedding,
                row.metadata,
                memory_type,
                status,
            )?),
        }
    }
}
