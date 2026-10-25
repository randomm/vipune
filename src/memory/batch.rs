//! Batch operations for the memory store.

use crate::embedding_profiles::EmbeddingRole;
use crate::errors::Error;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, BatchIngestItemResult, BatchIngestResult, IngestPolicy};
use crate::sqlite::NewRow;

use super::store::MemoryStore;

/// The caller-supplied fields of a memory the store is about to create on the
/// supersede path.
///
/// The sibling of [`NewMemory`](crate::memory::NewMemory) on the plain-add
/// side, and the store-layer counterpart of the storage layer's `UpdateOptions`.
/// `NewMemory` is `pub` and `Copy` and is built by name in external code, so
/// giving it a superseded-id field would break every struct-literal caller for
/// one extra column; this shape costs nothing for the same effect (issue #258).
///
/// `importance` and `status` are **required** rather than `Option`: the whole
/// point of the supersede body is that both lifecycle columns are named in the
/// INSERT, so a caller cannot express "no status" and quietly get the column
/// DEFAULT — the shape that made `--importance` a no-op in #238 and `--status`
/// a no-op in #258. The two level-less public wrappers
/// ([`MemoryStore::supersede`]) bind the defaults by name instead.
#[derive(Clone, Copy, Debug)]
pub struct NewSupersedingMemory<'a> {
    /// Project identifier (e.g., git repo URL or user-defined).
    pub project_id: &'a str,
    /// Text content to store (1 to 100,000 characters).
    pub content: &'a str,
    /// Optional JSON metadata string.
    pub metadata: Option<&'a str>,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: MemoryType,
    /// Id of the memory this new row replaces; marked `superseded` in the same
    /// transaction.
    pub old_id: &'a str,
    /// Operator-assigned importance of the new row.
    pub importance: MemoryImportance,
    /// Lifecycle status of the new row (active, candidate). Must satisfy
    /// [`MemoryStatus::is_valid_for_insert`].
    pub status: MemoryStatus,
}

impl MemoryStore {
    #[must_use = "handle the error or results may be lost"]
    /// Batch ingest multiple memories with conflict-aware per-item outcomes.
    ///
    /// This method is part of the public library API for external consumers,
    /// even though the CLI binary doesn't use it directly.
    ///
    /// Processes each item independently according to the specified policy.
    /// Returns a `BatchIngestResult` with deterministic mapping from input indices
    /// to per-item results (Added, Conflicts, or Error).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `items` - Vector of (content, optional_metadata) tuples to ingest
    /// * `policy` - Conflict handling policy (ConflictAware or Force)
    ///
    /// # Returns
    ///
    /// * `Ok(BatchIngestResult { results })` where results[i] corresponds to items[i]
    ///
    /// # Partial-Failure Semantics
    ///
    /// - **Added**: Item succeeded (Force policy always succeeds unless validation fails)
    /// - **Conflicts**: Similar memories found (only with ConflictAware policy)
    /// - **Error**: Item failed validation (empty, too long, embedding error, database error)
    ///
    /// All items are processed. No single item failure stops the batch.
    /// Result order matches input order for deterministic index-based mapping.
    ///
    /// # Consistency Guarantees
    ///
    /// - **Independent Processing**: Each item is processed independently
    /// - **No Early Termination**: Failures in earlier items do NOT prevent processing of later items
    /// - **Deterministic Index Mapping**: `results[i]` ALWAYS corresponds to `items[i]`
    /// - **Partial Success Possible**: No atomic or transactional semantics; some items may succeed while others fail
    /// - **Single-Threaded Safe**: vipune is fully synchronous with no concurrent access patterns
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let items = vec![
    ///     ("First memory", None),
    ///     ("Second memory", Some(r#"{"tag": "important"}"#)),
    /// ];
    /// let result = store.batch_ingest("my-project", items, IngestPolicy::ConflictAware)?;
    /// for (idx, item_result) in result.results.iter().enumerate() {
    ///     match item_result {
    ///         BatchIngestItemResult::Added { id } => println!("Item {}: Added {}", idx, id),
    ///         BatchIngestItemResult::Conflicts { .. } => println!("Item {}: Conflict", idx),
    ///         BatchIngestItemResult::Error { message } => println!("Item {}: Error {}", idx, message),
    ///     }
    /// }
    /// ```
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn batch_ingest(
        &mut self,
        project_id: &str,
        items: Vec<(&str, Option<&str>)>,
        policy: IngestPolicy,
    ) -> Result<BatchIngestResult, Error> {
        let mut results = Vec::with_capacity(items.len());

        let force = matches!(policy, IngestPolicy::Force);

        for (content, metadata) in items {
            let item_result = match Self::validate_input_length(content) {
                Err(e) => BatchIngestItemResult::Error {
                    message: format!("{}", e),
                },
                Ok(()) => match self.add_with_conflict(
                    project_id,
                    content,
                    metadata,
                    force,
                    MemoryType::Fact,
                    MemoryStatus::Active,
                ) {
                    Ok(AddResult::Added { id }) => BatchIngestItemResult::Added { id },
                    Ok(AddResult::Conflicts {
                        proposed,
                        conflicts,
                    }) => BatchIngestItemResult::Conflicts {
                        proposed,
                        conflicts,
                    },
                    Err(e) => BatchIngestItemResult::Error {
                        message: format!("{}", e),
                    },
                },
            };
            results.push(item_result);
        }

        Ok(BatchIngestResult { results })
    }

    #[allow(dead_code)] // Public API for library consumers (e.g., kide)
    #[must_use = "the new memory ID is needed for downstream operations"]
    /// Supersede an existing memory with a new one at the `medium` importance
    /// and `active` status defaults.
    ///
    /// Atomically replaces the old memory (marked as "superseded") with a new memory
    /// that supersedes it. Both memories remain in the database but the old one
    /// has status "superseded" and superseded_by pointing to the new ID. The new
    /// row is `active` by default; a caller that wants it `candidate` uses
    /// [`Self::supersede_with_lifecycle`].
    ///
    /// The level-carrying variant is [`Self::supersede_with_importance`] — the
    /// one `vipune add --supersedes <id> --importance <level>` drives.
    pub fn supersede(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        memory_type: MemoryType,
        old_id: &str,
    ) -> Result<String, Error> {
        self.supersede_impl(&NewSupersedingMemory {
            project_id,
            content,
            metadata,
            memory_type,
            old_id,
            importance: MemoryImportance::default(),
            status: MemoryStatus::Active,
        })
    }

    #[must_use = "the new memory ID is needed for downstream operations"]
    /// Supersede an existing memory with a new row carrying an explicit
    /// importance level and the `active` status — the signature `vipune add
    /// --supersedes --importance` was built on, kept for library callers.
    ///
    /// The CLI's supersedes branch now drives
    /// [`Self::supersede_with_lifecycle`], which takes the lifecycle status as
    /// well; this wrapper keeps its published signature and pins `active`
    /// explicitly, never the column DEFAULT — a caller of this method cannot
    /// ask for a `candidate` replacement, which is the point of the distinction
    /// rather than a silent drop.
    /// The parsed level (canonical `as_str` spelling) is what reaches the INSERT:
    /// an omitted column would silently reset it to the DEFAULT and cost the row
    /// the `high`/`critical` prune exemption (issue #238).
    #[cfg_attr(not(test), allow(dead_code))] // Public API for library consumers
    pub fn supersede_with_importance(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        memory_type: MemoryType,
        old_id: &str,
        importance: MemoryImportance,
    ) -> Result<String, Error> {
        self.supersede_with_lifecycle(&NewSupersedingMemory {
            project_id,
            content,
            metadata,
            memory_type,
            old_id,
            importance,
            status: MemoryStatus::Active,
        })
    }

    #[must_use = "the new memory ID is needed for downstream operations"]
    /// Supersede an existing memory with a new row carrying both an explicit
    /// importance level and an explicit lifecycle status — the path `vipune add
    /// --supersedes --importance <level> --status <status>` drives, and the MCP
    /// `store_memory` equivalent.
    ///
    /// Before issue #258 the supersede path hardcoded `active` into the row it
    /// built, so `--status candidate` was accepted, validated, and then dropped
    /// two frames above the INSERT. `is_valid_for_insert` is enforced here as
    /// well as at the CLI/MCP entry points: a superseding row is a *new* row, so
    /// `superseded`/`deprecated` are not insertable — that lifecycle belongs to
    /// the old row's UPDATE, which the store issues itself.
    ///
    /// A `candidate` replacement is invisible to default search/list (which bind
    /// `status = 'active'`) until it is promoted — there is no `promote` method
    /// on the store; promotion goes through [`Self::update`] with
    /// `status: Some(MemoryStatus::Active)`, which is what `vipune promote`
    /// drives. Documented in `docs/cli-reference.md`, not silently changed.
    pub fn supersede_with_lifecycle(
        &mut self,
        row: &NewSupersedingMemory<'_>,
    ) -> Result<String, Error> {
        if !row.status.is_valid_for_insert() {
            return Err(Error::InvalidInput(format!(
                "Status '{}' is not valid for new memory insertion. Must be 'active' or 'candidate'.",
                row.status.as_str()
            )));
        }
        self.supersede_impl(row)
    }

    /// Validate, embed, and insert the replacement row — the shared body of
    /// every public supersede entry point, so none can drift from the others.
    ///
    /// Every caller names both lifecycle columns: the status-carrying entry
    /// point forwards what it was given, and the level-less `supersede` wrapper
    /// binds `MemoryStatus::Active` / `MemoryImportance::default` by name rather
    /// than leaving either to the column DEFAULT. Both are required fields on
    /// [`NewSupersedingMemory`] precisely so a caller cannot bind one and drop
    /// the other — that is how `--importance` lost its value in #238 and
    /// `--status` in #258.
    fn supersede_impl(&mut self, row: &NewSupersedingMemory<'_>) -> Result<String, Error> {
        let NewSupersedingMemory {
            project_id,
            content,
            metadata,
            memory_type,
            old_id,
            importance,
            status,
        } = *row;

        Self::validate_input_length(content)?;

        // Validate metadata: reject empty strings and invalid JSON
        if let Some(meta) = metadata {
            if meta.trim().is_empty() {
                return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
            }
            serde_json::from_str::<serde_json::Value>(meta)
                .map_err(|e| Error::InvalidInput(format!("invalid metadata JSON: {}", e)))?;
        }

        let embedding = self.get_embedding(content, EmbeddingRole::Passage)?;

        // One INSERT shape for every supersede caller (issues #238, #258).
        Ok(self.db.supersede_with_importance(
            &NewRow {
                project_id,
                content,
                metadata,
                memory_type: memory_type.as_str(),
                status: status.as_str(),
                importance: importance.as_str(),
            },
            &embedding,
            old_id,
        )?)
    }
}
