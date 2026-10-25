//! Batch operations for the memory store.

use crate::embedding_profiles::EmbeddingRole;
use crate::errors::Error;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, BatchIngestItemResult, BatchIngestResult, IngestPolicy};
use crate::sqlite::NewRow;

use super::NewMemory;
use super::store::MemoryStore;

/// The caller-supplied fields of a memory the store is about to create on the
/// supersede path: [`NewMemory`] plus the replaced id and the level.
///
/// The shared columns are embedded rather than copied, so the two INSERT paths
/// cannot drift column-for-column — the defect class of #238 and #258. The
/// struct is `#[non_exhaustive]` because it is re-exported from the crate root
/// and built by name: that keeps a future field additive for external callers
/// instead of silently breaking every literal.
///
/// `importance` and `status` are **required** rather than `Option`: the whole
/// point of the supersede body is that both lifecycle columns are named in the
/// INSERT, so a caller cannot express "no status" and quietly get the column
/// DEFAULT — the shape that made `--importance` a no-op in #238 and `--status`
/// a no-op in #258. The level-less public wrapper
/// ([`MemoryStore::supersede`]) binds the defaults by name instead.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct NewSupersedingMemory<'a> {
    /// The new row's shared columns (project, content, metadata, type, status).
    pub row: NewMemory<'a>,
    /// Id of the memory this new row replaces; marked `superseded` in the same
    /// transaction.
    pub old_id: &'a str,
    /// Operator-assigned importance of the new row.
    pub importance: MemoryImportance,
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
    /// [`Self::supersede_with_lifecycle`], which also takes the importance level
    /// and is the entry point the CLI and MCP supersede paths drive.
    pub fn supersede(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        memory_type: MemoryType,
        old_id: &str,
    ) -> Result<String, Error> {
        self.supersede_impl(&NewSupersedingMemory {
            row: NewMemory {
                project_id,
                content,
                metadata,
                memory_type,
                status: MemoryStatus::Active,
            },
            old_id,
            importance: MemoryImportance::default(),
        })
    }

    #[must_use = "the new memory ID is needed for downstream operations"]
    /// Supersede an existing memory with a new row carrying both an explicit
    /// importance level and an explicit lifecycle status — the one supersede
    /// entry point the CLI (`vipune add --supersedes --importance <level>
    /// --status <status>`) and the MCP `store_memory` supersedes arm drive.
    ///
    /// Before issue #258 the supersede path hardcoded `active` into the row it
    /// built, so `--status candidate` was accepted, validated, and then dropped
    /// two frames above the INSERT. Insertability is enforced here and nowhere
    /// else: a superseding row is a *new* row, so `superseded`/`deprecated` are
    /// not insertable — that lifecycle belongs to the old row's UPDATE, which
    /// the store issues itself. The CLI and MCP layers pre-check the same rule
    /// with the same [`MemoryStatus::ensure_valid_for_insert`] for their own
    /// error shape; this is the copy that protects a library caller.
    ///
    /// A `candidate` replacement is invisible to default search/list (which bind
    /// `status = 'active'`), and supersede does not touch its retrieval
    /// telemetry, so `vipune promote` will not select it: the deterministic way
    /// to activate one is `vipune update <id> --status active` with the id this
    /// method returns. Promotion stays retrieval-driven and correct — see the
    /// `promote` module doc. Documented in `docs/cli-reference.md`.
    pub fn supersede_with_lifecycle(
        &mut self,
        row: &NewSupersedingMemory<'_>,
    ) -> Result<String, Error> {
        row.row.status.ensure_valid_for_insert()?;
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
        Self::validate_input_length(row.row.content)?;

        // Validate metadata: reject empty strings and invalid JSON
        if let Some(meta) = row.row.metadata {
            if meta.trim().is_empty() {
                return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
            }
            serde_json::from_str::<serde_json::Value>(meta)
                .map_err(|e| Error::InvalidInput(format!("invalid metadata JSON: {}", e)))?;
        }

        let embedding = self.get_embedding(row.row.content, EmbeddingRole::Passage)?;

        // One INSERT shape for every supersede caller (issues #238, #258).
        Ok(self.db.supersede_with_importance(
            &NewRow {
                project_id: row.row.project_id,
                content: row.row.content,
                metadata: row.row.metadata,
                memory_type: row.row.memory_type.as_str(),
                status: row.row.status.as_str(),
                importance: row.importance.as_str(),
            },
            &embedding,
            row.old_id,
        )?)
    }
}
