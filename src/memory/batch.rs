//! The store's supersede path and its batch ingest entry point.
//!
//! The supersede path owns an INSERT of its own — it writes the replacement row
//! and the old row's `superseded` UPDATE in one transaction — so it is kept
//! apart from `crud.rs`, which holds the plain add, read and update operations.
//! Every supersede caller funnels through [`MemoryStore::supersede_with`], the
//! one body that names the columns, because a second body is how `--importance`
//! became a no-op in issue #238 and `--status` in issue #258.

use crate::embedding_profiles::EmbeddingRole;
use crate::errors::Error;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, BatchIngestItemResult, BatchIngestResult, IngestPolicy};
use crate::sqlite::NewRow;

use super::NewMemory;
use super::store::MemoryStore;

/// The caller-supplied fields of a memory the store is about to create on the
/// supersede path: a [`NewMemory`] row composed with the replaced id and the
/// operator's level.
///
/// The shared columns are embedded rather than copied, so the two INSERT paths
/// cannot drift column-for-column — the defect class of #238 and #258. The
/// struct is `#[non_exhaustive]` because it is re-exported from the crate root
/// and built by name: that keeps a future field additive for external callers
/// instead of silently breaking every literal.
///
/// `importance` is **required** rather than `Option`, for the same reason: the
/// whole point of the supersede body is that both lifecycle columns are named in
/// the INSERT, so a caller cannot express "no level" and quietly get the column
/// DEFAULT. The level-less public wrapper ([`MemoryStore::supersede`]) binds the
/// default by name instead. The status is not duplicated here — it is the
/// `status` field of the embedded [`NewMemory`], the same shape the add path
/// carries, and [`MemoryStore::supersede_with`] validates it with
/// [`MemoryStatus::ensure_valid_for_insert`] before any INSERT.
///
/// The type is `MemoryStatus`, **not** the narrower `Option<MemoryStatus>` it
/// might seem to want: the released
/// [`MemoryStore::supersede_with_importance`](MemoryStore::supersede_with_importance)
/// builds one, and `None` there means "`active`", a value this field can already
/// express. An `Option` would invite a caller to write the status the old row
/// receives (`superseded`) as "no status", which is the #258 defect re-typed
/// rather than fixed.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct NewSupersedingMemory<'a> {
    /// The new row's shared columns (project, content, metadata, type, status).
    pub row: NewMemory<'a>,
    /// Id of the memory this new row replaces; marked `superseded` in the same
    /// transaction.
    pub old_id: &'a str,
    /// Operator-assigned importance of the new row. [`NewSupersedingMemory::new`]
    /// binds the default; [`NewSupersedingMemory::with_importance`] overrides it.
    pub importance: MemoryImportance,
}

impl<'a> NewSupersedingMemory<'a> {
    /// Compose a superseding row from the [`NewMemory`] that replaces `old_id`.
    ///
    /// The constructor exists because the struct is `#[non_exhaustive]` and
    /// re-exported from the crate root: outside this crate a field literal cannot
    /// name a field set, so without it [`MemoryStore::supersede_with`] was
    /// unreachable for an external consumer even though the method is public
    /// (issue #271).
    ///
    /// `importance` defaults to [`MemoryImportance::default`] (`medium`) — the
    /// same binding [`MemoryStore::supersede`] makes for the level it does not
    /// take; [`Self::with_importance`] names another. `row.status` is whatever
    /// the caller put in the [`NewMemory`]: the constructor does not default it,
    /// because [`MemoryStatus::ensure_valid_for_insert`] is what decides it and a
    /// constructor that quietly rewrote a `superseded` status to `active` would
    /// hide the mistake that rule exists to catch.
    #[must_use = "the composed row is the argument supersede_with takes"]
    // No dead_code exemption: the two defaulting wrappers below compose through
    // here, so the released constructor — not a field literal — is the way a
    // superseding row is built in and out of the crate.
    pub fn new(row: NewMemory<'a>, old_id: &'a str) -> Self {
        Self {
            row,
            old_id,
            importance: MemoryImportance::default(),
        }
    }

    /// Override the level [`Self::new`] defaulted to `medium`.
    ///
    /// The level reaches the INSERT in its canonical `as_str` spelling: an
    /// omitted column would reset it to the DEFAULT and cost the row the
    /// `high`/`critical` prune exemption (issue #238), so this setter is the way
    /// a caller built through [`Self::new`] changes it.
    #[must_use = "the returned row carries the level; the setter does not mutate in place"]
    pub fn with_importance(mut self, importance: MemoryImportance) -> Self {
        self.importance = importance;
        self
    }
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
    /// * `items` - Vector of (content, optional_metadata) tuples to ingest. A
    ///   present metadata value must be non-blank valid JSON, like on every other
    ///   write path (#271)
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
    /// - **Error**: Item failed validation (empty, too long, blank or invalid
    ///   JSON metadata, embedding error, database error)
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
    /// row is `active` at the `medium` level; a caller that names either column
    /// uses [`Self::supersede_with`] — it takes the status and the level and is
    /// what the CLI and MCP supersede paths drive — or
    /// [`Self::supersede_with_importance`], which names only the level.
    pub fn supersede(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        memory_type: MemoryType,
        old_id: &str,
    ) -> Result<String, Error> {
        self.supersede_with(&NewSupersedingMemory::new(
            NewMemory {
                project_id,
                content,
                metadata,
                memory_type,
                status: MemoryStatus::Active,
            },
            old_id,
        ))
    }

    #[must_use = "the new memory ID is needed for downstream operations"]
    /// Supersede an existing memory with a new row whose importance and
    /// lifecycle status are named by `row` — the body every supersede reaches:
    /// the CLI (`vipune add --supersedes --importance <level> --status
    /// <status>`) and the MCP `store_memory` supersedes arm call it directly,
    /// and the two defaulting wrappers, [`Self::supersede`] (no level, no
    /// status) and [`Self::supersede_with_importance`] (a level, `active`),
    /// forward to it with the columns they do not take bound by name. A
    /// supersede is not a family of entry points with their own argument lists
    /// any more; there is one body, so there is one place that can name the
    /// wrong column (issues #238, #258).
    ///
    /// Before issue #258 the supersede path hardcoded `active` into the row it
    /// built, so `--status candidate` was accepted, validated, and then dropped
    /// two frames above the INSERT. [`MemoryStatus::ensure_valid_for_insert`] is
    /// the rule and names every frame that calls it — the two store bodies and
    /// the CLI/MCP prechecks, which call it only to answer in their own error
    /// shape — so this is not the only copy of the check, it is the one that
    /// guards the write: a superseding row is a *new* row, so
    /// `superseded`/`deprecated` are not insertable and that lifecycle belongs to
    /// the old row's UPDATE, which this body issues itself. A library caller that
    /// skips the prechecks is stopped by this copy.
    ///
    /// A `candidate` replacement is invisible to default search/list (which bind
    /// `status = 'active'`), and supersede does not touch its retrieval
    /// telemetry, so `vipune promote` will not select it: the deterministic way
    /// to activate one is `vipune update <id> --status active` with the id this
    /// method returns. Promotion stays retrieval-driven and correct — see the
    /// `promote` module doc. Documented in `docs/cli-reference.md`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `row.row.status` is not insertable, if
    /// the content is empty or over the limit, if `row.row.metadata` is blank or
    /// invalid JSON, or if the row named by `row.old_id` is already `superseded`
    /// or `deprecated` — only `active` and `candidate` rows can be superseded, so
    /// re-superseding one of the other two writes nothing and the message names
    /// the row's current status (and its `superseded_by` when set). Returns a
    /// storage-layer error if `row.old_id` does not exist in `row.row.project_id`.
    pub fn supersede_with(&mut self, row: &NewSupersedingMemory<'_>) -> Result<String, Error> {
        // The insert-status rule, on the one body every supersede takes. Not the
        // only call site — see `MemoryStatus::ensure_valid_for_insert` for the
        // map — but the one a caller cannot get past.
        row.row.status.ensure_valid_for_insert()?;

        // Both cheap rules run before the embedding call and before the store
        // write, so a payload the caller must repair never pays for inference and
        // is never confused with a lock or a disk fault.
        Self::validate_input_length(row.row.content)?;
        Self::validate_metadata(row.row.metadata)?;

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

    // Same exemption as the level-less `supersede` above: this is the released
    // library surface, reached by external consumers (e.g. kide) and by the
    // tests in `memory/tests/crud.rs`, never by this crate's own binary since
    // #258 routed the CLI's `--supersedes --importance` through `supersede_with`.
    #[cfg_attr(not(test), allow(dead_code))] // Public API for library consumers
    #[must_use = "the new memory ID is needed for downstream operations"]
    /// Supersede an existing memory with a new row carrying an explicit
    /// importance level, leaving every other column of the replacement at the
    /// [`Self::supersede`] defaults — `active` status, [`MemoryType::Fact`]
    /// supplied by the caller as before.
    ///
    /// The path `vipune add --supersedes --importance` drove before issue #258
    /// replaced it with the status-carrying [`Self::supersede_with`]; it stays
    /// because it shipped in v0.15.4 and the public library API is additive-only
    /// (issue #258). It is a thin wrapper: it composes the same
    /// [`NewSupersedingMemory`] the other entry points build and forwards, so it
    /// cannot learn to write a different column set (#238, #258).
    ///
    /// The parsed level (canonical `as_str` spelling) is what reaches the INSERT:
    /// an omitted column would silently reset it to the DEFAULT and cost the row
    /// the `high`/`critical` prune exemption (issue #238).
    ///
    /// # Errors
    ///
    /// As [`Self::supersede_with`]: [`Error::InvalidInput`] for empty or
    /// over-long content, or blank or invalid JSON metadata; a storage-layer
    /// error if `old_id` does not exist in `project_id`.
    pub fn supersede_with_importance(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        memory_type: MemoryType,
        old_id: &str,
        importance: MemoryImportance,
    ) -> Result<String, Error> {
        self.supersede_with(
            &NewSupersedingMemory::new(
                NewMemory {
                    project_id,
                    content,
                    metadata,
                    memory_type,
                    status: MemoryStatus::Active,
                },
                old_id,
            )
            .with_importance(importance),
        )
    }
}
