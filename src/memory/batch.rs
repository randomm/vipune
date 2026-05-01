//! Batch operations for the memory store.

use crate::errors::Error;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, BatchIngestItemResult, BatchIngestResult, IngestPolicy};

use super::store::MemoryStore;

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
    /// Supersede an existing memory with a new one.
    ///
    /// Atomically replaces the old memory (marked as "superseded") with a new memory
    /// that supersedes it. Both memories remain in the database but the old one
    /// has status "superseded" and superseded_by pointing to the new ID.
    pub fn supersede(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        memory_type: MemoryType,
        old_id: &str,
    ) -> Result<String, Error> {
        Self::validate_input_length(content)?;

        // Validate metadata: reject empty strings and invalid JSON
        if let Some(meta) = metadata {
            if meta.trim().is_empty() {
                return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
            }
            serde_json::from_str::<serde_json::Value>(meta)
                .map_err(|e| Error::InvalidInput(format!("invalid metadata JSON: {}", e)))?;
        }

        let embedding = self.get_embedding(content)?;

        Ok(self.db.supersede(
            project_id,
            content,
            &embedding,
            metadata,
            memory_type.as_str(),
            old_id,
        )?)
    }
}
