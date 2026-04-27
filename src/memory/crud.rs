//! CRUD operations for the memory store.

use crate::errors::Error;
use crate::memory_types::{
    AddResult, BatchIngestItemResult, BatchIngestResult, ConflictMemory, IngestPolicy,
};
use crate::sqlite::Memory;

use super::store::MemoryStore;

/// Generate a deterministic mock embedding for specific content.
/// Uses the content's bytes to create a unique but consistent embedding.
/// This ensures that the same content always gets the same embedding.
pub(crate) fn mock_embedding_for_content(content: &str) -> Vec<f32> {
    let mut hash: u64 = 0x123456789abcdef; // Starting seed
    for byte in content.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }

    // Generate random-like embedding seeded by hash
    // Deterministic but produces low similarity between different content
    let mut embedding = Vec::with_capacity(384);
    for i in 0..384 {
        // Use hash + index to generate deterministic but varied values
        let mut dim_hash = hash.wrapping_add(i as u64);
        dim_hash ^= dim_hash >> 33;
        dim_hash = dim_hash.wrapping_mul(0xff51afd7ed558ccd);
        dim_hash ^= dim_hash >> 33;
        dim_hash = dim_hash.wrapping_mul(0xc4ceb9fe1a85ec53);

        // Normalize to [-1.0, 1.0]
        let value = ((dim_hash % 2000) as f32 - 1000.0) / 1000.0;
        embedding.push(value);
    }
    embedding
}

impl MemoryStore {
    #[must_use = "handle the error or results may be lost"]
    /// Add a memory with conflict detection.
    ///
    /// Checks for similar existing memories before adding. If conflicts are found
    /// (similarity >= threshold), returns conflicts details without storing.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `content` - Text content to store (1 to 100,000 characters)
    /// * `metadata` - Optional JSON metadata string
    /// * `force` - If true, bypass conflict detection and add regardless
    ///
    /// # Returns
    ///
    /// * `Ok(AddResult::Added { id })` if no conflicts or force=true
    /// * `Ok(AddResult::Conflicts { proposed, conflicts })` if conflicts found
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input is empty
    /// - Input exceeds 100,000 characters
    /// - Embedding generation fails
    /// - Database operations fail
    pub fn add_with_conflict(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        force: bool,
    ) -> Result<AddResult, Error> {
        Self::validate_input_length(content)?;

        // Use mock embedding if embedder is not loaded (test mode)
        let embedding = if self.embedder.is_none() {
            mock_embedding_for_content(content)
        } else {
            self.embedder()?.embed(content)?
        };

        if force {
            let id = self.db.insert(project_id, content, &embedding, metadata)?;
            return Ok(AddResult::Added { id });
        }

        let similars =
            self.db
                .find_similar(project_id, &embedding, self.config.similarity_threshold)?;
        let conflicts: Vec<ConflictMemory> = similars
            .into_iter()
            .map(|m| ConflictMemory {
                id: m.id,
                content: m.content,
                similarity: m.similarity.unwrap_or(0.0),
            })
            .collect();

        if conflicts.is_empty() {
            let id = self.db.insert(project_id, content, &embedding, metadata)?;
            Ok(AddResult::Added { id })
        } else {
            Ok(AddResult::Conflicts {
                proposed: content.to_string(),
                conflicts,
            })
        }
    }

    #[must_use = "handle the error or results may be lost"]
    /// Ingest a memory with explicit policy.
    ///
    /// Ergonomic single-method API for adding memories with configurable
    /// conflict handling behavior.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `content` - Text content to store (1 to 100,000 characters)
    /// * `metadata` - Optional JSON metadata string
    /// * `policy` - Conflict handling policy (ConflictAware or Force)
    ///
    /// # Returns
    ///
    /// * `Ok(AddResult::Added { id })` if memory was stored successfully
    /// * `Ok(AddResult::Conflicts { proposed, conflicts })` if Conflicts policy and similar memories exist
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input is empty
    /// - Input exceeds 100,000 characters
    /// - Embedding generation fails
    /// - Database operations fail
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Add with conflict detection (reject if similar exists)
    /// match store.ingest("my-project", "Alice works at Microsoft", None, IngestPolicy::ConflictAware)? {
    ///     AddResult::Added { id } => println!("Added: {}", id),
    ///     AddResult::Conflicts { conflicts, .. } => println!("Found {} conflicts", conflicts.len()),
    /// }
    ///
    /// // Force add regardless of conflicts
    /// let id = match store.ingest("my-project", "Duplicate content", None, IngestPolicy::Force)? {
    ///     AddResult::Added { id } => id,
    ///     AddResult::Conflicts { .. } => unreachable!(),
    /// };
    /// ```
    pub fn ingest(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        policy: IngestPolicy,
    ) -> Result<AddResult, Error> {
        match policy {
            IngestPolicy::ConflictAware => {
                self.add_with_conflict(project_id, content, metadata, false)
            }
            IngestPolicy::Force => self.add_with_conflict(project_id, content, metadata, true),
        }
    }

    #[must_use = "handle the error or results may be lost"]
    /// Get a specific memory by ID.
    ///
    /// Returns `None` if the memory doesn't exist.
    pub fn get(&self, id: &str) -> Result<Option<Memory>, Error> {
        Ok(self.db.get(id)?)
    }

    #[must_use = "handle the error or results may be lost"]
    /// List all memories for a project.
    ///
    /// Returns memories ordered by creation time (newest first).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `limit` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Limit is 0
    /// - Limit exceeds MAX_SEARCH_LIMIT
    pub fn list(&self, project_id: &str, limit: usize) -> Result<Vec<Memory>, Error> {
        use super::store::validate_limit;
        validate_limit(limit)?;
        Ok(self.db.list(project_id, limit)?)
    }

    #[must_use = "handle the error or results may be lost"]
    /// Update a memory's content and/or metadata.
    ///
    /// - If content is provided: generates a new embedding and updates content
    /// - If metadata is provided: updates metadata (full replacement, not merge)
    /// - If both provided: updates both content (with new embedding) and metadata
    /// - The memory ID, project ID, and creation timestamp remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID to update
    /// * `content` - Optional new content for the memory
    /// * `metadata` - Optional JSON metadata string (replaces existing metadata)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Both content and metadata are None
    /// - Content is provided and exceeds 100,000 characters
    /// - Memory doesn't exist
    pub fn update(
        &mut self,
        id: &str,
        content: Option<&str>,
        metadata: Option<&str>,
    ) -> Result<(), Error> {
        if content.is_none() && metadata.is_none() {
            return Err(Error::InvalidInput(
                "At least one of content or metadata must be provided".to_string(),
            ));
        }

        // Validate metadata: reject empty strings and invalid JSON
        if let Some(meta) = metadata {
            if meta.trim().is_empty() {
                return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
            }
            // Validate that metadata is valid JSON
            serde_json::from_str::<serde_json::Value>(meta).map_err(|e| {
                Error::InvalidInput(format!("invalid metadata JSON: {}", e))
            })?;
        }

        // If content is provided, validate and generate new embedding
        let embedding = if let Some(text) = content {
            Self::validate_input_length(text)?;
            Some(self.embedder()?.embed(text)?)
        } else {
            None
        };

        Ok(self
            .db
            .update(id, content, embedding.as_deref(), metadata)?)
    }

    #[must_use = "handle the error or results may be lost"]
    /// Delete a memory.
    ///
    /// Returns:
    /// - `Ok(true)` if memory was deleted
    /// - `Ok(false)` if memory didn't exist
    pub fn delete(&self, id: &str) -> Result<bool, Error> {
        Ok(self.db.delete(id)?)
    }

    #[allow(dead_code)] // Public API for library consumers (e.g., kide)
    #[must_use = "handle the error or results may be lost"]
    /// List memories for a project created since a given timestamp.
    ///
    /// Returns memories with `created_at > since_timestamp`, ordered by creation time (newest first).
    /// The timestamp comparison is exclusive (does not include memories created exactly at the timestamp).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `since_timestamp` - RFC3339-formatted timestamp (exclusive lower bound)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The timestamp is not valid RFC3339
    /// - Limit is 0 or exceeds MAX_SEARCH_LIMIT
    /// - Database query fails
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chrono::Utc;
    /// let one_hour_ago = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
    /// let recent = store.list_since("project", &one_hour_ago, 10)?;
    /// ```
    pub fn list_since(
        &self,
        project_id: &str,
        since_timestamp: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, Error> {
        use super::store::validate_limit;
        validate_limit(limit)?;
        Ok(self.db.list_since(project_id, since_timestamp, limit)?)
    }

    #[allow(dead_code)] // Public API for library consumers (e.g., kide)
    #[must_use = "handle the error or results may be lost"]
    /// Get multiple memories by their IDs.
    ///
    /// Returns results in the same order as the input IDs. Missing IDs are represented as `None`.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of memory IDs to retrieve
    ///
    /// # Returns
    ///
    /// Vector of `Option<Memory>` with the same length as `ids`. Each position corresponds
    /// to the ID at the same index in the input. `Some(memory)` if found, `None` if not found.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let results = store.get_many(&["id1", "id2", "missing-id"])?;
    /// assert_eq!(results.len(), 3);
    /// assert!(results[0].is_some()); // Found id1
    /// assert!(results[1].is_some()); // Found id2
    /// assert!(results[2].is_none()); // Missing ID
    /// ```
    pub fn get_many(&self, ids: &[&str]) -> Result<Vec<Option<Memory>>, Error> {
        Ok(self.db.get_many(ids)?)
    }

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
                Ok(()) => match self.add_with_conflict(project_id, content, metadata, force) {
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
}
