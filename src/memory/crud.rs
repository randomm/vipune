//! CRUD operations for the memory store.

#[cfg(test)]
use crate::embedding::l2_normalize; // test-only; pure function, pragmatic coupling
use crate::errors::Error;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, ConflictMemory, IngestPolicy};
use crate::sqlite::Memory;

use super::store::MemoryStore;

/// Generate a deterministic mock embedding for specific content.
/// Uses the content's bytes to create a unique but consistent embedding.
/// This ensures that the same content always gets the same embedding.
///
/// Only available in test builds — never ships in release binaries.
#[cfg(test)]
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

/// THE designated fake embedder for tests — L2-normalised mock vectors.
///
/// Returns L2-normalised vectors so the Phase 2 norm detector classifies them
/// as real rather than mock. Normalisation does not change conflict-detection
/// behaviour because `cosine_similarity` is scale-invariant.
#[cfg(test)]
pub(crate) fn test_fake_embedder(content: &str) -> Result<Vec<f32>, Error> {
    Ok(l2_normalize(&mock_embedding_for_content(content)))
}

impl MemoryStore {
    #[must_use = "the new memory ID is needed for downstream operations"]
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
    /// * `memory_type` - Memory type string (fact, preference, procedure, guard, observation)
    /// * `status` - Memory status string (active, candidate)
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
        memory_type: MemoryType,
        status: MemoryStatus,
    ) -> Result<AddResult, Error> {
        Self::validate_input_length(content)?;

        let embedding = self.get_embedding(content)?;
        let memory_type_str = memory_type.as_str();
        let status_str = status.as_str();

        if force {
            let id = self.db.insert(
                project_id,
                content,
                &embedding,
                metadata,
                memory_type_str,
                status_str,
            )?;
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
            let id = self.db.insert(
                project_id,
                content,
                &embedding,
                metadata,
                memory_type_str,
                status_str,
            )?;
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
    #[allow(dead_code)] // Library API: available for consumers
    pub fn ingest(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        policy: IngestPolicy,
    ) -> Result<AddResult, Error> {
        self.ingest_with_type_status(
            project_id,
            content,
            metadata,
            policy,
            MemoryType::Fact,
            MemoryStatus::Active,
        )
    }

    /// Ingest with explicit memory type and status.
    #[must_use = "handle the error or results may be lost"]
    pub fn ingest_with_type_status(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        policy: IngestPolicy,
        memory_type: MemoryType,
        status: MemoryStatus,
    ) -> Result<AddResult, Error> {
        match policy {
            IngestPolicy::ConflictAware => {
                self.add_with_conflict(project_id, content, metadata, false, memory_type, status)
            }
            IngestPolicy::Force => {
                self.add_with_conflict(project_id, content, metadata, true, memory_type, status)
            }
        }
    }

    #[must_use = "handle the error or results may be lost"]
    /// Get a specific memory by ID scoped to a project.
    ///
    /// Returns `None` if the memory doesn't exist or belongs to a different project.
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID to retrieve
    /// * `project_id` - Project identifier (scope guard)
    pub fn get(&self, id: &str, project_id: &str) -> Result<Option<Memory>, Error> {
        Ok(self.db.get(id, project_id)?)
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
    /// * `memory_types` - Optional filter by memory types
    /// * `statuses` - Optional filter by memory statuses
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Limit is 0
    /// - Limit exceeds MAX_SEARCH_LIMIT
    pub fn list(
        &self,
        project_id: &str,
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<Vec<Memory>, Error> {
        use super::store::validate_limit;
        validate_limit(limit)?;
        Ok(self.db.list(project_id, limit, memory_types, statuses)?)
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
    /// * `memory_type` - Optional memory type to update
    /// * `status` - Optional lifecycle status to update
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - All content, metadata, memory_type, and status are None
    /// - Content is provided and exceeds 100,000 characters
    /// - Memory doesn't exist
    /// - memory_type is invalid
    /// - status is "superseded" (use --supersedes flag instead)
    pub fn update(
        &mut self,
        id: &str,
        content: Option<&str>,
        metadata: Option<&str>,
        memory_type: Option<MemoryType>,
        status: Option<MemoryStatus>,
    ) -> Result<(), Error> {
        if content.is_none() && metadata.is_none() && memory_type.is_none() && status.is_none() {
            return Err(Error::InvalidInput(
                "At least one of content, metadata, memory_type, or status must be provided"
                    .to_string(),
            ));
        }

        // Validate metadata: reject empty strings and invalid JSON
        if let Some(meta) = metadata {
            if meta.trim().is_empty() {
                return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
            }
            // Validate that metadata is valid JSON
            serde_json::from_str::<serde_json::Value>(meta)
                .map_err(|e| Error::InvalidInput(format!("invalid metadata JSON: {}", e)))?;
        }

        // Validate status if provided - reject "superseded"
        if let Some(s) = status {
            if s == MemoryStatus::Superseded {
                return Err(Error::InvalidInput(
                    "Cannot set status to 'superseded'. Use --supersedes flag instead.".to_string(),
                ));
            }
        }

        // If content is provided, validate and generate new embedding
        let embedding = if let Some(text) = content {
            Self::validate_input_length(text)?;
            Some(self.get_embedding(text)?)
        } else {
            None
        };

        Ok(self.db.update(
            id,
            content,
            embedding.as_deref(),
            metadata,
            memory_type.map(|t| t.as_str()),
            status.map(|s| s.as_str()),
        )?)
    }

    #[must_use = "handle the error or results may be lost"]
    /// Delete a memory scoped to a project.
    ///
    /// Returns:
    /// - `Ok(true)` if memory was deleted
    /// - `Ok(false)` if memory didn't exist or belongs to a different project
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID to delete
    /// * `project_id` - Project identifier (scope guard)
    pub fn delete(&self, id: &str, project_id: &str) -> Result<bool, Error> {
        Ok(self.db.delete(id, project_id)?)
    }

    /// Increment retrieval_count and set last_retrieved_at for given memory IDs.
    ///
    /// Called after retrieving memories to track telemetry.
    #[allow(dead_code)] // Library API: unused when MCP feature is disabled
    pub fn touch_memories(&self, ids: &[&str]) -> Result<(), Error> {
        Ok(self.db.touch_memories(ids)?)
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
    /// * `memory_types` - Optional filter by memory types
    /// * `statuses` - Optional filter by memory statuses
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
    /// let recent = store.list_since("project", &one_hour_ago, 10, None, None)?;
    /// ```
    pub fn list_since(
        &self,
        project_id: &str,
        since_timestamp: &str,
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<Vec<Memory>, Error> {
        use super::store::validate_limit;
        validate_limit(limit)?;
        Ok(self
            .db
            .list_since(project_id, since_timestamp, limit, memory_types, statuses)?)
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

    /// Get the embedding for content.
    ///
    /// In test builds, uses the injected `test_embedder` if present.
    /// Otherwise, delegates to the real embedding engine.
    /// If the embedder is unavailable, returns `Error::EmbedderUnavailable`.
    pub(crate) fn get_embedding(&mut self, content: &str) -> Result<Vec<f32>, Error> {
        #[cfg(test)]
        {
            if let Some(f) = &self.test_embedder {
                return f(content);
            }
        }
        self.embedder()?.embed(content)
    }
}
