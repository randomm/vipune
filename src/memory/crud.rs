//! CRUD operations for the memory store.

use crate::errors::Error;
use crate::memory_types::{AddResult, ConflictMemory};
use crate::sqlite::Memory;

use super::store::MemoryStore;

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
        if force {
            let embedding = self.embedder()?.embed(content)?;
            let id = self.db.insert(project_id, content, &embedding, metadata)?;
            return Ok(AddResult::Added { id });
        }

        let embedding = self.embedder()?.embed(content)?;
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
    /// Update a memory's content.
    ///
    /// Generates a new embedding for the updated content and persists it.
    /// The memory ID, project ID, and creation timestamp remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID to update
    /// * `content` - New content for the memory
    ///
    /// # Errors
    ///
    /// Returns error if the memory doesn't exist.
    pub fn update(&mut self, id: &str, content: &str) -> Result<(), Error> {
        Self::validate_input_length(content)?;
        let embedding = self.embedder()?.embed(content)?;
        Ok(self.db.update(id, content, &embedding)?)
    }

    #[must_use = "handle the error or results may be lost"]
    /// Delete a memory.
    ///
    /// # Returns
    ///
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
}
