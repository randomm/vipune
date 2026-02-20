//! CRUD operations for the memory store.

use crate::errors::Error;
use crate::memory_types::{AddResult, ConflictMemory};
use crate::sqlite::Memory;

use super::store::MemoryStore;

impl MemoryStore {
    /// Add a memory to the store (legacy method without conflict detection).
    ///
    /// Generates an embedding for the content and stores it in SQLite.
    /// Returns the generated memory ID (UUID).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `content` - Text content to store
    /// * `metadata` - Optional JSON metadata string
    #[cfg(test)]
    pub fn add(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
    ) -> Result<String, Error> {
        let embedding = self.embedder.embed(content)?;
        Ok(self.db.insert(project_id, content, &embedding, metadata)?)
    }

    /// Add a memory with conflict detection.
    ///
    /// Checks for similar existing memories before adding. If conflicts are found
    /// (similarity >= threshold), returns conflicts details without storing.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `content` - Text content to store
    /// * `metadata` - Optional JSON metadata string
    /// * `force` - If true, bypass conflict detection and add regardless
    ///
    /// # Returns
    ///
    /// * `Ok(AddResult::Added { id })` if no conflicts or force=true
    /// * `Ok(AddResult::Conflicts { proposed, conflicts })` if conflicts found
    pub fn add_with_conflict(
        &mut self,
        project_id: &str,
        content: &str,
        metadata: Option<&str>,
        force: bool,
    ) -> Result<AddResult, Error> {
        if force {
            let embedding = self.embedder.embed(content)?;
            let id = self.db.insert(project_id, content, &embedding, metadata)?;
            return Ok(AddResult::Added { id });
        }

        let embedding = self.embedder.embed(content)?;
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

    /// Get a specific memory by ID.
    ///
    /// Returns `None` if the memory doesn't exist.
    pub fn get(&self, id: &str) -> Result<Option<Memory>, Error> {
        Ok(self.db.get(id)?)
    }

    /// List all memories for a project.
    ///
    /// Returns memories ordered by creation time (newest first).
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier
    /// * `limit` - Maximum number of results to return
    pub fn list(&self, project_id: &str, limit: usize) -> Result<Vec<Memory>, Error> {
        Ok(self.db.list(project_id, limit)?)
    }

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
        let embedding = self.embedder.embed(content)?;
        Ok(self.db.update(id, content, &embedding)?)
    }

    /// Delete a memory.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if memory was deleted
    /// - `Ok(false)` if memory didn't exist
    pub fn delete(&self, id: &str) -> Result<bool, Error> {
        Ok(self.db.delete(id)?)
    }
}
