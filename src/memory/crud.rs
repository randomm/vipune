//! CRUD operations for the memory store.

use crate::errors::Error;
use crate::memory_types::{
    AddResult, BatchIngestItem, BatchIngestOutcome, BatchIngestResult, ConflictMemory, IngestPolicy,
};
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

    #[must_use = "handle the error or results may be lost"]
    #[allow(dead_code)]
    /// Add multiple memories in a single batch operation.
    ///
    /// Processes all items independently; mixed outcomes (Added/Conflicts/Error) are supported.
    /// Results are returned in input order for direct mapping to original items.
    ///
    /// # Arguments
    ///
    /// * `project_id` - Project identifier (e.g., git repo URL or user-defined)
    /// * `items` - List of (content, metadata) pairs to ingest
    /// * `policy` - Conflict policy: ConflictAware checks for conflicts, Force bypasses detection
    ///
    /// # Returns
    ///
    /// * `Ok(BatchIngestOutcome)` containing per-item results and summary statistics
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - All items fail validation
    /// - Database backend fails catastrophically
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vipune::{MemoryStore, Config, memory_types::{BatchIngestItem, IngestPolicy}};
    /// # let config = Config::default();
    /// # let mut store = MemoryStore::new(config.database_path.as_path(), &config.embedding_model, config.clone()).unwrap();
    /// let items = vec![
    ///     BatchIngestItem::new("First memory".to_string()),
    ///     BatchIngestItem::with_metadata("Second memory".to_string(), r#"{"tag": "important"}"#.to_string()),
    /// ];
    /// let outcome = store.batch_ingest("my-project", &items, IngestPolicy::ConflictAware).unwrap();
    /// println!("Added: {}, Conflicts: {}", outcome.summary.added, outcome.summary.conflicts);
    /// ```
    pub fn batch_ingest(
        &mut self,
        project_id: &str,
        items: &[BatchIngestItem],
        policy: IngestPolicy,
    ) -> Result<BatchIngestOutcome, Error> {
        if items.is_empty() {
            return Ok(BatchIngestOutcome::new(Vec::new()));
        }

        let mut results = Vec::with_capacity(items.len());

        for item in items {
            let result = match Self::validate_input_length(&item.content) {
                Ok(()) => match self.process_single_item(project_id, item, policy) {
                    Ok(add_result) => match add_result {
                        AddResult::Added { id } => BatchIngestResult::Added { id },
                        AddResult::Conflicts {
                            proposed,
                            conflicts,
                        } => BatchIngestResult::Conflicts {
                            proposed,
                            conflicts,
                        },
                    },
                    Err(e) => BatchIngestResult::Error {
                        message: e.to_string(),
                    },
                },
                Err(e) => BatchIngestResult::Error {
                    message: e.to_string(),
                },
            };
            results.push(result);
        }

        Ok(BatchIngestOutcome::new(results))
    }

    /// Process a single batch item with the given policy.
    fn process_single_item(
        &mut self,
        project_id: &str,
        item: &BatchIngestItem,
        policy: IngestPolicy,
    ) -> Result<AddResult, Error> {
        if policy == IngestPolicy::Force {
            let embedding = self.embedder()?.embed(&item.content)?;
            let id = self.db.insert(
                project_id,
                &item.content,
                &embedding,
                item.metadata.as_deref(),
            )?;
            return Ok(AddResult::Added { id });
        }

        let embedding = self.embedder()?.embed(&item.content)?;
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
                &item.content,
                &embedding,
                item.metadata.as_deref(),
            )?;
            Ok(AddResult::Added { id })
        } else {
            Ok(AddResult::Conflicts {
                proposed: item.content.clone(),
                conflicts,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::memory_types::BatchIngestItem;
    use tempfile::NamedTempFile;

    fn create_test_store() -> MemoryStore {
        let temp_file = NamedTempFile::new().unwrap();
        let database_path = temp_file.path().to_path_buf();
        // Keep temp file alive so SQLite can use it
        std::mem::forget(temp_file);
        let embedding_model = Config::default().embedding_model;
        let config = Config {
            database_path: database_path.clone(),
            embedding_model: embedding_model.clone(),
            ..Default::default()
        };
        MemoryStore::new(database_path.as_path(), &embedding_model, config).unwrap()
    }

    #[test]
    fn batch_all_success() {
        let mut store = create_test_store();
        let items = vec![
            BatchIngestItem::new("The first unique test memory document".to_string()),
            BatchIngestItem::new("A completely different second document content".to_string()),
            BatchIngestItem::new("The third separate memory with unique text".to_string()),
        ];

        let outcome = store
            .batch_ingest("test-project", &items, IngestPolicy::ConflictAware)
            .unwrap();

        assert_eq!(outcome.summary.total, 3);
        assert_eq!(outcome.summary.added, 3);
        assert_eq!(outcome.summary.conflicts, 0);
        assert_eq!(outcome.summary.errors, 0);

        assert_eq!(outcome.results.len(), 3);
        for result in &outcome.results {
            assert!(matches!(result, BatchIngestResult::Added { .. }));
        }
    }

    #[test]
    fn batch_mixed_outcomes() {
        let mut store = create_test_store();

        // Add a base memory to create potential conflicts
        let base_content = "Duplicate content here";
        store
            .add_with_conflict("test-project", base_content, None, false)
            .unwrap();

        let items = vec![
            BatchIngestItem::new("The quick brown fox jumps over lazy dog".to_string()), // index 0: added
            BatchIngestItem::new(base_content.to_string()), // index 1: conflicts
            BatchIngestItem::new(
                "Gravity is a fundamental force that attracts objects with mass".to_string(),
            ), // index 2: added
            BatchIngestItem::new(
                "Photosynthesis converts light energy into chemical energy".to_string(),
            ), // index 3: added
        ];

        let outcome = store
            .batch_ingest("test-project", &items, IngestPolicy::ConflictAware)
            .unwrap();

        assert_eq!(outcome.summary.total, 4);
        assert_eq!(outcome.summary.added, 3);
        assert_eq!(outcome.summary.conflicts, 1);
        assert_eq!(outcome.summary.errors, 0);

        assert_eq!(outcome.results.len(), 4);

        // Index mapping: verify results match input order
        assert!(matches!(
            outcome.results[0],
            BatchIngestResult::Added { .. }
        ));
        assert!(matches!(
            outcome.results[1],
            BatchIngestResult::Conflicts { .. }
        ));
        assert!(matches!(
            outcome.results[2],
            BatchIngestResult::Added { .. }
        ));
        assert!(matches!(
            outcome.results[3],
            BatchIngestResult::Added { .. }
        ));

        // Verify conflict details
        if let BatchIngestResult::Conflicts { proposed, .. } = &outcome.results[1] {
            assert_eq!(proposed, base_content);
        }
    }

    #[test]
    fn batch_invalid_input_continues_others() {
        let mut store = create_test_store();

        // Empty string is invalid input
        let items = vec![
            BatchIngestItem::new("Valid memory 1".to_string()), // index 0: added
            BatchIngestItem::new(String::new()),                // index 1: error (empty)
            BatchIngestItem::new("Valid memory 2".to_string()), // index 2: added
        ];

        let outcome = store
            .batch_ingest("test-project", &items, IngestPolicy::Force)
            .unwrap();

        assert_eq!(outcome.summary.total, 3);
        assert_eq!(outcome.summary.added, 2);
        assert_eq!(outcome.summary.conflicts, 0);
        assert_eq!(outcome.summary.errors, 1);

        assert_eq!(outcome.results.len(), 3);

        assert!(matches!(
            outcome.results[0],
            BatchIngestResult::Added { .. }
        ));
        assert!(matches!(
            outcome.results[1],
            BatchIngestResult::Error { .. }
        ));
        assert!(matches!(
            outcome.results[2],
            BatchIngestResult::Added { .. }
        ));

        // Verify error message
        if let BatchIngestResult::Error { message } = &outcome.results[1] {
            assert!(message.to_lowercase().contains("input cannot be empty"));
        }
    }
}
