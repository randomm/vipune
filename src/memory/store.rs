//! Core memory store struct combining embedding generation and persistence.

use std::path::Path;

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::errors::Error;
use crate::sqlite::Database;

/// Maximum allowed input length (100,000 characters).
pub const MAX_INPUT_LENGTH: usize = 100_000;
/// Maximum allowed limit for search operations.
pub const MAX_SEARCH_LIMIT: usize = 10_000;

/// Validate that a limit parameter is within acceptable bounds.
///
/// Returns error if limit is 0 or exceeds MAX_SEARCH_LIMIT.
pub(crate) fn validate_limit(limit: usize) -> Result<(), Error> {
    if limit == 0 {
        return Err(Error::InvalidInput(
            "Limit must be greater than 0".to_string(),
        ));
    }
    if limit > MAX_SEARCH_LIMIT {
        return Err(Error::InvalidInput(format!(
            "Limit {} exceeds maximum allowed ({})",
            limit, MAX_SEARCH_LIMIT
        )));
    }
    Ok(())
}

/// Core memory store combining embedding generation and persistence.
///
/// Wraps a SQLite database and ONNX embedding engine to provide
/// semantic search capabilities for stored text memories.
///
/// # Mutability Requirements
///
/// Methods that generate embeddings (`add`, `search`, `update`) require
/// `&mut self` because `EmbeddingEngine::embed` internally mutates state
/// for ONNX tensor allocations.
pub struct MemoryStore {
    pub(crate) db: Database,
    pub(crate) embedder: EmbeddingEngine,
    pub(crate) config: Config,
}

impl MemoryStore {
    /// Initialize a new memory store with database path, model ID, and config.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the SQLite database file (created if it doesn't exist)
    /// * `model_id` - HuggingFace model ID (e.g., "BAAI/bge-small-en-v1.5")
    /// * `config` - Configuration including similarity threshold for conflict detection
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Database path contains path traversal sequences (e.g., "../")
    /// - Parent directory cannot be canonicalized
    /// - Database cannot be opened
    /// - Embedding model cannot be loaded
    pub fn new(db_path: &Path, model_id: &str, config: Config) -> Result<Self, Error> {
        use std::path::Component;

        // Path traversal guard: reject parent directory components (works on all platforms)
        for component in db_path.components() {
            if matches!(component, Component::ParentDir) {
                return Err(Error::Config(
                    "Invalid database path: contains '..' which may escape the intended directory"
                        .to_string(),
                ));
            }
        }

        // Canonicalize the full db_path to resolve any symlinks and get the real path
        // Use canonical parent + filename approach to handle non-existent paths
        let db_real_path = if db_path.exists() {
            std::fs::canonicalize(db_path).map_err(|e| {
                Error::Config(format!(
                    "Invalid database path: cannot canonicalize existing path: {}",
                    e
                ))
            })?
        } else {
            // For non-existent paths, canonicalize parent and reconstruct
            let parent = db_path.parent().ok_or_else(|| {
                Error::Config("Invalid database path: no parent directory".to_string())
            })?;
            let canonical_parent = std::fs::canonicalize(parent).map_err(|e| {
                Error::Config(format!(
                    "Invalid database path: parent directory not accessible: {}",
                    e
                ))
            })?;
            // Join canonical parent with just the filename (safe: no .. in filename extraction)
            let filename = db_path
                .file_name()
                .ok_or_else(|| Error::Config("Invalid database path: no filename".to_string()))?;
            canonical_parent.join(filename)
        };

        let db = Database::open(&db_real_path)?;
        let embedder = EmbeddingEngine::new(model_id)?;
        Ok(MemoryStore {
            db,
            embedder,
            config,
        })
    }

    /// Validate input length (rejects empty and whitespace-only inputs).
    pub(crate) fn validate_input_length(text: &str) -> Result<(), Error> {
        if text.trim().is_empty() {
            return Err(Error::EmptyInput);
        }
        if text.len() > MAX_INPUT_LENGTH {
            return Err(Error::InputTooLong {
                max_length: MAX_INPUT_LENGTH,
                actual_length: text.len(),
            });
        }
        Ok(())
    }
}
