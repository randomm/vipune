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
    pub(crate) embedder: Option<EmbeddingEngine>,
    pub(crate) model_id: String,
    pub(crate) config: Config,
    #[cfg(test)]
    pub(crate) test_embedder: Option<TestEmbedder>,
}

/// Test-only embedder function type.
/// Available in all builds so the `MemoryStore` struct can reference it,
/// but the field using this type is `#[cfg(test)]`-gated.
#[allow(dead_code)]
pub(crate) type TestEmbedder = Box<dyn Fn(&str) -> Result<Vec<f32>, Error> + Send + Sync>;

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
        Ok(MemoryStore {
            db,
            embedder: None,
            model_id: model_id.to_string(),
            config,
            #[cfg(test)]
            test_embedder: None,
        })
    }

    /// Lazily initialize and return a mutable reference to the embedding engine.
    ///
    /// Downloads the model on first call; subsequent calls return the cached engine.
    /// Returns errors from `EmbeddingEngine::new` directly — for download failures
    /// these already include the offline hint (see `wrap_download_err`).
    ///
    /// Note: the underlying error from `EmbeddingEngine::new` may contain local
    /// filesystem paths (e.g., model cache directories). This is acceptable for
    /// a local CLI tool — no sanitization is applied.
    pub(crate) fn embedder(&mut self) -> Result<&mut EmbeddingEngine, Error> {
        if self.embedder.is_none() {
            self.embedder = Some(EmbeddingEngine::new(&self.model_id)?);
        }
        Ok(self.embedder.as_mut().unwrap())
    }

    // Set an embedder that was pre-initialized externally (e.g., with a timeout).
    // Used by the MCP server which spawns the model download in a separate thread
    // and applies a timeout to prevent indefinite hangs.
    #[allow(dead_code)]
    pub(crate) fn set_preinitialized_embedder(&mut self, engine: EmbeddingEngine) {
        self.embedder = Some(engine);
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

    #[cfg(test)]
    /// Create a MemoryStore from an existing Database for testing.
    pub(crate) fn from_db(db: Database, config: Config) -> Self {
        MemoryStore {
            db,
            embedder: None,
            model_id: String::new(),
            config,
            test_embedder: None,
        }
    }

    #[cfg(test)]
    /// Create a MemoryStore from a pre-populated Database with the test embedder wired in.
    ///
    /// For tests that need to insert data before creating the store:
    /// 1. Call [`test_db_path`] to get a temp database path
    /// 2. Open a `Database`, insert data
    /// 3. Call this method to get a ready-to-use store
    pub(crate) fn from_db_with_test_embedder(db: Database) -> Self {
        let mut store = Self::from_db(db, Config::default());
        store.test_embedder = Some(Box::new(crate::memory::crud::test_fake_embedder));
        store
    }

    #[cfg(test)]
    /// Create a fully-configured MemoryStore for tests with TempDir, Database, and fake embedder.
    ///
    /// Replaces the 3-line boilerplate (TempDir + Database::open + from_db + test_embedder)
    /// that was duplicated across 12+ test functions.
    ///
    /// ⚠️ The returned TempDir is forgotten (not cleaned up on drop) to keep the database
    /// file alive for the duration of the test. Temp files are cleaned up by the OS.
    pub(crate) fn test_store() -> Self {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("test.db");
        std::mem::forget(dir);

        let db = Database::open(&path).expect("open test database");
        let mut store = Self::from_db(db, Config::default());
        store.test_embedder = Some(Box::new(crate::memory::crud::test_fake_embedder));
        store
    }

    #[cfg(test)]
    /// Return a temp database path for tests that need to pre-populate data.
    ///
    /// Callers open a `Database` on the returned path, insert data, then
    /// create a `MemoryStore` via `from_db` + `test_embedder`.
    ///
    /// ⚠️ The TempDir is forgotten, same as [`test_store`].
    pub(crate) fn test_db_path() -> std::path::PathBuf {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("test.db");
        std::mem::forget(dir);
        path
    }
}
