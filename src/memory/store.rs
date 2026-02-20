//! Core memory store struct combining embedding generation and persistence.

use std::path::Path;

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::errors::Error;
use crate::sqlite::Database;

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
    pub fn new(db_path: &Path, model_id: &str, config: Config) -> Result<Self, Error> {
        let db = Database::open(db_path)?;
        let embedder = EmbeddingEngine::new(model_id)?;
        Ok(MemoryStore {
            db,
            embedder,
            config,
        })
    }
}
