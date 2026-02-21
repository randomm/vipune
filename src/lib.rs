//! vipune - A minimal memory layer for AI agents.
//!
//! This crate provides a local, semantic memory store with conflict detection.
//! All operations are synchronous (no async/await required).
//!
//! # Example
//!
//! ```no_run
//! use vipune::{Config, MemoryStore, detect_project};
//!
//! // Initialize memory store
//! let config = Config::default();
//! let mut store = MemoryStore::new(
//!     config.database_path.as_path(),
//!     &config.embedding_model,
//!     config.clone()
//! ).expect("Failed to initialize store");
//!
//! // Detect project ID
//! let project_id = detect_project(None);
//!
//! // Add a memory with conflict detection
//! let result = store.add_with_conflict(&project_id, "Alice works at Microsoft", None, false);
//! match result {
//!     Ok(vipune::AddResult::Added { id }) => println!("Added memory: {}", id),
//!     Ok(vipune::AddResult::Conflicts { .. }) => println!("Conflict detected"),
//!     Err(e) => eprintln!("Error: {}", e),
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//!
//! // Search memories
//! let results = store.search(&project_id, "where does alice work", 10, 0.0);
//! for memory in results.unwrap() {
//!     println!("{:.2}: {}", memory.similarity.unwrap_or(0.0), memory.content);
//! }
//! ```
//!
//! # Mutability Requirements
//!
//! Methods that generate embeddings (`add`, `search`, `update`) require `&mut self`
//! because the embedding engine internally mutates state for ONNX tensor allocations.

pub mod config;
pub mod embedding;
pub mod errors;
pub mod memory;
pub mod memory_types;
pub mod project;
mod rrf;
mod temporal;
mod sqlite;

// Re-export public API
pub use config::Config;
pub use embedding::{EmbeddingEngine, EMBEDDING_DIMS};
pub use errors::Error;
pub use memory::MemoryStore;
pub use memory::store::{MAX_INPUT_LENGTH, MAX_SEARCH_LIMIT};
pub use memory_types::{AddResult, ConflictMemory};
pub use project::detect_project;
pub use sqlite::Memory;