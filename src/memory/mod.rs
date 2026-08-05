//! Core memory store orchestrating embedding and SQLite operations.
//!
//! Provides a high-level API for storing, searching, and retrieving memories
//! with automatic embedding generation via the ONNX model.

pub(crate) mod batch;
pub(crate) mod crud;
pub mod search;

// pub(crate): module internals hidden; public items re-exported explicitly via lib.rs
pub(crate) mod store;

pub mod lifecycle;

pub use crud::UpdateParams;
pub use search::SearchOptions;
pub use store::MemoryStore;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod batch_tests;
