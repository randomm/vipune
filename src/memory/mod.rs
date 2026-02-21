//! Core memory store orchestrating embedding and SQLite operations.
//!
//! Provides a high-level API for storing, searching, and retrieving memories
//! with automatic embedding generation via the ONNX model.

mod crud;
mod search;

// pub(crate): module internals hidden; public items re-exported explicitly via lib.rs
pub(crate) mod store;

pub use store::MemoryStore;

#[cfg(test)]
mod tests;
