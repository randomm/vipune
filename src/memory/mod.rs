//! Core memory store orchestrating embedding and SQLite operations.
//!
//! Provides a high-level API for storing, searching, and retrieving memories
//! with automatic embedding generation via the ONNX model.

pub(crate) mod add;
pub(crate) mod batch;
pub(crate) mod crud;
pub mod search;

// pub(crate): module internals hidden; public items re-exported explicitly via lib.rs
pub(crate) mod store;

pub mod lifecycle;

pub use add::NewMemory;
pub use batch::NewSupersedingMemory;
pub use crud::UpdateParams;
pub use search::SearchOptions;
pub use store::MemoryStore;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod batch_tests;

// Wired from `memory::tests` (it shares that module's `crud` test helpers), but
// declared here: the orphan guard in `lib.rs` only reads this directory's files.
#[cfg(test)]
#[path = "supersede_tests.rs"]
mod supersede_tests;

/// The re-supersede guard, split from `supersede_tests` for #271 so both stay
/// inside the per-file cap. Same wiring reason as above.
#[cfg(test)]
#[path = "resupersede_guard_tests.rs"]
mod resupersede_guard_tests;
