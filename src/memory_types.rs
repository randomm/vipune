//! Memory store data types.

use serde::Serialize;

/// Result type for conflict-aware add operations.
///
/// Returned by `MemoryStore::add_with_conflict()` to indicate whether
/// a memory was successfully added or conflicts were detected.
#[derive(Debug, Serialize)]
pub enum AddResult {
    /// Memory was successfully added.
    Added { id: String },
    /// Memory conflicts with existing similar memories.
    Conflicts {
        proposed: String,
        conflicts: Vec<ConflictMemory>,
    },
}

/// Details about a conflicting memory.
///
/// Provides information about memories that are similar to a proposed addition,
/// including their IDs, content, and similarity scores.
#[derive(Debug, Serialize)]
pub struct ConflictMemory {
    /// Unique identifier of the conflicting memory.
    pub id: String,
    /// Memory content that conflicts with the proposed addition.
    pub content: String,
    /// Similarity score indicating the degree of conflict (0.0 to 1.0).
    pub similarity: f64,
}
