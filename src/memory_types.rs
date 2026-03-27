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

/// Policy for handling conflicts during memory ingestion.
///
/// Determines whether similar existing memories should block addition
/// or be ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestPolicy {
    /// Detect and reject conflicts (similar to force=false in add_with_conflict).
    ///
    /// Returns conflict details without storing if similar memories exist
    /// with similarity >= threshold.
    ConflictAware,
    /// Force addition regardless of conflicts (similar to force=true).
    ///
    /// Bypasses conflict detection and stores the memory unconditionally.
    Force,
}
