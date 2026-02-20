//! Memory store data types.

/// Result type for conflict-aware add operations.
#[derive(Debug, serde::Serialize)]
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
#[derive(Debug, serde::Serialize)]
pub struct ConflictMemory {
    /// Unique identifier of the conflicting memory.
    pub id: String,
    /// Memory content that conflicts with the proposed addition.
    pub content: String,
    /// Similarity score indicating the degree of conflict (0.0 to 1.0).
    pub similarity: f64,
}
