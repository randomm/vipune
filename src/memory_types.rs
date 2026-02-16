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
    pub id: String,
    pub content: String,
    pub similarity: f64,
}
