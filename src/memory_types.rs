//! Memory store data types.
//!
//! # Library-Only Public API
//!
//! The batch ingest API (`IngestPolicy`, `BatchIngestItemResult`, `BatchIngestResult`)
//! is part of the public library interface for external consumers (integration tests, kide, etc.)
//! even though the CLI binary (`src/main.rs`) doesn't use them directly.
//! These types are re-exported from `lib.rs` for library users.

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
///
/// This enum is part of the public library API, re-exported from lib.rs.
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

/// Result of processing a single item in a batch ingest operation.
///
/// Each item maps to its input index, enabling deterministic partial-failure handling.
///
/// This enum is part of the public library API, re-exported from lib.rs.
#[derive(Debug, Serialize)]
pub enum BatchIngestItemResult {
    /// Item was successfully added.
    Added { id: String },
    /// Item conflicted with existing memories (only for ConflictAware policy).
    Conflicts {
        proposed: String,
        conflicts: Vec<ConflictMemory>,
    },
    /// Item failed to process (empty, too long, or other errors).
    Error { message: String },
}

/// Summary result for a batch ingest operation.
///
/// Provides deterministic mapping from input indices to per-item outcomes.
///
/// This struct is part of the public library API, re-exported from lib.rs.
#[derive(Debug, Serialize)]
pub struct BatchIngestResult {
    /// Vector of results, one per input item, in the same order as input.
    ///
    /// Each element's index corresponds to the input item's position,
    /// enabling callers to identify which items succeeded, conflicted, or failed.
    pub results: Vec<BatchIngestItemResult>,
}
