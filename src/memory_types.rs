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

/// Conflict policy for batch ingest operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum IngestPolicy {
    /// Check for conflicts and refuse to add if conflicts are found.
    ConflictAware,
    /// Bypass conflict detection and add regardless of conflicts.
    Force,
}

/// Input item for batch ingest operations.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BatchIngestItem {
    /// Text content to store (1 to 100,000 characters).
    pub content: String,
    /// Optional JSON metadata string.
    pub metadata: Option<String>,
}

#[allow(dead_code)]
impl BatchIngestItem {
    /// Create a new batch ingest item.
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: None,
        }
    }

    /// Create a new batch ingest item with metadata.
    pub fn with_metadata(content: String, metadata: String) -> Self {
        Self {
            content,
            metadata: Some(metadata),
        }
    }
}

/// Per-item result for batch ingest operations.
///
/// Each item in a batch ingest operation returns a `BatchIngestResult`
/// indicating whether it was added, had conflicts, or encountered an error.
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub enum BatchIngestResult {
    /// Memory was successfully added.
    Added { id: String },
    /// Memory conflicts with existing similar memories.
    Conflicts {
        proposed: String,
        conflicts: Vec<ConflictMemory>,
    },
    /// Error processing this item.
    Error { message: String },
}

/// Summary statistics for batch ingest operations.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[allow(dead_code)]
pub struct BatchIngestSummary {
    /// Number of items successfully added.
    pub added: usize,
    /// Number of items with conflicts.
    pub conflicts: usize,
    /// Number of items that encountered errors.
    pub errors: usize,
    /// Total number of items in the batch.
    pub total: usize,
}

impl BatchIngestSummary {
    /// Calculate summary from a slice of results.
    fn from_results(results: &[BatchIngestResult]) -> Self {
        let mut added = 0;
        let mut conflicts = 0;
        let mut errors = 0;

        for result in results {
            match result {
                BatchIngestResult::Added { .. } => added += 1,
                BatchIngestResult::Conflicts { .. } => conflicts += 1,
                BatchIngestResult::Error { .. } => errors += 1,
            }
        }

        Self {
            added,
            conflicts,
            errors,
            total: results.len(),
        }
    }
}

/// Complete result from a batch ingest operation.
///
/// Contains per-item results (in input order) and summary statistics.
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct BatchIngestOutcome {
    /// Per-item results, indexed by input order.
    pub results: Vec<BatchIngestResult>,
    /// Summary statistics.
    pub summary: BatchIngestSummary,
}

impl BatchIngestOutcome {
    /// Create a new batch ingest outcome from results.
    pub fn new(results: Vec<BatchIngestResult>) -> Self {
        let summary = BatchIngestSummary::from_results(&results);
        Self { results, summary }
    }
}
