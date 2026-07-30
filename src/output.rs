//! JSON response types and formatting for CLI output.

use serde::Serialize;

/// Response for successful memory addition.
#[derive(Serialize)]
pub struct AddResponse {
    /// Operation status ("added").
    pub status: String,
    /// Unique identifier of the added memory.
    pub id: String,
}

/// Response for search results.
#[derive(Serialize)]
pub struct SearchResponse {
    /// List of search results ranked by relevance.
    pub results: Vec<SearchResultItem>,
}

/// Individual search result item.
#[derive(Serialize)]
pub struct SearchResultItem {
    /// Unique identifier of the memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Relevance score (0.0 to 1.0, higher is better).
    pub similarity: f64,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
}

/// Response for retrieving a specific memory.
#[derive(Serialize)]
pub struct GetResponse {
    /// Unique identifier of the memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Project identifier for this memory.
    pub project_id: String,
    /// Optional user-provided metadata (JSON string).
    pub metadata: Option<String>,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Last update timestamp in RFC3339 format.
    pub updated_at: String,
}

/// Response for listing memories.
#[derive(Serialize)]
pub struct ListResponse {
    /// List of memories ordered by creation time (newest first).
    pub memories: Vec<ListItem>,
}

/// Individual list item.
#[derive(Serialize)]
pub struct ListItem {
    /// Unique identifier of the memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
}

/// Response for successful memory deletion.
#[derive(Serialize)]
pub struct DeleteResponse {
    /// Operation status ("deleted").
    pub status: String,
    /// Unique identifier of the deleted memory.
    pub id: String,
}

/// Response for successful memory update.
#[derive(Serialize)]
pub struct UpdateResponse {
    /// Operation status ("updated").
    pub status: String,
    /// Unique identifier of the updated memory.
    pub id: String,
}

/// Response for error cases.
#[derive(Serialize)]
pub struct ErrorResponse {
    /// Error message describing what went wrong.
    pub error: String,
}

/// Response for conflict detection.
#[derive(Serialize)]
pub struct ConflictsResponse {
    /// Operation status ("conflicts").
    pub status: String,
    /// The proposed memory content.
    pub proposed: String,
    /// List of conflicting memories.
    pub conflicts: Vec<ConflictMemoryResponse>,
}

/// Individual conflicting memory in response.
#[derive(Serialize)]
pub struct ConflictMemoryResponse {
    /// Unique identifier of the conflicting memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Similarity score indicating the degree of conflict (0.0 to 1.0).
    pub similarity: f64,
}

/// Response for token validation.
#[derive(Serialize)]
pub struct ValidateResponse {
    /// Token count of the validated text.
    pub token_count: usize,
    /// Maximum allowed tokens for embedding.
    pub max_tokens: usize,
    /// Whether the text is within the embedding limit.
    pub within_limit: bool,
}

/// Response for `doctor --embeddings` per-project report.
#[derive(Serialize)]
pub struct DoctorResponse {
    /// Project identifier.
    pub project_id: String,
    /// Total number of rows in the project.
    pub total_rows: usize,
    /// Number of rows with real (L2-normalised) embeddings.
    pub real_rows: usize,
    /// Number of rows with mock embeddings.
    pub mock_rows: usize,
    /// Number of rows with unknown/corrupted embeddings.
    pub unknown_rows: usize,
}

/// Response for `reindex` per-project report.
#[derive(Serialize)]
pub struct ReindexResponse {
    /// Project identifier.
    pub project_id: String,
    /// Number of rows successfully re-embedded.
    pub reindexed: usize,
    /// Number of rows skipped (unknown/corrupted embeddings).
    pub skipped: usize,
    /// Rows that failed during re-embedding.
    pub failed: Vec<ReindexFailure>,
}

/// A single failed row during reindex.
#[derive(Serialize)]
pub struct ReindexFailure {
    /// Memory ID that failed.
    pub id: String,
    /// Error message describing the failure.
    pub error: String,
}

/// Serialize a value as formatted JSON and print to stdout.
///
/// Exits with status 1 if serialization fails.
pub fn print_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("Failed to serialize JSON: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_add_response() {
        let response = AddResponse {
            status: "added".to_string(),
            id: "test-id".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"added\""));
        assert!(json.contains("\"id\":\"test-id\""));
    }

    #[test]
    fn test_serialize_search_response() {
        let response = SearchResponse {
            results: vec![SearchResultItem {
                id: "test-id".to_string(),
                content: "test content".to_string(),
                similarity: 0.95,
                created_at: "2024-01-01T00:00:00Z".to_string(),
            }],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"results\""));
        assert!(json.contains("\"similarity\":0.95"));
    }
}
