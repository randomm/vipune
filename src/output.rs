//! JSON response types and formatting for CLI output.

use serde::Serialize;

/// Response for successful memory addition.
#[derive(Serialize)]
pub struct AddResponse {
    pub status: String,
    pub id: String,
}

/// Response for search results.
#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
}

/// Individual search result item.
#[derive(Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub content: String,
    pub similarity: f64,
    pub created_at: String,
}

/// Response for retrieving a specific memory.
#[derive(Serialize)]
pub struct GetResponse {
    pub id: String,
    pub content: String,
    pub project_id: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Response for listing memories.
#[derive(Serialize)]
pub struct ListResponse {
    pub memories: Vec<ListItem>,
}

/// Individual list item.
#[derive(Serialize)]
pub struct ListItem {
    pub id: String,
    pub content: String,
    pub created_at: String,
}

/// Version response (reserved for future use).
#[derive(Serialize)]
#[allow(dead_code)] // Dead code justified: available for future CLI enhancement
pub struct VersionResponse {
    pub status: String,
    pub id: String,
}

/// Response for successful memory deletion.
#[derive(Serialize)]
pub struct DeleteResponse {
    pub status: String,
    pub id: String,
}

/// Response for successful memory update.
#[derive(Serialize)]
pub struct UpdateResponse {
    pub status: String,
    pub id: String,
}

/// Response for import operations.
#[derive(Serialize)]
pub struct ImportResponse {
    pub status: String,
    pub total_memories: usize,
    pub imported: usize,
    pub skipped_duplicates: usize,
    pub skipped_corrupted: usize,
    pub projects: usize,
}

/// Response for errors.
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Response for conflict detection.
#[derive(Serialize)]
pub struct ConflictsResponse {
    pub status: String,
    pub proposed: String,
    pub conflicts: Vec<ConflictMemoryResponse>,
}

/// Individual conflicting memory in response.
#[derive(Serialize)]
pub struct ConflictMemoryResponse {
    pub id: String,
    pub content: String,
    pub similarity: f64,
}

/// Print a value as formatted JSON to stdout.
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

    #[test]
    fn test_serialize_import_response() {
        let response = ImportResponse {
            status: "imported".to_string(),
            total_memories: 100,
            imported: 95,
            skipped_duplicates: 4,
            skipped_corrupted: 1,
            projects: 2,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"imported\":95"));
        assert!(json.contains("\"skipped_duplicates\":4"));
        assert!(json.contains("\"projects\":2"));
    }
}
