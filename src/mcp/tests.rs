//! Tests for MCP server and tools.

use crate::config::Config;
use crate::mcp::params::*;
use crate::mcp::tools::ToolHandler;
use crate::memory::MemoryStore;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

/// Create a test MemoryStore with in-memory database.
fn create_test_store() -> (MemoryStore, TempDir) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let config = Config::default();
    let store = MemoryStore::new(&path, &config.embedding_model, config.clone()).unwrap();
    (store, dir)
}

#[cfg(all(test, feature = "mcp"))]
mod tool_handler_tests {
    use super::*;

    /// Test that MCP server can be initialized.
    #[test]
    fn test_mcp_server_initialization() {
        let (store, _dir) = create_test_store();
        let _handler = ToolHandler::new(Arc::new(Mutex::new(store)), "test-project".to_string());
    }

    /// Test that parameter structs are valid for serde.
    #[test]
    fn test_store_memory_params_serde() {
        let params = StoreMemoryParams {
            text: "test memory".to_string(),
            metadata: Some(serde_json::json!({"topic": "test"})),
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: StoreMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text, "test memory");
        assert!(decoded.metadata.is_some());
    }

    /// Test search parameters serde.
    #[test]
    fn test_search_params_serde() {
        let params = SearchMemoriesParams {
            query: "test query".to_string(),
            limit: Some(10),
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: SearchMemoriesParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.query, "test query");
        assert_eq!(decoded.limit, Some(10));
    }

    /// Test list parameters serde.
    #[test]
    fn test_list_params_serde() {
        let params = ListMemoriesParams { limit: Some(5) };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: ListMemoriesParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.limit, Some(5));
    }

    /// Test list parameters with no limit (default behavior).
    #[test]
    fn test_list_params_default_limit() {
        let params = ListMemoriesParams { limit: None };
        let json = serde_json::to_string(&params).unwrap();
        let decoded: ListMemoriesParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.limit, None);
    }

    /// Test success response serialization.
    #[test]
    fn test_success_response_serde() {
        let response = SuccessResponse {
            id: "test-id".to_string(),
            status: "added".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        // Verify JSON contains expected fields
        assert!(json.contains("test-id"));
        assert!(json.contains("added"));
    }

    /// Test conflict memory serialization.
    #[test]
    fn test_conflict_memory_serde() {
        let memory = ConflictMemory {
            id: "test-id".to_string(),
            content: "test content".to_string(),
            similarity: 0.85,
        };

        let json = serde_json::to_string(&memory).unwrap();
        assert!(json.contains("test-id"));
        assert!(json.contains("test content"));
        assert!(json.contains("0.85"));
    }
}
