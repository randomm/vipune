//! Tests for MCP server and tools.

use crate::config::Config;
use crate::mcp::params::*;
use crate::mcp::store_wrapper::StoreWrapper;
use crate::mcp::tools::ToolHandler;
use crate::memory::{MemoryStore, UpdateParams};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

/// Create a test MemoryStore with a fake embedder (no ONNX model load).
///
/// Uses `from_db_with_test_embedder` so the shared `test_embedder` field is wired in,
/// avoiding the real ONNX session that caused race-condition flakes when multiple
/// async tests loaded the model concurrently.
///
/// ⚠️ The TempDir is forgotten to keep the database file alive for the test duration.
/// This matches the pattern used by `MemoryStore::test_store()`.
fn create_test_store() -> MemoryStore {
    use crate::sqlite::Database;

    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).expect("open test database");
    MemoryStore::from_db_with_test_embedder(db)
}

#[cfg(all(test, feature = "mcp"))]
mod tool_handler_tests {
    use super::*;

    /// Test that MCP server can be initialized.
    #[test]
    fn test_mcp_server_initialization() {
        let store = create_test_store();
        let config = Config::default();
        let _handler = ToolHandler::new(
            Arc::new(Mutex::new(store)),
            "test-project".to_string(),
            config,
        );
    }

    /// Test that parameter structs are valid for serde.
    #[test]
    fn test_store_memory_params_serde() {
        let params = StoreMemoryParams {
            text: "test memory".to_string(),
            metadata: Some(serde_json::json!({"topic": "test"})),
            memory_type: None,
            status: None,
            supersedes: None,
            force: None,
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: StoreMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text, "test memory");
        assert!(decoded.metadata.is_some());
        assert!(decoded.memory_type.is_none());
        assert!(decoded.status.is_none());
        assert!(decoded.supersedes.is_none());
        assert!(decoded.force.is_none());
    }

    /// Test that StoreMemoryParams with all fields serializes correctly.
    #[test]
    fn test_store_memory_params_with_all_fields() {
        let params = StoreMemoryParams {
            text: "test memory".to_string(),
            metadata: Some(serde_json::json!({"topic": "test"})),
            memory_type: Some("preference".to_string()),
            status: Some("active".to_string()),
            supersedes: Some("old-memory-id".to_string()),
            force: Some(true),
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: StoreMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text, "test memory");
        assert!(decoded.metadata.is_some());
        assert_eq!(decoded.memory_type, Some("preference".to_string()));
        assert_eq!(decoded.status, Some("active".to_string()));
        assert_eq!(decoded.supersedes, Some("old-memory-id".to_string()));
        assert_eq!(decoded.force, Some(true));
    }

    /// Test that SupersedeMemoryParams serializes and deserializes correctly.
    #[test]
    fn test_supersede_memory_params_serde() {
        let params = SupersedeMemoryParams {
            new_text: "updated memory".to_string(),
            old_memory_id: "old-memory-id".to_string(),
            memory_type: Some("fact".to_string()),
            metadata: Some(serde_json::json!({"updated": true})),
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: SupersedeMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_text, "updated memory");
        assert_eq!(decoded.old_memory_id, "old-memory-id");
        assert_eq!(decoded.memory_type, Some("fact".to_string()));
        assert_eq!(decoded.metadata, Some(serde_json::json!({"updated": true})));
    }

    /// Test that SupersedeMemoryParams with minimal fields works.
    #[test]
    fn test_supersede_memory_params_minimal() {
        let params = SupersedeMemoryParams {
            new_text: "updated memory".to_string(),
            old_memory_id: "old-memory-id".to_string(),
            memory_type: None,
            metadata: None,
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: SupersedeMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_text, "updated memory");
        assert_eq!(decoded.old_memory_id, "old-memory-id");
        assert!(decoded.memory_type.is_none());
        assert!(decoded.metadata.is_none());
    }

    /// Test search parameters serde.
    #[test]
    fn test_search_params_serde() {
        let params = SearchMemoriesParams {
            query: "test query".to_string(),
            limit: Some(10),
            memory_types: None,
            statuses: None,
            recency_weight: None,
            hybrid: None,
            no_touch: None,
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: SearchMemoriesParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.query, "test query");
        assert_eq!(decoded.limit, Some(10));
        assert!(decoded.memory_types.is_none());
        assert!(decoded.statuses.is_none());
    }

    /// Test list parameters serde.
    #[test]
    fn test_list_params_serde() {
        let params = ListMemoriesParams {
            limit: Some(5),
            memory_types: None,
            statuses: None,
            no_touch: None,
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: ListMemoriesParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.limit, Some(5));
        assert!(decoded.memory_types.is_none());
        assert!(decoded.statuses.is_none());
    }

    /// Test list parameters with no limit (default behavior).
    #[test]
    fn test_list_params_default_limit() {
        let params = ListMemoriesParams {
            limit: None,
            memory_types: None,
            statuses: None,
            no_touch: None,
        };
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

    /// Tests ToolHandler data path: ingest, search, and list operations through StoreWrapper.
    #[tokio::test]
    async fn test_store_and_search_integration() {
        let store = create_test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Ingest a memory
        let result = wrapper.ingest(project_id, "rust is awesome", "null", false);
        assert!(result.is_ok());
        let value = result.unwrap();
        let json_str = serde_json::to_string(&value).unwrap();
        assert!(json_str.contains("added"));

        // Search for it
        let search_result = wrapper.search(
            project_id,
            "rust programming",
            5,
            0.0,
            None::<Vec<&str>>,
            None,
        );
        assert!(search_result.is_ok());
        let search_value = search_result.unwrap();
        let search_str = serde_json::to_string(&search_value).unwrap();
        assert!(search_str.contains("rust is awesome"));

        // List it
        let list_result = wrapper.list(project_id, 10, None, None);
        assert!(list_result.is_ok());
        let list_value = list_result.unwrap();
        let list_str = serde_json::to_string(&list_value).unwrap();
        assert!(list_str.contains("rust is awesome"));
    }

    /// Test that search results include metadata, project_id, and updated_at fields.
    #[tokio::test]
    async fn test_search_results_include_metadata_and_project_id() {
        let store = create_test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Ingest a memory with metadata
        let metadata_json = serde_json::json!({"topic": "rust", "type": "fact"});
        let metadata_str = serde_json::to_string(&metadata_json).unwrap();
        let result = wrapper.ingest(project_id, "rust is awesome", &metadata_str, false);
        assert!(result.is_ok());

        // Search for it
        let search_result = wrapper.search(project_id, "rust", 5, 0.0, None::<Vec<&str>>, None);
        assert!(search_result.is_ok());
        let search_value = search_result.unwrap();

        // Parse JSON to verify all fields are present
        let search_str = serde_json::to_string(&search_value).unwrap();
        assert!(search_str.contains("\"id\""));
        assert!(search_str.contains("\"content\""));
        assert!(search_str.contains("\"similarity\""));
        assert!(search_str.contains("\"created_at\""));
        assert!(search_str.contains("\"updated_at\""));
        assert!(search_str.contains("\"project_id\""));
        assert!(search_str.contains("\"metadata\""));
        assert!(search_str.contains("rust is awesome"));
        assert!(search_str.contains("test-project"));

        // Verify the JSON structure by parsing it
        let parsed: serde_json::Value = serde_json::from_str(&search_str).unwrap();
        assert!(parsed.is_array());
        let results = parsed.as_array().unwrap();
        assert!(!results.is_empty());

        let first_result = &results[0];
        assert!(first_result.get("id").is_some());
        assert!(first_result.get("content").is_some());
        assert!(first_result.get("similarity").is_some());
        assert!(first_result.get("created_at").is_some());
        assert!(first_result.get("updated_at").is_some());
        assert!(first_result.get("project_id").is_some());
        assert!(first_result.get("metadata").is_some());

        // Verify project_id matches
        assert_eq!(
            first_result["project_id"],
            serde_json::Value::String("test-project".to_string())
        );

        // Verify metadata is parsed as a JSON object, not a string
        assert_eq!(first_result["metadata"], metadata_json);
    }

    /// Test that search results serialize correctly when metadata is None.
    #[tokio::test]
    async fn test_search_results_with_null_metadata() {
        let store = create_test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Ingest a memory without metadata (null)
        let result = wrapper.ingest(project_id, "simple fact", "null", false);
        assert!(result.is_ok());

        // Search for it
        let search_result = wrapper.search(project_id, "simple", 5, 0.0, None::<Vec<&str>>, None);
        assert!(search_result.is_ok());
        let search_value = search_result.unwrap();

        // Parse JSON to verify metadata field is present and null
        let search_str = serde_json::to_string(&search_value).unwrap();
        assert!(search_str.contains("\"metadata\":null"));

        // Verify the JSON structure by parsing it
        let parsed: serde_json::Value = serde_json::from_str(&search_str).unwrap();
        assert!(parsed.is_array());
        let results = parsed.as_array().unwrap();
        assert!(!results.is_empty());

        let first_result = &results[0];
        assert!(first_result.get("metadata").is_some());
        assert_eq!(first_result["metadata"], serde_json::Value::Null);
    }

    /// Test get_memory integration: store a memory and retrieve it by ID.
    #[tokio::test]
    async fn test_get_memory_integration() {
        let store = create_test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Store a memory via ingest
        let result = wrapper.ingest(project_id, "test memory content", "null", false);
        assert!(result.is_ok());
        let value = result.unwrap();
        let json_str = serde_json::to_string(&value).unwrap();
        assert!(json_str.contains("added"));

        // Extract the ID from the response
        let response: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let id = response["id"].as_str().unwrap();

        // Get the memory by ID
        let memory_result = wrapper.get(id, project_id);
        assert!(memory_result.is_ok());
        let memory_opt = memory_result.unwrap();
        assert!(memory_opt.is_some());
        let memory = memory_opt.unwrap();

        // Verify the returned memory has the expected content
        assert_eq!(memory.id, id);
        assert_eq!(memory.content, "test memory content");
        assert_eq!(memory.project_id, project_id);

        // Try to get a non-existent memory, verify it returns None
        let nonexistent_result = wrapper.get("nonexistent-id", project_id);
        assert!(nonexistent_result.is_ok());
        assert!(nonexistent_result.unwrap().is_none());
    }

    /// Test delete_memory integration: store, delete, and verify deletion.
    #[tokio::test]
    async fn test_delete_memory_integration() {
        let store = create_test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Store a memory
        let result = wrapper.ingest(project_id, "memory to delete", "null", false);
        assert!(result.is_ok());
        let value = result.unwrap();
        let json_str = serde_json::to_string(&value).unwrap();
        let response: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let id = response["id"].as_str().unwrap();

        // Delete the memory
        let delete_result = wrapper.delete(id, project_id);
        assert!(delete_result.is_ok());
        assert!(delete_result.unwrap(), "delete should return true");

        // Try to get the deleted memory, verify it's not found
        let get_result = wrapper.get(id, project_id);
        assert!(get_result.is_ok());
        assert!(
            get_result.unwrap().is_none(),
            "deleted memory should not be found"
        );

        // Try to delete again, verify it returns false
        let delete_again_result = wrapper.delete(id, project_id);
        assert!(delete_again_result.is_ok());
        assert!(
            !delete_again_result.unwrap(),
            "deleting non-existent memory should return false"
        );
    }

    /// Test update_memory integration: store, update, and verify update.
    #[tokio::test]
    async fn test_update_memory_integration() {
        use crate::memory::lifecycle::{MemoryStatus, MemoryType};

        let store = create_test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Store a memory
        let result = wrapper.ingest_with_type_status(
            project_id,
            "original content",
            "null",
            false,
            MemoryType::Fact,
            MemoryStatus::Active,
        );
        assert!(result.is_ok());
        let value = result.unwrap();
        let json_str = serde_json::to_string(&value).unwrap();
        let response: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let id = response["id"].as_str().unwrap();

        // Update the memory with new text
        let update_result = wrapper.update(
            id,
            project_id,
            UpdateParams {
                text: Some("updated content"),
                ..Default::default()
            },
        );
        assert!(update_result.is_ok(), "update should succeed");

        // Verify the content changed by getting the memory
        let get_result = wrapper.get(id, project_id);
        assert!(get_result.is_ok());
        let memory_opt = get_result.unwrap();
        assert!(memory_opt.is_some());
        let memory = memory_opt.unwrap();
        assert_eq!(memory.content, "updated content");

        // Update with new metadata
        let new_metadata = serde_json::json!({"topic": "test", "updated": true});
        let metadata_str = serde_json::to_string(&new_metadata).unwrap();
        let update_meta_result = wrapper.update(
            id,
            project_id,
            UpdateParams {
                metadata: Some(&metadata_str),
                ..Default::default()
            },
        );
        assert!(update_meta_result.is_ok());

        // Verify metadata updated
        let get_meta_result = wrapper.get(id, project_id);
        assert!(get_meta_result.is_ok());
        let memory_meta_opt = get_meta_result.unwrap();
        assert!(memory_meta_opt.is_some());
        let memory_meta = memory_meta_opt.unwrap();
        assert!(memory_meta.metadata.is_some());
        let stored_metadata: serde_json::Value =
            serde_json::from_str(memory_meta.metadata.as_ref().unwrap()).unwrap();
        assert_eq!(stored_metadata, new_metadata);

        // Try to update with all None optional fields, should fail
        let update_none_result = wrapper.update(id, project_id, UpdateParams::default());
        assert!(
            update_none_result.is_err(),
            "update with all None fields should fail"
        );

        // Update with new type and status
        let update_type_status_result = wrapper.update(
            id,
            project_id,
            UpdateParams {
                memory_type: Some(MemoryType::Preference),
                status: Some(MemoryStatus::Candidate),
                ..Default::default()
            },
        );
        assert!(update_type_status_result.is_ok());

        // Verify type and status updated
        let get_final_result = wrapper.get(id, project_id);
        assert!(get_final_result.is_ok());
        let memory_final_opt = get_final_result.unwrap();
        assert!(memory_final_opt.is_some());
        let memory_final = memory_final_opt.unwrap();
        assert_eq!(memory_final.memory_type, MemoryType::Preference.as_str());
        assert_eq!(memory_final.status, MemoryStatus::Candidate.as_str());
    }
}
