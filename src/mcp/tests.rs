//! Tests for MCP server and tools.

use crate::config::Config;
use crate::mcp::params::*;
use crate::mcp::store_wrapper::StoreWrapper;
use crate::mcp::tools::ToolHandler;
use crate::memory::lifecycle::MemoryImportance;
use crate::memory::{MemoryStore, NewMemory};
use std::sync::{Arc, Mutex};

/// Seed an active row at a chosen level through the wrapper, so a
/// handler-driven test starts from a known row without going through the
/// tool under test.
pub(super) fn seed_row_at(
    wrapper: &StoreWrapper,
    project: &str,
    text: &str,
    importance: MemoryImportance,
) -> crate::Memory {
    use crate::memory::lifecycle::{MemoryStatus, MemoryType};
    let row = NewMemory {
        project_id: project,
        content: text,
        metadata: None,
        memory_type: MemoryType::Fact,
        status: MemoryStatus::Active,
    };
    let seeded = wrapper
        .ingest_with_lifecycle(&row, true, importance)
        .expect("seed add");
    let id = seeded["id"].as_str().expect("seed id").to_string();
    wrapper
        .get(&id, project)
        .expect("get")
        .expect("seeded row readable")
}

/// Seed one active row at the default level. See [`seed_row_at`].
pub(super) fn seed_row(wrapper: &StoreWrapper, project: &str, text: &str) -> crate::Memory {
    seed_row_at(wrapper, project, text, MemoryImportance::Medium)
}

/// Seed one active row at the default level carrying `metadata` verbatim, so a
/// metadata test starts from a known document rather than a NULL column.
pub(super) fn seed_row_with_metadata(
    wrapper: &StoreWrapper,
    project: &str,
    text: &str,
    metadata: &str,
) -> crate::Memory {
    use crate::memory::lifecycle::{MemoryStatus, MemoryType};
    let row = NewMemory {
        project_id: project,
        content: text,
        metadata: Some(metadata),
        memory_type: MemoryType::Fact,
        status: MemoryStatus::Active,
    };
    let seeded = wrapper
        .ingest_with_lifecycle(&row, true, MemoryImportance::Medium)
        .expect("seed add");
    let id = seeded["id"].as_str().expect("seed id").to_string();
    wrapper
        .get(&id, project)
        .expect("get")
        .expect("seeded row readable")
}

/// Pull the `id` out of a tool result's JSON text content.
///
/// The tool handlers return `CallToolResult`, whose payload is a list of
/// text contents holding the serialised response, so a handler-driven test
/// has to unwrap that envelope once instead of per assertion.
pub(super) fn tool_result_id(result: &rmcp::model::CallToolResult) -> String {
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .next()
        .expect("tool result carries text content");
    serde_json::from_str::<serde_json::Value>(text)
        .expect("tool result text is JSON")
        .get("id")
        .and_then(|v| v.as_str())
        .expect("tool result JSON carries an id")
        .to_string()
}

/// The arguments an MCP client would send for a `store_memory` call.
pub(super) fn store_memory_args(
    text: &str,
    status: Option<&str>,
    supersedes: Option<&str>,
) -> serde_json::Value {
    store_memory_args_full(text, status, supersedes, None)
}

/// As [`store_memory_args`], with the `importance` an agent would send.
/// An absent key and `"importance": null` both deserialize to `None`, so
/// passing `None` here is the omitted-field case.
pub(super) fn store_memory_args_full(
    text: &str,
    status: Option<&str>,
    supersedes: Option<&str>,
    importance: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "text": text,
        "status": status,
        "supersedes": supersedes,
        "importance": importance,
    })
}

#[cfg(all(test, feature = "mcp"))]
mod tool_handler_tests {
    use super::*;

    /// Test that MCP server can be initialized.
    #[test]
    fn test_mcp_server_initialization() {
        let (_dir, store) = MemoryStore::test_store();
        let config = Config::default();
        let _handler = ToolHandler::new(
            Arc::new(Mutex::new(store)),
            "test-project".to_string(),
            config,
        );
    }

    /// Regression test for #149: `ToolHandler::new` must retain the exact
    /// config it is constructed with. `recency_weight` and `hybrid` are set
    /// to non-default values here so a regression that falls back to
    /// `Config::default()` (or drops a field via `..Config::default()`
    /// somewhere upstream of construction) is observable on the handler.
    #[test]
    fn test_tool_handler_retains_non_default_config() {
        let (_dir, store) = MemoryStore::test_store();
        let config = Config {
            recency_weight: 0.91,
            hybrid: true,
            ..Config::default()
        };

        let handler = ToolHandler::new(
            Arc::new(Mutex::new(store)),
            "test-project".to_string(),
            config,
        );

        assert_eq!(handler.config().recency_weight, 0.91);
        assert!(handler.config().hybrid);
    }

    /// Test that parameter structs are valid for serde.
    #[test]
    fn test_store_memory_params_serde() {
        let params = StoreMemoryParams {
            text: "test memory".to_string(),
            metadata: Some(serde_json::json!({"topic": "test"})),
            memory_type: None,
            status: None,
            importance: None,
            supersedes: None,
            force: None,
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: StoreMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text, "test memory");
        assert!(decoded.metadata.is_some());
        assert!(decoded.memory_type.is_none());
        assert!(decoded.status.is_none());
        assert!(decoded.importance.is_none());
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
            importance: Some("critical".to_string()),
            supersedes: Some("old-memory-id".to_string()),
            force: Some(true),
        };

        let json = serde_json::to_string(&params).unwrap();
        let decoded: StoreMemoryParams = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text, "test memory");
        assert!(decoded.metadata.is_some());
        assert_eq!(decoded.memory_type, Some("preference".to_string()));
        assert_eq!(decoded.status, Some("active".to_string()));
        assert_eq!(decoded.importance, Some("critical".to_string()));
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

    /// The tool router must expose exactly the tools the MCP server advertises.
    ///
    /// `#[tool_router]` generates `ToolHandler::tool_router()` from the `#[tool]`
    /// attributes, and `#[tool_handler]` routes `tools/list` through
    /// `Self::tool_router().list_all()` (rmcp-macros 1.8.0
    /// src/tool_handler.rs:66-77, src/tool_router.rs:62-70). An rmcp API change
    /// that silently drops a route would remove a tool from clients with no
    /// compile error and no failing test — this is that test. No embedding model
    /// is needed to list tools, so it runs in the default `cargo test`.
    #[test]
    fn test_tool_router_exposes_all_expected_tools() {
        let expected = [
            "delete_memory",
            "get_memory",
            "list_memories",
            "search_memories",
            "store_memory",
            "supersede_memory",
            "update_memory",
        ];

        let listed: Vec<String> = ToolHandler::test_tool_router()
            .list_all()
            .into_iter()
            .map(|tool| tool.name.into_owned())
            .collect();

        // `list_all` sorts by name (rmcp 1.8.0 src/handler/server/router/tool.rs:580-589),
        // so the comparison below is order-stable; assert it to keep that assumption visible.
        assert_eq!(
            listed,
            expected.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );

        // Every tool is also reachable by name, which is the path `tools/call`
        // takes through the router.
        for name in expected {
            assert!(
                ToolHandler::test_tool_router().get(name).is_some(),
                "tool '{name}' is missing from the MCP tool router"
            );
        }
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
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Ingest a memory
        let result = wrapper.ingest(project_id, "rust is awesome", None, false);
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
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Ingest a memory with metadata
        let metadata_json = serde_json::json!({"topic": "rust", "type": "fact"});
        let metadata_str = serde_json::to_string(&metadata_json).unwrap();
        let result = wrapper.ingest(project_id, "rust is awesome", Some(&metadata_str), false);
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

    /// Test delete_memory integration: store, delete, and verify deletion.
    #[tokio::test]
    async fn test_delete_memory_integration() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Store a memory
        let result = wrapper.ingest(project_id, "memory to delete", None, false);
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
}
