//! The row-read/write MCP surfaces — `get_memory` and `update_memory` — split
//! out of `tests.rs` for issue #271 so that file stays inside the per-file size
//! cap.
//!
//! The gate must keep the `feature = "mcp"` half: the tool handlers these tests
//! drive do not exist in the `--no-default-features` lane CI runs, and without
//! it the file compiles to zero tests while the orphan guard in `lib.rs` still
//! counts it as wired.
#[cfg(all(test, feature = "mcp"))]
mod mcp_update {
    use crate::mcp::store_wrapper::StoreWrapper;
    use crate::memory::lifecycle::MemoryImportance;
    use crate::memory::{MemoryStore, NewMemory, UpdateParams};
    use std::sync::{Arc, Mutex};

    /// Test get_memory integration: store a memory and retrieve it by ID.
    #[tokio::test]
    async fn test_get_memory_integration() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Store a memory via ingest
        let result = wrapper.ingest(project_id, "test memory content", None, false);
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

    /// Test update_memory integration: store, update, and verify update.
    #[tokio::test]
    async fn test_update_memory_integration() {
        use crate::memory::lifecycle::{MemoryStatus, MemoryType};

        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let project_id = "test-project";

        // Store a memory
        let result = wrapper.ingest_with_lifecycle(
            &NewMemory {
                project_id,
                content: "original content",
                metadata: None,
                memory_type: MemoryType::Fact,
                status: MemoryStatus::Active,
            },
            false,
            MemoryImportance::Medium,
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
