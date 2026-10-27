//! `store_memory` / `update_memory` metadata-shape tests for the MCP surface
//! (issue #309).
//!
//! New module (not split out of `tests.rs`): the metadata write rule moved into
//! one helper in #309 and these are the tool-driven cases that pin it. The gate
//! keeps the `feature = "mcp"` half for the same reason as `supersede_tests`:
//! the tool handler these tests drive does not exist in the
//! `--no-default-features` lane CI runs. The shared fixtures come from
//! `tests.rs` behind `pub(super)`.
#[cfg(test)]
mod mcp_store_metadata {
    use crate::config::Config;
    use crate::mcp::store_wrapper::StoreWrapper;
    use crate::mcp::tests::{seed_row_with_metadata, tool_result_id};
    use crate::mcp::tools::ToolHandler;
    use crate::memory::MemoryStore;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    /// A handler and a wrapper over the same store, so a test can drive the tool
    /// and then read the row back.
    fn handler_for(project: &str) -> (tempfile::TempDir, ToolHandler, StoreWrapper) {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let handler = ToolHandler::new(store.clone(), project.to_string(), Config::default());
        let wrapper = StoreWrapper::new(store);
        (_dir, handler, wrapper)
    }

    /// Issue #309: `store_memory` built the 4-character string `"null"` for an
    /// absent argument and then compared the serialised text back to that literal
    /// to recover `None`. The two spellings of "no metadata" must land on the same
    /// SQL NULL column and no other value may be collapsed into it — so the JSON
    /// *string* `"null"` keeps its quotes.
    ///
    /// The assertion reads the stored column back through `StoreWrapper::get`,
    /// which hands it over verbatim via `map_row_to_memory`: the read-side
    /// `trim() != "null"` rendering that collapses both spellings lives in
    /// `parse_metadata` / the tool response builders, which this path does not go
    /// through, so a regression that stores the text `null` instead of SQL NULL
    /// shows up here as `Some("null")` against an expected `None`.
    #[tokio::test]
    async fn store_memory_stores_no_metadata_for_omitted_or_null_metadata() {
        let (_dir, handler, wrapper) = handler_for("omitted-meta-proj");

        // (arguments sent to the tool, metadata column expected on the new row).
        // `force` because every case shares one text: without it the conflict
        // scan answers on the second row and the metadata never reaches the
        // store, so the assertions below would be vacuous.
        let cases = [
            (
                "omitted",
                json!({ "text": "no metadata", "force": true }),
                None,
            ),
            (
                "explicit null",
                json!({ "text": "no metadata", "metadata": null, "force": true }),
                None,
            ),
            (
                "json-null string",
                json!({ "text": "no metadata", "metadata": "null", "force": true }),
                Some("\"null\""),
            ),
            (
                "a document",
                json!({
                    "text": "no metadata",
                    "metadata": { "topic": "auth" },
                    "force": true
                }),
                Some(r#"{"topic":"auth"}"#),
            ),
        ];

        for (label, args, expected) in cases {
            let result = handler
                .call_tool_for_test("store_memory", args)
                .await
                .unwrap_or_else(|err| panic!("{label}: store_memory must succeed: {err:?}"));
            let id = tool_result_id(&result);
            let row = wrapper
                .get(&id, "omitted-meta-proj")
                .expect("get")
                .unwrap_or_else(|| panic!("{label}: row stored"));
            assert_eq!(
                row.metadata.as_deref(),
                expected,
                "{label}: the metadata column must hold what the handler was given"
            );
        }
    }

    /// Issue #309, the `update_memory` half of the rule: an absent `metadata`
    /// argument and `"metadata": null` both reach the handler as `None` — rmcp
    /// deserialises the JSON `null` into `None` for the `Option<Value>` field
    /// before the handler body runs — so both must be "leave the column alone",
    /// not "clear it". The row keeps the document it was seeded with while the
    /// text still updates.
    ///
    /// The helper's `Value::Null` arm is not what this pins (the serde boundary
    /// means the handler never sees it); that arm is covered by
    /// `metadata_column_maps_only_absent_and_json_null_to_none` in `helpers.rs`.
    #[tokio::test]
    async fn update_memory_leaves_metadata_untouched_when_it_is_omitted_or_null() {
        const SEEDED: &str = r#"{"keep":"me"}"#;
        const SEED_TEXT: &str = "row to reword";

        for label in ["omitted", "explicit null"] {
            let (_dir, handler, wrapper) = handler_for("upd-meta-proj");
            let seed = seed_row_with_metadata(&wrapper, "upd-meta-proj", SEED_TEXT, SEEDED);

            let mut args = json!({ "id": seed.id, "text": "reworded content" });
            if label == "explicit null" {
                args["metadata"] = json!(null);
            }

            handler
                .call_tool_for_test("update_memory", args)
                .await
                .unwrap_or_else(|err| panic!("{label}: update_memory must succeed: {err:?}"));

            let row = wrapper
                .get(&seed.id, "upd-meta-proj")
                .expect("get")
                .unwrap_or_else(|| panic!("{label}: row readable"));
            assert_eq!(row.content, "reworded content", "{label}: text updated");
            assert_eq!(
                row.metadata.as_deref(),
                Some(SEEDED),
                "{label}: an absent or null metadata must leave the column unchanged"
            );
        }
    }
}
