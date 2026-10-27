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
    /// The assertion reads the stored column back through `StoreWrapper::get`.
    /// That path is the plain row read (`MemoryStore::get` → `Database::get`,
    /// which selects the 13-column projection and decodes it with
    /// `map_row_to_memory`), so these tests pin the **write** side: what landed
    /// in the column. They would not catch a read-side normalisation added inside
    /// `get`, nor the `trim() != "null"` rendering in `parse_metadata` / the tool
    /// response builders, which this path does not go through. What they do catch
    /// is a regression that stores the text `null` instead of SQL NULL: that shows
    /// up here as `Some("null")` against an expected `None`.
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

    /// Issue #309, the `update_memory` half of the rule. The load-bearing arm is
    /// the last one: a document that *is* sent must replace the seeded column,
    /// which proves the handler passes `metadata` through to the write at all.
    /// The two arms before it are the serde boundary claim, nothing more: after
    /// rmcp deserialises the tool arguments an absent `metadata` and
    /// `"metadata": null` are *identical* — both are `None` on the
    /// `Option<Value>` field before the handler body runs — so between them they
    /// pin one thing only: that `None` means "leave the column unchanged" rather
    /// than "clear it". Their differing seed and new texts are there to prove the
    /// row really was written, not to distinguish the arms.
    ///
    /// The helper's `Value::Null` arm is not what any of this pins (the serde
    /// boundary means the handler never sees it); that arm is covered by
    /// `metadata_column_maps_only_absent_and_json_null_to_none` in `helpers.rs`.
    ///
    /// As in `store_memory_stores_no_metadata_for_omitted_or_null_metadata`, the
    /// read-back goes through the plain `StoreWrapper::get` row read: these are
    /// write-side assertions and would not catch a read-side normalisation.
    #[tokio::test]
    async fn update_memory_leaves_metadata_untouched_when_it_is_omitted_or_null() {
        const SEEDED: &str = r#"{"keep":"me"}"#;

        // Positive arm first — it is the point of this test: a real document
        // replaces the seed, so the handler demonstrably forwards `metadata`.
        // Without it the two arms below could pass on a handler that dropped the
        // argument entirely.
        let (_dir, handler, wrapper) = handler_for("upd-meta-proj");
        let seed = seed_row_with_metadata(&wrapper, "upd-meta-proj", "doc arm", SEEDED);
        handler
            .call_tool_for_test(
                "update_memory",
                json!({ "id": seed.id, "metadata": { "topic": "auth" } }),
            )
            .await
            .unwrap_or_else(|err| panic!("document: update_memory must succeed: {err:?}"));

        let row = wrapper
            .get(&seed.id, "upd-meta-proj")
            .expect("get")
            .unwrap_or_else(|| panic!("document: row readable"));
        assert_eq!(
            row.metadata.as_deref(),
            Some(r#"{"topic":"auth"}"#),
            "document: the given metadata must replace the seeded column"
        );

        // The serde-boundary arms: identical arguments after deserialisation, so
        // together they assert one thing — `None` leaves the column unchanged.
        for (label, new_text) in [
            ("omitted", "reworded (omitted)"),
            ("explicit null", "reworded (explicit null)"),
        ] {
            let (_dir, handler, wrapper) = handler_for("upd-meta-proj");
            let seed = seed_row_with_metadata(
                &wrapper,
                "upd-meta-proj",
                &format!("row to reword ({label})"),
                SEEDED,
            );

            let mut args = json!({ "id": seed.id, "text": new_text });
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
            // The seed text differs per label, so this assertion also proves the
            // update reached the row rather than comparing against itself.
            assert_eq!(row.content, new_text, "{label}: text updated");
            assert_eq!(
                row.metadata.as_deref(),
                Some(SEEDED),
                "{label}: an absent or null metadata must leave the column unchanged"
            );
        }
    }
}
