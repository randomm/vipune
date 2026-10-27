//! `store_memory` metadata-shape tests for the MCP surface (issue #309).
//!
//! Split out of `tests.rs` to keep that file inside the per-file size cap. The
//! gate keeps the `feature = "mcp"` half for the same reason as `supersede_tests`:
//! the tool handler these tests drive does not exist in the `--no-default-features`
//! lane CI runs. The shared fixtures come from `tests.rs` behind `pub(super)`.
#[cfg(all(test, feature = "mcp"))]
mod mcp_store_metadata {
    use crate::config::Config;
    use crate::mcp::store_wrapper::StoreWrapper;
    use crate::mcp::tests::tool_result_id;
    use crate::mcp::tools::ToolHandler;
    use crate::memory::MemoryStore;
    use crate::sqlite::Database;
    use std::sync::{Arc, Mutex};

    /// Reads the metadata column straight out of SQLite with no Rust type
    /// between the column and the assertion. `MemoryStore::get` and the wrapper's
    /// `get` both hand back an `Option<String>` decoded by `map_row_to_memory`, so
    /// a row whose column holds the 4-character text `null` is a `Some("null")`
    /// there — which the read-side `trim() != "null"` rendering then displays as
    /// JSON `null`, exactly like a real SQL NULL (#305). Only the raw column
    /// distinguishes the two spellings, so that is what this test asserts on.
    ///
    /// Re-opens the file for each read rather than sharing the store's handle:
    /// `Database` owns its `Connection` and the store holds it behind the `Mutex`,
    /// and SQLite's default rollback journal allows one reader alongside the
    /// writer — so the read only has to avoid the write critical section, which
    /// the loop's ordering does (the row is written before this is called).
    fn raw_metadata_column(db_path: &std::path::Path, id: &str) -> Option<String> {
        let db = Database::open(db_path).expect("open test database");
        db.conn()
            .query_row("SELECT metadata FROM memories WHERE id = ?1", [id], |row| {
                row.get::<_, Option<String>>(0)
            })
            .expect("row written")
    }

    /// Issue #309: `store_memory` built the 4-character string `"null"` for an
    /// absent argument and then compared the serialised text back to that literal
    /// to recover `None`. The two spellings of "no metadata" must land on the same
    /// SQL NULL column and no other value may be collapsed into it — so the JSON
    /// *string* `"null"` keeps its quotes. Both `StoreWrapper::get` and the raw
    /// column are asserted: the wrapper alone cannot tell SQL NULL from the text
    /// `null` (see [`raw_metadata_column`]).
    #[tokio::test]
    async fn store_memory_stores_no_metadata_for_omitted_or_null_metadata() {
        use serde_json::json;

        let (dir, db_path) = MemoryStore::test_db_path();
        let store = MemoryStore::from_db_with_test_embedder(
            Database::open(&db_path).expect("open test database"),
        );
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "omitted-meta-proj".to_string(), Config::default());

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
            assert_eq!(
                raw_metadata_column(&db_path, &id).as_deref(),
                expected,
                "{label}: the raw column must be SQL NULL, not the literal text null"
            );
        }
        // The TempDir must outlive the store handle, as everywhere else (#278):
        // the handler holds the last `Arc`, so it drops with it at the end of the
        // scope, before `dir`.
        drop(dir);
    }
}
