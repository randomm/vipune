//! `update_memory` importance tests — the level an agent names has to reach the
//! column, an omitted level must leave the row alone, and an invalid one must be
//! refused before the write (#280).
//!
//! `store_memory` gained its `importance` in #256, which left the MCP level
//! write-once: `prune` never demotes `high`/`critical`, so an over-rated row was
//! prune-exempt forever. These tests pin the demotion path, driven through the
//! real router so the wiring *inside* `ToolHandler::update_memory` is under test.
//!
//! Wired from `src/mcp/mod.rs` as its own module to keep `tests.rs` inside the
//! per-file size cap; the shared fixtures live in `tests.rs` behind `pub(super)`.
#[cfg(all(test, feature = "mcp"))]
mod update_memory_importance {
    use crate::config::Config;
    use crate::mcp::params::UpdateMemoryParams;
    use crate::mcp::store_wrapper::StoreWrapper;
    use crate::mcp::tests::{seed_row, seed_row_at, tool_result_id};
    use crate::mcp::tools::ToolHandler;
    use crate::memory::MemoryStore;
    use crate::memory::lifecycle::MemoryImportance;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    /// The arguments a client sends for an `update_memory` call that changes only
    /// the level — the case the "at least one field" guard used to reject.
    fn update_importance_args(id: &str, importance: Option<&str>) -> serde_json::Value {
        json!({ "id": id, "importance": importance })
    }

    /// A handler and a wrapper over the same store, so a test can drive the tool
    /// and then read the raw row back.
    fn handler_for(project: &str) -> (tempfile::TempDir, ToolHandler, StoreWrapper) {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let handler = ToolHandler::new(store.clone(), project.to_string(), Config::default());
        let wrapper = StoreWrapper::new(store);
        (_dir, handler, wrapper)
    }

    /// The MCP twin of `vipune update --importance`: a named level reaches the
    /// column on a promotion and a demotion alike.
    #[tokio::test]
    async fn update_memory_promotes_and_demotes_the_level() {
        let (_dir, handler, wrapper) = handler_for("upd-proj");
        let seed = seed_row(&wrapper, "upd-proj", "a row to re-rate");

        for level in ["critical", "low", "high", "medium"] {
            let result = handler
                .call_tool_for_test(
                    "update_memory",
                    update_importance_args(&seed.id, Some(level)),
                )
                .await
                .expect("update_memory with a valid importance must succeed");
            assert_eq!(tool_result_id(&result), seed.id);
            // Each round starts from the level the previous one set, so the
            // sequence covers promote (medium->critical, low->high) and demote
            // (critical->low, high->medium) in one pass.
            assert_eq!(
                wrapper
                    .get(&seed.id, "upd-proj")
                    .expect("get")
                    .expect("row readable")
                    .importance,
                level,
                "the level named in the tool call must reach the column"
            );
        }
    }

    /// Omitting `importance` must leave the column exactly as it was, not reset
    /// it to the tool default, while another field still updates.
    #[tokio::test]
    async fn update_memory_without_importance_leaves_the_level_unchanged() {
        let (_dir, handler, wrapper) = handler_for("upd-omit-proj");
        let seed = seed_row_at(
            &wrapper,
            "upd-omit-proj",
            "originally critical",
            MemoryImportance::Critical,
        );

        handler
            .call_tool_for_test(
                "update_memory",
                json!({ "id": seed.id, "text": "reworded content" }),
            )
            .await
            .expect("a text-only update must succeed");

        let row = wrapper
            .get(&seed.id, "upd-omit-proj")
            .expect("get")
            .expect("row readable");
        assert_eq!(row.content, "reworded content");
        assert_eq!(
            row.importance, "critical",
            "an omitted importance must not reset the level to the default"
        );
    }

    /// The guard enumerated text/metadata/memory_type/status only, so a client
    /// asking for nothing but a level change was refused even though the store
    /// below supports exactly that call.
    #[tokio::test]
    async fn update_memory_accepts_an_importance_only_call() {
        let (_dir, handler, wrapper) = handler_for("upd-only-proj");
        let seed = seed_row(&wrapper, "upd-only-proj", "level is all that changes");

        handler
            .call_tool_for_test(
                "update_memory",
                update_importance_args(&seed.id, Some("high")),
            )
            .await
            .expect("an importance-only update must be accepted by the guard");

        let row = wrapper
            .get(&seed.id, "upd-only-proj")
            .expect("get")
            .expect("row readable");
        assert_eq!(row.importance, "high");
        assert_eq!(row.content, "level is all that changes");
    }

    /// An invalid level is rejected as invalid input **before** the write: the
    /// column has no CHECK constraint, so an unvalidated string would sit there
    /// unparsable and make every later recency-scored search fail. Hence the row
    /// is asserted unchanged, not just the call errored.
    #[tokio::test]
    async fn update_memory_rejects_an_invalid_importance_without_writing() {
        let (_dir, handler, wrapper) = handler_for("upd-bad-proj");
        let seed = seed_row_at(
            &wrapper,
            "upd-bad-proj",
            "row that must survive the bad calls",
            MemoryImportance::High,
        );

        for bad in ["urgent", "", "normal", "hgh"] {
            let err = handler
                .call_tool_for_test("update_memory", update_importance_args(&seed.id, Some(bad)))
                .await
                .expect_err("an invalid importance must be rejected");
            assert!(
                err.message.contains("Invalid importance"),
                "the tool error must name the offending field ({bad:?}): {err:?}"
            );
            assert_eq!(
                wrapper
                    .get(&seed.id, "upd-bad-proj")
                    .expect("get")
                    .expect("row readable")
                    .importance,
                "high",
                "a rejected level must leave the column untouched ({bad:?})"
            );
        }
    }

    /// The field has to be visible in the schema a client discovers, spelled the
    /// way `store_memory` advertises it after #279: prose plus a closed `enum` of
    /// the canonical spellings, merged additively into the `Option<String>` type
    /// so an omitted level stays legal.
    #[test]
    fn update_memory_tool_schema_documents_the_importance_field() {
        let tool = ToolHandler::test_tool_definition("update_memory").expect("update_memory tool");
        assert!(
            tool.description
                .as_deref()
                .is_some_and(|d| d.contains("importance")),
            "the tool description must advertise the new field: {:?}",
            tool.description
        );

        let importance = tool
            .input_schema
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("input_schema properties")
            .get("importance")
            .expect("input_schema advertises an importance parameter");

        let types = importance
            .get("type")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("importance keeps its nullable type: {importance}"));
        assert_eq!(
            types,
            &vec![json!("string"), json!("null")],
            "the enum must be additive, not a replacement of the Option<String> type"
        );

        let description = importance
            .get("description")
            .and_then(|v| v.as_str())
            .expect("importance carries a description");
        for level in ["low", "medium", "high", "critical"] {
            assert!(
                description.contains(level),
                "the advertised description must name every level ({level}): {description}"
            );
        }

        let levels = importance
            .get("enum")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("importance advertises a closed enum: {importance}"));
        let advertised: Vec<&str> = levels
            .iter()
            .map(|v| v.as_str().expect("enum entry is a string"))
            .collect();
        assert_eq!(
            advertised,
            vec!["low", "medium", "high", "critical"],
            "the enum must name exactly the four levels, in canonical spelling"
        );
    }

    /// The wire format the tool accepts for the new field. There was no
    /// `UpdateMemoryParams` serde test at all before #280.
    #[test]
    fn update_memory_params_serde_round_trip() {
        let omitted: UpdateMemoryParams =
            serde_json::from_value(json!({ "id": "abc" })).expect("id only");
        assert_eq!(omitted.id, "abc");
        assert!(omitted.importance.is_none());

        let explicit_null: UpdateMemoryParams =
            serde_json::from_value(json!({ "id": "abc", "importance": null }))
                .expect("explicit null importance");
        assert!(explicit_null.importance.is_none());

        let named: UpdateMemoryParams =
            serde_json::from_value(json!({ "id": "abc", "importance": "critical" }))
                .expect("named importance");
        assert_eq!(named.importance.as_deref(), Some("critical"));

        let again: UpdateMemoryParams =
            serde_json::from_value(serde_json::to_value(&named).expect("serialize"))
                .expect("round trip");
        assert_eq!(again.importance.as_deref(), Some("critical"));
    }
}
