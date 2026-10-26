//! `store_memory` importance tests — the level an agent names in a
//! `store_memory` call has to reach the column on both write arms (plain and
//! `supersedes`) and be advertised in the tool schema the client discovers
//! (#256, the MCP twin of the CLI defect #238).
//!
//! Wired from `src/mcp/mod.rs` as its own module so `tests.rs` stays inside the
//! per-file size cap. The shared fixtures (`tool_result_id`
//! and the seed/argument builders) live in `tests.rs` behind `pub(super)` so
//! there is exactly one copy of each.
#[cfg(all(test, feature = "mcp"))]
mod store_memory_importance {
    use crate::config::Config;
    use crate::mcp::store_wrapper::StoreWrapper;
    // Shared fixtures: defined at the top of `tests.rs` and re-exported there
    // for this module, so they are not duplicated here.
    use crate::mcp::tests::{seed_row, store_memory_args, store_memory_args_full, tool_result_id};
    use crate::mcp::tools::ToolHandler;
    use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
    use crate::memory::{MemoryStore, NewMemory};
    use std::sync::{Arc, Mutex};

    /// Issue #256: `store_memory` exposed no `importance`, so every row an agent
    /// stored landed at `medium` and prune could demote a memory the operator had
    /// called critical — the MCP twin of the CLI defect #238.
    ///
    /// Drives the tool handler through `call_tool_for_test` (the router path a
    /// client takes) and reads the column back through `StoreWrapper::get`, which
    /// applies no status or importance filter.
    #[tokio::test]
    async fn store_memory_writes_the_importance_the_agent_named() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "imp-proj".to_string(), Config::default());

        // "medium" is the tool's own default, so the omitted case is the same
        // call — the level must reach the column either way, never be dropped
        // one frame down.
        for (i, level) in ["critical", "high", "low", "medium"].iter().enumerate() {
            let text = format!("importance probe {i}");
            let result = handler
                .call_tool_for_test(
                    "store_memory",
                    store_memory_args_full(&text, None, None, Some(level)),
                )
                .await
                .expect("store_memory with a valid importance must succeed");
            let id = tool_result_id(&result);
            assert_eq!(
                wrapper
                    .get(&id, "imp-proj")
                    .expect("get")
                    .expect("row stored")
                    .importance,
                *level,
                "the level named in the tool call must reach the column, not its DEFAULT"
            );
        }
    }

    /// Omitting the field keeps the default level, and the level an agent names
    /// is case-insensitive on the way in and canonical lowercase on the way out.
    #[tokio::test]
    async fn store_memory_importance_defaults_to_medium_and_stores_lowercase() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "imp-default-proj".to_string(), Config::default());

        let omitted = handler
            .call_tool_for_test(
                "store_memory",
                store_memory_args("a memory with no importance", None, None),
            )
            .await
            .expect("omitting importance must succeed");
        assert_eq!(
            wrapper
                .get(&tool_result_id(&omitted), "imp-default-proj")
                .expect("get")
                .expect("row stored")
                .importance,
            "medium",
            "an omitted importance is the named default, not the column DEFAULT"
        );

        for (input, expected) in [("HIGH", "high"), ("Critical", "critical"), ("Low", "low")] {
            let text = format!("uppercase probe for {input}");
            let result = handler
                .call_tool_for_test(
                    "store_memory",
                    store_memory_args_full(&text, None, None, Some(input)),
                )
                .await
                .unwrap_or_else(|e| panic!("{input} must be accepted: {e:?}"));
            assert_eq!(
                wrapper
                    .get(&tool_result_id(&result), "imp-default-proj")
                    .expect("get")
                    .expect("row stored")
                    .importance,
                expected,
                "{input} must be stored as the canonical spelling"
            );
        }
    }

    /// An invalid level is an input error answered before either write branch:
    /// nothing is inserted, on the plain path or the `supersedes` path.
    #[tokio::test]
    async fn store_memory_rejects_an_invalid_importance_without_writing() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "imp-bad-proj".to_string(), Config::default());

        let old = seed_row(&wrapper, "imp-bad-proj", "original fact");

        for bad in ["urgent", "", "normal"] {
            let text = format!("probe against {bad:?}");
            for supersedes in [None, Some(old.id.as_str())] {
                let err = handler
                    .call_tool_for_test(
                        "store_memory",
                        store_memory_args_full(&text, None, supersedes, Some(bad)),
                    )
                    .await
                    .expect_err("an invalid importance must be rejected");
                assert!(
                    err.message.contains("Invalid importance"),
                    "the tool error must name the offending field ({bad:?}): {err:?}"
                );
            }
        }

        // Nothing written anywhere: only the seeded row remains, and it is
        // untouched — no replacement row, no level change.
        let listed: Vec<serde_json::Value> =
            serde_json::from_value(wrapper.list("imp-bad-proj", 20, None, None).expect("list"))
                .expect("list payload is an array");
        assert_eq!(
            listed.len(),
            1,
            "a rejected call must insert no row on either branch"
        );
        let original = wrapper
            .get(&old.id, "imp-bad-proj")
            .expect("get")
            .expect("original still readable");
        assert_eq!(original.status, "active");
        assert_eq!(original.importance, "medium");
    }

    /// With `supersedes` the level belongs to the NEW row, matching the CLI
    /// (`vipune add --supersedes <id> --importance critical`): the replaced row
    /// keeps its own level and its superseded lifecycle.
    #[tokio::test]
    async fn store_memory_supersedes_applies_the_importance_to_the_new_row() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "imp-sup-proj".to_string(), Config::default());

        let old = seed_row(&wrapper, "imp-sup-proj", "original fact");

        let result = handler
            .call_tool_for_test(
                "store_memory",
                store_memory_args_full("replacement fact", None, Some(&old.id), Some("critical")),
            )
            .await
            .expect("store_memory with supersedes and importance must succeed");
        let new_id = tool_result_id(&result);

        let replacement = wrapper
            .get(&new_id, "imp-sup-proj")
            .expect("get")
            .expect("replacement row stored");
        assert_eq!(
            replacement.importance, "critical",
            "the superseding row must carry the level the agent asked for"
        );
        assert_eq!(replacement.content, "replacement fact");

        let replaced = wrapper
            .get(&old.id, "imp-sup-proj")
            .expect("get")
            .expect("superseded row still readable");
        assert_eq!(replaced.status, "superseded");
        assert_eq!(
            replaced.importance, "medium",
            "the new row's level must not leak into the row it replaced"
        );
        assert_eq!(replaced.superseded_by.as_deref(), Some(new_id.as_str()));
    }

    /// The `importance` parameter has to be visible in the schema a client
    /// discovers, with the vocabulary spelled out — the tool schema is plain JSON
    /// and an agent only sees what is advertised there (acceptance criterion: the
    /// MCP tool schema/description documents the field).
    ///
    /// `#[schemars(extend("enum" = ...))]` on the field, which schemars 1.2.3
    /// merges into the `Option<String>`'s `{"type":["string","null"]}` schema
    /// rather than replacing it. The vocabulary is advertised twice: in the
    /// description (prose) and as a closed `enum` (machine-checkable). The enum
    /// names the canonical lowercase spellings only; the handler's `from_str`
    /// remains the validator and still accepts any casing, so an enum-strict
    /// client that rejects `"HIGH"` is the accepted trade-off (#279 round 1).
    #[test]
    fn store_memory_tool_schema_documents_the_importance_field() {
        let tool = ToolHandler::test_tool_definition("store_memory").expect("store_memory tool");
        let properties = tool
            .input_schema
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("input_schema properties");
        let importance = properties
            .get("importance")
            .expect("input_schema advertises an importance parameter");
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

    /// `store_memory` accepted `supersedes` together with `force`, which the CLI
    /// refuses: the supersede arm never runs the conflict gate, so `force` was a
    /// no-op the client believed had meaning. The handler now rejects the pair
    /// before any write, with the CLI's own wording (#279 round 1).
    #[tokio::test]
    async fn store_memory_rejects_supersedes_together_with_force() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "sup-force-proj".to_string(), Config::default());

        let seed = seed_row(&wrapper, "sup-force-proj", "original fact");

        let mut args = store_memory_args_full(
            "replacement fact",
            None,
            Some(seed.id.as_str()),
            Some("critical"),
        );
        args["force"] = serde_json::json!(true);

        let err = handler
            .call_tool_for_test("store_memory", args)
            .await
            .expect_err("supersedes together with force must be refused");
        assert!(
            err.message.contains("--supersedes") && err.message.contains("--force"),
            "the refusal must use the CLI's wording, got: {}",
            err.message
        );

        // Nothing was written: no replacement row, and the seed row is exactly
        // as it was — still active, still at the default level, unlinked.
        let seed_after = wrapper
            .get(&seed.id, "sup-force-proj")
            .expect("get")
            .expect("seed row still present");
        assert_eq!(seed_after.status, "active");
        assert_eq!(seed_after.importance, "medium");
        assert_eq!(seed_after.superseded_by, None);

        let listed: Vec<serde_json::Value> = serde_json::from_value(
            wrapper
                .list("sup-force-proj", 20, None, None)
                .expect("list"),
        )
        .expect("list payload is an array");
        assert_eq!(
            listed.len(),
            1,
            "the refused call must not have stored a row: {listed:?}"
        );
        assert_eq!(listed[0]["content"], "original fact");
    }

    /// `StoreWrapper::supersede_with_row` takes a row whose `status` is a
    /// `MemoryStatus`, wider than the `active`/`candidate` vocabulary an inserted
    /// row may carry, so the wrapper itself cannot be the thing that rejects the
    /// rest: the store must. Issue #258 was a status that travelled the wrong
    /// frame, and a `superseded` replacement row would be the same class of silent
    /// corruption seen from the MCP surface, where the tool schema is just a
    /// string.
    #[tokio::test]
    async fn supersede_with_row_refuses_a_status_that_is_not_insertable() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());

        let old_id = seed_row(&wrapper, "guard-proj", "original content").id;

        for status in [MemoryStatus::Superseded, MemoryStatus::Deprecated] {
            let err = wrapper
                .supersede_with_row(
                    &NewMemory {
                        project_id: "guard-proj",
                        content: "replacement content",
                        metadata: None,
                        memory_type: MemoryType::Fact,
                        status,
                    },
                    &old_id,
                    MemoryImportance::Medium,
                )
                .expect_err("the widened wrapper must not reach the INSERT");
            assert!(
                err.message
                    .contains("is not valid for new memory insertion"),
                "{status:?} reached the supersede INSERT: {err:?}"
            );
        }

        // Nothing was replaced and nothing was inserted: the original is still
        // the live row, and the only rows in scope are the original.
        let replaced_row = wrapper
            .get(&old_id, "guard-proj")
            .expect("get")
            .expect("original row still readable");
        assert_eq!(replaced_row.status, MemoryStatus::Active.as_str());
        let listed: Vec<serde_json::Value> =
            serde_json::from_value(wrapper.list("guard-proj", 20, None, None).expect("list"))
                .expect("list payload is an array");
        assert_eq!(
            listed.len(),
            1,
            "a refused supersede must insert no replacement row"
        );
    }
}
