//! Supersede-path tests for the MCP surface — the `store_memory` tool an agent
//! calls, which is the frame that decides which supersede wrapper a `supersedes`
//! call reaches.
//!
//! Split out of `tests.rs` for issue #271 so both files stay inside the
//! per-file size cap. The gate must keep the `feature = "mcp"` half: the tool
//! handler these tests drive does not exist in the `--no-default-features` lane
//! CI runs. The shared fixtures (`seed_row`, `store_memory_args`,
//! `tool_result_id`) come from `tests.rs` behind `pub(super)`.
#[cfg(all(test, feature = "mcp"))]
mod mcp_supersede {
    use crate::config::Config;
    use crate::mcp::store_wrapper::StoreWrapper;
    use crate::mcp::tests::{seed_row, store_memory_args, tool_result_id};
    use crate::mcp::tools::ToolHandler;
    use crate::memory::lifecycle::MemoryImportance;
    use crate::memory::{MemoryStore, NewMemory};
    use std::sync::{Arc, Mutex};

    /// Issue #258: `store_memory`'s `supersedes` arm parsed and validated
    /// `status` and then called the status-less `StoreWrapper::supersede`, so an
    /// agent passing `status: "candidate"` together with `supersedes` got an
    /// `active` replacement row with no warning — the MCP twin of the CLI drop.
    ///
    /// Drives the `store_memory` TOOL HANDLER itself (the frame where the status
    /// was dropped, and the frame that decides which wrapper the supersedes arm
    /// reaches), not the wrapper below it, and reads the stored column back via
    /// `StoreWrapper::get`, the read path that applies no status filter.
    #[tokio::test]
    async fn store_memory_supersedes_honours_the_requested_status() {
        use crate::memory::lifecycle::{MemoryStatus, MemoryType};

        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "status-proj".to_string(), Config::default());

        let seeded = seed_row(&wrapper, "status-proj", "original content");
        let old_id = seeded.id.clone();

        // The MCP equivalent of `add --supersedes <id> --status candidate`,
        // called the way an agent calls it: through the tool router, so the
        // wiring inside `store_memory` is what is under test.
        let result = handler
            .call_tool_for_test(
                "store_memory",
                store_memory_args("replacement content", Some("candidate"), Some(&old_id)),
            )
            .await
            .expect("store_memory with supersedes must succeed");
        let new_id = tool_result_id(&result);

        let replacement = wrapper
            .get(&new_id, "status-proj")
            .expect("get")
            .expect("replacement row stored");
        assert_eq!(
            replacement.status,
            MemoryStatus::Candidate.as_str(),
            "the tool handler must hand its validated status to the store, not the wrapper's default"
        );
        assert_eq!(replacement.content, "replacement content");

        // The old row keeps the lifecycle the supersede transaction owns.
        let replaced_row = wrapper
            .get(&old_id, "status-proj")
            .expect("get")
            .expect("superseded row still readable");
        assert_eq!(replaced_row.status, "superseded");
        assert_eq!(replaced_row.superseded_by.as_deref(), Some(new_id.as_str()));

        // Identical outcome to the CLI: a candidate replacement is invisible to
        // the active-only default list on this surface too. A second, unrelated
        // active row is seeded as the control — the superseded original is not
        // listed either — so the guard cannot be satisfied by a list call that
        // returned nothing.
        let control = wrapper
            .ingest_with_lifecycle(
                &NewMemory {
                    project_id: "status-proj",
                    content: "unrelated active control",
                    metadata: None,
                    memory_type: MemoryType::Fact,
                    status: MemoryStatus::Active,
                },
                true,
                MemoryImportance::Medium,
            )
            .expect("seed control");
        let control_id = control["id"].as_str().expect("control id").to_string();

        let listed: Vec<serde_json::Value> =
            serde_json::from_value(wrapper.list("status-proj", 20, None, None).expect("list"))
                .expect("list payload is an array");
        let listed_ids: Vec<&str> = listed
            .iter()
            .map(|m| m["id"].as_str().expect("listed id"))
            .collect();
        assert!(
            listed_ids.contains(&control_id.as_str()),
            "the default list itself works, so its silence about the candidate is meaningful: {listed_ids:?}"
        );
        assert!(
            !listed_ids.contains(&new_id.as_str()),
            "a candidate replacement must stay out of the default list: {listed_ids:?}"
        );

        // And the status-less entry point (`supersede_memory`, which has no
        // `status` parameter at all) still writes `active` by name — the same
        // column value the `status: "active"` case above pins from the tool.
        let plain = wrapper
            .supersede(
                "status-proj",
                "plain replacement",
                "null",
                MemoryType::Fact,
                &new_id,
            )
            .expect("level-less supersede");
        let plain_id = plain["id"].as_str().expect("plain id").to_string();
        assert_eq!(
            wrapper
                .get(&plain_id, "status-proj")
                .expect("get")
                .expect("plain row stored")
                .status,
            MemoryStatus::Active.as_str(),
            "the status-less wrapper keeps writing active explicitly"
        );
    }

    /// The `status` an agent names on a `supersedes` call must reach the new
    /// row for **both** insertable spellings. `store_memory_supersedes_honours_the
    /// _requested_status` pins `candidate`; without an `active` case a handler
    /// that hardcoded `active` — the #258 defect — would still pass, since
    /// `active` is also the wrapper's own default. Driven through the router, so
    /// the frame under test is the one that chooses the wrapper.
    #[tokio::test]
    async fn store_memory_supersedes_honours_an_explicit_active_status() {
        use crate::memory::lifecycle::{MemoryStatus, MemoryType};

        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "active-proj".to_string(), Config::default());

        let old = seed_row(&wrapper, "active-proj", "original content");
        let result = handler
            .call_tool_for_test(
                "store_memory",
                store_memory_args("replacement content", Some("active"), Some(&old.id)),
            )
            .await
            .expect("store_memory with supersedes must succeed");
        let new_id = tool_result_id(&result);

        let replacement = wrapper
            .get(&new_id, "active-proj")
            .expect("get")
            .expect("replacement row stored");
        assert_eq!(replacement.status, MemoryStatus::Active.as_str());
        assert_eq!(replacement.content, "replacement content");
        assert_eq!(replacement.memory_type, MemoryType::Fact.as_str());

        let replaced_row = wrapper
            .get(&old.id, "active-proj")
            .expect("get")
            .expect("superseded row still readable");
        assert_eq!(replaced_row.status, "superseded");
        assert_eq!(replaced_row.superseded_by.as_deref(), Some(new_id.as_str()));
    }

    /// Issue #271: the re-supersede guard has to answer an agent in the tool's
    /// own error shape, not as a bare database message. Superseding a row the
    /// supersede transaction already retired would fork the chain, so the tool
    /// refuses and writes nothing.
    #[tokio::test]
    async fn store_memory_refuses_to_supersede_a_superseded_row() {
        let (_dir, store) = MemoryStore::test_store();
        let store = Arc::new(Mutex::new(store));
        let wrapper = StoreWrapper::new(store.clone());
        let handler = ToolHandler::new(store, "chain-proj".to_string(), Config::default());

        let old = seed_row(&wrapper, "chain-proj", "original content");
        let first = handler
            .call_tool_for_test(
                "store_memory",
                store_memory_args("first replacement", None, Some(&old.id)),
            )
            .await
            .expect("first supersede must succeed");
        let first_id = tool_result_id(&first);

        let err = handler
            .call_tool_for_test(
                "store_memory",
                store_memory_args("second replacement", None, Some(&old.id)),
            )
            .await
            .expect_err("re-superseding a superseded row must be refused");
        assert!(
            err.message.contains("superseded") && err.message.contains(&first_id),
            "the refusal must name the current status and its superseded_by: {err:?}"
        );

        // Nothing written: the chain still points at the first replacement and
        // the project holds exactly those two rows.
        let replaced = wrapper
            .get(&old.id, "chain-proj")
            .expect("get")
            .expect("superseded row still readable");
        assert_eq!(replaced.superseded_by.as_deref(), Some(first_id.as_str()));
        let listed: Vec<serde_json::Value> =
            serde_json::from_value(wrapper.list("chain-proj", 20, None, None).expect("list"))
                .expect("list payload is an array");
        assert_eq!(
            listed.len(),
            1,
            "only the live replacement is listed: {listed:?}"
        );
        let all: Vec<serde_json::Value> = serde_json::from_value(
            wrapper
                .list(
                    "chain-proj",
                    20,
                    None,
                    Some(["active", "candidate", "superseded", "deprecated"].as_slice()),
                )
                .expect("list"),
        )
        .expect("list payload is an array");
        assert_eq!(all.len(), 2, "the refused call inserted nothing: {all:?}");
    }
}
