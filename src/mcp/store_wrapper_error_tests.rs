//! Tests for the `From<Error> for rmcp::ErrorData` conversion in
//! `store_wrapper` — the machine-readable `data.type` discriminators an MCP
//! client branches on.
//!
//! Wired from `store_wrapper.rs` via `#[path = "store_wrapper_error_tests.rs"]`
//! to keep that file under the project 500-line cap.

use crate::config::Config;
use crate::mcp::store_wrapper::StoreWrapper;
use crate::mcp::tools::ToolHandler;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::{MemoryStore, NewMemory};
use crate::memory_types::IngestPolicy;
use std::sync::{Arc, Mutex};

use super::*;

/// The `data.type` field is the discriminator an MCP client must read: rmcp's
/// `ErrorCode` set has no "retry later", so a retryable lock and a server
/// defect share `INTERNAL_ERROR`. If the fields stop being emitted the client
/// can only string-match the message — which is what issue #255 set out to
/// remove.
#[test]
fn database_errors_carry_a_machine_readable_type_discriminator() {
    let cases: Vec<(Error, &str)> = vec![
        (Error::DatabaseLocked, "database_locked"),
        (
            Error::CorruptEmbedding {
                id: "mem-1".to_string(),
                reason: "not enough bytes".to_string(),
            },
            "corrupt_embedding",
        ),
        (
            Error::InvalidBlobSize {
                expected: crate::embedding::EMBEDDING_DIMS * 4,
                actual: 4,
            },
            "invalid_blob_size",
        ),
        (
            Error::MismatchedDimensions {
                expected: 384,
                actual: 768,
            },
            "mismatched_dimensions",
        ),
    ];
    for (err, expected_type) in cases {
        let data = rmcp::ErrorData::from(err)
            .data
            .unwrap_or_else(|| panic!("{expected_type} must carry a data payload"));
        assert_eq!(
            data["type"].as_str(),
            Some(expected_type),
            "expected the {expected_type} discriminator"
        );
    }
}

#[test]
fn the_corrupt_embedding_discriminator_carries_the_row_id() {
    // The #186 payload a client needs to locate the corrupt row must survive
    // the MCP conversion, not just the Display text.
    let err = Error::CorruptEmbedding {
        id: "mem-42".to_string(),
        reason: "invalid BLOB size".to_string(),
    };
    let data = rmcp::ErrorData::from(err).data.expect("data payload");
    assert_eq!(data["id"].as_str(), Some("mem-42"));
    assert_eq!(data["expected"], serde_json::Value::Null);
}

/// Poison `store` by panicking in a thread while holding its guard, and block
/// until the poison is observable.
///
/// The join comes before any assertion: without it a case could run while the
/// panic has not happened yet (mutex still unpoisoned) or after the guard
/// dropped on unwind, which makes the test flaky rather than red.
///
/// The `.expect()` here is the one acquisition under `src/mcp/` allowed to
/// panic: it exists to *cause* the poison, never to serve a request. A plain
/// `.lock()?` would leave the mutex unpoisoned and every case below green for
/// the wrong reason.
fn poison_store(store: &Arc<Mutex<MemoryStore>>) {
    let poisoner = {
        let store = store.clone();
        std::thread::spawn(move || {
            let _guard = store.lock().expect("fresh mutex must lock");
            panic!("deliberate panic to poison the store mutex");
        })
    };
    let joined = poisoner.join();
    assert!(
        joined.is_err(),
        "the poisoner must have panicked, otherwise the mutex is not poisoned: {joined:?}"
    );
    assert!(store.is_poisoned(), "the fixture must leave it poisoned");
}

/// A wrapper over a store holding one seeded row, whose mutex is poisoned. All
/// three fields are bound: the `TempDir` must outlive the SQLite handle, and
/// `store` is the `Arc` the wrapper shares — a fresh `Arc` per test, since
/// poison is process-sticky and a sibling test must never observe it.
struct Poisoned {
    _dir: tempfile::TempDir,
    store: Arc<Mutex<MemoryStore>>,
    wrapper: StoreWrapper,
}

fn poisoned_store() -> Poisoned {
    let (dir, mut store) = MemoryStore::test_store();
    let row = NewMemory {
        project_id: "poison-proj",
        content: "seeded before the poison",
        metadata: None,
        memory_type: MemoryType::Fact,
        status: MemoryStatus::Active,
    };
    store
        .ingest_with_importance(&row, IngestPolicy::Force, MemoryImportance::Medium)
        .expect("seed before poisoning");
    let store = Arc::new(Mutex::new(store));
    let wrapper = StoreWrapper::new(store.clone());
    poison_store(&store);
    Poisoned {
        _dir: dir,
        store,
        wrapper,
    }
}

/// The contract a poisoned store answers with, whichever helper acquired the
/// lock: `INTERNAL_ERROR`, the `internal_error` discriminator every server-side
/// failure carries, and the "store lock poisoned" words the helpers share.
///
/// Matched by `contains` rather than exactly: what a caller sees is this
/// crate's text around std's `PoisonError` Display, and for the `lock_store`
/// paths also around `Error::Database`'s `Database error:` frame — none of
/// which is ours to pin beyond the poison wording itself.
fn assert_poison_answer(err: &rmcp::ErrorData, what: &str) {
    assert_eq!(
        err.code,
        rmcp::model::ErrorCode::INTERNAL_ERROR,
        "{what} must answer INTERNAL_ERROR"
    );
    // The wrapper methods that answer in the library `Error` arrive through
    // `From<Error> for rmcp::ErrorData`, which prepends `Error::Database`'s
    // frame, so the poison wording is matched rather than the whole message.
    assert!(
        err.message.contains("store lock poisoned"),
        "{what} must say so, got {:?}",
        err.message
    );
    let data = err
        .data
        .clone()
        .unwrap_or_else(|| panic!("{what} must carry a data payload like every internal error"));
    assert_eq!(
        data["type"].as_str(),
        Some("internal_error"),
        "{what} must carry the internal_error discriminator"
    );
}

/// Issue #275: 8 of the 12 acquisitions were a bare `.lock().unwrap()`, so the
/// first panic in the long-lived stdio server poisoned the shared mutex and
/// every later request panicked in the handler — a permanent outage instead of
/// one failed request. The bare-unwrap paths the tools actually call, plus the
/// previously-guarded `get`, must all answer instead.
#[test]
fn a_poisoned_store_mutex_is_answered_as_an_internal_error_instead_of_panicking() {
    let poisoned = poisoned_store();

    // search_raw / list return the library Error, so their answer is the
    // internal-style variant of the #255 surface rather than an ErrorData.
    let err = poisoned
        .wrapper
        .list("poison-proj", 10, None, None)
        .expect_err("list must answer, not panic");
    assert!(
        !err.is_locked(),
        "a poison must not be reported as the retryable locked verdict: {err}"
    );
    let err = rmcp::ErrorData::from(err);
    assert_poison_answer(&err, "list");

    let err = poisoned
        .wrapper
        .search_raw(
            "poison-proj",
            "seeded",
            5,
            0.0,
            crate::memory::SearchOptions {
                memory_types: None,
                statuses: None,
            },
        )
        .expect_err("search_raw must answer, not panic");
    assert!(!err.is_locked(), "a poison is not a retryable lock: {err}");
    assert_poison_answer(&rmcp::ErrorData::from(err), "search_raw");

    // get was already guarded, but with `data: None` — this is the wire-shape
    // half of the fix: uniform with every other internal error.
    assert_poison_answer(
        &poisoned
            .wrapper
            .get("whatever", "poison-proj")
            .expect_err("get must answer with the uniform error"),
        "get",
    );

    assert!(
        poisoned.store.is_poisoned(),
        "answering must not clear the poison — the guard is never taken back out"
    );
}

/// The same answer has to survive the `tools.rs` boundary, which is where an
/// MCP client reads it: `list_memories` and `search_memories` map the library
/// `Error` themselves, so a poison must still arrive framed as
/// `data.type == "internal_error"`.
#[tokio::test]
async fn a_poisoned_store_is_answered_at_the_tool_boundary() {
    let poisoned = poisoned_store();
    let handler = ToolHandler::new(
        poisoned.store.clone(),
        "poison-proj".to_string(),
        Config::default(),
    );

    let err = handler
        .call_tool_for_test("list_memories", serde_json::json!({"limit": 10}))
        .await
        .expect_err("list_memories must answer, not panic");
    assert_poison_answer(&err, "list_memories");

    let err = handler
        .call_tool_for_test("search_memories", serde_json::json!({"query": "seeded"}))
        .await
        .expect_err("search_memories must answer, not panic");
    assert_poison_answer(&err, "search_memories");
}
