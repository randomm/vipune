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
use crate::memory::{MemoryStore, NewMemory, UpdateParams};
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
///
/// Nothing is ever written after the poison: `supersede_with_row` is driven with
/// a status the store rejects before it reaches SQLite, so no sibling test can
/// observe a `superseded` row from this fixture.
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

/// `SearchOptions` is taken by value by every search wrapper, so each arm of the
/// table below needs its own. `None` on both filters is what the MCP tools send
/// when the client names no filter.
fn options() -> crate::memory::SearchOptions<'static> {
    crate::memory::SearchOptions {
        memory_types: None,
        statuses: None,
    }
}

/// A row whose `status` is `superseded` — the one value the store's own
/// insert/update rules reject. The write arms of the call table pass it, and
/// each acquires the lock *before* the store reaches that rejection, so those
/// arms exercise the acquisition without ever writing a row a sibling test could
/// observe or returning `Ok`.
fn poison_row() -> NewMemory<'static> {
    NewMemory {
        project_id: "poison-proj",
        content: "never written",
        metadata: None,
        memory_type: MemoryType::Fact,
        status: MemoryStatus::Superseded,
    }
}

/// The 12 acquisitions of the shared store mutex in `StoreWrapper`, by the name
/// `lock_mcp`/`lock_store` stamp into the poison message — which is also the
/// `op` tag each arm's message has to carry. One entry per acquisition, not per
/// method: `ingest_with_lifecycle` and `supersede_with_row` are the two
/// acquisitions their sibling wrappers (`ingest`, `supersede`) reach by
/// delegation, so the roster is complete only with those names in it.
///
/// The test below asserts its call table against this roster by name, so a tag
/// that is not an acquisition — a typo, or a method renamed while its
/// `lock_mcp`/`lock_store` tag was not — fails there rather than matching itself.
const POISONED_ACQUISITIONS: [&str; 12] = [
    "ingest",
    "ingest_with_lifecycle",
    "supersede_with_row",
    "search_raw",
    "search_hybrid_raw",
    "search",
    "search_hybrid",
    "list",
    "get",
    "delete",
    "update",
    "touch_memories",
];

/// The five library-`Error` acquisitions — the two `_raw` searches the tools
/// drive plus the three JSON-returning wrappers no tool calls today, since the
/// tools build their own JSON from the `_raw` results. All in roster order, with
/// every `Ok` payload mapped to a `usize` so one vector holds them. That payload
/// is never read: a poisoned store always answers `Err`, and `take_err` fails the
/// test if it does not.
fn library_error_arms(w: &StoreWrapper) -> Vec<Result<usize, Error>> {
    vec![
        w.search_raw("poison-proj", "seeded", 5, 0.0, options())
            .map(|memories| memories.len()),
        w.search_hybrid_raw("poison-proj", "seeded", 5, 0.0, options())
            .map(|memories| memories.len()),
        w.search("poison-proj", "seeded", 5, 0.0, None, None)
            .map(|value| value.to_string().len()),
        w.search_hybrid("poison-proj", "seeded", 5, 0.0, None, None)
            .map(|value| value.to_string().len()),
        w.list("poison-proj", 10, None, None)
            .map(|value| value.to_string().len()),
    ]
}

/// The five library-`Error` arms tagged with their roster names, in the order
/// `POISONED_ACQUISITIONS` lists them. Built once per test because each call
/// consumes its arguments, and drained with `.remove(0)` by the call table, which
/// cannot move a `Result` out of a `Vec` element.
fn library_error_arms_with_names(w: &StoreWrapper) -> Vec<(&'static str, Result<usize, Error>)> {
    [
        "search_raw",
        "search_hybrid_raw",
        "search",
        "search_hybrid",
        "list",
    ]
    .into_iter()
    .zip(library_error_arms(w))
    .collect()
}

/// The `Err` of one library-`Error` arm, checked against the roster name the call
/// table pairs it with. A poisoned arm always takes the `Err` branch, so an `Ok`
/// — a real answer read off an unpoisoned store — or a pairing that drifted from
/// `POISONED_ACQUISITIONS` is the one thing that can make this panic.
fn take_err(arm: (&'static str, Result<usize, Error>), what: &str) -> Error {
    assert_eq!(arm.0, what, "the call table pairs a tag with the wrong arm");
    arm.1
        .expect_err("a poisoned arm must answer, not return a result")
}

/// Issue #275: 8 of the 12 acquisitions were a bare `.lock().unwrap()`, so the
/// first panic in the long-lived stdio server poisoned the shared mutex and
/// every later request panicked in the handler — a permanent outage instead of
/// one failed request. Every acquisition has to answer, so this drives all 12
/// against one poisoned store: the bare-unwrap paths the tools actually call,
/// the two JSON-returning wrappers no tool reaches today, the previously
/// unguarded `touch_memories`, and the previously-guarded `get`.
#[test]
fn a_poisoned_store_mutex_is_answered_as_an_internal_error_instead_of_panicking() {
    let poisoned = poisoned_store();
    let w = &poisoned.wrapper;
    let row = poison_row();
    // `superseded` is the status `MemoryStore::update` refuses, so the update arm
    // answers without writing (and `update` takes the lock before the check).
    let update = UpdateParams {
        status: Some(MemoryStatus::Superseded),
        ..UpdateParams::default()
    };
    // Built before the table and drained with `.remove(0)` by it: each call
    // consumes its arguments, and a `Result` cannot be moved out of a `Vec`
    // element in place. `take_err` checks each pairing against the roster name.
    let mut library_errors = library_error_arms_with_names(w);

    // Every arm answers; the library-`Error` arms are converted through the
    // same `From` the tool boundary uses, so all 12 are asserted with one helper.
    let answers: [(&str, rmcp::ErrorData); 12] = [
        // `force`, so the store reaches its own status check instead of the
        // conflict scan, which would answer `Ok(Conflicts)` for a non-empty store.
        (
            "ingest",
            w.ingest("poison-proj", "never written", "null", true)
                .expect_err("ingest must answer, not panic"),
        ),
        (
            "ingest_with_lifecycle",
            w.ingest_with_lifecycle(&row, true, MemoryImportance::High)
                .expect_err("ingest_with_lifecycle must answer, not panic"),
        ),
        (
            "supersede_with_row",
            w.supersede_with_row(&row, "whatever", MemoryImportance::High)
                .expect_err("supersede_with_row must answer, not panic"),
        ),
        (
            "search_raw",
            rmcp::ErrorData::from(take_err(library_errors.remove(0), "search_raw")),
        ),
        (
            "search_hybrid_raw",
            rmcp::ErrorData::from(take_err(library_errors.remove(0), "search_hybrid_raw")),
        ),
        (
            "search",
            rmcp::ErrorData::from(take_err(library_errors.remove(0), "search")),
        ),
        (
            "search_hybrid",
            rmcp::ErrorData::from(take_err(library_errors.remove(0), "search_hybrid")),
        ),
        (
            "list",
            rmcp::ErrorData::from(take_err(library_errors.remove(0), "list")),
        ),
        (
            "get",
            w.get("whatever", "poison-proj")
                .expect_err("get must answer, not panic"),
        ),
        (
            "delete",
            w.delete("whatever", "poison-proj")
                .expect_err("delete must answer, not panic"),
        ),
        (
            "update",
            w.update("whatever", "poison-proj", update)
                .expect_err("update must answer, not panic"),
        ),
        (
            "touch_memories",
            w.touch_memories(&["whatever"])
                .expect_err("touch_memories must answer, not panic"),
        ),
    ];

    let mut covered: Vec<&str> = Vec::new();
    for (what, err) in &answers {
        assert_poison_answer(err, what);
        assert!(
            err.message.contains(&format!("({what})")),
            "{what} must name its own acquisition in the message, got {:?}",
            err.message
        );
        covered.push(*what);
    }
    // Zip against the roster rather than compare two literals: the tags are the
    // acquisition names, so a table that drops, duplicates or renames an arm
    // fails here instead of agreeing with itself.
    assert_eq!(
        covered, POISONED_ACQUISITIONS,
        "the roster and the call table must be the same 12 acquisitions in order"
    );
    for (what, _) in &library_errors {
        assert!(
            POISONED_ACQUISITIONS.contains(what),
            "the library-Error arms must be acquisition names: {what}"
        );
    }

    assert!(
        poisoned.store.is_poisoned(),
        "answering must not clear the poison — the guard is never taken back out"
    );
}

/// `lock_store` answers with `Error::Database`, never the retryable
/// `Error::DatabaseLocked` a client would retry forever — asserted for every
/// arm whose return type is the library `Error`, i.e. the five the tools reach
/// through `tools.rs`. `lock_mcp`'s arms cannot get it wrong: they build the
/// `ErrorData` in `store_lock.rs` and never construct an `Error`.
#[test]
fn a_poisoned_lock_store_answer_is_never_the_retryable_locked_verdict() {
    let poisoned = poisoned_store();
    for (what, arm) in library_error_arms_with_names(&poisoned.wrapper) {
        let err = arm.expect_err("the library-Error arms must answer, not panic");
        assert!(
            !err.is_locked(),
            "{what}: a poison must not be reported as the retryable locked verdict: {err}"
        );
    }
}

/// The same answer has to survive the `tools.rs` boundary, which is where an
/// MCP client reads it: `list_memories`, `search_memories` and `get_memory` map
/// the library `Error` or wrap the store call themselves, so a poison must still
/// arrive framed as `data.type == "internal_error"`.
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

    // config.hybrid defaults to false, so this arm takes `search_raw`;
    // `search_hybrid_raw` has the same acquisition shape and is covered at the
    // wrapper level below.
    let err = handler
        .call_tool_for_test("search_memories", serde_json::json!({"query": "seeded"}))
        .await
        .expect_err("search_memories must answer, not panic");
    assert_poison_answer(&err, "search_memories");

    // `get_memory` is the tool that drives `get`'s wire-shape half of the fix:
    // that acquisition was already guarded, but with `data: None`.
    let err = handler
        .call_tool_for_test("get_memory", serde_json::json!({"id": "whatever"}))
        .await
        .expect_err("get_memory must answer, not panic");
    assert_poison_answer(&err, "get_memory");
}

/// `search_hybrid_raw` is the other arm of the `search_memories` tool — the one
/// `hybrid: true` selects (the `hybrid` config field or the tool's own argument).
/// Without this case a regression there would be invisible to the tool-level
/// test above, which only ever takes `search_raw`.
#[tokio::test]
async fn a_poisoned_store_is_answered_by_the_hybrid_search_tool() {
    let poisoned = poisoned_store();
    let config = Config {
        hybrid: true,
        ..Config::default()
    };
    let handler = ToolHandler::new(poisoned.store.clone(), "poison-proj".to_string(), config);

    let err = handler
        .call_tool_for_test("search_memories", serde_json::json!({"query": "seeded"}))
        .await
        .expect_err("the hybrid arm must answer, not panic");
    assert_poison_answer(&err, "search_memories (hybrid)");
    assert!(
        err.message.contains("(search_hybrid_raw)"),
        "the hybrid arm must name its own acquisition, got {:?}",
        err.message
    );
}

/// The `touch_memories` acquisition is the one no tool call reports: both
/// callers in `tools.rs` log a failure and continue serving. It is therefore the
/// arm the other two tests could never notice regressing — and the arm whose
/// regression is worst, because `search_memories` calls it on every non-`no_touch`
/// request. Asserted directly on the wrapper, which is the acquisition's frame.
#[test]
fn a_poisoned_store_is_answered_by_the_telemetry_acquisition_its_callers_swallow() {
    let poisoned = poisoned_store();
    let err = poisoned
        .wrapper
        .touch_memories(&["whatever"])
        .expect_err("telemetry must answer, not panic");
    assert_poison_answer(&err, "touch_memories");
    assert!(
        err.message.contains("(touch_memories)"),
        "the op tag is the only clue an operator gets in a swallowed-telemetry log, got {:?}",
        err.message
    );
}
