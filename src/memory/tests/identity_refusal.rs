//! Mismatch / in-flight-migration refusal at the MemoryStore chokepoint
//! (issue #217, decision 3: add, update, search — and by extension
//! batch_ingest / supersede, which all route through `get_embedding` —
//! refuse with an error naming recorded vs configured identity, or the
//! interrupted migration target, and pointing at `vipune reindex --force`).
//!
//! Also covers the prefix-application contract (M1): the recording embedder
//! receives the profile-prefixed text exactly as the real engine would (e5:
//! `passage: <content>` / `query: <text>`; bge: unprefixed), while the
//! stored content and FTS stay unprefixed.

use crate::config::Config;
use crate::embedding_profiles::EmbeddingRole;
use crate::memory::{MemoryStore, SearchOptions};
use crate::sqlite::Database;

/// Write a migration marker (marker-first step: touches ONLY the marker
/// column; the recorded identity is left untouched).
fn write_marker(conn: &rusqlite::Connection, target: &crate::sqlite::identity::ModelIdentity) {
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, NULL, NULL, ?1)
         ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
        [format!("migrating to {}", target.display())],
    )
    .unwrap();
}

/// Record a model identity (marker cleared; the only step that replaces the
/// recorded identity).
fn record_identity(conn: &rusqlite::Connection, identity: &crate::sqlite::identity::ModelIdentity) {
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)
         ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision,
                                      migration_marker = NULL",
        (&identity.model_id, &identity.revision),
    )
    .unwrap();
}

/// Open a fresh temp-path store configured for `model_id`, with the
/// identity/marker state applied via `prepare`. The returned db path lets the
/// test inspect the recorded state afterwards.
fn prepared_store(
    model_id: &str,
    prepare: impl FnOnce(&Database),
) -> (tempfile::TempDir, MemoryStore) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let db = Database::open(&path).unwrap();
    prepare(&db);
    let store = MemoryStore {
        db,
        embedder: None,
        model_id: model_id.to_string(),
        config: Config::default(),
        #[cfg(test)]
        test_embedder: Some(Box::new(crate::memory::crud::test_fake_embedder)),
    };
    (dir, store)
}

fn e5_model_id() -> &'static str {
    "intfloat/multilingual-e5-small"
}

#[test]
fn add_refuses_unrecorded_store_with_nondefault_model() {
    // A not-yet-recorded (bge-default) store with e5 configured is a
    // mismatch: effective identity bge ≠ configured e5.
    let (_dir, mut store) = prepared_store(e5_model_id(), |_| {});
    let result = store.add_with_conflict(
        "p",
        "hello",
        None,
        false,
        crate::memory::lifecycle::MemoryType::Fact,
        crate::memory::lifecycle::MemoryStatus::Active,
    );
    let err = result.expect_err("add must refuse");
    let msg = err.to_string();
    assert!(msg.contains("model identity mismatch"), "{msg}");
    assert!(msg.contains(e5_model_id()), "{msg}");
    assert!(msg.contains("vipune reindex --force"), "{msg}");
}

#[test]
fn add_refuses_while_migration_marker_present() {
    let (_dir, mut store) = prepared_store(e5_model_id(), |db| {
        write_marker(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity {
                model_id: e5_model_id().to_string(),
                revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
            },
        );
    });
    let result = store.add_with_conflict(
        "p",
        "hello",
        None,
        false,
        crate::memory::lifecycle::MemoryType::Fact,
        crate::memory::lifecycle::MemoryStatus::Active,
    );
    let err = result.expect_err("add must refuse while migrating");
    let msg = err.to_string();
    assert!(msg.contains("migrating"), "{msg}");
    assert!(msg.contains("vipune reindex --force"), "{msg}");
}

#[test]
fn search_refuses_on_mismatch() {
    let (_dir, mut store) = prepared_store(e5_model_id(), |_| {});
    let result = store.search("p", "query", 5, 0.0, SearchOptions::default());
    let err = result.expect_err("search must refuse");
    let msg = err.to_string();
    assert!(msg.contains("model identity mismatch"), "{msg}");
    assert!(msg.contains("vipune reindex --force"), "{msg}");
}

#[test]
fn search_refuses_while_migration_marker_present() {
    let (_dir, mut store) = prepared_store(e5_model_id(), |db| {
        write_marker(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity {
                model_id: e5_model_id().to_string(),
                revision: "rev".to_string(),
            },
        );
    });
    let result = store.search("p", "query", 5, 0.0, SearchOptions::default());
    let err = result.expect_err("search must refuse while migrating");
    assert!(err.to_string().contains("migrating"));
}

#[test]
fn update_refuses_on_mismatch_when_content_changes() {
    // Update with content must embed (passage role) → refuses on mismatch.
    let (_dir, mut store) = prepared_store(e5_model_id(), |_| {});
    let db_path = crate::memory::store::MemoryStore::test_db_path();
    let db = Database::open(&db_path).unwrap();
    let emb = crate::memory::crud::test_fake_embedder("orig").unwrap();
    let id = db
        .insert("p", "orig", &emb, None, "fact", "active")
        .unwrap();
    // Seed a row into this store's own db so the update targets a real row.
    let emb2 = crate::memory::crud::test_fake_embedder("orig2").unwrap();
    store
        .db
        .insert("p", "orig2", &emb2, None, "fact", "active")
        .unwrap();
    let result = store.update(
        &id,
        "p",
        crate::memory::UpdateParams {
            text: Some("new content"),
            ..Default::default()
        },
    );
    let err = result.expect_err("update with content must refuse");
    assert!(err.to_string().contains("model identity mismatch"));
    let _ = db;
}

#[test]
fn batch_ingest_refuses_each_item_on_mismatch() {
    let (_dir, mut store) = prepared_store(e5_model_id(), |_| {});
    let items = vec![("one", None), ("two", None)];
    let result = store.batch_ingest("p", items, crate::memory_types::IngestPolicy::Force);
    // batch_ingest maps per-item failures to Error results.
    let batch = result.expect("batch_ingest returns per-item results");
    assert_eq!(batch.results.len(), 2);
    for item in &batch.results {
        let msg = match item {
            crate::memory_types::BatchIngestItemResult::Error { message } => message.clone(),
            other => panic!("expected per-item Error, got {other:?}"),
        };
        assert!(msg.contains("model identity mismatch"), "{msg}");
    }
}

#[test]
fn add_succeeds_when_identity_matches() {
    // Recorded identity == configured → no refusal (zero-change contract for
    // a healthy store).
    let (_dir, mut store) = prepared_store(crate::embedding::EMBED_MODEL_ID, |db| {
        record_identity(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity::default_identity(),
        );
    });
    let result = store
        .add_with_conflict(
            "p",
            "hello world",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed when identity matches");
    assert!(
        matches!(result, crate::memory_types::AddResult::Added { .. }),
        "expected Added, got {result:?}"
    );
}

#[test]
fn fresh_unrecorded_store_with_default_model_adds() {
    // No recorded row + default (bge) configured → Ok (zero-change contract).
    let (_dir, mut store) = prepared_store(crate::embedding::EMBED_MODEL_ID, |_| {});
    let result = store
        .add_with_conflict(
            "p",
            "fresh store content",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed on a fresh store with the default model");
    assert!(matches!(
        result,
        crate::memory_types::AddResult::Added { .. }
    ));
}

#[test]
fn refusal_does_not_require_the_real_embedder() {
    // The refusal happens before the (test) embedder is consulted: a mismatch
    // store configured for e5 refuses even though the test embedder would
    // happily embed.
    let (_dir, mut store) = prepared_store(e5_model_id(), |_| {});
    let _ = store.get_embedding("hello", EmbeddingRole::Passage);
    let err = store
        .get_embedding("hello", EmbeddingRole::Passage)
        .expect_err("must refuse before embedding");
    assert!(err.to_string().contains("model identity mismatch"));
}

// ---- Prefix application (issue #217 M1: stored content/FTS unprefixed) ----
//
// The engine input is `prefix + text` exactly — the profile's passage/query
// prefix applied before the embedder is consulted, and never after. A
// recording test embedder proves the exact engine input for both profiles:
// e5 receives `passage: <content>` / `query: <text>`; bge receives the text
// unprefixed. The stored content and FTS stay unprefixed in either case.

/// A test embedder that records the exact text it is called with (and returns
/// a deterministic L2-normalised vector derived from that text).
fn recording_embedder(
    recorded: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
) -> crate::memory::store::TestEmbedder {
    Box::new(move |content: &str| {
        recorded.lock().unwrap().push(content.to_string());
        crate::memory::crud::test_fake_embedder(content)
    })
}

fn e5_store_with_real_prefix_contract() -> (tempfile::TempDir, MemoryStore) {
    let (dir, store) = prepared_store(e5_model_id(), |_| {});
    // Record the e5 identity so the identity check passes and add/search
    // actually run through the (test) embedder path.
    record_identity(
        store.db.conn(),
        &crate::sqlite::identity::ModelIdentity {
            model_id: e5_model_id().to_string(),
            revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
        },
    );
    (dir, store)
}

#[test]
fn add_stores_content_without_passage_prefix() {
    let (_dir, mut store) = e5_store_with_real_prefix_contract();
    let result = store
        .add_with_conflict(
            "p",
            "the quick brown fox",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed when identity matches");
    let crate::memory_types::AddResult::Added { id } = result else {
        panic!("expected Added");
    };
    let memory = store.get(&id, "p").unwrap().expect("row exists");
    // The stored content must be exactly what was passed in — no passage:
    // prefix may leak into the DB (prefixes live only at embed time).
    assert_eq!(memory.content, "the quick brown fox");
}

#[test]
fn e5_engine_input_is_exactly_passage_prefixed() {
    // Under the e5 profile, the engine input for add / update / batch / search
    // is exactly `passage: <content>` / `query: <text>` — the prefix is
    // applied before the (test) embedder is consulted, so the recording
    // embedder sees it.
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (_dir, mut store) = e5_store_with_real_prefix_contract();
    store.set_test_embedder(recording_embedder(recorded.clone()));

    // add
    let result = store
        .add_with_conflict(
            "p",
            "fox content",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed when identity matches");
    let crate::memory_types::AddResult::Added { id } = result else {
        panic!("expected Added");
    };
    assert_eq!(
        recorded.lock().unwrap().clone(),
        vec!["passage: fox content"]
    );

    // update (content change re-embeds)
    store
        .update(
            &id,
            "p",
            crate::memory::UpdateParams {
                text: Some("new fox content"),
                ..Default::default()
            },
        )
        .expect("update must succeed when identity matches");
    assert_eq!(
        recorded.lock().unwrap().clone(),
        vec!["passage: fox content", "passage: new fox content"]
    );

    // search
    store
        .search("p", "fox", 5, 0.0, SearchOptions::default())
        .expect("search must succeed when identity matches");
    assert_eq!(
        recorded.lock().unwrap().clone(),
        vec![
            "passage: fox content",
            "passage: new fox content",
            "query: fox"
        ]
    );

    // batch ingest
    let batch = store
        .batch_ingest(
            "p",
            vec![("batch item one", None), ("batch item two", None)],
            crate::memory_types::IngestPolicy::Force,
        )
        .expect("batch_ingest must succeed when identity matches");
    assert_eq!(batch.results.len(), 2);
    let recorded_after_batch = recorded.lock().unwrap().clone();
    assert_eq!(recorded_after_batch.len(), 5);
    assert_eq!(recorded_after_batch[3], "passage: batch item one");
    assert_eq!(recorded_after_batch[4], "passage: batch item two");
}

#[test]
fn bge_engine_input_is_unprefixed() {
    // Under the bge profile (no prefixes), the engine input is exactly the
    // raw content / query — zero-change contract.
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (_dir, mut store) = prepared_store(crate::embedding::EMBED_MODEL_ID, |_| {});
    store.set_test_embedder(recording_embedder(recorded.clone()));

    let result = store
        .add_with_conflict(
            "p",
            "plain bge content",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed on a fresh store with the default model");
    let crate::memory_types::AddResult::Added { .. } = result else {
        panic!("expected Added");
    };
    assert_eq!(recorded.lock().unwrap().clone(), vec!["plain bge content"]);

    store
        .search("p", "plain query", 5, 0.0, SearchOptions::default())
        .expect("search must succeed on a fresh store with the default model");
    assert_eq!(
        recorded.lock().unwrap().clone(),
        vec!["plain bge content", "plain query"]
    );
}

#[test]
fn search_does_not_store_prefix_in_fts_or_content() {
    let (_dir, mut store) = e5_store_with_real_prefix_contract();
    let result = store
        .add_with_conflict(
            "p",
            "remember to water the plants",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed");
    let crate::memory_types::AddResult::Added { id } = result else {
        panic!("expected Added");
    };
    let memories = store
        .search("p", "water plants", 5, 0.0, SearchOptions::default())
        .expect("search must succeed when identity matches");
    // The row's content (as returned by search) must be unprefixed.
    let row = memories.iter().find(|m| m.id == id).expect("row found");
    assert_eq!(row.content, "remember to water the plants");
    // FTS: the BM25 index is built from the stored content, which is
    // unprefixed — search by the raw word must hit.
    let hybrid = store
        .search_hybrid("p", "water", 5, 0.0, SearchOptions::default())
        .expect("hybrid search must succeed");
    assert!(
        hybrid.iter().any(|m| m.id == id),
        "BM25 search on raw word must find the unprefixed row"
    );
}

#[test]
fn bge_add_stores_content_without_prefix() {
    // bge has no prefixes — stored content must be byte-identical to the
    // input (zero-change contract).
    let (_dir, mut store) = prepared_store(crate::embedding::EMBED_MODEL_ID, |_| {});
    let result = store
        .add_with_conflict(
            "p",
            "plain text no prefix",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed");
    let crate::memory_types::AddResult::Added { id } = result else {
        panic!("expected Added");
    };
    let memory = store.get(&id, "p").unwrap().expect("row exists");
    assert_eq!(memory.content, "plain text no prefix");
}
