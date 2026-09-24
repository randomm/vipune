//! Mismatch / in-flight-migration refusal at the MemoryStore chokepoint
//! (issue #217, decision 3: add, update, search — and by extension
//! batch_ingest / supersede, which all route through `get_embedding` —
//! refuse with an error naming recorded vs configured identity, or the
//! interrupted migration target, and pointing at `vipune reindex --force`).
//!
//! Also covers the prefix-application contract (M1): the test embedder
//! receives the raw stored content (no prefix — prefixes live only in the
//! real engine path), but the identity check and the prefix-free raw text
//! are still observable through the stored content and FTS.

use crate::config::Config;
use crate::embedding_profiles::EmbeddingRole;
use crate::memory::{MemoryStore, SearchOptions};
use crate::sqlite::Database;

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
        identity_checked: false,
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
        crate::sqlite::identity::write_marker(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity {
                model_id: e5_model_id().to_string(),
                revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
            },
        )
        .unwrap();
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
        crate::sqlite::identity::write_marker(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity {
                model_id: e5_model_id().to_string(),
                revision: "rev".to_string(),
            },
        )
        .unwrap();
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
        crate::sqlite::identity::record_identity_and_clear_marker(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity::default_identity(),
        )
        .unwrap();
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
// The real engine path applies the profile's passage/query prefix before
// embedding (see `get_embedding`). The test embedder receives the raw stored
// content (no prefix), so the prefix itself is not observable through the
// fake — but we can prove the stored content and FTS never carry a prefix
// even when the e5 profile is configured, which is the contract that matters.

fn e5_store_with_real_prefix_contract() -> (tempfile::TempDir, MemoryStore) {
    let (dir, store) = prepared_store(e5_model_id(), |_| {});
    // Record the e5 identity so the identity check passes and add/search
    // actually run through the (test) embedder path.
    crate::sqlite::identity::record_identity_and_clear_marker(
        store.db.conn(),
        &crate::sqlite::identity::ModelIdentity {
            model_id: e5_model_id().to_string(),
            revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
        },
    )
    .unwrap();
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
