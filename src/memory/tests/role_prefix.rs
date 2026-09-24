//! Role-aware engine method delegation (issue #220).
//!
//! The public API is the four role-aware methods, all delegating to the
//! engine's crate-private role helper. These model-free tests pin the
//! delegation through the store's `#[cfg(test)]` embedder dispatch (the
//! allowed second, test-only prefix site): the recording embedder receives
//! exactly `role.prefix(profile) + content` for the store's model, so the
//! role-aware entry points and the test-embedder contract cannot drift
//! apart. The engine's own pure `prefix_input` helper is pinned in
//! `src/embedding/tests/model_free.rs`.

use crate::embedding_profiles::EmbeddingRole;
use crate::memory::SearchOptions;
use crate::memory::tests::identity_refusal::{e5_model_id, e5_store_with_real_prefix_contract};

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

/// `MemoryStore::add` routes through `get_embedding` (passage role): the
/// recording embedder must receive exactly `passage: <content>` under e5.
#[test]
fn add_delegates_to_passage_role_engine_method() {
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (_dir, mut store) = e5_store_with_real_prefix_contract();
    store.set_test_embedder(recording_embedder(recorded.clone()));

    let result = store
        .add_with_conflict(
            "p",
            "passage role add",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed");
    let crate::memory_types::AddResult::Added { id } = result else {
        panic!("expected Added");
    };

    let content = "passage role add";
    let profile = crate::embedding_profiles::profile_for(e5_model_id()).expect("e5 profile");
    let expected = format!("{}{}", EmbeddingRole::Passage.prefix(profile), content);
    assert_eq!(recorded.lock().unwrap().clone(), vec![expected.clone()]);
    // And the stored content stays unprefixed.
    let memory = store.get(&id, "p").unwrap().expect("row exists");
    assert_eq!(memory.content, content);
}

/// `MemoryStore::update` routes through `get_embedding` (passage role):
/// the recording embedder must receive exactly `passage: <content>` under e5.
#[test]
fn update_delegates_to_passage_role_engine_method() {
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (_dir, mut store) = e5_store_with_real_prefix_contract();
    store.set_test_embedder(recording_embedder(recorded.clone()));

    let result = store
        .add_with_conflict(
            "p",
            "update target content",
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed");
    let crate::memory_types::AddResult::Added { id } = result else {
        panic!("expected Added");
    };
    recorded.lock().unwrap().clear();

    store
        .update(
            &id,
            "p",
            crate::memory::UpdateParams {
                text: Some("updated passage content"),
                ..Default::default()
            },
        )
        .expect("update must succeed");

    let profile = crate::embedding_profiles::profile_for(e5_model_id()).expect("e5 profile");
    let expected = format!(
        "{}updated passage content",
        EmbeddingRole::Passage.prefix(profile)
    );
    assert_eq!(recorded.lock().unwrap().clone(), vec![expected]);
}

/// `MemoryStore::search` routes through `get_embedding` (query role): the
/// recording embedder must receive exactly `query: <text>` under e5.
#[test]
fn search_delegates_to_query_role_engine_method() {
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (_dir, mut store) = e5_store_with_real_prefix_contract();
    store.set_test_embedder(recording_embedder(recorded.clone()));

    store
        .search("p", "passage role add", 5, 0.0, SearchOptions::default())
        .expect("search must succeed");

    let profile = crate::embedding_profiles::profile_for(e5_model_id()).expect("e5 profile");
    let expected = format!("{}passage role add", EmbeddingRole::Query.prefix(profile));
    assert_eq!(recorded.lock().unwrap().clone(), vec![expected]);
}
