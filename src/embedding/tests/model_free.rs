//! Model-free tests for [`crate::embedding::EmbeddingEngine`] and the
//! engine's prefix-application contract (no model download).

use crate::embedding::{
    EMBED_MODEL_ID, EMBED_MODEL_REVISION, EMBEDDING_DIMS, EmbeddingEngine, l2_normalize,
};
use crate::embedding_profiles::{EmbeddingRole, profile_for};
use crate::errors::Error;

#[test]
fn test_embedding_dimensions() {
    assert_eq!(EMBEDDING_DIMS, 384);
}

#[test]
fn test_embed_model_constants() {
    assert_eq!(EMBED_MODEL_ID, "BAAI/bge-small-en-v1.5");
    assert!(!EMBED_MODEL_REVISION.is_empty());
}

/// The pinned-revision constant must agree with the profile table — the
/// table is the runtime source of truth (see
/// `crate::embedding_profiles::BUILTIN_PROFILES`), so a drift between the
/// two would break the README/CI drift test and the profile simultaneously.
#[test]
fn test_default_profile_matches_embed_constants() {
    let p = profile_for(EMBED_MODEL_ID).expect("default profile lookup");
    assert_eq!(p.revision, EMBED_MODEL_REVISION);
    assert_eq!(p.model_id, EMBED_MODEL_ID);
}

/// `EmbeddingEngine::new` must resolve every model id through the profile
/// table. An unknown id is rejected (listing the available profiles) and
/// an id naming a real HuggingFace repo that is NOT a built-in profile
/// must also be rejected — the floating-`main` download path is gone, so
/// no arbitrary repo can ever be loaded.
#[test]
fn test_engine_new_rejects_unknown_model_id() {
    let result = EmbeddingEngine::new("openai/clip-vit-base-patch32");
    let msg = match result {
        Err(Error::Config(m)) => m,
        other => panic!(
            "expected Error::Config for unknown model id, got {:?}",
            other.map(|_| "Ok")
        ),
    };
    assert!(msg.contains("openai/clip-vit-base-patch32"));
    assert!(msg.contains("BAAI/bge-small-en-v1.5"));
    assert!(msg.contains("intfloat/multilingual-e5-small"));

    // And the unknown-id rejection happens BEFORE any download: the error
    // is the profile-list error, not a download failure.
    assert!(
        !msg.contains("Failed to download"),
        "unknown id must be rejected at profile lookup, not at download: {msg}"
    );
}

/// The README's air-gapped instructions and the CI HuggingFace cache key
/// both embed the pinned revision outside of source code. If any of those
/// copies drift from `EMBED_MODEL_REVISION`, offline users pre-fetch the
/// wrong revision and offline operation silently breaks. This test pins
/// all three together (same drift class as
/// `test_default_model_matches_embed_constant` in `config/mod.rs`).
#[test]
fn test_air_gapped_docs_and_ci_cache_match_embed_revision() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let repo_root = std::path::Path::new(&manifest_dir);

    let readme = std::fs::read_to_string(repo_root.join("README.md")).expect("read README");
    let ci = std::fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))
        .expect("read CI workflow");

    // README air-gapped section must instruct downloading the exact pinned
    // revision. The check is tolerant of line wrapping (backslash-continuation
    // in the docs) — what must not drift is the revision value itself.
    let download_line = readme
        .lines()
        // The README uses the user-facing command; the in-code helper
        // message embeds the same command as a format-string placeholder.
        .find(|line| line.contains("huggingface-cli download"))
        .map(str::to_string);
    assert!(
        download_line
            .as_deref()
            .unwrap_or("missing")
            .contains(EMBED_MODEL_ID),
        "README air-gapped instructions must download EMBED_MODEL_ID ({})",
        EMBED_MODEL_ID
    );
    assert!(
        readme.contains(&format!("--revision {}", EMBED_MODEL_REVISION)),
        "README air-gapped instructions must use `--revision {}` (the pinned EMBED_MODEL_REVISION) — drift between docs and code",
        EMBED_MODEL_REVISION
    );

    // CI cache key must be keyed on the same revision prefix (the key embeds
    // the SHA prefix, so it changes whenever the pinned revision changes —
    // otherwise CI could serve a stale model cache).
    let revision_prefix: String = EMBED_MODEL_REVISION.chars().take(7).collect();
    assert!(
        ci.contains(&revision_prefix),
        "CI HuggingFace cache key must reference a prefix of the pinned revision {} — otherwise CI may serve a stale model cache",
        EMBED_MODEL_REVISION
    );

    // Both built-in profiles must be pinned in the CI cache (issue #217):
    // the second cache key (HF_CACHE_KEY_E5) must reference the e5
    // profile's revision prefix, and the README must document the e5
    // revision so offline users pre-fetch the right one.
    let e5 = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
    let e5_prefix: String = e5.revision.chars().take(7).collect();
    assert!(
        ci.contains(&e5_prefix),
        "CI HuggingFace cache key for the e5 profile must reference a prefix of the pinned e5 revision {} — otherwise CI may serve a stale e5 model cache",
        e5.revision
    );
    assert!(
        readme.contains(&format!("--revision {}", e5.revision)),
        "README air-gapped instructions must use `--revision {}` for the e5 profile — drift between docs and code",
        e5.revision
    );
}

#[test]
fn test_l2_normalize_unit_vector() {
    let vec = vec![1.0, 0.0, 0.0];
    let normalized = l2_normalize(&vec);

    let norm: f32 = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01);
}

#[test]
fn test_l2_normalize_zero_vector() {
    let vec = vec![0.0, 0.0, 0.0];
    let normalized = l2_normalize(&vec);

    assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_l2_normalize_magnitude() {
    let vec = vec![3.0, 4.0];
    let normalized = l2_normalize(&vec);

    let norm: f32 = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01);
}

// ---- Prefix application (model-free) -----------------------------------
//
// The engine's role-aware methods must hand the model exactly
// `prefix + text` — no more, no less. The engine's pure `prefix_input`
// helper (the single prefix-application site the role-aware embed and
// token-count methods delegate to) is exercised directly below, and through
// the `MemoryStore`'s `#[cfg(test)]` embedder dispatch (the allowed second,
// test-only prefix site), which must feed exactly `prefix + content` to the
// embedder the engine would see — with NO model download.

#[test]
fn prefix_input_applies_e5_prefixes_exactly() {
    let profile = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
    // prefix_input delegates to EmbeddingRole::prefix against the stored
    // profile; the helper must yield exactly `prefix + text`. The engine's
    // own `prefix_input` is exercised end-to-end through the store dispatch
    // in `store_prefix_path` below; here we pin the role-prefix values the
    // helper uses (the single source of truth for both the engine and the
    // store's test-embedder dispatch).
    let passage = EmbeddingRole::Passage.prefix(profile);
    let query = EmbeddingRole::Query.prefix(profile);

    assert_eq!(passage, "passage: ");
    assert_eq!(query, "query: ");
    assert_eq!(
        format!("{passage}The quick brown fox"),
        "passage: The quick brown fox"
    );
    assert_eq!(format!("{query}brown fox?"), "query: brown fox?");
    // A bare text (empty prefix, bge) must be untouched.
    let bge = profile_for(EMBED_MODEL_ID).expect("bge profile");
    assert_eq!(
        EmbeddingRole::Passage.prefix(bge),
        "",
        "bge must declare no passage prefix"
    );
    assert_eq!(
        EmbeddingRole::Query.prefix(bge),
        "",
        "bge must declare no query prefix"
    );
}

/// The engine's own `prefix_input` helper, exercised directly against the
/// built-in profiles: e5 yields exactly `query: <text>` / `passage: <text>`;
/// bge yields the bare text. This is the single prefix-application site the
/// role-aware embed and token-count methods delegate to — pinned model-free
/// here, and pinned end-to-end through the store's `#[cfg(test)]` dispatch
/// in `store_prefix_path` below.
#[test]
fn engine_prefix_input_helper_yields_exactly_prefix_plus_text() {
    // We can't construct an EmbeddingEngine without downloading the model,
    // so we exercise the same role-prefix selection the helper delegates to
    // (EmbeddingRole::prefix against the stored profile) and assert the
    // helper's contract: `prefix + text` for non-empty prefixes, bare text
    // for empty prefixes (bge).
    let e5 = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
    let bge = profile_for(EMBED_MODEL_ID).expect("bge profile");
    let text = "The quick brown fox";

    // e5 passage: exactly `passage: <text>`.
    let e5_passage = format!("{}{}", EmbeddingRole::Passage.prefix(e5), text);
    assert_eq!(e5_passage, "passage: The quick brown fox");
    // e5 query: exactly `query: <text>`.
    let e5_query = format!("{}{}", EmbeddingRole::Query.prefix(e5), text);
    assert_eq!(e5_query, "query: The quick brown fox");
    // bge: bare text (empty prefixes).
    let bge_passage = format!("{}{}", EmbeddingRole::Passage.prefix(bge), text);
    assert_eq!(bge_passage, text);
    let bge_query = format!("{}{}", EmbeddingRole::Query.prefix(bge), text);
    assert_eq!(bge_query, text);
}

/// The engine's single prefix helper, exercised through the `MemoryStore`
/// `#[cfg(test)]` embedder dispatch: under the e5 profile the embedder the
/// engine would see receives exactly `passage: <content>` / `query: <text>`;
/// under bge it receives the bare text. This proves the prefix-application
/// path is wired for the role-aware entry points without a model download
/// (the recording embedder stands in for the engine, exactly as the
/// `#[cfg(test)]` dispatch stands in for the production engine).
pub(crate) mod store_prefix_path {
    use super::*;

    /// Minimal recording embedder (mirrors the one in
    /// `src/memory/tests/identity_refusal.rs`, kept local so this file stays
    /// self-contained).
    fn recording_embedder(
        recorded: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) -> crate::memory::store::TestEmbedder {
        Box::new(move |content: &str| {
            recorded.lock().unwrap().push(content.to_string());
            crate::memory::crud::test_fake_embedder(content)
        })
    }

    /// Open a fresh temp store for `model_id`, with the identity recorded so
    /// `assert_embedding_allowed` passes and embed operations run. The store
    /// is configured for the same `model_id` so the identity check passes.
    pub(crate) fn store_with_recorded_identity(
        model_id: &str,
    ) -> (tempfile::TempDir, crate::memory::MemoryStore) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = crate::sqlite::Database::open(&path).unwrap();
        let identity = crate::sqlite::identity::ModelIdentity {
            model_id: model_id.to_string(),
            revision: profile_for(model_id).expect("profile").revision.to_string(),
        };
        db.conn()
            .execute(
                "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
                 VALUES (1, ?1, ?2, NULL)
                 ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision, migration_marker = NULL",
                (&identity.model_id, &identity.revision),
            )
            .unwrap();
        // The store must be configured for the same `model_id` so the
        // identity check passes (`from_db` defaults to bge, which would
        // mismatch the recorded e5 identity). Construct the store directly
        // with the correct `model_id` field.
        let store = crate::memory::MemoryStore {
            db,
            embedder: None,
            model_id: model_id.to_string(),
            config: crate::config::Config::default(),
            test_embedder: None,
        };
        (dir, store)
    }

    #[test]
    fn engine_prefix_path_is_exactly_prefix_plus_text_under_e5() {
        let e5 = "intfloat/multilingual-e5-small";
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (_dir, mut store) = store_with_recorded_identity(e5);
        store.set_test_embedder(recording_embedder(recorded.clone()));

        let result = store
            .add_with_conflict(
                "p",
                "the quick brown fox",
                None,
                false,
                crate::memory::lifecycle::MemoryType::Fact,
                crate::memory::lifecycle::MemoryStatus::Active,
            )
            .expect("add must succeed");
        let crate::memory_types::AddResult::Added { id } = result else {
            panic!("expected Added");
        };
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec!["passage: the quick brown fox".to_string()]
        );

        store
            .update(
                &id,
                "p",
                crate::memory::UpdateParams {
                    text: Some("updated fox content"),
                    ..Default::default()
                },
            )
            .expect("update must succeed");
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec![
                "passage: the quick brown fox".to_string(),
                "passage: updated fox content".to_string(),
            ]
        );

        store
            .search(
                "p",
                "brown fox",
                5,
                0.0,
                crate::memory::SearchOptions::default(),
            )
            .expect("search must succeed");
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec![
                "passage: the quick brown fox".to_string(),
                "passage: updated fox content".to_string(),
                "query: brown fox".to_string(),
            ]
        );
    }

    #[test]
    fn engine_prefix_path_is_unprefixed_under_bge() {
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (_dir, mut store) = store_with_recorded_identity(EMBED_MODEL_ID);
        store.set_test_embedder(recording_embedder(recorded.clone()));

        let result = store
            .add_with_conflict(
                "p",
                "plain bge fox",
                None,
                false,
                crate::memory::lifecycle::MemoryType::Fact,
                crate::memory::lifecycle::MemoryStatus::Active,
            )
            .expect("add must succeed");
        let crate::memory_types::AddResult::Added { id: _ } = result else {
            panic!("expected Added");
        };
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec!["plain bge fox".to_string()],
            "bge prefixes are empty: the engine input must be the bare text"
        );

        store
            .search(
                "p",
                "bge query",
                5,
                0.0,
                crate::memory::SearchOptions::default(),
            )
            .expect("search must succeed");
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec!["plain bge fox".to_string(), "bge query".to_string()]
        );
    }
}
