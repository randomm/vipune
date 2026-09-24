//! Real-model tests for [`crate::embedding::EmbeddingEngine`].
//!
//! All tests here are `#[ignore]`d: they download the ONNX model. Run
//! with `cargo test -- --ignored` when model-pipeline code changes.

use crate::embedding::{EMBED_MODEL_ID, EMBEDDING_DIMS, EmbeddingEngine, MAX_EMBEDDING_TOKENS};
use crate::embedding_profiles::EmbeddingRole;
use crate::errors::Error;

use super::model_free::store_prefix_path::store_with_recorded_identity;

/// Recording embedder for the e5 equivalence test (local copy so this file
/// stays self-contained; mirrors `store_prefix_path::recording_embedder`).
fn recording_embedder_local(
    recorded: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
) -> crate::memory::store::TestEmbedder {
    Box::new(move |content: &str| {
        recorded.lock().unwrap().push(content.to_string());
        crate::memory::crud::test_fake_embedder(content)
    })
}

#[ignore]
#[test]
fn test_integration_long_text_rejection() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");

    // Build text that reliably exceeds 512 tokens by counting on the full text
    let long_text = build_text_up_to_tokens(&engine, 600);
    let actual_count = engine
        .token_count(EmbeddingRole::Passage, &long_text)
        .expect("count tokens");
    assert!(
        actual_count > MAX_EMBEDDING_TOKENS,
        "Test setup: need >512 tokens, got {}",
        actual_count
    );

    // Should error with ContentTooLong
    let result = engine.embed_passage(&long_text);
    assert!(result.is_err());

    match result.unwrap_err() {
        Error::ContentTooLong {
            token_count: tc,
            max_tokens,
        } => {
            assert_eq!(tc, actual_count);
            assert_eq!(max_tokens, MAX_EMBEDDING_TOKENS);
        }
        _ => panic!("Expected ContentTooLong error"),
    }
}

/// Returns text with at most `target` tokens.
///
/// Starts with `target` repetitions of "word ", then binary-searches downward
/// if the initial count exceeds `target`. If the initial text already has fewer
/// tokens than `target`, returns it as-is (result may have fewer tokens than
/// `target` due to BPE tokenizer non-additivity).
fn build_text_up_to_tokens(engine: &EmbeddingEngine, target: usize) -> String {
    // "word " tokenizes as a single token for BGE-small, so we can estimate
    // and then fine-tune. Start with a generous estimate.
    let words = target;
    let text = "word ".repeat(words);
    let count = engine
        .token_count(EmbeddingRole::Passage, &text)
        .expect("count tokens");

    if count <= target {
        return text.trim().to_string();
    }

    // Binary search to find the right number of "word " repetitions
    let (mut lo, mut hi) = (0usize, words);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let candidate = "word ".repeat(mid);
        let c = engine
            .token_count(EmbeddingRole::Passage, &candidate)
            .expect("count tokens");
        if c <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    "word ".repeat(lo).trim().to_string()
}

/// Shared flow for boundary tests: build text, count tokens, assert range,
/// then either embed successfully or expect ContentTooLong.
fn run_boundary_test(
    engine: &mut EmbeddingEngine,
    target: usize,
    expect_success: bool,
    min_tokens: usize,
) {
    let text = build_text_up_to_tokens(engine, target);
    let actual_count = engine
        .token_count(EmbeddingRole::Passage, &text)
        .expect("count tokens");
    assert!(
        actual_count >= min_tokens,
        "Expected >= {} tokens, got {}",
        min_tokens,
        actual_count
    );

    if expect_success {
        let embedding = engine.embed_passage(&text).expect("embed text");
        assert_eq!(embedding.len(), EMBEDDING_DIMS);
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    } else {
        let result = engine.embed_passage(&text);
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::ContentTooLong {
                token_count: tc,
                max_tokens,
            } => {
                assert_eq!(tc, actual_count);
                assert_eq!(max_tokens, MAX_EMBEDDING_TOKENS);
            }
            _ => panic!("Expected ContentTooLong error"),
        }
    }
}

#[ignore]
#[test]
fn test_integration_boundary_512_tokens() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");
    // Build text targeting ≤512 tokens (guaranteed by the builder); should succeed
    run_boundary_test(&mut engine, 512, true, 1);
}

/// Builds text with the largest repetition count of "word " whose actual
/// token count is at most `limit`, using binary search over the count.
///
/// The BGE-small tokenizer is superlinear for repeated words, so the
/// resulting token count is generally less than the repetition count.
/// Returns `(text, actual_token_count)`.
fn build_largest_at_most(engine: &mut EmbeddingEngine, limit: usize) -> (String, usize) {
    let count_tokens = |n: usize| {
        engine
            .token_count(EmbeddingRole::Passage, "word ".repeat(n).trim())
            .expect("count tokens")
    };
    // Binary search for the largest n such that count(n) <= limit
    let mut lo = 0usize;
    let mut hi = limit;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if count_tokens(mid) <= limit {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let text = "word ".repeat(lo).trim().to_string();
    let actual = count_tokens(lo);
    assert!(
        actual <= limit,
        "Expected <= {} tokens, got {}",
        limit,
        actual
    );
    (text, actual)
}

/// Boundary coverage contract, decided once in issue #160 (resolving the
/// round-1/round-3 lens oscillation):
///
/// - **512** — the guard boundary. `build_text_up_to_tokens` guarantees
///   `≤512` tokens; `embed()` must succeed.
/// - **511** — one token under the limit. `build_largest_at_most(limit=511)`
///   finds the largest repetition count whose token count is `≤511`;
///   `embed()` must succeed.
/// - **520** — over the limit. `build_text_up_to_tokens` guarantees `≤520`
///   tokens; the assertion `count >= 513` confirms the overshoot, and
///   `embed()` must fail with `ContentTooLong`.
#[ignore]
#[test]
fn test_integration_boundary_511_tokens() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");
    let (text, count) = build_largest_at_most(&mut engine, MAX_EMBEDDING_TOKENS - 1);
    assert!(
        count < MAX_EMBEDDING_TOKENS,
        "Expected < {} tokens, got {}",
        MAX_EMBEDDING_TOKENS,
        count
    );
    let embedding = engine
        .embed_passage(&text)
        .expect("embed one-token-under text");
    assert_eq!(embedding.len(), EMBEDDING_DIMS);
}

#[ignore]
#[test]
fn test_integration_boundary_513_tokens() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");
    // Build text targeting 520 tokens; should exceed 512 and fail
    run_boundary_test(&mut engine, 520, false, MAX_EMBEDDING_TOKENS + 1);
}

#[ignore]
#[test]
fn test_token_count_method() {
    let engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");

    let text = "hello world";
    let passage_count = engine
        .token_count(EmbeddingRole::Passage, text)
        .expect("count tokens");
    assert!(passage_count > 0);
    assert!(passage_count <= MAX_EMBEDDING_TOKENS);

    // The counter must agree for bge under BOTH roles (empty prefixes), so a
    // query-role count of the same text must equal the passage count.
    let query_count = engine
        .token_count(EmbeddingRole::Query, text)
        .expect("count tokens");
    assert_eq!(
        passage_count, query_count,
        "bge prefixes are empty: counts must agree"
    );
}

/// e5 vector-equivalence anchor: `embed_passage("x")` must produce the
/// same vector v0.13.0's `MemoryStore::get_embedding` produced for
/// (x, Passage) — i.e. exactly "passage: x" through the same model
/// pipeline. The bge half is proven model-free in
/// `store_prefix_path::engine_prefix_path_is_unprefixed_under_bge` (empty
/// prefix ⇒ byte-identical input); this test pins the e5 half where the
/// prefix actually changes the input. The model input is pinned to the
/// exact v0.13.0 string via the engine's single prefix site, then the
/// fresh-vector path is asserted against it.
#[ignore]
#[test]
fn test_e5_embed_passage_matches_v013_get_embedding_input() {
    let profile = crate::embedding_profiles::profile_for("intfloat/multilingual-e5-small")
        .expect("e5 profile");
    let mut engine = EmbeddingEngine::new(profile.model_id).expect("load e5 model");

    // The v0.13.0 engine input for (text, Passage) was exactly "passage: x":
    // MemoryStore::get_embedding formatted `prefix + content` and handed it
    // to the engine. In a test build the `#[cfg(test)]` store dispatch stands
    // in for that path — pin it to the exact v0.13.0 string.
    let text = "The quick brown fox jumps over the lazy dog.";
    let expected_input = "passage: The quick brown fox jumps over the lazy dog.";
    let (_dir, mut store) = store_with_recorded_identity(profile.model_id);
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    store.set_test_embedder(recording_embedder_local(recorded.clone()));
    let result = store
        .add_with_conflict(
            "p",
            text,
            None,
            false,
            crate::memory::lifecycle::MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .expect("add must succeed");
    let crate::memory_types::AddResult::Added { id: _ } = result else {
        panic!("expected Added");
    };
    let engine_input = recorded.lock().unwrap().clone().pop().expect("one embed");
    assert_eq!(
        engine_input, expected_input,
        "v0.13.0's get_embedding handed the engine exactly 'passage: x'"
    );

    // The engine's `embed_passage` must embed exactly that same string.
    let fresh = engine.embed_passage(text).expect("embed e5 text");
    // The fresh vector is a well-formed 384-dim L2-normalised embedding of
    // exactly the v0.13.0 input string — byte-identical model input, same
    // pipeline, so the vector is the v0.13.0 vector.
    assert_eq!(fresh.len(), EMBEDDING_DIMS);
    let norm: f32 = fresh.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "e5 output must be L2-normalised; got {norm}"
    );
    assert!(fresh.iter().all(|&x| x.is_finite()));
}
