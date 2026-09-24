//! Tests for [`crate::embedding::EmbeddingEngine`].
//!
//! Real-model tests are `#[ignore]`d; run with `cargo test -- --ignored`.

#![cfg(test)]

use crate::embedding::{
    EMBED_MODEL_ID, EMBED_MODEL_REVISION, EMBEDDING_DIMS, EmbeddingEngine, MAX_EMBEDDING_TOKENS,
    l2_normalize,
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
// `prefix + text` — no more, no less. These tests exercise the pure
// `prefix_input` helper (the single prefix-application site the engine
// delegates to) against both built-in profiles, with NO model download.

#[test]
fn prefix_input_applies_e5_prefixes_exactly() {
    let profile = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
    // prefix_input delegates to EmbeddingRole::prefix against the stored
    // profile; the helper must yield exactly `prefix + text`.
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

// ---- Real-model tests (#[ignore]d: download the ONNX model) ------------
//
// Migrated to the role-aware API (embed_query / embed_passage /
// token_count_query / token_count_passage). Run with
// `cargo test -- --ignored` when model-pipeline code changes.

#[ignore]
#[test]
fn test_integration_whitespace_only() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");
    let embedding = engine
        .embed_passage("   \t\n  ")
        .expect("embed whitespace text");

    // Whitespace-only input should produce a valid embedding
    assert_eq!(embedding.len(), EMBEDDING_DIMS);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[ignore]
#[test]
fn test_integration_simple_text() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");
    let embedding = engine.embed_passage("hello world").expect("embed text");

    assert_eq!(embedding.len(), EMBEDDING_DIMS);

    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding should be L2-normalized"
    );

    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[ignore]
#[test]
fn test_integration_empty_string() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");
    // Empty input keeps today's behaviour under BOTH roles: a zero vector,
    // the empty check running BEFORE any prefix is applied.
    let embedding = engine.embed_passage("").expect("embed empty text");
    assert_eq!(embedding.len(), EMBEDDING_DIMS);
    assert_eq!(embedding, vec![0.0f32; EMBEDDING_DIMS]);

    let embedding = engine.embed_query("").expect("embed empty text (query)");
    assert_eq!(embedding, vec![0.0f32; EMBEDDING_DIMS]);
}

/// Decision 5 (issue #217): the e5 model card specifies mean pooling over
/// `last_hidden_state` followed by L2 normalisation, the same pipeline as
/// bge. This real-model test asserts the e5 output has L2 norm ≈ 1.0 so
/// `classify_embedding`'s Real band [0.99, 1.01] still holds for e5
/// vectors (the Mock > 2.0 band is unaffected). The passage prefix comes
/// from the profile via `embed_passage` — no hand-written prefix here.
/// Ignored because it downloads the ~470 MB e5 model.
#[ignore]
#[test]
fn test_integration_e5_output_norm_is_one() {
    let profile = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
    let mut engine = EmbeddingEngine::new(profile.model_id).expect("load e5 model");

    // Embed a short passage with the e5 passage prefix (the role that
    // stored content uses) and assert the L2 normalisation holds.
    let embedding = engine
        .embed_passage("The quick brown fox jumps over the lazy dog.")
        .expect("embed e5 text");
    assert_eq!(
        embedding.len(),
        EMBEDDING_DIMS,
        "e5 must produce 384-dim vectors"
    );

    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "e5 output must be L2-normalised (norm ≈ 1.0) so the Real band [0.99, 1.01] in classify_embedding holds; got {norm}"
    );
    assert!(
        embedding.iter().all(|&x| x.is_finite()),
        "e5 output must be all-finite"
    );
}

/// e5 vector-equivalence anchor: `embed_passage("x")` must produce the
/// same vector v0.13.0's `MemoryStore::get_embedding` produced for
/// (x, Passage) — i.e. exactly "passage: x" through the same model
/// pipeline. The bge equivalence is guaranteed structurally (empty
/// prefix ⇒ identical input); this test pins the e5 half where the
/// prefix actually changes the input.
#[ignore]
#[test]
fn test_e5_embed_passage_equals_v013_get_embedding_output() {
    let profile = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
    let mut engine = EmbeddingEngine::new(profile.model_id).expect("load e5 model");

    // The v0.13.0 engine input for (text, Passage) was "passage: text".
    // prefix_input is the engine's single prefix site: assert the helper
    // yields exactly that string, then that embedding it is well-formed.
    let text = "The quick brown fox jumps over the lazy dog.";
    let input = engine.prefix_input(EmbeddingRole::Passage, text);
    assert_eq!(
        input,
        "passage: The quick brown fox jumps over the lazy dog."
    );

    let embedding = engine.embed_passage(text).expect("embed e5 text");
    assert_eq!(embedding.len(), EMBEDDING_DIMS);
    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "e5 output must be L2-normalised; got {norm}"
    );
}

#[ignore]
#[test]
fn test_integration_long_text_rejection() {
    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID).expect("load model");

    // Build text that reliably exceeds 512 tokens by counting on the full text
    let long_text = build_text_up_to_tokens(&engine, 600);
    let actual_count = engine
        .token_count_passage(&long_text)
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
    let count = engine.token_count_passage(&text).expect("count tokens");

    if count <= target {
        return text.trim().to_string();
    }

    // Binary search to find the right number of "word " repetitions
    let (mut lo, mut hi) = (0usize, words);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let candidate = "word ".repeat(mid);
        let c = engine
            .token_count_passage(&candidate)
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
    let actual_count = engine.token_count_passage(&text).expect("count tokens");
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
            .token_count_passage("word ".repeat(n).trim())
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
    let passage_count = engine.token_count_passage(text).expect("count tokens");
    assert!(passage_count > 0);
    assert!(passage_count <= MAX_EMBEDDING_TOKENS);

    // Role-aware counters must agree for bge (empty prefixes), and the
    // query counter must also work.
    let query_count = engine.token_count_query(text).expect("count tokens");
    assert_eq!(
        passage_count, query_count,
        "bge prefixes are empty: counts must agree"
    );
}
