//! Synchronous ONNX embedding engine for text-to-vector conversion.
//!
//! Loads one of the built-in model profiles from
//! [`crate::embedding_profiles`] (each pinned to an exact revision) and
//! produces 384-dimensional vectors with mean pooling and L2 normalization.
//! Query/passage prefixes are applied by callers at the embedding chokepoint
//! *before* calling [`EmbeddingEngine::embed`], so the 512-token check sees
//! the prefixed text; the engine itself is prefix-unaware.

use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::errors::Error;
use tokenizers::TruncationParams;

/// Embedding dimensions for bge-small-en-v1.5 model.
///
/// All generated embeddings are 384-dimensional vectors.
pub const EMBEDDING_DIMS: usize = 384;

/// Maximum number of tokens allowed for embedding.
///
/// Content exceeding this limit will be rejected instead of silently truncated.
pub const MAX_EMBEDDING_TOKENS: usize = 512;

/// HuggingFace model ID for the embedding model.
pub const EMBED_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";

/// Pinned commit SHA for the bge-small-en-v1.5 ONNX model on HuggingFace.
/// To update: verify new SHA resolves onnx/model.onnx (200 OK), update the
/// HF_CACHE_KEY env var in .github/workflows/ci.yml, re-embed all stored memories.
///
/// The revision value itself is also declared in the profile table
/// (`crate::embedding_profiles::BUILTIN_PROFILES`); this constant is kept as
/// the stable public re-export the library API and drift tests reference.
pub const EMBED_MODEL_REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";

/// ONNX embedding engine for synchronous text-to-vector conversion.
///
/// Uses the bge-small-en-v1.5 model to generate 384-dimensional embeddings
/// with mean pooling and L2 normalization. All methods are synchronous,
/// matching vipune's no-async policy.
///
/// # Mutability Requirements
///
/// The `embed()` method requires `&mut self` because ONNX internally mutates
/// state for tensor allocations during inference.
pub struct EmbeddingEngine {
    session: Session,
    tokenizer: Tokenizer,
    /// Truncation-free tokenizer cloned once at startup for accurate token counting.
    count_tokenizer: Tokenizer,
    requires_token_type_ids: bool,
}

impl EmbeddingEngine {
    /// Creates a new embedding engine for the given built-in model id.
    ///
    /// The id must name a built-in profile (see
    /// [`crate::embedding_profiles::BUILTIN_PROFILES`]); an unknown id is
    /// rejected with an error listing the available profiles. No profile ever
    /// loads a floating revision — each is pinned to an exact commit SHA.
    ///
    /// Files are cached locally in the HF Hub cache
    /// (`~/.cache/huggingface/hub/` by default, or `$HF_HOME/hub` when `HF_HOME`
    /// is set), only downloaded once.
    pub fn new(model_id: &str) -> Result<Self, Error> {
        use crate::embedding_profiles::profile_for;

        // Resolve through the profile table: unknown ids are rejected here, and
        // every profile carries its exact pinned revision.
        let profile = profile_for(model_id)?;
        let api = ApiBuilder::new().build()?;

        let repo = api.repo(Repo::with_revision(
            profile.model_id.to_string(),
            RepoType::Model,
            profile.revision.to_string(),
        ));

        // Helper function for error messaging (every profile is revision-pinned,
        // so the hint always carries `--revision`)
        let wrap_download_err = move |e: hf_hub::api::sync::ApiError| {
            Error::Config(format!(
                "Failed to download embedding model '{}': {}.\n\nIf running in an air-gapped environment, pre-fetch the model before going offline:\n  huggingface-cli download {} --revision {} --cache-dir ~/.cache/huggingface/hub",
                profile.model_id, e, profile.model_id, profile.revision
            ))
        };

        let model_path = repo
            .get(profile.onnx_path)
            .or_else(|_| repo.get("onnx/model.onnx"))
            .or_else(|_| repo.get("model.onnx"))
            .map_err(&wrap_download_err)?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(&wrap_download_err)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)?;
        tokenizer
            .with_padding(None)
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .commit_from_file(&model_path)?;

        // Check if model requires token_type_ids input
        let requires_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");

        // Pre-initialize a truncation-free tokenizer for accurate token counting.
        // Cloning once at startup avoids a 700KB+ deep copy on every embed() call.
        let mut count_tokenizer = tokenizer.clone();
        count_tokenizer.with_truncation(None)?;

        Ok(EmbeddingEngine {
            session,
            tokenizer,
            count_tokenizer,
            requires_token_type_ids,
        })
    }

    /// Count tokens in text without generating an embedding.
    ///
    /// Returns the number of tokens that would be generated for the given text.
    /// This is used for validation before embedding operations.
    ///
    /// Uses a separate truncation-free tokenizer (pre-cloned at startup) so the
    /// true count is returned. The main tokenizer has truncation at 512 enabled
    /// for safe inference, but that would silently cap counts and make the
    /// `ContentTooLong` guard in `embed()` unreachable.
    pub fn token_count(&self, text: &str) -> Result<usize, Error> {
        let encoding = self.count_tokenizer.encode(text, true)?;
        Ok(encoding.get_ids().len())
    }

    /// Generate embedding for a single text.
    ///
    /// Returns exactly 384-dimensional f32 vector, L2-normalized.
    ///
    /// # Empty Input Handling
    ///
    /// Empty strings return a zero vector. This provides graceful handling
    /// without requiring error recovery from callers.
    ///
    /// # Token Limit
    ///
    /// Texts exceeding 512 tokens are rejected with a ContentTooLong error instead
    /// of being silently truncated.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        if text.is_empty() {
            return Ok(vec![0.0f32; EMBEDDING_DIMS]);
        }

        // Validate token count BEFORE embedding
        let token_count = self.token_count(text)?;
        if token_count > MAX_EMBEDDING_TOKENS {
            return Err(Error::ContentTooLong {
                token_count,
                max_tokens: MAX_EMBEDDING_TOKENS,
            });
        }

        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        if input_ids.is_empty() {
            return Ok(vec![0.0f32; EMBEDDING_DIMS]);
        }

        let seq_len = input_ids.len();

        let input_ids_vec: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let attention_mask_vec: Vec<i64> = attention_mask.iter().map(|&m| m as i64).collect();

        let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids_vec))?;
        let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask_vec))?;

        // Only include token_type_ids if the model requires it
        let outputs = if self.requires_token_type_ids {
            let token_type_ids_vec: Vec<i64> = vec![0i64; seq_len]; // Single sentence, all zeros
            let token_type_ids_tensor =
                Tensor::from_array(([1usize, seq_len], token_type_ids_vec))?;
            let inputs = inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor
            ];
            self.session.run(inputs?)?
        } else {
            let inputs = inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ];
            self.session.run(inputs?)?
        };

        let last_hidden_state = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("token_embeddings"))
            .ok_or_else(|| {
                Error::Inference(
                    "Output tensor 'last_hidden_state' or 'token_embeddings' not found".to_string(),
                )
            })?
            .try_extract_tensor::<f32>()?;

        let shape = last_hidden_state.shape();
        let data = last_hidden_state.as_slice().unwrap();
        if shape.len() != 3 {
            return Err(Error::Inference(format!(
                "Expected 3D output (batch, seq_len, hidden), got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let hidden_dim = shape[2];

        if batch_size != 1 || hidden_dim != EMBEDDING_DIMS {
            return Err(Error::Inference(format!(
                "Unexpected output shape: {:?}, batch=1, hidden=384 expected",
                shape
            )));
        }

        let mut pooled = vec![0.0f32; EMBEDDING_DIMS];

        for (token_idx, chunk) in data.chunks(hidden_dim).take(seq_len).enumerate() {
            let mask_value = attention_mask.get(token_idx).copied().unwrap_or(0) as f32;

            for (dim, pooled_value) in pooled.iter_mut().enumerate() {
                *pooled_value += chunk[dim] * mask_value;
            }
        }

        let mask_sum: f32 = attention_mask
            .iter()
            .take(seq_len)
            .map(|&m| m as f32)
            .sum::<f32>()
            .max(1e-9);

        for value in pooled.iter_mut() {
            *value /= mask_sum;
        }

        let normalized = l2_normalize(&pooled);
        Ok(normalized)
    }
}

pub(crate) fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm = norm.max(1e-9);

    vec.iter().map(|&x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        use crate::embedding_profiles::{default_profile, profile_for};
        let p = profile_for(EMBED_MODEL_ID).expect("default profile lookup");
        assert_eq!(p.revision, EMBED_MODEL_REVISION);
        assert_eq!(default_profile().model_id, EMBED_MODEL_ID);
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
        let e5 = crate::embedding_profiles::profile_for("intfloat/multilingual-e5-small")
            .expect("e5 profile");
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

    #[ignore]
    #[test]
    fn test_integration_whitespace_only() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");
        let embedding = engine.embed("   \t\n  ").expect("embed whitespace text");

        // Whitespace-only input should produce a valid embedding
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[ignore]
    #[test]
    fn test_integration_simple_text() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");
        let embedding = engine.embed("hello world").expect("embed text");

        assert_eq!(embedding.len(), 384);

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
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");
        let embedding = engine.embed("").expect("embed empty text");

        assert_eq!(embedding.len(), 384);
        assert_eq!(embedding, vec![0.0f32; 384]);
    }

    /// Decision 5 (issue #217): the e5 model card specifies mean pooling over
    /// `last_hidden_state` followed by L2 normalisation, the same pipeline as
    /// bge. This real-model test asserts the e5 output has L2 norm ≈ 1.0 so
    /// `classify_embedding`'s Real band [0.99, 1.01] still holds for e5
    /// vectors (the Mock > 2.0 band is unaffected). Ignored because it
    /// downloads the ~470 MB e5 model.
    #[ignore]
    #[test]
    fn test_integration_e5_output_norm_is_one() {
        use crate::embedding_profiles::profile_for;

        let profile = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
        let mut engine = EmbeddingEngine::new(profile.model_id).expect("load e5 model");

        // Embed a short passage with the e5 passage prefix (the role that
        // stored content uses) and assert the L2 normalisation holds.
        let text = "passage: The quick brown fox jumps over the lazy dog.";
        let embedding = engine.embed(text).expect("embed e5 text");
        assert_eq!(embedding.len(), 384, "e5 must produce 384-dim vectors");

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

    #[ignore]
    #[test]
    fn test_integration_long_text_rejection() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");

        // Build text that reliably exceeds 512 tokens by counting on the full text
        let long_text = build_text_up_to_tokens(&engine, 600);
        let actual_count = engine.token_count(&long_text).expect("count tokens");
        assert!(
            actual_count > MAX_EMBEDDING_TOKENS,
            "Test setup: need >512 tokens, got {}",
            actual_count
        );

        // Should error with ContentTooLong
        let result = engine.embed(&long_text);
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
        let count = engine.token_count(&text).expect("count tokens");

        if count <= target {
            return text.trim().to_string();
        }

        // Binary search to find the right number of "word " repetitions
        let (mut lo, mut hi) = (0usize, words);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            let candidate = "word ".repeat(mid);
            let c = engine.token_count(&candidate).expect("count tokens");
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
        let actual_count = engine.token_count(&text).expect("count tokens");
        assert!(
            actual_count >= min_tokens,
            "Expected >= {} tokens, got {}",
            min_tokens,
            actual_count
        );

        if expect_success {
            let embedding = engine.embed(&text).expect("embed text");
            assert_eq!(embedding.len(), 384);
            let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01);
        } else {
            let result = engine.embed(&text);
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
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");
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
                .token_count("word ".repeat(n).trim())
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
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");
        let (text, count) = build_largest_at_most(&mut engine, MAX_EMBEDDING_TOKENS - 1);
        assert!(
            count < MAX_EMBEDDING_TOKENS,
            "Expected < {} tokens, got {}",
            MAX_EMBEDDING_TOKENS,
            count
        );
        let embedding = engine.embed(&text).expect("embed one-token-under text");
        assert_eq!(embedding.len(), EMBEDDING_DIMS);
    }

    #[ignore]
    #[test]
    fn test_integration_boundary_513_tokens() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");
        // Build text targeting 520 tokens; should exceed 512 and fail
        run_boundary_test(&mut engine, 520, false, MAX_EMBEDDING_TOKENS + 1);
    }

    #[ignore]
    #[test]
    fn test_token_count_method() {
        let engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");

        let text = "hello world";
        let token_count = engine.token_count(text).expect("count tokens");
        assert!(token_count > 0);
        assert!(token_count <= 512);
    }
}
