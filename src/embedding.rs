//! Synchronous ONNX embedding engine for text-to-vector conversion.
//!
//! Uses bge-small-en-v1.5 model (384 dimensions) with mean pooling and L2 normalization.

use hf_hub::{Repo, RepoType, api::sync::Api};
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
    requires_token_type_ids: bool,
}

impl EmbeddingEngine {
    /// Creates a new embedding engine with the specified model.
    ///
    /// When `model_id` is the default model (`EMBED_MODEL_ID`), the pinned revision
    /// `EMBED_MODEL_REVISION` is used to ensure reproducibility. Custom model IDs
    /// fall back to the `main` branch.
    ///
    /// Files are cached locally in HF Hub cache, only downloaded once.
    pub fn new(model_id: &str) -> Result<Self, Error> {
        let api = Api::new()?;

        // Use pinned revision for default model, "main" for custom models
        let revision = if model_id == EMBED_MODEL_ID {
            EMBED_MODEL_REVISION.to_string()
        } else {
            "main".to_string()
        };

        // Capture the actual model_id and revision for the error message
        let err_model_id = model_id.to_string();
        let err_revision = revision.clone();

        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision,
        ));

        // Helper function for error messaging
        let wrap_download_err = move |e: hf_hub::api::sync::ApiError| {
            let revision_hint = if err_model_id == EMBED_MODEL_ID {
                format!(" --revision {}", err_revision)
            } else {
                String::new()
            };
            Error::Config(format!(
                "Failed to download embedding model '{}': {}.\n\nIf running in an air-gapped environment, pre-fetch the model before going offline:\n  huggingface-cli download {}{} --cache-dir ~/.cache/huggingface/hub",
                err_model_id, e, err_model_id, revision_hint
            ))
        };

        let model_path = repo
            .get("onnx/model.onnx")
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

        Ok(EmbeddingEngine {
            session,
            tokenizer,
            requires_token_type_ids,
        })
    }

    /// Count tokens in text without generating an embedding.
    ///
    /// Returns the number of tokens that would be generated for the given text.
    /// This is used for validation before embedding operations.
    ///
    /// **Important:** we clone the tokenizer and disable truncation so the true
    /// token count is returned. The main tokenizer has truncation at 512 enabled
    /// for safe inference, but that would silently cap counts and make the
    /// `ContentTooLong` guard in `embed()` unreachable.
    pub fn token_count(&self, text: &str) -> Result<usize, Error> {
        let mut count_tokenizer = self.tokenizer.clone();
        count_tokenizer.with_truncation(None)?;
        let encoding = count_tokenizer.encode(text, true)?;
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

    #[ignore]
    #[test]
    fn test_integration_long_text_rejection() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");

        // Build text that reliably exceeds 512 tokens by counting on the full text
        let long_text = build_text_with_tokens(&engine, 600);
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

    /// Builds text with approximately `target` tokens by adding words in large
    /// batches and checking `engine.token_count()` on the full accumulated text.
    /// This avoids the BPE tokenizer non-additivity bug where tokenizing individual
    /// words and summing differs from tokenizing the full text.
    fn build_text_with_tokens(engine: &EmbeddingEngine, target: usize) -> String {
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
            let mid = lo + (hi - lo + 1) / 2;
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

    #[ignore]
    #[test]
    fn test_integration_boundary_511_tokens() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");

        // Construct text approaching 512 tokens, then verify actual count
        let text = build_text_with_tokens(&engine, 512);
        let actual_count = engine.token_count(&text).expect("count tokens");
        assert!(
            actual_count <= MAX_EMBEDDING_TOKENS,
            "Expected ≤512 tokens, got {}",
            actual_count
        );

        // Should succeed (≤512 tokens)
        let embedding = engine.embed(&text).expect("embed text");
        assert_eq!(embedding.len(), 384);

        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[ignore]
    #[test]
    fn test_integration_boundary_512_tokens() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");

        // Construct text with exactly 512 tokens (or the maximum achievable ≤512)
        let text = build_text_with_tokens(&engine, 512);
        let actual_count = engine.token_count(&text).expect("count tokens");
        assert!(
            actual_count > 0 && actual_count <= MAX_EMBEDDING_TOKENS,
            "Expected 1..=512 tokens, got {}",
            actual_count
        );

        // Should succeed (≤512 tokens)
        let embedding = engine.embed(&text).expect("embed 512-token text");
        assert_eq!(embedding.len(), 384);

        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[ignore]
    #[test]
    fn test_integration_boundary_513_tokens() {
        let mut engine = EmbeddingEngine::new("BAAI/bge-small-en-v1.5").expect("load model");

        // Construct text that exceeds 512 tokens
        // Use a higher target to guarantee we exceed MAX_EMBEDDING_TOKENS
        let text = build_text_with_tokens(&engine, 520);
        let actual_count = engine.token_count(&text).expect("count tokens");
        assert!(
            actual_count > MAX_EMBEDDING_TOKENS,
            "Expected >512 tokens, got {}",
            actual_count
        );

        // Should fail with ContentTooLong
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
