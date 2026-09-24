//! Synchronous ONNX embedding engine for text-to-vector conversion.
//!
//! Loads one of the built-in model profiles from
//! [`crate::embedding_profiles`] (each pinned to an exact revision) and
//! produces 384-dimensional vectors with mean pooling and L2 normalization.
//!
//! The engine is prefix-aware: it stores the resolved model profile and the
//! role-aware methods ([`EmbeddingEngine::embed_query`] /
//! [`EmbeddingEngine::embed_passage`], [`EmbeddingEngine::token_count_query`]
//! / [`EmbeddingEngine::token_count_passage`]) prepend the profile's
//! query/passage prefix *before* tokenisation, so the 512-token check and any
//! pre-flight token count always see the prefixed text. Exactly one
//! production site applies prefixes — these methods. Empty input is
//! short-circuited to a zero vector *before* the prefix is applied, so an
//! empty string never reaches the model as a bare prefix.

use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::embedding_profiles::{EmbeddingRole, ModelProfile};
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
/// Uses one of the built-in model profiles (see
/// [`crate::embedding_profiles::BUILTIN_PROFILES`]) to generate 384-dimensional
/// embeddings with mean pooling and L2 normalization. All methods are
/// synchronous, matching vipune's no-async policy.
///
/// # Role-aware embedding
///
/// Some profiles (e5-style models) require a task prefix in the input:
/// `query: ` for search text and `passage: ` for stored text. The engine
/// stores the profile resolved in [`EmbeddingEngine::new`] and applies the
/// right prefix itself:
///
/// - Use [`EmbeddingEngine::embed_passage`] for stored text (content being
///   added, updated, or re-indexed).
/// - Use [`EmbeddingEngine::embed_query`] for search/classification text.
///
/// The role-aware token counters ([`EmbeddingEngine::token_count_passage`] /
/// [`EmbeddingEngine::token_count_query`]) count the *prefixed* text through
/// the same prefix path as the embedders, so a pre-flight count and the
/// actual embed always agree.
///
/// # No model-identity check
///
/// The engine performs **no** database model-identity check — direct engine
/// callers bypass the database entirely, and the identity refusal stays in
/// `MemoryStore::get_embedding`. If you keep vector caches **outside**
/// vipune, key them by [`crate::current_identity`]'s `ModelIdentity`
/// (model id + revision) so a model switch invalidates them.
///
/// # Mutability Requirements
///
/// The `embed_query()` / `embed_passage()` methods require `&mut self`
/// because ONNX internally mutates state for tensor allocations during
/// inference.
pub struct EmbeddingEngine {
    session: Session,
    tokenizer: Tokenizer,
    /// Truncation-free tokenizer cloned once at startup for accurate token counting.
    count_tokenizer: Tokenizer,
    requires_token_type_ids: bool,
    /// The resolved built-in profile — its prefixes make the engine role-aware.
    profile: &'static ModelProfile,
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
            }))
            .map_err(Error::Tokenization)?;

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
        count_tokenizer
            .with_truncation(None)
            .map_err(Error::Tokenization)?;

        Ok(EmbeddingEngine {
            session,
            tokenizer,
            count_tokenizer,
            requires_token_type_ids,
            profile,
        })
    }

    /// The resolved built-in profile (id, pinned revision, prefixes).
    pub fn profile(&self) -> &'static ModelProfile {
        self.profile
    }

    /// Count the tokens the engine would see for `text` with the given role's
    /// prefix applied (the same prefix path the embedders use), so a
    /// pre-flight count and the actual embed always agree.
    ///
    /// Uses a separate truncation-free tokenizer (pre-cloned at startup) so the
    /// true count is returned. The main tokenizer has truncation at 512 enabled
    /// for safe inference, but that would silently cap counts and make the
    /// `ContentTooLong` guard in the embed methods unreachable.
    pub fn token_count_role(&self, text: &str, role: EmbeddingRole) -> Result<usize, Error> {
        let input = self.prefix_input(role, text);
        self.count_tokens(&input)
    }

    /// Count the tokens of `text` as stored-content (passage) input: the
    /// profile's passage prefix is prepended before counting.
    pub fn token_count_passage(&self, text: &str) -> Result<usize, Error> {
        self.token_count_role(text, EmbeddingRole::Passage)
    }

    /// Count the tokens of `text` as search (query) input: the profile's query
    /// prefix is prepended before counting.
    pub fn token_count_query(&self, text: &str) -> Result<usize, Error> {
        self.token_count_role(text, EmbeddingRole::Query)
    }

    /// Embed stored-content (passage) text: the profile's passage prefix is
    /// prepended before tokenisation. Use for content being added, updated,
    /// or re-indexed.
    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        self.embed_role(text, EmbeddingRole::Passage)
    }

    /// Embed search (query) text: the profile's query prefix is prepended
    /// before tokenisation. Use for search and classification text.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        self.embed_role(text, EmbeddingRole::Query)
    }

    /// Embed a single text **without** applying any prefix (the raw text is
    /// fed to the model as-is).
    ///
    /// Internal use only — public callers must use [`Self::embed_passage`] or
    /// [`Self::embed_query`] so the profile's prefix is applied. This method
    /// is `pub(crate)` so the public API exposes only the role-aware methods.
    ///
    /// # Empty Input Handling
    ///
    /// Empty strings return a zero vector.
    ///
    /// # Token Limit
    ///
    /// Texts exceeding 512 tokens are rejected with a `ContentTooLong` error
    /// instead of being silently truncated.
    pub(crate) fn embed(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        if text.is_empty() {
            return Ok(vec![0.0f32; EMBEDDING_DIMS]);
        }

        let token_count = self.count_tokens(text)?;
        if token_count > MAX_EMBEDDING_TOKENS {
            return Err(Error::ContentTooLong {
                token_count,
                max_tokens: MAX_EMBEDDING_TOKENS,
            });
        }

        self.encode_and_infer(text)
    }

    /// The engine input for `text` with the role's profile prefix applied.
    ///
    /// This is the single prefix-application helper: the role-aware embed and
    /// token-count methods all route through it, so the prefixed input the
    /// model sees is exactly `prefix + text` (the bare text when the profile
    /// declares no prefix, e.g. bge).
    pub fn prefix_input(&self, role: EmbeddingRole, text: &str) -> String {
        let prefix = role.prefix(self.profile);
        if prefix.is_empty() {
            text.to_string()
        } else {
            format!("{prefix}{text}")
        }
    }

    /// Generate embedding for a single text with the role's prefix applied.
    ///
    /// Returns exactly 384-dimensional f32 vector, L2-normalized.
    ///
    /// # Empty Input Handling
    ///
    /// Empty strings return a zero vector — the check runs on the caller's
    /// text *before* the prefix is applied, so an empty string never reaches
    /// the model as a bare prefix.
    ///
    /// # Token Limit
    ///
    /// Texts whose *prefixed* form exceeds 512 tokens are rejected with a
    /// ContentTooLong error instead of being silently truncated.
    pub fn embed_role(&mut self, text: &str, role: EmbeddingRole) -> Result<Vec<f32>, Error> {
        if text.is_empty() {
            return Ok(vec![0.0f32; EMBEDDING_DIMS]);
        }

        // The prefixed text — the exact input the model sees.
        let input = self.prefix_input(role, text);

        // Validate token count BEFORE embedding (the prefixed text, counted
        // through the same truncation-free tokenizer the model path uses).
        let token_count = self.count_tokens(&input)?;
        if token_count > MAX_EMBEDDING_TOKENS {
            return Err(Error::ContentTooLong {
                token_count,
                max_tokens: MAX_EMBEDDING_TOKENS,
            });
        }

        self.encode_and_infer(&input)
    }

    /// Count the tokens of an already-finalized (possibly prefixed) engine
    /// input through the truncation-free count tokenizer.
    ///
    /// Internal use only — public callers must use [`Self::token_count_passage`]
    /// or [`Self::token_count_query`] so the profile's prefix is counted.
    pub(crate) fn token_count(&self, text: &str) -> Result<usize, Error> {
        self.count_tokens(text)
    }

    fn count_tokens(&self, input: &str) -> Result<usize, Error> {
        let encoding = self.count_tokenizer.encode(input, true)?;
        Ok(encoding.get_ids().len())
    }

    /// Tokenise `input`, run the model, and pool + L2-normalize the output.
    fn encode_and_infer(&mut self, input: &str) -> Result<Vec<f32>, Error> {
        let encoding = self.tokenizer.encode(input, true)?;
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
#[path = "embedding_tests.rs"]
mod embedding_tests;
