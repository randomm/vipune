//! Synchronous ONNX embedding engine for text-to-vector conversion.
//!
//! Loads one of the built-in model profiles from
//! [`crate::embedding_profiles`] (each pinned to an exact revision) and
//! produces 384-dimensional vectors with mean pooling and L2 normalization.
//!
//! The engine is prefix-aware: it stores the resolved model profile and the
//! role-aware methods ([`EmbeddingEngine::embed_query`] /
//! [`EmbeddingEngine::embed_passage`], [`EmbeddingEngine::token_count`])
//! prepend the profile's
//! query/passage prefix *before* tokenisation, so the 512-token check and any
//! pre-flight token count always see the prefixed text. Exactly one
//! production site applies prefixes — these methods. Empty input is
//! short-circuited to a zero vector *before* the prefix is applied, so an
//! empty string never reaches the model as a bare prefix.
//!
//! The ONNX run + pooling live in [`inference`]; constants and the
//! `EmbeddingEngine` struct are re-exported from [`engine`].

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use tokenizers::Tokenizer;

use super::inference;
use crate::embedding_profiles::{EmbeddingRole, ModelProfile};
use crate::errors::{Error, with_prefix};
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

/// The home directory used for default paths: `dirs::home_dir()` with an
/// `env::var("HOME")` fallback, the same pattern `Config::default()` uses.
fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    })
}

/// Resolves the hf-hub cache directory: `$HF_HOME/hub` when `HF_HOME` is set
/// to a non-empty value, `<home>/.cache/huggingface/hub` otherwise (hf-hub's
/// own rule; an empty `HF_HOME` counts as unset, and the env value is appended
/// to verbatim — no normalisation, so a relative value resolves against the
/// process cwd). Pure: the env value and home dir are arguments, so the
/// resolution is testable without a model or network.
pub(crate) fn resolve_hf_cache_dir(hf_home: Option<&str>, home: &Path) -> PathBuf {
    match hf_home {
        Some(hf_home) if !hf_home.is_empty() => Path::new(hf_home).join("hub"),
        _ => home.join(".cache").join("huggingface").join("hub"),
    }
}

/// The process-wide ONNX Runtime environment, built on first use.
///
/// ONNX Runtime permits exactly one environment per process: rc.15's
/// `EnvironmentBuilder::build` fails with "only one environment is allowed per
/// process" once one exists. Every `EmbeddingEngine` therefore shares this one
/// `Environment` (it is `Clone` over an `Arc`), so constructing a second
/// engine — e5 beside bge in a test, a store that swaps its embedder, or the
/// MCP server — cannot fail on the second build.
///
/// The `OnceLock` holds the `Result`, not the `Environment`: `get_or_init` runs
/// the initializer on exactly one thread and every other caller blocks in
/// `get_or_init` until that call returns, so `build` is never called twice
/// concurrently (and a failed build is latched rather than retried — the
/// message keeps the `ort` `ErrorCode`, which `Display` alone drops).
fn ort_environment() -> Result<Environment, Error> {
    static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();
    let environment = ENVIRONMENT.get_or_init(|| {
        // Telemetry is off: vipune is a local CLI, and the builder would
        // otherwise turn ONNX Runtime's telemetry on for every run.
        ort::init()
            .with_telemetry(false)
            .build()
            .map_err(|e| format!("[{:?}] {}", e.code(), e))
    });
    match environment {
        Ok(environment) => Ok(environment.clone()),
        Err(message) => Err(Error::Onnx {
            message: format!("ort environment: {message}"),
        }),
    }
}

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
/// The role-aware counter [`EmbeddingEngine::token_count`] counts the
/// *prefixed* text through the same prefix path as the embedders, so a
/// pre-flight count and the actual embed always agree.
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
    /// The environment this engine's `session` was built against. Never read
    /// after construction — it exists so the ORT env outlives the session:
    /// the field drops *after* `session` (declaration order), and the shared
    /// handle keeps the last `ReleaseEnv` from running while any session lives.
    /// Declared before `session` for exactly that reason. Belt-and-braces, not
    /// the only keep-alive: rc.15 `Session`s hold an `Environment` clone of
    /// their own, so the ORT env already outlives every session built from it.
    _environment: Environment,
    pub(super) session: Session,
    pub(super) tokenizer: Tokenizer,
    /// Truncation-free tokenizer cloned once at startup for accurate token counting.
    count_tokenizer: Tokenizer,
    pub(super) requires_token_type_ids: bool,
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
    /// Files are cached locally in the HF Hub cache, only downloaded once:
    /// `$HF_HOME/hub` when `HF_HOME` is set to a non-empty value, otherwise
    /// `~/.cache/huggingface/hub` (see [`resolve_hf_cache_dir`], which is
    /// vipune's own resolver — hf-hub's `ApiBuilder::new()` ignores `HF_HOME`).
    /// Because the cache dir moves with `HF_HOME`, hf-hub then reads the
    /// access token from `$HF_HOME/token` (its own convention).
    pub fn new(model_id: &str) -> Result<Self, Error> {
        use crate::embedding_profiles::profile_for;

        // Resolve through the profile table: unknown ids are rejected here, and
        // every profile carries its exact pinned revision.
        let profile = profile_for(model_id)?;

        // Build/latch the process-wide ORT environment (see
        // [`ort_environment`]) BEFORE touching the Hub: an environment failure
        // is deterministic (rc.15 refuses a second environment in a process
        // that already has one), so it must surface first instead of behind a
        // multi-megabyte model download that is pure waste on a doomed call.
        let environment = ort_environment()?;

        // hf-hub's ApiBuilder::new() ignores HF_HOME, so the cache directory
        // is resolved here and handed to the client explicitly. The home
        // directory is obtained the same way Config::default() does it.
        let home = home_dir();
        let cache_dir = resolve_hf_cache_dir(std::env::var("HF_HOME").ok().as_deref(), &home);
        let air_gapped_hint = format!(
            "If running in an air-gapped environment, pre-fetch the model before going offline:\n  huggingface-cli download {} --revision {} --cache-dir {}",
            profile.model_id,
            profile.revision,
            cache_dir.display()
        );

        // Same air-gapped remedy as the download sites below: a builder
        // failure is a Hub/network failure too, so it must not arrive as a
        // bare `HfHub` message that names no remedy.
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .build()
            .map_err(|e| {
                Error::Config(format!(
                    "Failed to initialise the HuggingFace Hub client: {e}.\n\n{air_gapped_hint}"
                ))
            })?;

        let repo = api.repo(Repo::with_revision(
            profile.model_id.to_string(),
            RepoType::Model,
            profile.revision.to_string(),
        ));

        // Helper function for error messaging (every profile is revision-pinned,
        // so the hint always carries `--revision`). Borrows the shared hint, so
        // it stays callable for both download sites below.
        let wrap_download_err = |e: hf_hub::api::sync::ApiError| {
            Error::Config(format!(
                "Failed to download embedding model '{}': {}.\n\n{air_gapped_hint}",
                profile.model_id, e
            ))
        };

        let model_path = repo
            .get(profile.onnx_path)
            .or_else(|_| repo.get("onnx/model.onnx"))
            .or_else(|_| repo.get("model.onnx"))
            .map_err(wrap_download_err)?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(wrap_download_err)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::from)?;
        tokenizer
            .with_padding(None)
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(Error::from)?;

        let session = Session::builder(&environment)
            .map_err(Error::from)?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            // rc.15 returns the builder back on failure (`Error<SessionBuilder>`);
            // `From<Error<SessionBuilder>> for Error<()>` discards the recoverable
            // builder and keeps the message, so the error still reaches the caller
            // (a `recover()` here would swallow it).
            .map_err(|e| Error::from(ort::error::Error::from(e)))?
            .commit_from_file(&model_path)
            .map_err(Error::from)?;

        // Check if model requires token_type_ids input
        let requires_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        // Pre-initialize a truncation-free tokenizer for accurate token counting.
        // Cloning once at startup avoids a 700KB+ deep copy on every embed() call.
        let mut count_tokenizer = tokenizer.clone();
        count_tokenizer.with_truncation(None).map_err(Error::from)?;

        Ok(EmbeddingEngine {
            _environment: environment,
            session,
            tokenizer,
            count_tokenizer,
            requires_token_type_ids,
            profile,
        })
    }

    /// Count the tokens the engine would see for `text` with the given role's
    /// prefix applied (the same prefix path the embedders use), so a
    /// pre-flight count and the actual embed always agree.
    ///
    /// Pass [`EmbeddingRole::Passage`] for stored content (added, updated, or
    /// re-indexed text) and [`EmbeddingRole::Query`] for search text. Pass the
    /// stored, unprefixed text — the role's prefix is applied here.
    ///
    /// Uses a separate truncation-free tokenizer (pre-cloned at startup) so the
    /// true count is returned. The main tokenizer has truncation at 512 enabled
    /// for safe inference, but that would silently cap counts and make the
    /// `ContentTooLong` guard in the embed methods unreachable.
    pub fn token_count(&self, role: EmbeddingRole, text: &str) -> Result<usize, Error> {
        let input = self.prefix_input(role, text);
        self.count_tokens(&input)
    }

    /// Embed stored-content (passage) text: the profile's passage prefix is
    /// prepended before tokenisation. Use for content being added, updated,
    /// or re-indexed.
    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        self.embed_role(text, EmbeddingRole::Passage)
    }

    /// Embed search (query) text: the profile's query prefix is prepended
    /// before tokenisation. Use for search and classification text.
    ///
    /// Counterpart of [`Self::embed_passage`] for the query role (the role-
    /// less `embed` has been removed).
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        self.embed_role(text, EmbeddingRole::Query)
    }

    /// The engine input for `text` with the role's profile prefix applied.
    ///
    /// This is the single prefix-application helper: the role-aware embed and
    /// token-count methods all route through it, so the prefixed input the
    /// model sees is exactly `prefix + text` (the bare text when the profile
    /// declares no prefix, e.g. bge).
    ///
    /// Crate-visible (not public) so library callers cannot bypass the
    /// empty-input short-circuit and the `ContentTooLong` check in
    /// `embed_role`.
    pub(crate) fn prefix_input(&self, role: EmbeddingRole, text: &str) -> String {
        let prefix = role.prefix(self.profile);
        if prefix.is_empty() {
            text.to_string()
        } else {
            format!("{prefix}{text}")
        }
    }

    /// Shared prefix-aware embed path (crate-internal: called from
    /// `embed_passage` / `embed_query`). Prepends the role's profile prefix,
    /// enforces the token limit on the prefixed text, then runs the model.
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
    fn embed_role(&mut self, text: &str, role: EmbeddingRole) -> Result<Vec<f32>, Error> {
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

        inference::encode_and_infer(self, &input)
    }

    fn count_tokens(&self, input: &str) -> Result<usize, Error> {
        // `Error::from` is the single boundary a `tokenizers::Error` crosses;
        // the call-site context is prefixed afterwards, so the underlying
        // Display text is carried through that one conversion.
        let encoding = self
            .count_tokenizer
            .encode(input, true)
            .map_err(|e| with_prefix("count tokenizer:")(Error::from(e)))?;
        Ok(encoding.get_ids().len())
    }
}
