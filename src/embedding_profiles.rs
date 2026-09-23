//! Built-in embedding model profiles.
//!
//! A profile declares everything that is model-specific: the HuggingFace model
//! id, the exact pinned revision, the ONNX file path inside the repo, and the
//! query/passage prefixes the model requires. Prefixes are a property of the
//! model (e5-style models need "query: " / "passage: " in the input for the
//! model to work well), not a user setting — vipune applies them automatically
//! at the embedding chokepoint, so stored content and search input never carry
//! them.
//!
//! Every profile is pinned to an exact revision. No profile ever loads a
//! floating revision.

use crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION, EMBEDDING_DIMS};
use crate::errors::Error;

/// Built-in embedding model profiles.
pub const BUILTIN_PROFILES: &[ModelProfile] = &[
    ModelProfile {
        // Default profile: the id and revision are the single source of truth
        // (`EMBED_MODEL_ID` / `EMBED_MODEL_REVISION` in `crate::embedding`),
        // which keeps this table and the config default from drifting apart.
        model_id: EMBED_MODEL_ID,
        revision: EMBED_MODEL_REVISION,
        onnx_path: "onnx/model.onnx",
        query_prefix: "",
        passage_prefix: "",
    },
    ModelProfile {
        model_id: "intfloat/multilingual-e5-small",
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        onnx_path: "onnx/model.onnx",
        query_prefix: "query: ",
        passage_prefix: "passage: ",
    },
];

/// A built-in embedding model profile.
///
/// Declares the model id, the exact pinned revision (never floating), the ONNX
/// file path inside the HuggingFace repo, the query/passage prefixes the model
/// requires, and the embedding dimension (384 for every built-in profile).
#[derive(Debug)]
pub struct ModelProfile {
    /// HuggingFace model id (e.g. `BAAI/bge-small-en-v1.5`).
    pub model_id: &'static str,
    /// Pinned commit SHA on HuggingFace for reproducible downloads.
    pub revision: &'static str,
    /// Path to the ONNX model file inside the HuggingFace repo.
    pub onnx_path: &'static str,
    /// Prefix applied to search queries before embedding. Empty for models
    /// that take bare queries (bge).
    pub query_prefix: &'static str,
    /// Prefix applied to stored content before embedding. Empty for models
    /// that take bare text (bge).
    pub passage_prefix: &'static str,
}

impl ModelProfile {
    /// Embedding dimension for this profile.
    ///
    /// All built-in profiles are 384-dimensional; this method is the single
    /// place callers should read the dimension from, so a future profile with
    /// different geometry can declare it here.
    pub fn dims(&self) -> usize {
        EMBEDDING_DIMS
    }
}

/// The default built-in profile: bge-small-en-v1.5 at its pinned revision.
///
/// A database with no recorded model identity is treated as this profile at
/// its pinned revision.
pub fn default_profile() -> &'static ModelProfile {
    &BUILTIN_PROFILES[0]
}

/// The role an embedding is computed for.
///
/// Some models require different input prefixes depending on whether the
/// text is a stored passage or a search query (e5-style models). The role is
/// declared explicitly at the embedding chokepoint so the right prefix is
/// applied before `EmbeddingEngine::embed` is called; the prefixes are a
/// property of the model profile, never of the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRole {
    /// The text is content being stored (add / update / re-index).
    Passage,
    /// The text is a search query.
    Query,
}

impl EmbeddingRole {
    /// The model prefix to prepend for this role (empty string when the model
    /// uses no prefixes — e.g. bge). Prefixes are applied by callers at the
    /// embedding chokepoint, never inside `EmbeddingEngine::embed`.
    pub fn prefix(self, profile: &ModelProfile) -> &'static str {
        match self {
            EmbeddingRole::Passage => profile.passage_prefix,
            EmbeddingRole::Query => profile.query_prefix,
        }
    }
}

/// List the available built-in profile ids, for error messages.
pub fn available_profile_ids() -> Vec<&'static str> {
    BUILTIN_PROFILES.iter().map(|p| p.model_id).collect()
}

/// Look up a built-in profile by model id.
///
/// Returns the profile if the id names a built-in profile, or an
/// `Error::Config` naming the configured id and listing every available
/// built-in id if it does not. Unknown model ids are rejected — no arbitrary
/// HuggingFace repo may be loaded (that path would silently fall back to a
/// floating revision).
pub fn profile_for(model_id: &str) -> Result<&'static ModelProfile, Error> {
    BUILTIN_PROFILES
        .iter()
        .find(|p| p.model_id == model_id)
        .ok_or_else(|| {
            let available = available_profile_ids().join(", ");
            Error::Config(format!(
                "Unknown embedding model '{model_id}'. Available built-in models: {available}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bge_profile_declares_expected_values() {
        let p = profile_for("BAAI/bge-small-en-v1.5").expect("bge profile");
        assert_eq!(p.model_id, "BAAI/bge-small-en-v1.5");
        assert_eq!(p.revision, "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a");
        assert_eq!(p.onnx_path, "onnx/model.onnx");
        assert_eq!(p.query_prefix, "");
        assert_eq!(p.passage_prefix, "");
        assert_eq!(p.dims(), 384);
    }

    #[test]
    fn e5_profile_declares_expected_values() {
        let p = profile_for("intfloat/multilingual-e5-small").expect("e5 profile");
        assert_eq!(p.model_id, "intfloat/multilingual-e5-small");
        assert_eq!(p.revision, "614241f622f53c4eeff9890bdc4f31cfecc418b3");
        assert_eq!(p.onnx_path, "onnx/model.onnx");
        assert_eq!(p.query_prefix, "query: ");
        assert_eq!(p.passage_prefix, "passage: ");
        assert_eq!(p.dims(), 384);
    }

    #[test]
    fn unknown_model_id_is_rejected_listing_available_profiles() {
        let err = profile_for("someone/else-model").unwrap_err();
        let msg = match err {
            Error::Config(msg) => msg,
            other => panic!("expected Error::Config, got {other:?}"),
        };
        assert!(
            msg.contains("someone/else-model"),
            "error must name the bad id: {msg}"
        );
        assert!(
            msg.contains("BAAI/bge-small-en-v1.5"),
            "error must list the bge profile: {msg}"
        );
        assert!(
            msg.contains("intfloat/multilingual-e5-small"),
            "error must list the e5 profile: {msg}"
        );
    }

    #[test]
    fn no_profile_uses_a_floating_revision() {
        for p in BUILTIN_PROFILES {
            assert!(
                p.revision != "main" && !p.revision.is_empty(),
                "profile {} must be pinned to an exact revision, not {}",
                p.model_id,
                p.revision
            );
        }
    }

    #[test]
    fn default_profile_is_bge() {
        assert_eq!(default_profile().model_id, "BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn available_profile_ids_lists_all_builtins() {
        let ids = available_profile_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"BAAI/bge-small-en-v1.5"));
        assert!(ids.contains(&"intfloat/multilingual-e5-small"));
    }
}
