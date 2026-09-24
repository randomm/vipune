//! Synchronous ONNX embedding engine for text-to-vector conversion.
//!
//! The engine (see [`engine`]) loads one of the built-in model profiles from
//! [`crate::embedding_profiles`] (each pinned to an exact revision) and
//! produces 384-dimensional vectors with mean pooling and L2 normalization.
//! Prefix-awareness and the role-aware API live in [`engine::EmbeddingEngine`];
//! the ONNX run + pooling live in [`inference`].

mod engine;
mod inference;

#[cfg(test)]
mod tests;

// Public items re-exported so `crate::embedding::{...}` paths (and the crate
// root's `pub use embedding::...` list) are unchanged by the module split.
pub use engine::{
    EMBED_MODEL_ID, EMBED_MODEL_REVISION, EMBEDDING_DIMS, EmbeddingEngine, MAX_EMBEDDING_TOKENS,
};

#[cfg(test)]
pub(crate) use inference::l2_normalize;
