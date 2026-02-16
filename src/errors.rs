//! Error types for vipune.

use std::path::PathBuf;

use thiserror::Error;

/// Main error type for vipune operations.
#[derive(Error, Debug)]
#[allow(dead_code)] // Dead code justified: public API for CLI integration
pub enum Error {
    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite error.
    #[error("SQLite error: {0}")]
    SQLite(#[from] rusqlite::Error),

    /// ONNX inference error.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Tokenization error.
    #[error("Tokenization error: {0}")]
    Tokenization(#[from] tokenizers::Error),

    /// ONNX session error.
    #[error("ONNX session error: {0}")]
    Onnx(#[from] ort::Error),

    /// HuggingFace Hub error.
    #[error("HuggingFace Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Invalid date/time.
    #[error("Invalid date/time: {0}")]
    Chrono(#[from] chrono::ParseError),

    /// Invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// ndarray shape error.
    #[error("Array shape error: {0}")]
    Shape(String),
}
