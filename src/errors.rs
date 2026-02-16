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

    /// Memory not found.
    #[error("Memory not found: {0}")]
    NotFound(String),

    /// SQLite module error (from sqlite::Error).
    #[error("SQLite module error: {0}")]
    SqliteModule(String),
}

impl From<crate::sqlite::Error> for Error {
    fn from(err: crate::sqlite::Error) -> Self {
        // Convert specific SQLite errors to NotFound when applicable
        let err_str = err.to_string();
        if err_str.contains("No memory found with id:") {
            if let Some(id) = err_str.split("No memory found with id: ").nth(1) {
                return Error::NotFound(id.trim().to_string());
            }
            return Error::NotFound("unknown".to_string());
        }
        Error::SqliteModule(err_str)
    }
}
