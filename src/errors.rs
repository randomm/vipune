//! Error types for vipune.

use thiserror::Error;

/// Main error type for vipune operations.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite error.
    #[error("Database error")]
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

    /// Empty input cannot be processed.
    #[error("Input cannot be empty")]
    EmptyInput,

    /// Input exceeds maximum allowed length.
    #[error("Input too long: {actual_length} characters (max: {max_length})")]
    InputTooLong {
        max_length: usize,
        actual_length: usize,
    },

    /// Invalid timestamp in database record.
    #[error("Invalid timestamp for memory {id}: {timestamp} ({error})")]
    InvalidTimestamp {
        id: String,
        timestamp: String,
        error: String,
    },

    /// Memory not found.
    #[error("Memory not found: {0}")]
    NotFound(String),

    /// SQLite module error (from sqlite::Error).
    #[error("Database error")]
    SqliteModule(String),

    /// Validation error (for parameter validation).
    #[error("Validation error: {0}")]
    Validation(String),
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

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::InvalidInput(s)
    }
}
