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

    /// Content exceeds maximum embedding token limit.
    #[error("content exceeds {max_tokens}-token embedding limit (measured: {token_count} tokens)")]
    ContentTooLong {
        token_count: usize,
        max_tokens: usize,
    },

    /// Invalid timestamp in database record.
    #[error("Invalid timestamp format: {timestamp} ({error})")]
    InvalidTimestamp { timestamp: String, error: String },

    /// Memory not found.
    #[error("Memory not found: {0}")]
    NotFound(String),

    /// SQLite module error (from sqlite::Error).
    #[error("Database error")]
    SqliteModule(String),

    /// Validation error (for parameter validation).
    #[error("Validation error: {0}")]
    Validation(String),

    /// Embedder unavailable — model download failed, cache corrupt, or offline.
    /// Used by the MCP server which wraps errors with context before returning.
    /// Not constructed directly by the CLI path (which returns the inner Error::Config
    /// with its offline hint intact), hence the allow.
    #[allow(dead_code)]
    #[error("Embedder unavailable: {reason}")]
    EmbedderUnavailable { reason: String },
}

impl From<crate::sqlite::Error> for Error {
    fn from(err: crate::sqlite::Error) -> Self {
        match err {
            // Sanitize: don't leak memory IDs in error messages to library consumers.
            // The inner message (e.g. the UUID) is stripped; callers see only a generic hint.
            crate::sqlite::Error::NotFound(_) => Error::NotFound("memory not found".to_string()),
            // Preserve the InvalidInput message so validation context is not lost.
            crate::sqlite::Error::InvalidInput(msg) => Error::InvalidInput(msg),
            _ => Error::SqliteModule(err.to_string()),
        }
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::InvalidInput(s)
    }
}

#[cfg(test)]
mod error_conversion_tests {
    use super::*;

    #[test]
    fn sqlite_not_found_converts_to_error_not_found() {
        // The message content varies across call sites (UUID, "No memory found for...", etc.)
        // The conversion must NOT depend on message text — only on the variant.
        let sqlite_err = crate::sqlite::Error::NotFound("any-arbitrary-message".to_string());
        let err: Error = sqlite_err.into();
        assert!(
            matches!(err, Error::NotFound(_)),
            "sqlite::Error::NotFound must convert to Error::NotFound regardless of message text"
        );
        // Confirm sanitisation: the original message is NOT leaked
        let Error::NotFound(msg) = err else {
            unreachable!()
        };
        assert_eq!(msg, "memory not found");
    }

    #[test]
    fn sqlite_invalid_input_converts_to_error_invalid_input() {
        let sqlite_err =
            crate::sqlite::Error::InvalidInput("At least one field must be provided".to_string());
        let err: Error = sqlite_err.into();
        match err {
            Error::InvalidInput(msg) => {
                assert_eq!(msg, "At least one field must be provided");
            }
            other => panic!("Expected InvalidInput, got {:?}", other),
        }
    }

    #[test]
    fn sqlite_other_errors_convert_to_sqlite_module() {
        let sqlite_err = crate::sqlite::Error::Sqlite("disk I/O error".to_string());
        let err: Error = sqlite_err.into();
        assert!(matches!(err, Error::SqliteModule(_)));
    }
}
