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

    /// Pre-flight token scan found rows exceeding the embedding token limit.
    /// Carries the offending memory ids structurally so callers can act
    /// without parsing a message. No marker is written and no rows are
    /// changed when this error is returned.
    #[error("migration refused: {} row(s) exceed the {}-token embedding limit: {}",
        offending.len(), crate::embedding::MAX_EMBEDDING_TOKENS, offending.join(", "))]
    MigrationRefused { offending: Vec<String> },

    /// The re-embed pass ran but at least one row failed to embed.
    /// The migration marker is left in place and the old recorded identity
    /// is kept; re-running after the failures are fixed performs a full pass.
    #[error("migration incomplete: {} row(s) failed to embed; the marker is left in place and the old identity was not replaced — fix the errors and re-run the migration",
        report.failures.len())]
    MigrationIncomplete {
        report: crate::sqlite::migration_types::MigrationReport,
    },

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
            // Structured migration errors: carry the offending ids / report
            // through unchanged.
            crate::sqlite::Error::MigrationRefused { offending } => {
                Error::MigrationRefused { offending }
            }
            crate::sqlite::Error::MigrationIncomplete { report } => {
                Error::MigrationIncomplete { report }
            }
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
    fn migration_refused_message_names_the_token_limit_constant() {
        // The rendered wording must track MAX_EMBEDDING_TOKENS (already
        // interpolated via the #[error] attribute) — this pins "512-token"
        // to the constant so the two cannot drift.
        let err = Error::MigrationRefused {
            offending: vec!["mem-1".to_string()],
        };
        let msg = err.to_string();
        assert!(
            msg.contains(&crate::embedding::MAX_EMBEDDING_TOKENS.to_string()),
            "expected the rendered message to contain the token-limit constant: {msg}"
        );
    }

    #[test]
    fn sqlite_migration_refused_converts_to_migration_refused() {
        let offending = vec!["mem-1".to_string(), "mem-2".to_string()];
        let sqlite_err = crate::sqlite::Error::MigrationRefused {
            offending: offending.clone(),
        };
        let err: Error = sqlite_err.into();
        match err {
            Error::MigrationRefused { offending: ids } => {
                assert_eq!(
                    ids, offending,
                    "offending ids must be carried through unchanged"
                );
            }
            other => panic!("Expected MigrationRefused, got {:?}", other),
        }
    }

    #[test]
    fn sqlite_migration_incomplete_converts_to_migration_incomplete() {
        let report = crate::sqlite::migration_types::MigrationReport {
            reindexed: 3,
            skipped: 0,
            failures: vec![crate::sqlite::migration_types::MigrationRowFailure {
                id: "mem-1".to_string(),
                error: "boom".to_string(),
            }],
        };
        let sqlite_err = crate::sqlite::Error::MigrationIncomplete {
            report: report.clone(),
        };
        let err: Error = sqlite_err.into();
        match err {
            Error::MigrationIncomplete { report: r } => {
                assert_eq!(r, report, "report must be carried through unchanged");
            }
            other => panic!("Expected MigrationIncomplete, got {:?}", other),
        }
    }

    #[test]
    fn sqlite_other_errors_convert_to_sqlite_module() {
        let sqlite_err = crate::sqlite::Error::Sqlite("disk I/O error".to_string());
        let err: Error = sqlite_err.into();
        assert!(matches!(err, Error::SqliteModule(_)));
    }
}
