//! Error types for vipune.
//!
//! The public `vipune::Error` enum is vipune-owned end to end: no variant carries a
//! third-party error type, so upgrading `rusqlite`, `tokenizers`, `ort`,
//! `hf_hub`, `serde_json` or `chrono` cannot change the public API. Third-party
//! errors are flattened into a vipune-owned message string at the boundary
//! (their `Display` text is preserved; they are the documented exceptions to
//! the `source()` chain). Errors vipune owns keep their structured fields —
//! [`Error::CorruptEmbedding`], [`Error::InvalidBlobSize`],
//! [`Error::MismatchedDimensions`], the migration variants — so callers read
//! them instead of parsing prose. The enum is `#[non_exhaustive]`, so adding a
//! variant later is not a breaking change.
//!
//! Two internal conversions do exist — [`From<rusqlite::Error>`] and
//! [`From<crate::sqlite::Error>`] — because the storage layer sits on both
//! sides of the boundary; neither leaks its argument type into a variant.
//!
//! A locked database is the dedicated `DatabaseLocked` variant, testable with
//! `Error::is_locked()` — the replacement for matching the Display text
//! "database is locked".

use thiserror::Error;

/// Actionable hint appended to a locked/busy database error (issue #255).
///
/// A busy database is almost always the MCP server holding a write lock, so
/// the hint names the remedy. Library consumers normally branch on
/// [`Error::is_locked`] and render their own text.
pub const LOCKED_DB_HINT: &str =
    "Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.";

/// The full user-facing text command handlers use for a locked database.
///
/// The CLI prints `Error: {}`, so this string is the user-visible contract for
/// every busy-database path. `concat!` cannot splice a `&str` const, so the
/// composition is spelled out here rather than derived from
/// [`Error::DatabaseLocked`]'s `#[error]` attribute: the two literals live in
/// separate places and the equality is enforced by
/// `errors_public_api_tests::locked_database_message_stays_byte_identical_to_the_cli_text`.
pub const LOCKED_DB_MESSAGE: &str = "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.";

/// Prefix a [`Error::Tokenization`] message with call-site context.
///
/// The two tokenizer call sites (`embedding/engine.rs`, `embedding/inference.rs`)
/// already convert a third-party failure through [`From<tokenizers::Error>`]
/// and want to say *which* tokenizer failed without reaching around that
/// conversion to build the variant by hand. Crate-internal on purpose: the
/// variant-matching behaviour would be a permanent public commitment otherwise.
pub(crate) fn with_prefix(prefix: &str) -> impl Fn(Error) -> Error + '_ {
    move |err| match err {
        Error::Tokenization { message } => Error::Tokenization {
            message: format!("{prefix} {message}"),
        },
        other => other,
    }
}

/// Main error type for vipune operations.
///
/// `#[non_exhaustive]`: match it with a wildcard arm. No variant type is
/// third-party; see the [module docs](self) for the source() contract.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Database error. Replaces the old `SQLite(rusqlite::Error)` and
    /// `SqliteModule(String)` variants: the message is vipune-owned and the
    /// underlying message is included in it rather than dropped.
    ///
    /// The `Database error:` prefix is a match target for consumers outside
    /// this repo (the pi-rukas `ctx7` wrapper greps the CLI's stderr for the
    /// framed text), so it is pinned by
    /// `errors_public_api_tests::database_display_prefix_is_pinned_because_classification_depends_on_it`.
    /// Note a lock must never be routed through this variant: it renders like
    /// the busy text but [`Error::is_locked`] reads the variant, so such a
    /// lock would be invisible to both the typed API and the stderr matchers.
    #[error("Database error: {message}")]
    Database { message: String },

    /// The database is locked or busy (`SQLITE_BUSY` / `SQLITE_LOCKED`).
    ///
    /// Match this variant (or call [`Error::is_locked`]) instead of matching
    /// the Display text. The actionable MCP-server hint lives in
    /// [`LOCKED_DB_HINT`], outside the message, so the message can be used as
    /// a regression fixture while the remedy text stays editable.
    #[error("Database is locked. {LOCKED_DB_HINT}")]
    DatabaseLocked,

    /// ONNX inference error.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Tokenization error.
    #[error("Tokenization error: {message}")]
    Tokenization { message: String },

    /// HuggingFace Hub failure. Rendered by the `From<hf_hub::api::sync::ApiError>`
    /// impl below — the download paths in `embedding/engine.rs` convert
    /// `ApiError` through that impl (or re-render it as `Config` with the
    /// air-gapped hint), so this is the shape a Hub failure takes when it is
    /// not given the download hint.
    #[error("HuggingFace Hub error: {message}")]
    HfHub { message: String },

    /// ONNX session failure. Rendered by the `From<ort::error::Error>` impl
    /// below — the inference path converts `ort::error::Error` through it.
    #[error("ONNX session error: {message}")]
    Onnx { message: String },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

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

    /// Validation error (for parameter validation).
    #[error("Validation error: {0}")]
    Validation(String),

    /// Embedding BLOB for a specific memory failed to decode.
    ///
    /// Carries the affected memory `id` so library consumers can locate the
    /// corrupt row instead of parsing a message (issue #186; carried through
    /// the public boundary instead of flattening, issue #255).
    ///
    /// Unlike [`Error::NotFound`] — whose id is deliberately stripped by the
    /// storage conversion — this id *is* rendered into CLI and MCP output: a
    /// corrupt row is unlocatable without it, which is the point of #186. The
    /// ids are the same UUIDs `list`/`search` already return, so naming one
    /// discloses nothing the caller cannot already enumerate. That argument
    /// covers the `id` only: `reason` is the decoder's own message and a
    /// hostile DB file can influence what it says, so it is not held to the
    /// same "already enumerable" standard.
    #[error("Corrupt embedding for memory id: {id} ({reason})")]
    CorruptEmbedding { id: String, reason: String },

    /// Embedding BLOB has unexpected size.
    #[error("Invalid BLOB size: expected {expected} bytes, got {actual} bytes")]
    InvalidBlobSize { expected: usize, actual: usize },

    /// Embedding vector dimensions do not match model dimensions.
    #[error("Mismatched dimensions: expected {expected} dimensions, got {actual} dimensions")]
    MismatchedDimensions { expected: usize, actual: usize },

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

impl Error {
    /// True when this error means the database was locked or busy, i.e.
    /// another process holds a lock.
    ///
    /// Pure variant test — the programmatic replacement for matching the
    /// Display text "database is locked". The classification happens once, at
    /// the storage boundary (`sqlite::classify_db_error`), from the SQLite
    /// result code, and travels as the [`Error::DatabaseLocked`] variant: no
    /// arm here inspects a message, so store- or dependency-controlled text
    /// can never steer this decision (issue #255).
    #[must_use]
    pub fn is_locked(&self) -> bool {
        matches!(self, Self::DatabaseLocked)
    }
}

impl From<rusqlite::Error> for Error {
    /// Maps a raw rusqlite error at the boundary: the vipune-owned
    /// `CorruptEmbedding` domain error that `sqlite::query_mod` attaches as
    /// the source of a row-conversion failure is unwrapped to its structured
    /// public variant, and a locked/busy error becomes
    /// [`Error::DatabaseLocked`]. Everything else is flattened to a message,
    /// so `rusqlite::Error` never becomes part of the public API.
    fn from(err: rusqlite::Error) -> Self {
        match crate::sqlite::db_error::classify_db_error(&err) {
            crate::sqlite::db_error::DbErrorKind::Locked => Self::DatabaseLocked,
            crate::sqlite::db_error::DbErrorKind::Domain(domain) => (*domain).clone().into(),
            crate::sqlite::db_error::DbErrorKind::Other => Self::Database {
                message: err.to_string(),
            },
        }
    }
}

impl From<crate::sqlite::Error> for Error {
    fn from(err: crate::sqlite::Error) -> Self {
        match err {
            // Sanitize: don't leak memory IDs in error messages to library consumers.
            // The inner message (e.g. the UUID) is stripped; callers see only a generic hint.
            crate::sqlite::Error::NotFound(_) => Error::NotFound("memory not found".to_string()),
            // Preserve the InvalidInput message so validation context is not lost.
            crate::sqlite::Error::InvalidInput(msg) => Error::InvalidInput(msg),
            // Input-domain predicates, not storage faults: the callers pass
            // the vector/limit that produced them, so they are reported as
            // invalid input rather than under a variant that asserts a
            // database failure.
            crate::sqlite::Error::EmptyVector => Error::InvalidInput(err.to_string()),
            crate::sqlite::Error::InvalidEmbedding(msg) => Error::InvalidInput(msg),
            crate::sqlite::Error::InvalidLimit(msg) => Error::InvalidInput(msg),
            // Structured embedding failures keep their fields across the
            // public boundary (issue #255).
            crate::sqlite::Error::CorruptEmbedding { id, reason } => {
                Error::CorruptEmbedding { id, reason }
            }
            crate::sqlite::Error::InvalidBlobSize { expected, actual } => {
                Error::InvalidBlobSize { expected, actual }
            }
            crate::sqlite::Error::MismatchedDimensions { expected, actual } => {
                Error::MismatchedDimensions { expected, actual }
            }
            // Structured migration errors: carry the offending ids / report
            // through unchanged.
            crate::sqlite::Error::MigrationRefused { offending } => {
                Error::MigrationRefused { offending }
            }
            crate::sqlite::Error::MigrationIncomplete { report } => {
                Error::MigrationIncomplete { report }
            }
            // A locked/busy database is its own variant so callers never have
            // to match the message (issue #255).
            crate::sqlite::Error::Locked => Error::DatabaseLocked,
            crate::sqlite::Error::Sqlite(msg) => Error::Database { message: msg },
        }
    }
}

impl From<tokenizers::Error> for Error {
    fn from(err: tokenizers::Error) -> Self {
        Self::Tokenization {
            message: err.to_string(),
        }
    }
}

impl From<ort::error::Error> for Error {
    fn from(err: ort::error::Error) -> Self {
        Self::Onnx {
            message: err.to_string(),
        }
    }
}

impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(err: hf_hub::api::sync::ApiError) -> Self {
        Self::HfHub {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for Error {
    /// Flattened onto `InvalidInput`: a serde failure is always a bad payload
    /// the caller controls (an invalid `--metadata` document, an unserialisable
    /// response), which is what that variant means. A dedicated `Json` variant
    /// had no construction site, so it would have been public surface for a
    /// shape nothing produces.
    fn from(err: serde_json::Error) -> Self {
        Self::InvalidInput(err.to_string())
    }
}

impl From<chrono::ParseError> for Error {
    /// Flattened onto `InvalidInput` for the same reason as the serde
    /// conversion: a timestamp that will not parse is bad input.
    fn from(err: chrono::ParseError) -> Self {
        Self::InvalidInput(err.to_string())
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
    fn sqlite_other_errors_convert_to_database_with_their_message() {
        // The old `SqliteModule(String)` rendered the bare "Database error"
        // literal; the replacement must keep the underlying message.
        let sqlite_err = crate::sqlite::Error::Sqlite("disk I/O error".to_string());
        let err: Error = sqlite_err.into();
        match err {
            Error::Database { message } => {
                assert_eq!(message, "disk I/O error");
                assert!(
                    message.contains("disk I/O error"),
                    "Display must keep the underlying message, got: {message}"
                );
            }
            other => panic!("Expected Database, got {other:?}"),
        }
    }

    #[test]
    fn database_display_keeps_the_underlying_message() {
        // Regression for the old `#[error("Database error")]` that dropped the
        // rusqlite message entirely.
        let err = Error::Database {
            message: "no such table: memories".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("no such table: memories"),
            "Display must carry the underlying message, got: {msg}"
        );
        assert!(msg.contains("Database error"), "got: {msg}");
    }

    #[test]
    fn locked_errors_convert_to_the_dedicated_locked_variant() {
        // Both producers of locked-database errors — the storage layer's
        // dedicated `Locked` variant and the direct rusqlite path — must land
        // on `DatabaseLocked`, never on a message a caller would have to
        // substring-match.
        let from_sqlite: Error = crate::sqlite::Error::Locked.into();
        assert!(
            matches!(from_sqlite, Error::DatabaseLocked),
            "sqlite::Error::Locked must be classified locked, got {from_sqlite:?}"
        );
        assert!(from_sqlite.is_locked());

        let busy = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".to_string()),
        );
        let from_rusqlite: Error = busy.into();
        assert!(
            matches!(from_rusqlite, Error::DatabaseLocked),
            "a raw rusqlite SQLITE_BUSY must map to DatabaseLocked, got {from_rusqlite:?}"
        );
        assert!(from_rusqlite.is_locked());
    }

    #[test]
    fn locked_display_keeps_underlying_text_and_hint() {
        let err = Error::DatabaseLocked;
        let msg = err.to_string();
        assert!(msg.starts_with("Database is locked."), "got: {msg}");
        assert!(msg.contains("MCP server"), "got: {msg}");
        assert!(err.is_locked(), "DatabaseLocked is always locked");
    }
}

#[cfg(test)]
#[path = "errors_public_api_tests.rs"]
mod errors_public_api_tests;
