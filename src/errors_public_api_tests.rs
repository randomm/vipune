//! Public-API-surface tests for `vipune::Error` (issue #255).
//!
//! Lives beside `src/errors.rs` (included with `#[path]`) rather than under
//! `tests/`: asserting the shape of the public enum requires naming the
//! third-party error types, and the integration tests cannot — `Cargo.toml`
//! dev-dependencies are only criterion/tempfile/tokio, so adding rusqlite or
//! tokenizers there would be exactly the coupling this change removes.

use super::*;

#[test]
fn public_error_exposes_no_third_party_types() {
    // No third-party error type may appear in the public API, so upgrading
    // rusqlite / tokenizers / ort / hf_hub / serde_json / chrono cannot break
    // a library consumer.
    //
    // The check is a type-level one, which is the only form that actually
    // fails: each `fn` below names a public constructor's exact shape. If a
    // variant ever stores a third-party type again — the old
    // `SQLite(#[from] rusqlite::Error)`, for instance — the vipune-owned
    // message payload it replaces is gone and the assignment stops compiling
    // with a mismatched-function-pointer error. The shape stays pinned the
    // other way too: renaming `Database { message }` fails here as well.
    //
    // `Error::Io(std::io::Error)` is retained deliberately: std is not a
    // third-party dependency.
    let _: fn(String) -> Error = |message| Error::Database { message };
    let _: fn() -> Error = || Error::DatabaseLocked;
    let _: fn(String) -> Error = |message| Error::Tokenization { message };
    let _: fn(String) -> Error = |message| Error::Onnx { message };
    let _: fn(String) -> Error = |message| Error::HfHub { message };
    let _: fn(String) -> Error = |message| Error::Json { message };
    let _: fn(String) -> Error = |message| Error::Chrono { message };
    let _: fn(String, String) -> Error = |id, reason| Error::CorruptEmbedding { id, reason };
    let _: fn(usize, usize) -> Error =
        |expected, actual| Error::InvalidBlobSize { expected, actual };
    let _: fn(usize, usize) -> Error =
        |expected, actual| Error::MismatchedDimensions { expected, actual };

    // And the enum is `#[non_exhaustive]`, so consumers must carry a wildcard
    // arm: this match compiles only because of that attribute.
    let err = Error::DatabaseLocked;
    let classified = match err {
        Error::DatabaseLocked => "locked",
        Error::Database { .. } => "database",
        _ => "other",
    };
    assert_eq!(classified, "locked");
}

#[test]
fn third_party_conversions_carry_the_message() {
    // The internal boundary conversions keep the underlying text — the old
    // `#[error("Database error")]` dropped it entirely — while dropping the
    // third-party type.
    let busy = rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE),
        Some("UNIQUE constraint failed: memories.id".to_string()),
    );
    let err: Error = busy.into();
    match err {
        Error::Database { message } => {
            assert!(
                message.contains("UNIQUE constraint failed"),
                "underlying message must survive, got: {message}"
            );
        }
        other => panic!("Expected Database, got {other:?}"),
    }

    let json = serde_json::from_str::<serde_json::Value>("{oops").unwrap_err();
    let err: Error = json.into();
    match err {
        Error::Json { message } => assert!(!message.is_empty(), "message must not be empty"),
        other => panic!("Expected Json, got {other:?}"),
    }
}

#[test]
fn locked_database_message_stays_byte_identical_to_the_cli_text() {
    // The CLI prints `Error: {}`, so this wording is the user-visible contract
    // for every busy-database path. Pinning it here lets the locked-DB command
    // tests assert the structured variant instead of re-parsing the string,
    // and a wording change fails loudly.
    let err = Error::DatabaseLocked;
    assert_eq!(
        err.to_string(),
        "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry."
    );
    assert_eq!(LOCKED_DB_MESSAGE, err.to_string());
}

#[test]
fn plain_database_error_with_busy_message_is_still_reported_locked() {
    // The producers that flatten a database failure into a message (the
    // identity reads in `sqlite/identity.rs` and `commands/import.rs`) must
    // not regress to an unclassified error.
    let err = Error::Database {
        message: "Database error: database is locked".to_string(),
    };
    assert!(err.is_locked(), "got: {err:?}");

    // …and a genuine non-busy failure is not.
    let other = Error::Database {
        message: "disk I/O error".to_string(),
    };
    assert!(!other.is_locked());
}

#[test]
fn structured_embedding_errors_survive_the_public_conversion() {
    // These three used to flatten into `SqliteModule(String)` and lose their
    // fields on the way to a library consumer.
    let corrupt: Error = crate::sqlite::Error::CorruptEmbedding {
        id: "mem-1".to_string(),
        reason: "not enough bytes".to_string(),
    }
    .into();
    match corrupt {
        Error::CorruptEmbedding { id, reason } => {
            assert_eq!(id, "mem-1");
            assert_eq!(reason, "not enough bytes");
        }
        other => panic!("Expected CorruptEmbedding, got {other:?}"),
    }

    let blob: Error = crate::sqlite::Error::InvalidBlobSize {
        expected: crate::embedding::EMBEDDING_DIMS * 4,
        actual: 4,
    }
    .into();
    match blob {
        Error::InvalidBlobSize { expected, actual } => {
            assert_eq!(expected, crate::embedding::EMBEDDING_DIMS * 4);
            assert_eq!(actual, 4);
        }
        other => panic!("Expected InvalidBlobSize, got {other:?}"),
    }

    let dims: Error = crate::sqlite::Error::MismatchedDimensions {
        expected: 384,
        actual: 768,
    }
    .into();
    match dims {
        Error::MismatchedDimensions { expected, actual } => {
            assert_eq!((expected, actual), (384, 768));
        }
        other => panic!("Expected MismatchedDimensions, got {other:?}"),
    }
}
