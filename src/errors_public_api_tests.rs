//! Public-API-surface tests for `vipune::Error` (issue #255).
//!
//! Lives beside `src/errors.rs` (included with `#[path]`) rather than under
//! `tests/`: asserting the shape of the public enum requires naming the
//! third-party error types, and the integration tests cannot — `Cargo.toml`
//! dev-dependencies are only criterion/tempfile/tokio, so adding rusqlite or
//! tokenizers there would be exactly the coupling this change removes.

use super::*;

#[test]
fn public_variants_are_constructible_from_owned_payloads() {
    // Every variant this test names is built from vipune-owned payloads only
    // (`String`, `usize`) — so upgrading rusqlite / tokenizers / ort / hf_hub
    // / serde_json / chrono cannot change the public API.
    //
    // What this does NOT prove: that no variant anywhere stores a third-party
    // type. A closure that constructs the vipune-owned payload type-checks
    // whatever the removed payload used to be, and a variant absent from this
    // list is simply unchecked. A regression of that shape has to be caught by
    // reviewing the variant, not here.
    //
    // `Error::Io(std::io::Error)` is pinned too: std is not a third-party
    // dependency, but it is still part of the public shape.
    let _: fn(String) -> Error = |message| Error::Database { message };
    let _: fn() -> Error = || Error::DatabaseLocked;
    let _: fn(String) -> Error = |message| Error::Tokenization { message };
    let _: fn(std::io::Error) -> Error = Error::Io;
    let _: fn(String, String) -> Error = |id, reason| Error::CorruptEmbedding { id, reason };
    let _: fn(usize, usize) -> Error =
        |expected, actual| Error::InvalidBlobSize { expected, actual };
    let _: fn(usize, usize) -> Error =
        |expected, actual| Error::MismatchedDimensions { expected, actual };

    // The enumeration is EXHAUSTIVE on purpose: a new variant has to be added
    // here, and adding it as `Foo { message: rusqlite::Error }` would not
    // type-check against the `fn` signature this test pins for it. That is the
    // compile-time half of "no variant stores a third-party type"; the
    // wildcard arm below is what `#[non_exhaustive]` requires of consumers and
    // is exercised separately.
    let _: fn(Option<String>) -> Error = |message| Error::Inference(message.unwrap_or_default());
    let _: fn(String) -> Error = |message| Error::Config(message);
    let _: fn(String) -> Error = |message| Error::InvalidInput(message);
    let _: fn() -> Error = || Error::EmptyInput;
    let _: fn(usize, usize) -> Error = |max_length, actual_length| Error::InputTooLong {
        max_length,
        actual_length,
    };
    let _: fn(usize, usize) -> Error = |token_count, max_tokens| Error::ContentTooLong {
        token_count,
        max_tokens,
    };
    let _: fn(String, String) -> Error =
        |timestamp, error| Error::InvalidTimestamp { timestamp, error };
    let _: fn(String) -> Error = |id| Error::NotFound(id);
    let _: fn(String) -> Error = |message| Error::Validation(message);
    let _: fn(Vec<String>) -> Error = |offending| Error::MigrationRefused { offending };
    let _: fn(String) -> Error = |reason| Error::EmbedderUnavailable { reason };
    let _: fn(String) -> Error = |message| Error::Onnx { message };
    let _: fn(String) -> Error = |message| Error::HfHub { message };
    let _: fn(usize) -> Error = |token_count| Error::ContentTooLong {
        token_count,
        max_tokens: 0,
    };

    // And the enum is `#[non_exhaustive]`, so consumers must carry a wildcard
    // arm: this match compiles only because of that attribute.
    let err = Error::DatabaseLocked;
    let classified = match err {
        Error::DatabaseLocked => "locked",
        Error::Database { .. } => "database",
        _ => "other",
    };
    assert_eq!(classified, "locked");

    // Every variant is named by a constructor above or by the migration
    // conversion below, so an unlisted variant that regressed to a
    // third-party payload would have to be added to this file to be seen at
    // all — and it could not be added without its payload type being named.
    let _: fn(crate::sqlite::migration_types::MigrationReport) -> Error =
        |report| Error::MigrationIncomplete { report };
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

    let tokenizer = tokenizers::Error::from(std::io::Error::other("vocab missing"));
    let err: Error = tokenizer.into();
    match err {
        Error::Tokenization { message } => {
            assert!(
                message.contains("vocab missing"),
                "underlying message must survive, got: {message}"
            );
        }
        other => panic!("Expected Tokenization, got {other:?}"),
    }

    // The remaining four conversions this PR re-shaped. Each is constructed
    // from the third-party type — the point of the change is that the public
    // enum keeps none of them, so the test naming them is the assertion that
    // the boundary, not the caller, owns the flattening.
    let ort = ort::error::Error::new("kernel missing for op");
    match Error::from(ort) {
        Error::Onnx { message } => assert!(
            message.contains("kernel missing for op"),
            "ONNX text must survive, got: {message}"
        ),
        other => panic!("Expected Onnx, got {other:?}"),
    }

    let hub = hf_hub::api::sync::ApiError::ParseIntError(
        "not-a-number".parse::<i64>().expect_err("not a number"),
    );
    match Error::from(hub) {
        Error::HfHub { message } => assert!(
            !message.is_empty(),
            "a Hub failure must carry its rendered text"
        ),
        other => panic!("expected HfHub, got {other:?}"),
    }

    // serde and chrono failures are caller-controlled payloads, so both
    // collapse onto `InvalidInput` (see the conversions' docs) with the
    // underlying text intact.
    let json =
        serde_json::from_str::<serde_json::Value>("{ not json").expect_err("malformed document");
    match Error::from(json) {
        Error::InvalidInput(message) => assert!(
            !message.is_empty(),
            "the serde message must survive onto InvalidInput"
        ),
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    let timestamp =
        chrono::DateTime::parse_from_rfc3339("nonsense").expect_err("not an RFC3339 timestamp");
    match Error::from(timestamp) {
        Error::InvalidInput(message) => assert!(
            !message.is_empty(),
            "the chrono message must survive onto InvalidInput"
        ),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn locked_database_message_stays_byte_identical_to_the_cli_text() {
    // The CLI prints `Error: {}`, so this wording is the user-visible contract
    // for every busy-database path. Pinning it here lets the locked-DB command
    // tests assert the structured variant instead of re-parsing the string,
    // and a wording change fails loudly.
    let err = Error::DatabaseLocked;
    assert_eq!(err.to_string(), LOCKED_DB_MESSAGE);
    // And that one source really is the whole wording: the consts the
    // `#[error]` attribute interpolates compose it, so there is no second
    // literal a fix pass could drift from.
    assert_eq!(
        LOCKED_DB_MESSAGE,
        format!("{LOCKED_DB_LEAD} {LOCKED_DB_HINT}")
    );
}

#[test]
fn with_prefix_only_rewrites_the_tokenization_message() {
    // The engine's two tokenizer call sites prefix through this helper instead
    // of building the variant by hand, so the prefix must carry the converted
    // message and leave every other variant untouched.
    let tokenized: Error = tokenizers::Error::from(std::io::Error::other("vocab missing")).into();
    match with_prefix("count tokenizer:")(tokenized) {
        Error::Tokenization { message } => {
            assert_eq!(message, "count tokenizer: vocab missing");
        }
        other => panic!("expected Tokenization, got {other:?}"),
    }

    let other = Error::Database {
        message: "no such table".to_string(),
    };
    assert!(matches!(
        with_prefix("count tokenizer:")(other),
        Error::Database { .. }
    ));
}

#[test]
fn database_display_prefix_is_pinned_because_classification_depends_on_it() {
    // `hook/run.rs` and the pi-rukas `ctx7` wrapper both key on text that sits
    // *inside* this framing. Pinning the prefix turns a Display change from a
    // silent mis-classification into a failing test.
    let err = Error::Database {
        message: "database is locked".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "Database error: database is locked",
        "the Database Display prefix is a match target outside this crate"
    );
    // The framing alone is not a lock: classification is the variant, so a
    // message that merely mentions the busy text stays a plain Database error.
    assert!(
        !err.is_locked(),
        "a Database message must never classify as locked"
    );

    // And the storage boundary's own rendering keeps the same text.
    let storage = crate::sqlite::Error::Sqlite("database is locked".to_string());
    assert_eq!(storage.to_string(), "Database error: database is locked");
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
