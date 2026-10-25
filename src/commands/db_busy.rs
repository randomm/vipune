//! Locked-database presentation shared by the command handlers (issue #255).
//!
//! Eight handlers touch a possibly-locked store (backup, merge, reindex,
//! prune, export, promote, and the two doctor modes). Before this module each
//! carried its own `wrap_busy` that matched the Display text
//! `"database is locked"`, so a dependency rewording its message would
//! silently turn every fast-fail path into an unclassified database error.
//! They now share [`wrap_busy_error`], which decides on the
//! [`Error::DatabaseLocked`] variant — itself set once, from the SQLite result
//! code, by `sqlite::classify_db_error`. Nothing here reads message text.
//!
//! The helpers take and return the public [`Error`] so no classification
//! decision is made before the storage boundary's own `From` impls have run.
//! The two `rusqlite::Error` call sites (the read-only doctor opens) convert
//! first and then come through the same door.
//!
//! They live here rather than as `Error` methods because their callers are all
//! in the binary target: as library-target items they would read as dead code.

use crate::errors::{Error, LOCKED_DB_MESSAGE};

/// Wrap a result that carries the public error, turning a locked/busy database
/// into the actionable [`LOCKED_DB_MESSAGE`] `Config` error and passing
/// everything else through untouched.
pub(crate) fn wrap_busy_error<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(e) if e.is_locked() => Err(Error::Config(LOCKED_DB_MESSAGE.to_string())),
        Err(e) => Err(e),
    }
}

/// Convert and wrap a storage-layer failure. Thin composition of the public
/// `From` impl (which is where the locked verdict is carried as
/// [`Error::DatabaseLocked`]) over [`wrap_busy_error`].
pub(crate) fn wrap_busy<T>(result: Result<T, crate::sqlite::Error>) -> Result<T, Error> {
    wrap_busy_error(result.map_err(Error::from))
}

/// Convert and wrap a raw `rusqlite::Error` — the shape the read-only
/// diagnostic opens and the Online Backup API return.
pub(crate) fn wrap_rusqlite_busy<T>(result: Result<T, rusqlite::Error>) -> Result<T, Error> {
    wrap_busy_error(result.map_err(Error::from))
}

/// Map a failure raised while opening the database to a `Config` error.
///
/// A locked store becomes the actionable [`LOCKED_DB_MESSAGE`]; any other
/// failure keeps its own message. Deliberately flattens the variant: the CLI
/// prints every open-path failure (including migrations) as a configuration
/// error, so the byte-identical text is preserved at the eight call sites.
/// Callers that want the structured verdict use [`open_db_error_structured`].
#[must_use]
pub(crate) fn open_db_error(err: crate::sqlite::Error) -> Error {
    match open_db_error_structured(err) {
        Error::DatabaseLocked => Error::Config(LOCKED_DB_MESSAGE.to_string()),
        other => Error::Config(other.to_string()),
    }
}

/// [`open_db_error`]'s classification without the `Config` flattening, for
/// callers that want to keep the structured variant.
#[must_use]
pub(crate) fn open_db_error_structured(err: crate::sqlite::Error) -> Error {
    err.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A locked error built the way the storage layer produces one: from the
    /// rusqlite result code, through the boundary's `From` impl.
    fn busy_sqlite_error() -> crate::sqlite::Error {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".to_string()),
        )
        .into()
    }
    #[test]
    fn busy_sqlite_error_maps_to_the_actionable_config_message() {
        match wrap_busy(Err::<(), _>(busy_sqlite_error())) {
            Err(Error::Config(msg)) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }
    }

    #[test]
    fn a_lock_that_arrives_as_a_raw_rusqlite_error_is_still_wrapped() {
        // The Online Backup API (`backup`) and the read-only doctor opens
        // surface locks as `rusqlite::Error` without a SqliteFailure code, so
        // this path must classify too — it does, through the same
        // result-code classifier.
        let raw = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            None,
        );
        match wrap_rusqlite_busy(Err::<(), _>(raw)) {
            Err(Error::Config(msg)) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }
    }

    #[test]
    fn other_database_errors_keep_their_own_message() {
        let err = crate::sqlite::Error::Sqlite("disk I/O error".to_string());
        match open_db_error(err) {
            Error::Config(msg) => assert_eq!(msg, "Database error: disk I/O error"),
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[test]
    fn the_open_path_keeps_the_structured_locked_variant() {
        // `open_db_error_structured` is what a library-shaped caller uses; it
        // must not lose the variant the storage boundary produced.
        let structured = open_db_error_structured(busy_sqlite_error());
        assert!(matches!(structured, Error::DatabaseLocked));
        assert!(structured.is_locked());
    }

    #[test]
    fn a_locked_public_error_is_wrapped_by_the_shared_entry_point() {
        let err = Error::DatabaseLocked;
        match wrap_busy_error(Err::<(), _>(err)) {
            Err(Error::Config(msg)) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }
    }

    #[test]
    fn a_plain_database_error_is_never_reported_locked() {
        // Text is not an input to this decision: a message that merely
        // mentions the busy phrase (a store value echoed by an identity
        // refusal, say) must not be rewritten into the MCP-server hint.
        let err = Error::Database {
            message: "Database error: database is locked".to_string(),
        };
        match wrap_busy_error(Err::<(), _>(err)) {
            Err(Error::Database { message }) => {
                assert!(message.starts_with("Database error: database is locked"));
            }
            other => panic!("a non-locked failure must pass through unchanged, got {other:?}"),
        }
    }
}
