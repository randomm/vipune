//! Locked-database classification shared by the command handlers (issue #255).
//!
//! Eight handlers touch a possibly-locked store (backup, merge, reindex,
//! prune, export, promote, and the two doctor modes). Before this module each
//! carried its own `wrap_busy` that matched the Display text
//! `"database is locked"`, so a dependency rewording its message would
//! silently turn every fast-fail path into an unclassified database error.
//! They now share these four functions, which decide via
//! [`vipune::Error::is_locked`] / the SQLite result code instead.
//!
//! They live here rather than as `Error` methods because their callers are all
//! in the binary target: as library-target items they would read as dead code.

use crate::errors::{Error, LOCKED_DB_MESSAGE};

/// Canonical SQLite text for a busy database, as the storage layer surfaces it
/// (`sqlite::classify_db_error` maps both `SQLITE_BUSY` and `SQLITE_LOCKED` to
/// this text inside `crate::sqlite::Error::Sqlite`).
const SQLITE_BUSY_MESSAGE: &str = "database is locked";

/// True for a storage-layer error that means the database was locked or busy.
///
/// Decides before conversion on purpose: `crate::sqlite::Error::Sqlite` keeps
/// the rusqlite message verbatim, while the producers that flatten an arbitrary
/// database failure into a message (the identity reads in `sqlite/identity.rs`
/// and `commands/import.rs`) reach the public enum as `Error::Database`, not as
/// the dedicated [`Error::DatabaseLocked`] variant.
#[must_use]
pub(crate) fn sqlite_is_locked(err: &crate::sqlite::Error) -> bool {
    matches!(err, crate::sqlite::Error::Sqlite(msg) if msg.contains(SQLITE_BUSY_MESSAGE))
}

/// Map a failure raised while opening the database to a `Config` error.
///
/// A locked store becomes the actionable [`LOCKED_DB_MESSAGE`]; any other
/// failure keeps its own message.
#[must_use]
pub(crate) fn open_db_error(err: crate::sqlite::Error) -> Error {
    if sqlite_is_locked(&err) {
        Error::Config(LOCKED_DB_MESSAGE.to_string())
    } else {
        Error::Config(err.to_string())
    }
}

/// Wrap a result over the storage layer, turning a locked/busy database error
/// into the actionable [`LOCKED_DB_MESSAGE`] `Config` error and mapping
/// everything else through the public conversion.
pub(crate) fn wrap_busy<T>(result: Result<T, crate::sqlite::Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(err) if sqlite_is_locked(&err) => Err(Error::Config(LOCKED_DB_MESSAGE.to_string())),
        Err(err) => Err(err.into()),
    }
}

/// Wrap a result that already carries the public error, turning a locked/busy
/// database error into the actionable [`LOCKED_DB_MESSAGE`] `Config` error and
/// passing everything else through untouched.
///
/// Use this for helpers whose own signature is already `Result<_, Error>`;
/// [`wrap_busy`] is the variant for the storage layer's own errors.
pub(crate) fn wrap_busy_error<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(e) if e.is_locked() => Err(Error::Config(LOCKED_DB_MESSAGE.to_string())),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_sqlite_error_maps_to_the_actionable_config_message() {
        let err = crate::sqlite::Error::Sqlite("database is locked".to_string());
        match wrap_busy(Err::<(), _>(err)) {
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
    fn a_locked_public_database_error_is_still_wrapped() {
        // The identity-read producers flatten into `Error::Database`, so the
        // public-error wrapper has to catch that shape too.
        let err = Error::Database {
            message: "Database error: database is locked".to_string(),
        };
        match wrap_busy_error(Err::<(), _>(err)) {
            Err(Error::Config(msg)) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }
    }
}
