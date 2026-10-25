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
//! That guarantee is the producer's to keep: a call site that flattens a
//! `rusqlite::Error` into a message string before reaching here destroys the
//! result code and the verdict with it (`sqlite::identity::read_identity_row`
//! used to do exactly that — it now converts through `From`, and
//! `identity_tests::test_lock_on_identity_read_stays_classified_locked` pins
//! it). The `rusqlite::Error` call sites — the read-only diagnostic opens and
//! the Online Backup API — convert first and then come through the same door.
//!
//! They live here rather than as `Error` methods because they are CLI
//! presentation helpers: they render the operator-facing locked message, and
//! are deliberately not part of the public library API.

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
/// failure keeps its own message text. Deliberately flattens the variant: the
/// CLI prints every open-path failure (including migrations) as a
/// configuration error. A library consumer that wants the structured verdict
/// uses `Error::from(sqlite::Error)` directly rather than this helper.
#[must_use]
pub(crate) fn open_db_error(err: crate::sqlite::Error) -> Error {
    match err {
        crate::sqlite::Error::Locked => Error::Config(LOCKED_DB_MESSAGE.to_string()),
        other => Error::Config(Error::from(other).to_string()),
    }
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
        // The Online Backup API (`backup`) and the read-only diagnostic opens
        // surface locks as `rusqlite::Error`. The second argument is the
        // optional message, not the result code: `classify_db_error` reads
        // `sqlite_error_code()` from the `SqliteFailure` payload, so this
        // fixture carries the code and no message.
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
    fn a_lock_without_a_result_code_stays_a_plain_database_error() {
        // The wrapper must not turn every database failure into the busy hint:
        // only the classified verdict does that, so a plain `Sqlite` message
        // (which is what a code-less rusqlite failure becomes) passes through.
        let err = crate::sqlite::Error::Sqlite("disk I/O error".to_string());
        match wrap_busy(Err::<(), _>(err)) {
            Err(Error::Database { message }) => assert_eq!(message, "disk I/O error"),
            other => panic!("a non-locked failure must pass through unchanged, got {other:?}"),
        }
    }

    #[test]
    fn a_real_writer_holding_the_lock_classifies_a_statement_failure_as_locked() {
        // The synthetic fixtures above hand-build a variant, so they cannot
        // catch the verdict being lost between a real statement and the
        // wrapper. This drives an actual `BEGIN EXCLUSIVE` writer and fails a
        // mid-statement UPDATE (not an open), then asserts the structured
        // verdict `wrap_busy` classifies on.
        use crate::memory::lifecycle::MemoryStatus;
        use crate::sqlite::UpdateOptions;

        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("memories.db");
        let db = crate::sqlite::Database::open(&db_path).unwrap();
        let emb = crate::memory::crud::test_fake_embedder("contended").unwrap();
        let id = db
            .insert("proj-a", "contended", &emb, None, "fact", "active")
            .unwrap();
        db.set_busy_timeout(std::time::Duration::ZERO).unwrap();

        let lock_conn = rusqlite::Connection::open(&db_path).unwrap();
        lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();

        let result = wrap_busy(db.update(
            &id,
            "proj-a",
            UpdateOptions {
                content: None,
                embedding: None,
                metadata: None,
                memory_type: None,
                status: Some(MemoryStatus::Deprecated.as_str()),
                importance: None,
            },
        ));
        match result {
            Err(Error::Config(msg)) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the locked Config error, got {other:?}"),
        }
        // And the verdict really came from the classifier, not from text.
        let raw: Error = db
            .update(
                &id,
                "proj-a",
                UpdateOptions {
                    content: None,
                    embedding: None,
                    metadata: None,
                    memory_type: None,
                    status: Some(MemoryStatus::Deprecated.as_str()),
                    importance: None,
                },
            )
            .map_err(Error::from)
            .expect_err("the writer still holds the lock");
        assert!(raw.is_locked(), "a real lock must classify, got {raw:?}");

        lock_conn.execute("ROLLBACK", []).unwrap();
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
