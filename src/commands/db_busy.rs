//! Locked-database presentation shared by the command handlers (issue #255).
//!
//! Eight handlers touch a possibly-locked store — backup, merge, reindex,
//! prune, export, promote, and the two doctor modes — and `import` opens the
//! same store with a deliberately NONZERO busy timeout (see
//! `commands::import`), so it is not one of these fast-fail callers. Before
//! this module each of the eight carried its own `wrap_busy` that matched the
//! Display text `"database is locked"`, so a dependency rewording its message
//! would silently turn every fast-fail path into an unclassified database
//! error. They now share [`wrap_busy`], which decides on the
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
//! the Online Backup API — convert through the same `From` impls and then come
//! through the same door, which is why there is one wrapper rather than one
//! per input error type.
//!
//! They live here rather than as `Error` methods because they are CLI
//! presentation helpers: they render the operator-facing locked message, and
//! are deliberately not part of the public library API.

use crate::errors::{Error, LOCKED_DB_MESSAGE};

/// Wrap a result whose error converts into the public enum — `rusqlite::Error`,
/// `crate::sqlite::Error`, or `Error` itself — turning a locked/busy database
/// into the actionable [`LOCKED_DB_MESSAGE`] `Config` error and passing
/// everything else through untouched, variant and payload included.
///
/// One entry point rather than a sibling per input type: the caller's `Result`
/// already names its error type, so the conversion is `From`'s job. The
/// conversion runs first and the decision second, which is what keeps the
/// structured variants (`CorruptEmbedding`, the migration variants) intact.
///
/// # Pinning
///
/// The two assertions below are the contract tests for this helper: a lock
/// becomes the actionable message, and a non-lock passes through with its
/// variant unchanged.
pub(crate) fn wrap_busy<T, E: Into<Error>>(result: Result<T, E>) -> Result<T, Error> {
    match result.map_err(Into::into) {
        Ok(v) => Ok(v),
        Err(e) if e.is_locked() => Err(Error::Config(LOCKED_DB_MESSAGE.to_string())),
        Err(e) => Err(e),
    }
}

/// Map a failure raised while opening the database to the `Config` error the
/// CLI prints for every open-path failure.
///
/// The two arms are the whole contract, pinned by
/// [`open_db_error_preserves_the_structured_variants`] and
/// [`open_db_error_flattens_only_the_message_carrying_variants`]:
///
/// * A locked store becomes the actionable [`LOCKED_DB_MESSAGE`] `Config`
///   error. `Error::is_locked()` is `false` for it: the CLI's locked contract
///   is the operator-facing text (see `docs/cli-reference.md`), and a library
///   consumer that wants the typed verdict calls `Database::open` (or
///   [`wrap_busy`]) rather than a CLI handler.
/// * A storage error whose public form is message-only — `Sqlite`, the input
///   predicates, `NotFound` — is flattened to `Config` with the same rendered
///   text, so open-path failures read uniformly as a configuration problem.
/// * A storage error whose public form is *structured* (`CorruptEmbedding`,
///   `InvalidBlobSize`, `MismatchedDimensions`, `MigrationRefused`,
///   `MigrationIncomplete`) is returned as its own public variant instead: a
///   corrupt BLOB surfaced at open time must keep the `id` that locates the
///   row (issue #186), which is the whole point of #255 carrying those fields.
///
/// `promote`/`reindex`/`doctor --fts` open through [`wrap_busy`] instead, which
/// preserves every variant including the structured ones; that is the stricter
/// contract of the two, and both are pinned so neither can silently regress.
#[must_use]
pub(crate) fn open_db_error(err: crate::sqlite::Error) -> Error {
    match err {
        crate::sqlite::Error::Locked => Error::Config(LOCKED_DB_MESSAGE.to_string()),
        structured @ (crate::sqlite::Error::CorruptEmbedding { .. }
        | crate::sqlite::Error::InvalidBlobSize { .. }
        | crate::sqlite::Error::MismatchedDimensions { .. }
        | crate::sqlite::Error::MigrationRefused { .. }
        | crate::sqlite::Error::MigrationIncomplete { .. }) => Error::from(structured),
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
        match wrap_busy(Err::<(), _>(raw)) {
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
    fn open_db_error_preserves_the_structured_variants() {
        // The open-path wrapper must not erase the payloads this PR carries
        // through the public boundary: a corrupt BLOB hit at open time still
        // names the row (#186), and a migration failure keeps its ids/report.
        let corrupt = open_db_error(crate::sqlite::Error::CorruptEmbedding {
            id: "mem-1".to_string(),
            reason: "not enough bytes".to_string(),
        });
        match corrupt {
            Error::CorruptEmbedding { id, reason } => {
                assert_eq!(id, "mem-1");
                assert_eq!(reason, "not enough bytes");
            }
            other => panic!("expected CorruptEmbedding, got {other:?}"),
        }

        let refused = open_db_error(crate::sqlite::Error::MigrationRefused {
            offending: vec!["mem-9".to_string()],
        });
        match refused {
            Error::MigrationRefused { offending } => assert_eq!(offending, vec!["mem-9"]),
            other => panic!("expected MigrationRefused, got {other:?}"),
        }

        // The lock still gets the actionable operator framing, not a variant.
        match open_db_error(crate::sqlite::Error::Locked) {
            Error::Config(msg) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }
    }

    #[test]
    fn open_db_error_flattens_only_the_message_carrying_variants() {
        // The other half of the open_db_error contract: what has no public
        // fields to lose is rendered as a Config string, so open failures read
        // uniformly. `NotFound` also exercises the sanitisation the storage
        // conversion performs (the id must not reach the operator text).
        match open_db_error(crate::sqlite::Error::NotFound("secret-uuid".to_string())) {
            Error::Config(msg) => {
                assert_eq!(msg, "Memory not found: memory not found");
                assert!(
                    !msg.contains("secret-uuid"),
                    "the id sanitisation must survive the open path: {msg}"
                );
            }
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[test]
    fn wrap_busy_preserves_the_structured_variants_it_is_handed() {
        // `promote`/`reindex`/`doctor --fts` open through `wrap_busy`, so the
        // strict contract — every non-lock variant travels unchanged — is
        // pinned here rather than only asserted in a doc comment.
        let wrapped = wrap_busy(Err::<(), _>(crate::sqlite::Error::MismatchedDimensions {
            expected: 384,
            actual: 768,
        }))
        .expect_err("a storage failure stays a failure");
        match wrapped {
            Error::MismatchedDimensions { expected, actual } => {
                assert_eq!((expected, actual), (384, 768));
            }
            other => panic!("expected MismatchedDimensions, got {other:?}"),
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
        // verdict the shared entry point classifies on — the same call with
        // the `sqlite::Error` the real statement returns.
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
        // And the verdict really came from the classifier, not from text. Same
        // statement again (idempotent: it sets the status it already has).
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
        // And the storage layer carries the same verdict, so no layer above
        // has to re-derive it: `is_structured_embedding_error` cannot be what
        // promoted this row (that predicate admits only the three decoding
        // variants), which is what makes the allow-list in
        // `db_error::is_trusted_domain_error` observable rather than vacuous.
        let storage: crate::sqlite::Error = db
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
            .expect_err("the writer still holds the lock");
        assert!(
            matches!(storage, crate::sqlite::Error::Locked)
                && !storage.is_structured_embedding_error(),
            "a real lock must reach the boundary as the storage Locked variant, got {storage:?}"
        );

        lock_conn.execute("ROLLBACK", []).unwrap();
    }

    #[test]
    fn a_locked_public_error_is_wrapped_by_the_shared_entry_point() {
        let err = Error::DatabaseLocked;
        match wrap_busy(Err::<(), _>(err)) {
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
        match wrap_busy(Err::<(), _>(err)) {
            Err(Error::Database { message }) => {
                assert!(message.starts_with("Database error: database is locked"));
            }
            other => panic!("a non-locked failure must pass through unchanged, got {other:?}"),
        }
    }
}
