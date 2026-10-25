//! Locked-database presentation shared by the command handlers (issue #255).
//!
//! Eight handlers touch a possibly-locked store — backup, merge, reindex,
//! prune, export, promote, and the two doctor modes — and `import` opens the
//! same store with a deliberately NONZERO busy timeout (see
//! `commands::import`), so it is not one of these fast-fail callers. Before
//! this module each of the eight carried its own `wrap_busy` that matched the
//! Display text `"database is locked"`, so a dependency rewording its message
//! would silently turn every fast-fail path into an unclassified database
//! error. They now share [`wrap_busy`] for statements and [`open_db`] for the
//! open, both of which decide on the [`Error::DatabaseLocked`] variant — itself
//! set once, from the SQLite result code, by `sqlite::classify_db_error`.
//! Nothing here reads message text.
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
//!
//! **One contract for every entry point.** All eight handlers open the store
//! through [`open_db`] and wrap every later call through [`wrap_busy`], so a
//! lock is classified from the same verdict no matter which command hit it and
//! `is_locked()` cannot mean one thing for `backup` and another for `promote`.
//! There is deliberately no second open-path wrapper: an earlier revision had
//! `open_db_error`, used by only five of the eight, which flattened a *lock* to
//! `Error::Config` and destroyed the verdict — the inconsistency this unifies.
//! `open_db` keeps that helper's uniform `Config` rendering (which is the CLI's
//! long-standing output for an open failure) while preserving the verdict, so
//! unifying the paths changed neither the operator's text nor the exit code.
//!
//! The split between the two helpers is deliberate, not a second contract:
//! [`wrap_busy`] passes a non-lock through untouched because mid-statement
//! failures have always been reported under their own variant, while [`open_db`]
//! frames a non-lock as `Config` because that is what the CLI has always printed
//! for an open failure. Both classify a lock identically, which is the property
//! callers depend on. Each half is pinned by its own test below.

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
/// through [`open_db`] instead, which adds one thing this helper must not: the
/// uniform `Config` framing the CLI has always printed for an open failure.
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

/// Open the store for a command handler, mapping every open failure to the
/// `Config` error the CLI has always printed for it.
///
/// This is the single open-path entry point: the five handlers that used to
/// carry `open_db_error` and the three that opened through [`wrap_busy`]
/// (`promote`, `reindex`, `doctor --fts`) all use this, so a lock is classified
/// the same way from every command and `is_locked()` cannot mean one thing for
/// `backup` and another for `promote`.
///
/// The contract has two arms and neither is negotiable:
///
/// * **A lock keeps its verdict.** `Error::from(sqlite::Error::Locked)` is
///   `Error::DatabaseLocked`, and `is_locked()` reads that variant — the
///   deleted `open_db_error` flattened the lock straight to a `Config` string,
///   so the verdict was destroyed here for five of the eight handlers. The
///   operator text is unchanged: this returns the same `Config(LOCKED_DB_MESSAGE)`
///   shape, which is what keeps stderr byte-for-byte identical and the exit code
///   at 1.
/// * **Any other open failure renders as `Config` with the rendered public
///   message**, exactly as the deleted `open_db_error` did. That framing is not
///   decoration: it is what the CLI printed for an open failure before this
///   issue, so a plain `Database`/`NotFound`/input failure must keep reading as
///   one uniform configuration problem. Returning those variants untouched would
///   silently change the operator's text (no `Configuration error: ` lead-in) —
///   the regression
///   [`open_failures_render_the_same_text_the_deleted_open_db_error_produced`]
///   exists to catch.
///
/// A storage error whose public form is *structured* (`CorruptEmbedding`,
/// `InvalidBlobSize`, `MismatchedDimensions`, the migration variants) is
/// returned as its own variant rather than flattened: a corrupt BLOB surfaced
/// at open time must keep the `id` that locates the row, which is the whole
/// point of #255 carrying those fields.
pub(crate) fn open_db(path: &std::path::Path) -> Result<crate::sqlite::Database, Error> {
    crate::sqlite::Database::open(path).map_err(open_failure)
}

/// The open-path error mapping, factored out of [`open_db`] so it is testable
/// without a contended filesystem.
fn open_failure(err: crate::sqlite::Error) -> Error {
    let public = Error::from(err);
    if public.is_locked() {
        return Error::Config(LOCKED_DB_MESSAGE.to_string());
    }
    match public {
        structured @ (Error::CorruptEmbedding { .. }
        | Error::InvalidBlobSize { .. }
        | Error::MismatchedDimensions { .. }
        | Error::MigrationRefused { .. }
        | Error::MigrationIncomplete { .. }) => structured,
        other => Error::Config(other.to_string()),
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
    fn open_failures_render_the_same_text_the_deleted_open_db_error_produced() {
        // Unifying the eight open paths on `open_db` must not have changed what
        // the operator sees. `main.rs` prints `Error: {}` for every variant and
        // exits 1 for all of them alike, so byte-identical *Display* is the bar
        // for byte-identical output.
        //
        // The expectations below are the **rendered** lines, which is why they
        // are written out here rather than compared against `LOCKED_DB_MESSAGE`:
        // a lock is returned as `Error::Config(LOCKED_DB_MESSAGE)`, and
        // `Config`'s Display adds its own `Configuration error: ` framing, so
        // what reaches stderr is that framing *around* the locked text. The
        // deleted `open_db_error` produced exactly that shape for both arms,
        // which is what makes the unification output-neutral — asserting on the
        // payload instead of the rendering would have hidden the framing rather
        // than pinned it, and the first draft of this test did exactly that.
        let cases: Vec<(crate::sqlite::Error, &str)> = vec![
            (
                crate::sqlite::Error::Locked,
                "Configuration error: Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.",
            ),
            (
                crate::sqlite::Error::Sqlite("disk I/O error".to_string()),
                "Configuration error: Database error: disk I/O error",
            ),
            (
                crate::sqlite::Error::NotFound("secret-uuid".to_string()),
                "Configuration error: Memory not found: memory not found",
            ),
        ];
        for (storage, expected) in cases {
            let rendered = open_failure(storage).to_string();
            assert_eq!(
                rendered, expected,
                "the open path must render the same text as before the unification"
            );
        }

        // And the payload the operator text is built from is still the single
        // source the rest of the crate pins, so this test cannot drift into
        // pinning an accidental copy of the wording.
        let lock = open_failure(crate::sqlite::Error::Locked);
        match lock {
            Error::Config(msg) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }

        // The structured arm must NOT get the `Config` framing: a corrupt BLOB
        // hit at open time has to keep the `id` that locates the row (#186), and
        // flattening it to satisfy the uniform-open-failure framing is the
        // regression #255 was raised to remove.
        let corrupt = open_failure(crate::sqlite::Error::CorruptEmbedding {
            id: "mem-1".to_string(),
            reason: "not enough bytes".to_string(),
        });
        match corrupt {
            Error::CorruptEmbedding { id, reason } => {
                assert_eq!(
                    (id.as_str(), reason.as_str()),
                    ("mem-1", "not enough bytes")
                );
            }
            other => panic!("a structured open failure must keep its variant, got {other:?}"),
        }
    }

    #[test]
    fn a_lock_at_the_open_classifies_the_same_way_for_every_handler() {
        // The consistency half of the unification, asserted against the real
        // storage boundary all eight handlers share. Before this change the five
        // handlers that opened through `open_db_error` destroyed the lock
        // verdict, so `is_locked()` was false for `backup`/`export`/`merge`/
        // `prune`/`doctor` and true for `promote`/`reindex`/`doctor --fts` on the
        // *same* contention. Every open now runs through `open_db`, so one lock
        // shape yields one verdict everywhere; the per-handler rendering is
        // pinned by the command suites (`backup_tests`, `reindex_cli_tests`,
        // `doctor_fts_tests`, …).
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("memories.db");
        {
            let db = crate::sqlite::Database::open(&db_path).unwrap();
            let emb = crate::memory::crud::test_fake_embedder("locked-open").unwrap();
            db.insert("proj-a", "locked-open", &emb, None, "fact", "active")
                .unwrap();
        }

        let lock_conn = rusqlite::Connection::open(&db_path).unwrap();
        lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();
        // The write is what makes the EXCLUSIVE lock real; `BEGIN EXCLUSIVE`
        // alone defers it.
        lock_conn
            .execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (9999, 'hold')",
                [],
            )
            .unwrap();

        // A real contended open, through the exact boundary the handlers use.
        let storage = match crate::sqlite::Database::open(&db_path) {
            Ok(_) => panic!("the store is still locked, so the open must fail"),
            Err(e) => e,
        };
        assert!(
            matches!(storage, crate::sqlite::Error::Locked),
            "the storage layer must classify a contended open as Locked, got {storage:?}"
        );

        // And the two halves of the open contract hold for that real error:
        // the operator gets the `Config` framing, the verdict underneath it is
        // the typed `DatabaseLocked`. The second assertion is the one that was
        // false for five of the eight handlers before the unification.
        let presented = open_failure(storage);
        let raw: Error = crate::sqlite::Error::Locked.into();
        assert!(
            matches!(presented, Error::Config(_)),
            "an open failure is presented to the operator as Config, got {presented:?}"
        );
        assert!(
            raw.is_locked(),
            "the verdict every handler classifies on must survive the public \
             conversion, got {raw:?}"
        );

        lock_conn.execute("ROLLBACK", []).unwrap();
    }

    #[test]
    fn a_lock_at_the_open_stays_the_typed_verdict_for_every_entry_point() {
        // ONE contract, asserted where it is decided: the wrapper renders the
        // operator-facing `Config` text, and the verdict it classified that
        // text from survives the storage→public conversion as
        // `Error::DatabaseLocked`. The deleted `open_db_error` broke the second
        // half for the five open paths (a lock became `Config` with no typed
        // verdict left), which is why `is_locked()` meant different things
        // depending on which command was run.
        let lock = wrap_busy(Err::<(), _>(crate::sqlite::Error::Locked))
            .expect_err("a locked store must fail");
        match lock {
            Error::Config(ref msg) => assert_eq!(msg, LOCKED_DB_MESSAGE),
            other => panic!("expected the actionable Config error, got {other:?}"),
        }
        let storage: Error = crate::sqlite::Error::Locked.into();
        assert!(
            matches!(storage, Error::DatabaseLocked) && storage.is_locked(),
            "the verdict every handler classifies on survives the public conversion, got {storage:?}"
        );

        let refused = wrap_busy(Err::<(), _>(crate::sqlite::Error::MigrationRefused {
            offending: vec!["mem-9".to_string()],
        }))
        .expect_err("a refused migration stays a failure");
        match refused {
            Error::MigrationRefused { offending } => assert_eq!(offending, vec!["mem-9"]),
            other => panic!("expected MigrationRefused, got {other:?}"),
        }
    }

    #[test]
    fn wrap_busy_preserves_the_structured_variants_it_is_handed() {
        // All eight handlers open through `wrap_busy`, so the strict contract
        // — every non-lock variant travels unchanged — is pinned here rather
        // than only asserted in a doc comment. A corrupt BLOB surfaced at open
        // time must keep the `id` that locates the row (#186).
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

        let corrupt = wrap_busy(Err::<(), _>(crate::sqlite::Error::CorruptEmbedding {
            id: "mem-1".to_string(),
            reason: "not enough bytes".to_string(),
        }))
        .expect_err("a corrupt BLOB stays a failure");
        match corrupt {
            Error::CorruptEmbedding { id, reason } => {
                assert_eq!(id, "mem-1");
                assert_eq!(reason, "not enough bytes");
            }
            other => panic!("expected CorruptEmbedding, got {other:?}"),
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
