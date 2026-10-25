//! Contract tests for `commands::db_busy` (issue #255), split out to keep the
//! helper module under the project 500-line cap. Wired from `db_busy.rs` with
//! `#[path]`, so `super::*` here is the module under test: one `#[test]` per
//! arm of the two helpers — a lock becomes the actionable `Config` message, a
//! non-lock keeps its variant and payload.

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
    let raw =
        rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY), None);
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
    // regression #255 was raised to remove. One fixture per arm of
    // `Error::is_structured` — a variant the predicate admits must survive
    // the mapping, not only the one that inspired it.
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

    type Kept = (crate::sqlite::Error, fn(&Error) -> bool);
    let cases: Vec<Kept> = vec![
        (
            crate::sqlite::Error::InvalidBlobSize {
                expected: crate::embedding::EMBEDDING_DIMS * 4,
                actual: 4,
            },
            |e| {
                matches!(
                    e,
                    Error::InvalidBlobSize {
                        expected,
                        actual: 4
                    } if *expected == crate::embedding::EMBEDDING_DIMS * 4
                )
            },
        ),
        (
            crate::sqlite::Error::MismatchedDimensions {
                expected: 384,
                actual: 768,
            },
            |e| {
                matches!(
                    e,
                    Error::MismatchedDimensions {
                        expected: 384,
                        actual: 768
                    }
                )
            },
        ),
        (
            crate::sqlite::Error::MigrationRefused {
                offending: vec!["mem-2".to_string()],
            },
            |e| matches!(e, Error::MigrationRefused { offending } if offending == &vec!["mem-2".to_string()]),
        ),
        (
            crate::sqlite::Error::MigrationIncomplete {
                report: crate::sqlite::migration_types::MigrationReport {
                    reindexed: 1,
                    skipped: 0,
                    failures: vec![crate::sqlite::migration_types::MigrationRowFailure {
                        id: "mem-3".to_string(),
                        error: "boom".to_string(),
                    }],
                },
            },
            |e| {
                matches!(e, Error::MigrationIncomplete { report } if report.failures.len() == 1
                && report.failures[0].id == "mem-3")
            },
        ),
    ];
    for (storage, is_kept) in cases {
        let kept = open_failure(storage);
        assert!(
            is_kept(&kept),
            "a structured open failure must keep its variant and payload, got {kept:?}"
        );
        // The other half of the same arm: it was never framed as Config.
        assert!(
            !matches!(kept, Error::Config(_)),
            "a structured open failure must not be flattened to Config, got {kept:?}"
        );
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
        matches!(storage, crate::sqlite::Error::Locked) && !storage.is_structured_embedding_error(),
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
