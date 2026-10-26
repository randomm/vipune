//! The error-shape half of the `initialize_fts` FTS5-migration suite (issue
//! #278 / #255): what a *classified* FTS failure looks like on the way out, as
//! opposed to the contention tests in `fts_migration_tests.rs` that produce one
//! from a real `BEGIN EXCLUSIVE` writer.
//!
//! Wired from `sqlite/tests.rs` alongside its siblings. `with_context` is the
//! function under test: it prepends call-site context *after* classification,
//! so the two properties that matter are that a genuine (non-lock) failure
//! keeps its message and that a verdict keeps its variant — an earlier
//! revision flattened the structured variants into `Locked` here, which turned
//! a corrupt BLOB into a retryable lock.

/// The context string every case below prepends — the same one
/// `initialize_fts`'s rebuild arm uses, so these cases exercise the real call
/// shape rather than a paraphrase of it.
const MIGRATION_CONTEXT: &str = "FTS5 schema migration failed";

#[test]
fn an_unclassified_fts_migration_failure_keeps_its_context_and_its_message() {
    // The other half of `with_context`: the context is prepended after
    // classification, so a genuine (non-lock) failure still reads as the
    // operator saw it before, and a non-`Sqlite` verdict is never
    // flattened to make room for prose.
    let plain = crate::sqlite::Error::Sqlite("no such table: nope".to_string())
        .with_context(MIGRATION_CONTEXT);
    match plain {
        crate::sqlite::Error::Sqlite(msg) => {
            assert_eq!(msg, "FTS5 schema migration failed: no such table: nope");
        }
        other => panic!("expected Sqlite with the context prepended, got {other:?}"),
    }

    // A verdict has no message to grow, so the context must not swallow it.
    let locked = crate::sqlite::Error::Locked.with_context(MIGRATION_CONTEXT);
    assert!(
        matches!(locked, crate::sqlite::Error::Locked),
        "context must never flatten the locked verdict, got {locked:?}"
    );
}

#[test]
fn with_context_leaves_every_non_sqlite_variant_untouched() {
    // The arms `with_context` must NOT touch. An earlier revision mapped the
    // three structured decoding variants to `Error::Locked` here, which turned
    // a corrupt BLOB into a retryable lock and dropped the `id`/`expected`/
    // `actual` a caller acts on — so each is pinned individually, plus the
    // verdict arm in the test above, rather than covered by a blanket wildcard.
    // Each case pairs a storage error with what its public form must be; the
    // `fn` pointer is factored out because the pair type trips
    // `clippy::type_complexity` when written inline.
    type Kept = (crate::sqlite::Error, fn(&crate::sqlite::Error) -> bool);
    let cases: Vec<Kept> = vec![
        (
            crate::sqlite::Error::CorruptEmbedding {
                id: "mem-1".to_string(),
                reason: "not enough bytes".to_string(),
            },
            |e| {
                matches!(e, crate::sqlite::Error::CorruptEmbedding { id, reason }
                if id == "mem-1" && reason == "not enough bytes")
            },
        ),
        (
            crate::sqlite::Error::InvalidBlobSize {
                expected: crate::embedding::EMBEDDING_DIMS * 4,
                actual: 4,
            },
            |e| {
                matches!(
                    e,
                    crate::sqlite::Error::InvalidBlobSize {
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
                    crate::sqlite::Error::MismatchedDimensions {
                        expected: 384,
                        actual: 768
                    }
                )
            },
        ),
        (crate::sqlite::Error::Locked, |e| {
            matches!(e, crate::sqlite::Error::Locked)
        }),
    ];
    for (err, is_kept) in cases {
        let expected = err.to_string();
        let after = err.with_context(MIGRATION_CONTEXT);
        assert!(
            is_kept(&after),
            "a non-Sqlite variant must survive with_context with its payload, got {after:?}"
        );
        assert_eq!(
            after.to_string(),
            expected,
            "with_context must not rewrite a non-Sqlite variant's rendering"
        );
    }
}
