//! Classification of raw `rusqlite::Error` at the storage boundary (issue #255).
//!
//! Lives in its own module because it is the single place that decides what a
//! database failure *means* to the rest of the program. The public
//! `vipune::Error` deliberately carries no rusqlite type, so the question
//! "is this a locked/busy database?" has to be answered here, once, while the
//! SQLite result code is still in hand — not by the eight command handlers
//! matching the Display text, which is what they used to do and which breaks
//! silently the moment a dependency rewords its message.

use rusqlite::ffi;

use super::Error;

/// How a raw `rusqlite::Error` should be presented to callers.
#[derive(Debug)]
pub(crate) enum DbErrorKind<'a> {
    /// `SQLITE_BUSY` / `SQLITE_LOCKED` — another process holds a lock.
    Locked,
    /// A vipune domain error surfaced through a row-conversion failure
    /// (`query_mod::corrupt_embedding_error`); carries the structured variant.
    Domain(&'a Error),
    /// Any other database failure.
    Other,
}

/// Classify a raw rusqlite error at the storage boundary.
///
/// A locked database is recognised by the SQLite result code rather than by
/// message text: rusqlite rewrites or re-wraps several transient states, so
/// the code is the only stable signal.
#[must_use]
pub(crate) fn classify_db_error(err: &rusqlite::Error) -> DbErrorKind<'_> {
    // rusqlite's own error-code accessor, so this does not depend on the ffi
    // struct's fields staying shaped any particular way.
    if matches!(
        err.sqlite_error_code(),
        Some(ffi::ErrorCode::DatabaseBusy) | Some(ffi::ErrorCode::DatabaseLocked)
    ) {
        return DbErrorKind::Locked;
    }
    // Row-level `FromSqlConversionFailure`s built by
    // `query_mod::corrupt_embedding_error` carry a `CorruptEmbedding` domain
    // error (naming the affected memory id) as their source — surface it
    // directly so the id reaches the caller (issue #186).
    if let rusqlite::Error::FromSqlConversionFailure(_, _, boxed) = err
        && let Some(source) = (**boxed).source()
        && let Some(domain) = source.downcast_ref::<Error>()
    {
        return DbErrorKind::Domain(domain);
    }
    DbErrorKind::Other
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_and_locked_codes_classify_as_locked() {
        for code in [rusqlite::ffi::SQLITE_BUSY, rusqlite::ffi::SQLITE_LOCKED] {
            let err = rusqlite::Error::SqliteFailure(ffi::Error::new(code), None);
            assert!(
                matches!(classify_db_error(&err), DbErrorKind::Locked),
                "code {code} must classify as Locked"
            );
        }
    }

    #[test]
    fn unrelated_failures_are_other() {
        let err = rusqlite::Error::SqliteFailure(
            ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE),
            Some("UNIQUE constraint failed: memories.id".to_string()),
        );
        assert!(matches!(classify_db_error(&err), DbErrorKind::Other));
    }

    #[test]
    fn a_row_conversion_failure_surfaces_its_domain_error() {
        // The fabrication in `query_mod::corrupt_embedding_error` must keep
        // round-tripping: the domain error is the source of the failure, and
        // this is what lets the affected memory id reach the caller (#186).
        let domain = Error::CorruptEmbedding {
            id: "mem-1".to_string(),
            reason: "not enough bytes".to_string(),
        };
        let err = crate::sqlite::query_mod::corrupt_embedding_error("mem-1".to_string(), domain);
        match classify_db_error(&err) {
            DbErrorKind::Domain(Error::CorruptEmbedding { id, .. }) => assert_eq!(id, "mem-1"),
            other => panic!("expected the CorruptEmbedding domain error, got {other:?}"),
        }
    }
}
