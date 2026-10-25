//! Classification of raw `rusqlite::Error` at the storage boundary (issue #255).
//!
//! The one place that decides what a database failure *means*, from the SQLite
//! result code while it is still in hand. The verdict travels as a typed
//! variant (`sqlite::Error::Locked` / `vipune::Error::DatabaseLocked`), never
//! as message text.

use rusqlite::ffi;

use super::Error;

/// The public rendering of a locked store: the storage-layer Display text for
/// [`DbErrorKind::Locked`]. Kept next to the classifier that produces it so the
/// text and the verdict stay one decision (issue #255).
pub(crate) const LOCKED_DISPLAY: &str = "Database error: database is locked";

/// How a raw `rusqlite::Error` should be presented to callers.
#[derive(Debug)]
pub(crate) enum DbErrorKind<'a> {
    /// `SQLITE_BUSY` / `SQLITE_LOCKED` — another process holds a lock.
    ///
    /// The canonical busy text the storage layer renders for this verdict.
    /// `hook/run.rs` and the pi-rukas `ctx7` wrapper both match it, so
    /// `sqlite::Error::Locked`'s Display and the tests below are pinned to it.
    Locked,
    /// A vipune domain error surfaced through a row-conversion failure
    /// (`query_mod::corrupt_embedding_error`); carries the structured variant.
    ///
    /// Only the structured embedding-decoding variants qualify — see
    /// [`is_trusted_domain_error`] — so an unrelated error that happens to
    /// expose a `crate::sqlite::Error` on its source chain is never echoed
    /// verbatim to the CLI or to MCP clients.
    Domain(&'a Error),
    /// Any other database failure.
    Other,
}

/// The domain errors a row-conversion failure is allowed to carry.
///
/// `query_mod::corrupt_embedding_error` is the only producer, and it wraps
/// exactly these three decoding failures. Anything else on the source chain is
/// not ours to trust, so the classifier falls through to `Other` rather than
/// promoting its payload into the public error.
fn is_trusted_domain_error(err: &Error) -> bool {
    matches!(
        err,
        Error::CorruptEmbedding { .. }
            | Error::InvalidBlobSize { .. }
            | Error::MismatchedDimensions { .. }
    )
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
        && is_trusted_domain_error(domain)
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

    #[test]
    fn an_untrusted_domain_error_on_the_source_chain_is_other() {
        // Only the three decoding variants are trusted. A `NotFound` (whose
        // payload is sanitised on the way out) smuggled in the same shape must
        // not be promoted to a public domain error.
        let smuggled = rusqlite::Error::FromSqlConversionFailure(
            crate::sqlite::EMBEDDING_COLUMN,
            rusqlite::types::Type::Blob,
            Box::new(rusqlite::types::FromSqlError::other(Error::NotFound(
                "secret-uuid".to_string(),
            ))),
        );
        assert!(matches!(classify_db_error(&smuggled), DbErrorKind::Other));
    }

    #[test]
    fn locked_storage_error_renders_the_canonical_busy_text() {
        // The storage boundary's rendering is what `hook/run.rs` and the
        // pi-rukas `ctx7` wrapper match. Pinning it here turns a Display
        // change from silent mis-classification into a failing test.
        let busy = rusqlite::Error::SqliteFailure(
            ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            Some("database is locked".to_string()),
        );
        let storage: Error = busy.into();
        assert!(matches!(storage, Error::Locked));
        assert_eq!(storage.to_string(), LOCKED_DISPLAY);
    }
}
