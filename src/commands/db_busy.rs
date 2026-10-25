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
/// Callers that FAIL TO OPEN do not use this helper: they go through
/// [`open_db`] instead, which adds one thing this helper must not — the uniform
/// `Config` framing the CLI has always printed for an open failure.
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
    // The predicate, not a hand-written list: `is_structured_embedding_error`
    // is the single enumeration of the variants whose payload a caller acts on
    // (the same one the public `From` impl and the MCP `data` arms key off), so
    // a structured variant added beside it is admitted here without a fourth
    // list to remember. The migration pair is named separately because that
    // predicate covers only the embedding-decoding set.
    let structured = err.is_structured_embedding_error()
        || matches!(
            err,
            crate::sqlite::Error::MigrationRefused { .. }
                | crate::sqlite::Error::MigrationIncomplete { .. }
        );
    if structured {
        return Error::from(err);
    }
    let public = Error::from(err);
    if public.is_locked() {
        return Error::Config(LOCKED_DB_MESSAGE.to_string());
    }
    Error::Config(public.to_string())
}

#[cfg(test)]
#[path = "db_busy_tests.rs"]
mod db_busy_tests;
