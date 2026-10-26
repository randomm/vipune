//! Shared test-only storage fixtures for `src/sqlite/`.
//!
//! Wired from `src/sqlite/mod.rs` under `#[cfg(test)]`. The `TempDir` is
//! *returned* rather than forgotten (issue #278), so the caller's binding owns
//! it and the store's file is removed when the test returns — the leaked dirs
//! were what exhausted the operator's temp quota and made unrelated runs fail
//! with SQLite `disk I/O error`.
//!
//! Call sites bind both halves (`let (_dir, db) = create_test_db();`) and the
//! dir must outlive the handle, which is why the pair is returned together
//! rather than the helper dropping the dir on the way out.

use super::Database;
use tempfile::TempDir;

/// A store opened on an owned temp dir, with schema and migrations applied.
pub(crate) fn create_test_db() -> (TempDir, Database) {
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("test.db");
    let db = Database::open(&path).expect("open test database");
    (dir, db)
}
