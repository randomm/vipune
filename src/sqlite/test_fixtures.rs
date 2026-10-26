//! Test-only storage fixtures shared by every test in `src/sqlite/` (see the
//! module doc on `sqlite::test_fixtures`).

use super::Database;
use tempfile::TempDir;

/// An empty owned temp dir and a database path inside it, for tests that build
/// the file themselves or reopen it through several opens.
pub(crate) fn create_test_db_path() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("test.db");
    (dir, path)
}

/// A store opened on an owned temp dir, with schema and migrations applied.
pub(crate) fn create_test_db() -> (TempDir, Database) {
    let (dir, path) = create_test_db_path();
    let db = Database::open(&path).expect("open test database");
    (dir, db)
}
