//! Schema migration framework for vipune SQLite databases.
//!
//! Uses SQLite's built-in `PRAGMA user_version` to track schema version.
//! Migrations run automatically on database open, wrapped in transactions.
//!
//! # Migration Workflow
//!
//! 1. Read current schema version from `PRAGMA user_version`
//! 2. Run migrations from (current_version + 1) to LATEST
//! 3. Each migration runs in its own transaction (BEGIN → migrate → COMMIT/ROLLBACK)
//! 4. Update `user_version` only after successful migration
//!
//! # Adding a New Migration
//!
//! Add a migration function to the `migrations()` vector:
//!
//! ```ignore
//! fn migrate_v2(conn: &Connection) -> SqliteResult<()> {
//!     conn.execute("ALTER TABLE memories ADD COLUMN type TEXT", [])?;
//!     Ok(())
//! }
//!
//! fn migrations() -> Vec<MigrationFn> {
//!     vec![
//!         migrate_v1,  // Migration 1 (baseline)
//!         migrate_v2,  // Migration 2 (add type column)
//!     ]
//! }
//! ```

use rusqlite::{Connection, Error as RusqliteError, Result as SqliteResult};
use std::fmt;

/// Migration function type: takes a connection and performs schema changes.
type MigrationFn = fn(&Connection) -> SqliteResult<()>;

/// Error type for migration-specific failures.
#[derive(Debug)]
pub enum MigrationError {
    /// Database schema version is newer than this binary supports.
    UnsupportedVersion {
        current_version: i32,
        max_supported: i32,
    },
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MigrationError::UnsupportedVersion {
                current_version,
                max_supported,
            } => write!(
                f,
                "Database schema version {} is newer than this vipune binary supports (max: {}). Upgrade vipune.",
                current_version, max_supported
            ),
        }
    }
}

impl std::error::Error for MigrationError {}

impl From<MigrationError> for RusqliteError {
    fn from(err: MigrationError) -> Self {
        RusqliteError::ToSqlConversionFailure(Box::new(err))
    }
}

/// Migration 1: Baseline schema version.
///
/// This is a no-op migration that establishes the schema versioning system.
/// Fresh databases created with the current schema are marked as version 1.
/// Existing v0.2.x databases (user_version=0) will upgrade to version 1 here.
///
/// Future migrations (v0.3 type/status columns) will be migration 2, 3, etc.
fn migrate_v1(_conn: &Connection) -> SqliteResult<()> {
    // No-op: current schema is already v1
    Ok(())
}

/// Returns all migrations in order. Index 0 = migration 1, etc.
fn migrations() -> Vec<MigrationFn> {
    vec![migrate_v1]
}

/// Returns the total number of migrations available (i.e., the max supported schema version).
fn total_migrations() -> i32 {
    migrations().len() as i32
}

/// Run pending migrations. Call this on every database open.
///
/// # Migration Process
///
/// 1. Read current schema version from `PRAGMA user_version`
/// 2. Check if version is supported (not newer than this build)
/// 3. For each migration with version > current:
///    - Begin EXCLUSIVE transaction (locks DB for concurrent safety)
///    - Run migration function
///    - Commit (on success) or rollback (on failure)
///    - Update `user_version` only after commit succeeds (pragma is NOT transactional!)
/// 4. Returns error if any migration fails or version is unsupported
///
/// # Errors
///
/// Returns SQLite error if migration fails or database version is unsupported.
/// Failed migrations roll back, leaving the database in its previous state.
///
/// # Example
///
/// ```ignore
/// let conn = Connection::open(&path)?;
/// run_migrations(&conn)?;  // Runs all pending migrations
/// ```
pub fn run_migrations(conn: &Connection) -> SqliteResult<()> {
    let current: i32 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;

    // Check: database has newer schema than this binary supports
    if current > total_migrations() {
        return Err(MigrationError::UnsupportedVersion {
            current_version: current,
            max_supported: total_migrations(),
        }
        .into());
    }

    let all = migrations();

    for (i, migration) in all.iter().enumerate() {
        let version = (i + 1) as i32;
        if version > current {
            // Begin EXCLUSIVE transaction (prevents concurrent writers during migration)
            conn.execute_batch("BEGIN EXCLUSIVE;")?;

            match migration(conn) {
                Ok(()) => {
                    // Success: commit first, then update version
                    // PRAGMA user_version is NOT transactional, so must happen after commit
                    conn.execute_batch("COMMIT;")?;
                    conn.pragma_update(None, "user_version", version)?;
                }
                Err(e) => {
                    // Failure: rollback and propagate error
                    conn.execute_batch("ROLLBACK;")?;
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a fresh in-memory database for testing.
    fn create_test_db() -> Connection {
        Connection::open_in_memory().unwrap()
    }

    /// Helper: initialize the current vipune schema (without migrations).
    fn init_schema(conn: &Connection) -> SqliteResult<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    #[test]
    fn test_fresh_db_version_becomes_1() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();

        // Initial user_version is 0
        let initial: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(initial, 0);

        // Run migrations
        run_migrations(&conn).unwrap();

        // Version should now be 1
        let final_version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(final_version, 1);
    }

    #[test]
    fn test_already_at_version_1_is_noop() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();

        // Run migrations first time: 0 → 1
        run_migrations(&conn).unwrap();

        let version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 1);

        // Run again: should be no-op (already at version 1)
        run_migrations(&conn).unwrap();

        let version_after: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version_after, 1);
    }

    #[test]
    fn test_upgrade_from_v0_to_v1() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();

        // Explicitly set user_version to 0 (simulating v0.2.x database)
        conn.pragma_update(None, "user_version", 0).unwrap();
        let initial: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(initial, 0);

        // Run migrations
        run_migrations(&conn).unwrap();

        // Should upgrade from 0 to 1
        let final_version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(final_version, 1);
    }

    #[test]
    fn test_migration_framework_idempotent() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();

        // Run migrations multiple times
        for _ in 0..5 {
            run_migrations(&conn).unwrap();
        }

        // Version should be 1 (not incrementing on re-run)
        let version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 1);
    }

    #[test]
    fn test_migration_transaction_rollback_on_error() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();

        // Create a migration function that intentionally fails
        fn failing_migration(_conn: &Connection) -> SqliteResult<()> {
            Err(RusqliteError::InvalidQuery)
        }

        // Set user_version to 0 so failing_migration would run
        conn.pragma_update(None, "user_version", 0).unwrap();

        // Verify initial state
        let initial_version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(initial_version, 0);

        // Manually run the failing migration logic
        conn.execute_batch("BEGIN EXCLUSIVE;").unwrap();
        let result = failing_migration(&conn);
        assert!(result.is_err()); // Should fail
        conn.execute_batch("ROLLBACK;").unwrap();

        // Verify user_version was NOT incremented after the failure
        let version_after: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version_after, 0);

        // Verify database is still usable by running a successful migration
        run_migrations(&conn).unwrap();
        let final_version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(final_version, 1);
    }

    #[test]
    fn test_future_version_database_error() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();

        // Set user_version to a value higher than total_migrations (simulating a newer database)
        conn.pragma_update(None, "user_version", 999).unwrap();

        // Running migrations should fail because the database version is too new
        let result = run_migrations(&conn);
        assert!(result.is_err());

        // Error message should mention database version and upgrade
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("schema version"));
        assert!(err_msg.contains("999"));
        assert!(err_msg.contains("Upgrade vipune"));

        // Verify version is still 999 (unchanged)
        let version: i32 = conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 999);
    }
}
