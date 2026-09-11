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

fn migrate_v1(_conn: &Connection) -> SqliteResult<()> {
    Ok(())
}

fn migrate_v2(conn: &Connection) -> SqliteResult<()> {
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN type TEXT NOT NULL DEFAULT 'fact';
         ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
         ALTER TABLE memories ADD COLUMN superseded_by TEXT;
         CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
         CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
         CREATE INDEX IF NOT EXISTS idx_memories_project_status ON memories(project_id, status);",
    )?;
    Ok(())
}

fn migrate_v3(conn: &Connection) -> SqliteResult<()> {
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN retrieval_count INTEGER NOT NULL DEFAULT 0;
         ALTER TABLE memories ADD COLUMN last_retrieved_at TEXT;",
    )?;
    Ok(())
}

/// Name of the unique dedup index created by migration 4. Shared with the hook
/// insert path so it can distinguish a dedup violation from other constraint
/// failures when mapping a `SqliteError` to a silent skip.
pub const DEDUP_INDEX_NAME: &str = "idx_memories_dedup";

use super::hash::content_hash;

/// Migration 4: add `content_hash` column + `idx_memories_dedup` unique index.
///
/// All steps run inside the migration transaction (rolled back on failure):
/// 1. `ALTER TABLE memories ADD COLUMN content_hash TEXT`
/// 2. Backfill `content_hash` for all existing rows using the shared `content_hash`
/// 3. Deduplicate pre-existing duplicate `(project_id, content_hash)` rows:
///    keep the newest row (by `created_at`) per group, delete the older rows.
///    This handles real-world DBs that accumulated duplicate content before
///    the unique constraint existed.
/// 4. `CREATE UNIQUE INDEX idx_memories_dedup ON memories(project_id, content_hash)`
///
/// The unique index enforces at the database level that no two rows in the same
/// project have identical normalised content — the dedup guarantee for hooks.
fn migrate_v4(conn: &Connection) -> SqliteResult<()> {
    conn.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT", [])?;
    backfill_content_hash(conn)?;
    // Deduplicate: keep, per (project_id, content_hash) group, the newest row
    // by created_at (rowid breaks ties); delete every other row in the group.
    // Rows whose group has no duplicates keep their own rowid and are untouched.
    // This handles real-world DBs with pre-existing duplicate content.
    conn.execute(
        "DELETE FROM memories WHERE content_hash IS NOT NULL
         AND rowid NOT IN (
            SELECT rowid FROM (
                SELECT rowid,
                       ROW_NUMBER() OVER (
                           PARTITION BY project_id, content_hash
                           ORDER BY created_at DESC, rowid DESC
                       ) AS rn
                FROM memories
                WHERE content_hash IS NOT NULL
            )
            WHERE rn = 1
         )",
        [],
    )?;
    conn.execute(
        &format!("CREATE UNIQUE INDEX {DEDUP_INDEX_NAME} ON memories(project_id, content_hash)"),
        [],
    )?;
    Ok(())
}

/// Backfill `content_hash` for all rows that currently have `NULL`.
///
/// The hash is computed in Rust via the shared `content_hash` function
/// (`crate::sqlite::hash::content_hash`) so the migration and the hook path
/// always produce identical values.
fn backfill_content_hash(conn: &Connection) -> SqliteResult<()> {
    let rows: Vec<(String, String)> = {
        let mut stmt =
            conn.prepare("SELECT id, content FROM memories WHERE content_hash IS NULL")?;
        let mut out = Vec::new();
        for row_result in
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        {
            out.push(row_result?);
        }
        out
    };
    let mut upd = conn.prepare("UPDATE memories SET content_hash = ?1 WHERE id = ?2")?;
    for (id, content) in rows {
        let hash = content_hash(&content);
        upd.execute((hash, id))?;
    }
    Ok(())
}

/// Migration 5: add `importance` column (low/medium/high/critical, default medium).
///
/// The operator-assigned importance level is a HARD exclusion in the prune
/// command (high/critical rows are never demoted) and scales the temporal
/// decay rate (issue #194, sub-issue 2).
fn migrate_v5(conn: &Connection) -> SqliteResult<()> {
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN importance TEXT NOT NULL DEFAULT 'medium';",
    )?;
    Ok(())
}

fn migrations() -> Vec<MigrationFn> {
    vec![migrate_v1, migrate_v2, migrate_v3, migrate_v4, migrate_v5]
}

fn total_migrations() -> i32 {
    migrations().len() as i32
}

/// Run pending migrations on every database open.
///
/// # Migration Process
///
/// 1. Read current schema version from `PRAGMA user_version`
/// 2. Check if version is supported (not newer than this build)
/// 3. For each migration with version > current:
///    - BEGIN EXCLUSIVE transaction (locks DB for concurrent safety)
///    - Run migration function
///    - COMMIT (on success) or ROLLBACK (on failure)
///    - Update `user_version` only after commit succeeds (pragma is NOT transactional!)
/// 4. Returns error if any migration fails or version is unsupported
pub fn run_migrations(conn: &Connection) -> SqliteResult<()> {
    let current: i32 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;

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
            conn.execute_batch("BEGIN EXCLUSIVE;")?;
            match migration(conn) {
                Ok(()) => {
                    conn.execute_batch("COMMIT;")?;
                    conn.pragma_update(None, "user_version", version)?;
                }
                Err(e) => {
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

    fn create_test_db() -> Connection {
        Connection::open_in_memory().unwrap()
    }

    fn init_schema(conn: &Connection) -> SqliteResult<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, content TEXT NOT NULL,
                embedding BLOB NOT NULL, metadata TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);",
        )?;
        Ok(())
    }

    fn version_of(conn: &Connection) -> i32 {
        conn.pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap()
    }

    /// Insert a bare row (pre-migration shape) for testing backfill.
    fn insert_row(conn: &Connection, id: &str, project_id: &str, content: &str) {
        conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at)
             VALUES (?1, ?2, ?3, X'00', 't', 't')",
            (id, project_id, content),
        )
        .unwrap();
    }

    // --- Version tests (parameterised via total_migrations()) ---

    #[test]
    fn test_fresh_db_version_reaches_latest() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        run_migrations(&conn).unwrap();
        assert_eq!(version_of(&conn), total_migrations());
    }

    #[test]
    fn test_already_at_latest_is_noop() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        run_migrations(&conn).unwrap();
        run_migrations(&conn).unwrap(); // second run: no-op
        assert_eq!(version_of(&conn), total_migrations());
    }

    #[test]
    fn test_upgrade_from_v0_reaches_latest() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        conn.pragma_update(None, "user_version", 0).unwrap();
        run_migrations(&conn).unwrap();
        assert_eq!(version_of(&conn), total_migrations());
    }

    #[test]
    fn test_migration_framework_idempotent() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        for _ in 0..5 {
            run_migrations(&conn).unwrap();
        }
        assert_eq!(version_of(&conn), total_migrations());
    }

    #[test]
    fn test_migration_transaction_rollback_on_error() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        conn.pragma_update(None, "user_version", 0).unwrap();
        conn.execute_batch("BEGIN EXCLUSIVE;").unwrap();
        fn failing_migration(_conn: &Connection) -> SqliteResult<()> {
            Err(RusqliteError::InvalidQuery)
        }
        assert!(failing_migration(&conn).is_err());
        conn.execute_batch("ROLLBACK;").unwrap();
        assert_eq!(version_of(&conn), 0); // version unchanged after rollback
        run_migrations(&conn).unwrap(); // db still usable
        assert_eq!(version_of(&conn), total_migrations());
    }

    #[test]
    fn test_future_version_database_error() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        conn.pragma_update(None, "user_version", 999).unwrap();
        let err = run_migrations(&conn).unwrap_err().to_string();
        assert!(err.contains("schema version"));
        assert!(err.contains("999"));
        assert!(err.contains("Upgrade vipune"));
        assert_eq!(version_of(&conn), 999);
    }

    // --- Migration 5: importance column ---

    fn setup_v4_with_row(conn: &Connection, project_id: &str, content: &str) {
        init_schema(conn).unwrap();
        insert_row(conn, "r1", project_id, content);
        migrate_v2(conn).unwrap();
        migrate_v3(conn).unwrap();
        migrate_v4(conn).unwrap();
        conn.pragma_update(None, "user_version", 4).unwrap();
    }

    #[test]
    fn test_migration_5_adds_importance_column_default_medium() {
        let conn = create_test_db();
        setup_v4_with_row(&conn, "proj-a", "Some memory content");
        migrate_v5(&conn).unwrap();
        let importance: String = conn
            .query_row("SELECT importance FROM memories WHERE id = 'r1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(importance, "medium", "new column must default to 'medium'");
    }

    #[test]
    fn test_migration_5_bumps_user_version_to_5() {
        let conn = create_test_db();
        setup_v4_with_row(&conn, "proj-a", "content");
        run_migrations(&conn).unwrap();
        assert_eq!(version_of(&conn), 5);
    }

    #[test]
    fn test_migration_5_idempotent_via_run_migrations() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        run_migrations(&conn).unwrap();
        run_migrations(&conn).unwrap();
        assert_eq!(version_of(&conn), 5);
    }

    // --- content_hash parity tests (shared hash.rs::content_hash) ---

    #[test]
    fn test_content_hash_normalises_case_and_whitespace() {
        assert_eq!(
            content_hash("Hello   World"),
            content_hash("hello world"),
            "case + whitespace must be normalised"
        );
        assert_eq!(
            content_hash("  Leading and  trailing  "),
            content_hash("leading and trailing")
        );
    }

    #[test]
    fn test_content_hash_different_content_different_hash() {
        assert_ne!(content_hash("foo"), content_hash("bar"));
    }

    #[test]
    fn test_content_hash_is_lowercase_hex_16_chars() {
        let h = content_hash("test content");
        assert_eq!(h.len(), 16, "expected 16 hex chars, got {:?}", h);
        assert!(
            h.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "expected lowercase hex, got {:?}",
            h
        );
    }

    #[test]
    fn test_content_hash_deterministic() {
        assert_eq!(content_hash("same input"), content_hash("same input"));
    }

    // --- Migration 4: dedup behaviour tests ---

    fn setup_v3_with_row(conn: &Connection, project_id: &str, content: &str) {
        init_schema(conn).unwrap();
        insert_row(conn, "r1", project_id, content);
        conn.pragma_update(None, "user_version", 3).unwrap();
    }

    #[test]
    fn test_migration_4_backfills_content_hash_for_existing_rows() {
        let conn = create_test_db();
        setup_v3_with_row(&conn, "proj-a", "Some memory content");
        insert_row(&conn, "r2", "proj-a", "Other memory content");
        migrate_v4(&conn).unwrap();
        let null_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE content_hash IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(null_count, 0, "all rows should be backfilled");
    }

    #[test]
    fn test_migration_4_creates_unique_dedup_index() {
        let conn = create_test_db();
        setup_v3_with_row(&conn, "proj-a", "unique content here");
        migrate_v4(&conn).unwrap();
        let idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_memories_dedup'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1, "idx_memories_dedup should exist after migration");
    }

    #[test]
    fn test_migration_4_dedup_blocks_duplicate_normalized_content() {
        let conn = create_test_db();
        setup_v3_with_row(&conn, "proj-a", "Hello World");
        migrate_v4(&conn).unwrap();
        let hash = content_hash("hello world");
        let result = conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at, content_hash)
             VALUES ('r2', 'proj-a', 'hello world', X'00', 't', 't', ?1)",
            [hash],
        );
        assert!(
            result.is_err(),
            "duplicate normalised content must be rejected"
        );
    }

    #[test]
    fn test_migration_4_same_content_different_project_is_allowed() {
        let conn = create_test_db();
        setup_v3_with_row(&conn, "proj-a", "shared content");
        migrate_v4(&conn).unwrap();
        let hash = content_hash("shared content");
        let result = conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at, content_hash)
             VALUES ('r2', 'proj-b', 'shared content', X'00', 't', 't', ?1)",
            [hash],
        );
        assert!(
            result.is_ok(),
            "same content in different project must be allowed"
        );
    }

    #[test]
    fn test_migration_4_existing_duplicates_are_deduplicated() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        insert_row(&conn, "r1", "proj-a", "duplicate content here");
        insert_row(&conn, "r2", "proj-a", "Duplicate   Content Here");
        conn.pragma_update(None, "user_version", 3).unwrap();
        migrate_v4(&conn).expect("migration must succeed despite duplicates");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project_id = 'proj-a'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "only one row should remain after dedup");
        let idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_memories_dedup'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1, "dedup index should exist after migration");
    }

    #[test]
    fn test_migration_4_dedup_keeps_one_row_per_group_only() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        // Two duplicate pairs + one unique row.
        insert_row(&conn, "r1", "proj-a", "dup one");
        insert_row(&conn, "r2", "proj-a", "Dup   One");
        insert_row(&conn, "r3", "proj-a", "dup two");
        insert_row(&conn, "r4", "proj-a", "dup   TWO");
        insert_row(&conn, "r5", "proj-a", "unique row");
        conn.pragma_update(None, "user_version", 3).unwrap();
        migrate_v4(&conn).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 3, "3 unique contents should remain, got {count}");
        let unique_gone: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE content = 'unique row'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(unique_gone, 1, "non-duplicate rows must not be deleted");
    }

    #[test]
    fn test_migration_4_dedup_does_not_cross_project_boundary() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        insert_row(&conn, "r1", "proj-a", "shared content");
        insert_row(&conn, "r2", "proj-b", "shared content");
        conn.pragma_update(None, "user_version", 3).unwrap();
        migrate_v4(&conn).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            count, 2,
            "same content in different projects must both survive"
        );
    }

    #[test]
    fn test_migration_4_dedup_keeps_newest_by_created_at() {
        let conn = create_test_db();
        init_schema(&conn).unwrap();
        // Insert duplicate rows with different created_at timestamps; the
        // row with the latest created_at must survive.
        conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at)
             VALUES ('r1', 'proj-a', 'old dup', X'00', '2024-01-01T00:00:00Z', 't')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at)
             VALUES ('r2', 'proj-a', 'old dup', X'00', '2025-06-15T12:30:00Z', 't')",
            [],
        )
        .unwrap();
        conn.pragma_update(None, "user_version", 3).unwrap();
        migrate_v4(&conn).unwrap();
        let survivor: String = conn
            .query_row("SELECT id FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(survivor, "r2", "the newest row by created_at must survive");
    }
}
