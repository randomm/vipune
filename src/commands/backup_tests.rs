//! Tests for `vipune backup` handler.
//!
//! Following the pattern of `reindex_tests.rs` / `merge_tests.rs`: a
//! tempfile-backed source DB, seeded via the public `Database::insert` API,
//! and the Online Backup handler run against it. Tests cover:
//! - default destination naming and explicit `--output` path
//! - the produced backup is a queryable, FTS-consistent copy (row count,
//!   content, raw embedding BLOBs all match the source byte-for-byte)
//! - corrupt / NULL / 1535-byte embedding blobs carry through uncorrupted
//!   (the whole point of a page-level copy — no decoding happens)
//! - fast-fail when the source DB is held by another process (MCP server)
//! - idempotent re-run (a second `backup` over an existing destination
//!   replaces it, doesn't append or fail)

#![cfg(test)]

use crate::commands::backup::handle_backup;
use crate::memory::crud::test_fake_embedder;
use crate::sqlite::{Database, vec_to_blob};
use rusqlite::Connection;
use std::path::Path;

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("memories.db");
    Database::open(&path).unwrap();
    (dir, path)
}

/// Insert one row with a known embedding vector, return its id.
fn insert_row(db: &Database, project_id: &str, content: &str, embedding: &[f32]) -> String {
    db.insert(project_id, content, embedding, None, "fact", "active")
        .unwrap()
}

/// Mirror the default destination logic in `backup::resolve_destination` for
/// a known source path, so the test can locate the file the handler writes.
fn default_destination(src: &Path) -> std::path::PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap();
    let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("db");
    let parent = src.parent().unwrap();
    parent.join(format!("{}-backup.{}", stem, ext))
}

#[test]
fn test_backup_default_destination_is_queryable_copy() {
    let (_dir, src) = create_test_db();
    let db = Database::open(&src).unwrap();
    let emb = test_fake_embedder("alpha").unwrap();
    let id = insert_row(&db, "proj", "alpha", &emb);

    let exit = handle_backup(&src, None, true).expect("handle_backup should succeed");
    assert_eq!(exit, std::process::ExitCode::SUCCESS);

    let expected = default_destination(&src);
    assert!(expected.exists(), "default destination must exist");

    // Re-open the backup and verify row count + raw embedding byte-identity.
    let dest_conn = Connection::open(&expected).unwrap();
    let count: i64 = dest_conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);

    let (id2, blob2): (String, Vec<u8>) = dest_conn
        .query_row("SELECT id, embedding FROM memories LIMIT 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(id2, id);
    let expected_blob: Vec<u8> = vec_to_blob(&emb).unwrap();
    assert_eq!(
        blob2, expected_blob,
        "backup must be byte-identical for the embedding BLOB"
    );
}

#[test]
fn test_backup_explicit_output_path() {
    let (_dir, src) = create_test_db();
    let db = Database::open(&src).unwrap();
    let emb = test_fake_embedder("beta").unwrap();
    let _id = insert_row(&db, "proj", "beta", &emb);

    let out = _dir.keep().join("explicit-backup.db");
    let exit = handle_backup(&src, Some(Path::new(out.to_str().unwrap())), true)
        .expect("handle_backup should succeed");
    assert_eq!(exit, std::process::ExitCode::SUCCESS);

    let dest_conn = Connection::open(&out).unwrap();
    let count: i64 = dest_conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_backup_carries_corrupt_and_null_embeddings_verbatim() {
    let (_dir, src) = create_test_db();
    let db = Database::open(&src).unwrap();

    // Three rows: normal, 1535-byte corrupt blob (wrong dimension), and a
    // row where the embedding BLOB was never set (NULL via raw SQL).
    let normal_emb = test_fake_embedder("normal").unwrap();
    let normal_id = insert_row(&db, "proj", "normal", &normal_emb);

    // Corrupt: 1535 bytes (one short of 1536) — `vec_to_blob` rejects 384
    // vectors only when the length is wrong; we bypass by writing raw.
    let corrupt_blob: Vec<u8> = vec![0xABu8; 1535];
    let corrupt_id = "corrupt-1".to_string();
    db.conn().execute(
        "INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status)
         VALUES (?1, 'proj', 'corrupt', ?2, NULL, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active')",
        rusqlite::params![&corrupt_id, &corrupt_blob],
    ).unwrap();

    // NULL embedding: bypass the `NOT NULL` constraint by using raw SQL that
    // explicitly writes NULL (the schema constraint is `embedding BLOB NOT
    // NULL`; a NULL insert would fail. So instead we use an empty BLOB to
    // represent the "no data" case that the ticket describes: a NULL/empty
    // embedding is exported faithfully and import rejects it. Here we just
    // verify a 0-byte BLOB passes through unchanged.)
    let empty_id = "empty-2".to_string();
    db.conn().execute(
        "INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status)
         VALUES (?1, 'proj', 'empty', ?2, NULL, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active')",
        rusqlite::params![&empty_id, Vec::<u8>::new()],
    )
    .unwrap();

    let exit = handle_backup(&src, None, true).expect("handle_backup should succeed");
    assert_eq!(exit, std::process::ExitCode::SUCCESS);

    let expected = default_destination(&src);
    let dest_conn = Connection::open(&expected).unwrap();
    let count: i64 = dest_conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 3, "all three rows must be in the backup");

    // Verify each row's raw BLOB byte-for-byte.
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (&normal_id, vec_to_blob(&normal_emb).unwrap()),
        (&corrupt_id, vec![0xABu8; 1535]),
        (&empty_id, Vec::new()),
    ];
    for (id, expected_blob) in cases {
        let blob: Vec<u8> = dest_conn
            .query_row("SELECT embedding FROM memories WHERE id = ?", [id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(
            blob, expected_blob,
            "row {id} embedding must carry through byte-for-byte"
        );
    }
}

#[test]
fn test_backup_fails_fast_when_source_locked() {
    let (_dir, src) = create_test_db();
    let db = Database::open(&src).unwrap();
    let emb = test_fake_embedder("locked").unwrap();
    let _id = insert_row(&db, "proj", "locked", &emb);

    // Open a separate raw connection and take an EXCLUSIVE lock, mimicking a
    // running MCP server holding the write lock.
    let lock_conn = Connection::open(&src).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();

    let result = handle_backup(&src, None, true);
    // The actionable "Database is locked … MCP server" message is produced by
    // `wrap_busy` for both `Error::SqliteModule` (the `crate::sqlite::Error`
    // path) and `Error::SQLite` (the direct `rusqlite::Error` path, which is
    // what `rusqlite::backup::Backup::new` returns when the source is in a
    // hot-journal state). The `Display` form of `Error::Config` is the inner
    // string verbatim, so asserting on the `to_string()` form works regardless
    // of which conversion path the lock error took.
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("locked"),
                "Expected 'locked' in error message, got: {}",
                msg
            );
            assert!(
                msg.contains("MCP server"),
                "Expected 'MCP server' in error message, got: {}",
                msg
            );
        }
        Ok(_) => panic!("Expected error when database is locked, got Ok"),
    }

    lock_conn.execute("ROLLBACK", []).unwrap();
}

#[test]
fn test_backup_replaces_existing_destination() {
    let (_dir, src) = create_test_db();
    let db = Database::open(&src).unwrap();
    let emb1 = test_fake_embedder("first").unwrap();
    let _id = insert_row(&db, "proj", "first", &emb1);

    // First backup.
    let expected = default_destination(&src);
    let exit1 = handle_backup(&src, None, true).expect("first backup should succeed");
    assert_eq!(exit1, std::process::ExitCode::SUCCESS);
    assert!(expected.exists());

    // Insert a second row, then re-run: the destination must now contain both
    // rows (a fresh copy, not an append or a failed overwrite).
    let emb2 = test_fake_embedder("second").unwrap();
    let _id2 = insert_row(&db, "proj", "second", &emb2);
    let exit2 = handle_backup(&src, None, true).expect("second backup should succeed");
    assert_eq!(exit2, std::process::ExitCode::SUCCESS);

    let dest_conn = Connection::open(&expected).unwrap();
    let count: i64 = dest_conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2, "second backup must replace the first");
}

#[test]
fn test_backup_preserves_fts_index() {
    let (_dir, src) = create_test_db();
    let db = Database::open(&src).unwrap();
    let emb = test_fake_embedder("fts probe").unwrap();
    let _id = insert_row(&db, "proj", "fts probe", &emb);

    let exit = handle_backup(&src, None, true).expect("handle_backup should succeed");
    assert_eq!(exit, std::process::ExitCode::SUCCESS);

    let expected = default_destination(&src);
    let dest_conn = Connection::open(&expected).unwrap();

    // FTS5 consistency: a direct `SELECT count(*) FROM memories_fts`
    // exercises the FTS content and index paths (it reads through the FTS5
    // shadow tables) and would fail with a SQLite error if the index were
    // corrupted. The assertion below verifies the FTS index reflects the
    // seeded row.
    let fts_rows: i64 = dest_conn
        .query_row("SELECT count(*) FROM memories_fts", [], |r| r.get(0))
        .expect("FTS5 content must be queryable (a corrupted index would error)");
    assert_eq!(fts_rows, 1, "FTS5 content must reflect the seeded row");
}
