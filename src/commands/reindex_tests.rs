//! Tests for `vipune reindex` handler.

#![cfg(test)]

use crate::commands::reindex::*;
use crate::errors::Error;
use crate::memory::crud::{mock_embedding_for_content, test_fake_embedder};
use crate::memory::store::MemoryStore;
use crate::output::ReindexFailure;
use crate::sqlite::Database;
use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};
use rusqlite::Connection;

fn reindex_project_with_fake_embedder(
    db: &Database,
    project_id: &str,
) -> Result<(usize, usize, Vec<ReindexFailure>), Error> {
    let mut embed_callback = |content: &str| test_fake_embedder(content);
    reindex_project(db, &mut embed_callback, project_id, false)
}

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn get_embedding(db: &Database, id: &str) -> Vec<f32> {
    db.list_all_rows_for_project("proj")
        .unwrap()
        .iter()
        .find(|(i, _, _)| i == id)
        .map(|(_, _, e)| e.clone())
        .unwrap()
}

#[test]
fn test_mock_rows_are_repaired() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let content = "mock memory";
    let mock_vec = mock_embedding_for_content(content);
    let id = db
        .insert("proj", content, &mock_vec, None, "fact", "active")
        .unwrap();

    assert_eq!(
        classify_embedding(&get_embedding(&db, &id)),
        EmbeddingClass::Mock
    );

    let (reindexed, skipped, failed) = reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(reindexed, 1);
    assert_eq!(skipped, 0);
    assert_eq!(failed.len(), 0);

    let emb_after = get_embedding(&db, &id);
    assert_ne!(mock_vec, emb_after);
    assert_eq!(classify_embedding(&emb_after), EmbeddingClass::Real);
}

#[test]
fn test_real_rows_untouched() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let real_vec = test_fake_embedder("real memory").unwrap();
    let id = db
        .insert("proj", "real memory", &real_vec, None, "fact", "active")
        .unwrap();

    let (reindexed, _, _) = reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(reindexed, 0);
    assert_eq!(get_embedding(&db, &id), real_vec);
}

#[test]
fn test_unknown_rows_skipped() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "real",
        &test_fake_embedder("real").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    let zero_vec = vec![0.0; 384];
    let id = db
        .insert("proj", "unknown", &zero_vec, None, "fact", "active")
        .unwrap();

    assert_eq!(
        classify_embedding(&get_embedding(&db, &id)),
        EmbeddingClass::Unknown
    );

    let (_, skipped, _) = reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(skipped, 1);
    assert_eq!(get_embedding(&db, &id), zero_vec);
}

#[test]
fn test_idempotency_second_run_zero_changes() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let id1 = db
        .insert(
            "proj",
            "A",
            &mock_embedding_for_content("A"),
            None,
            "fact",
            "active",
        )
        .unwrap();
    let id2 = db
        .insert(
            "proj",
            "B",
            &mock_embedding_for_content("B"),
            None,
            "fact",
            "active",
        )
        .unwrap();

    let (r1, _, _) = reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(r1, 2);

    let emb1_first = get_embedding(&db, &id1);
    let emb2_first = get_embedding(&db, &id2);

    let (r2, _, _) = reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(r2, 0);
    assert_eq!(get_embedding(&db, &id1), emb1_first);
    assert_eq!(get_embedding(&db, &id2), emb2_first);
}

#[test]
fn test_timestamps_preserved() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let mock_vec = mock_embedding_for_content("mock");
    let id = db
        .insert("proj", "mock", &mock_vec, None, "fact", "active")
        .unwrap();

    let conn = db.conn();
    conn.execute(
        "UPDATE memories SET retrieval_count = 7, last_retrieved_at = '2024-03-20T14:30:00Z' WHERE id = ?",
        [&id],
    ).unwrap();

    let (u_before, rc_before, lr_before) = conn
        .query_row(
            "SELECT updated_at, retrieval_count, last_retrieved_at FROM memories WHERE id = ?",
            [&id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .unwrap();

    reindex_project_with_fake_embedder(&db, "proj").unwrap();

    let (u_after, rc_after, lr_after) = conn
        .query_row(
            "SELECT updated_at, retrieval_count, last_retrieved_at FROM memories WHERE id = ?",
            [&id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .unwrap();

    assert_eq!(u_before, u_after);
    assert_eq!(rc_before, rc_after);
    assert_eq!(lr_before, lr_after);
}

#[test]
fn test_counters_match_seeded_data() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "m1",
        &mock_embedding_for_content("m1"),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert(
        "proj",
        "m2",
        &mock_embedding_for_content("m2"),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert(
        "proj",
        "m3",
        &mock_embedding_for_content("m3"),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert(
        "proj",
        "r1",
        &test_fake_embedder("r1").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert(
        "proj",
        "r2",
        &test_fake_embedder("r2").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert("proj", "u", &vec![0.0; 384], None, "fact", "active")
        .unwrap();

    let (reindexed, skipped, failed) = reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(reindexed, 3);
    assert_eq!(skipped, 1);
    assert_eq!(failed.len(), 0);
}

#[test]
fn test_database_update_embedding_does_not_touch_updated_at() {
    let db = MemoryStore::test_store();
    let emb = test_fake_embedder("orig").unwrap();
    let id = db
        .db
        .insert("proj", "orig", &emb, None, "fact", "active")
        .unwrap();

    let t1: String = db
        .db
        .conn()
        .query_row("SELECT updated_at FROM memories WHERE id = ?", [&id], |r| {
            r.get(0)
        })
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));
    let emb2 = test_fake_embedder("up").unwrap();
    db.db.update_embedding(&id, &emb2).unwrap();

    let t2: String = db
        .db
        .conn()
        .query_row("SELECT updated_at FROM memories WHERE id = ?", [&id], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(t1, t2);
}

#[test]
fn test_locked_database_fast_fails() {
    let (_dir, db_path) = create_test_db();

    // Open a raw SQLite connection and begin an exclusive transaction to lock the database
    let lock_conn = Connection::open(&db_path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();

    // Now try to open the database with handle_reindex - it should fail fast
    let result = handle_reindex(&db_path, "BAAI/bge-small-en-v1.5", None, false);

    // Verify the error is the actionable Config error, not a timeout or hang
    match result {
        Err(Error::Config(msg)) => {
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
        Err(e) => panic!(
            "Expected Error::Config with locked database message, got: {:?}",
            e
        ),
        Ok(_) => panic!("Expected error when database is locked, got Ok"),
    }

    // Release lock
    lock_conn.execute("ROLLBACK", []).unwrap();
}
