//! Tests for `vipune project merge` handler.

use crate::commands::merge::build_mcp_restart_notice;
use crate::memory::crud::test_fake_embedder;
use crate::sqlite::Database;

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn count_project_rows(db: &Database, project_id: &str) -> usize {
    db.conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?",
            [project_id],
            |row| row.get::<_, i64>(0),
        )
        .unwrap() as usize
}

fn get_embedding(db: &Database, project_id: &str, id: &str) -> Vec<u8> {
    db.conn()
        .query_row(
            "SELECT embedding FROM memories WHERE project_id = ? AND id = ?",
            [project_id, id],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .unwrap()
}

fn get_row_fields(
    db: &Database,
    project_id: &str,
    id: &str,
) -> (String, String, i64, Option<String>) {
    db.conn()
        .query_row(
            "SELECT content, updated_at, retrieval_count, last_retrieved_at FROM memories WHERE project_id = ? AND id = ?",
            [project_id, id],
            |row| {
                Ok((
                    row.get::<_, String>(0).unwrap(),
                    row.get::<_, String>(1).unwrap(),
                    row.get::<_, i64>(2).unwrap(),
                    row.get::<_, Option<String>>(3).unwrap(),
                ))
            },
        )
        .unwrap()
}

#[test]
fn test_rows_are_moved() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("memory one").unwrap();
    let id1 = db
        .insert("src", "memory one", &emb, None, "fact", "active")
        .unwrap();
    let emb2 = test_fake_embedder("memory two").unwrap();
    let id2 = db
        .insert("src", "memory two", &emb2, None, "fact", "active")
        .unwrap();

    assert_eq!(count_project_rows(&db, "src"), 2);
    assert_eq!(count_project_rows(&db, "dst"), 0);

    let rows_moved = db.merge_project_ids("src", "dst").unwrap();
    assert_eq!(rows_moved, 2);

    assert_eq!(count_project_rows(&db, "src"), 0);
    assert_eq!(count_project_rows(&db, "dst"), 2);

    // Verify moved rows exist in target
    let (content, _, _, _) = get_row_fields(&db, "dst", &id1);
    assert_eq!(content, "memory one");
    let (content2, _, _, _) = get_row_fields(&db, "dst", &id2);
    assert_eq!(content2, "memory two");
}

#[test]
fn test_timestamps_preserved() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("test").unwrap();
    let id = db
        .insert("src", "test", &emb, None, "fact", "active")
        .unwrap();

    // Set retrieval telemetry
    db.conn()
        .execute(
            "UPDATE memories SET retrieval_count = 7, last_retrieved_at = '2024-03-20T14:30:00Z' WHERE id = ?",
            [&id],
        )
        .unwrap();

    let (_, u_before, rc_before, lr_before) = get_row_fields(&db, "src", &id);

    db.merge_project_ids("src", "dst").unwrap();

    let (_, u_after, rc_after, lr_after) = get_row_fields(&db, "dst", &id);
    assert_eq!(u_before, u_after);
    assert_eq!(rc_before, rc_after);
    assert_eq!(lr_before, lr_after);
}

#[test]
fn test_idempotency_second_run_zero_changes() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("A").unwrap();
    db.insert("src", "A", &emb, None, "fact", "active").unwrap();
    let emb2 = test_fake_embedder("B").unwrap();
    db.insert("src", "B", &emb2, None, "fact", "active")
        .unwrap();

    let first = db.merge_project_ids("src", "dst").unwrap();
    assert_eq!(first, 2);

    let second = db.merge_project_ids("src", "dst").unwrap();
    assert_eq!(second, 0);
    assert_eq!(count_project_rows(&db, "src"), 0);
    assert_eq!(count_project_rows(&db, "dst"), 2);
}

#[test]
fn test_embeddings_byte_identical() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("preserve me").unwrap();
    let id = db
        .insert("src", "preserve me", &emb, None, "fact", "active")
        .unwrap();

    let emb_blob_before = get_embedding(&db, "src", &id);

    db.merge_project_ids("src", "dst").unwrap();

    let emb_blob_after = get_embedding(&db, "dst", &id);
    assert_eq!(emb_blob_before, emb_blob_after);
}

#[test]
fn test_from_equals_to_returns_zero_rows() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("self").unwrap();
    db.insert("same", "self", &emb, None, "fact", "active")
        .unwrap();

    let rows = db.merge_project_ids("same", "same").unwrap();
    assert_eq!(rows, 0);
}

#[test]
fn test_from_equals_to_performs_no_db_writes() {
    // Snapshot total_changes() before and after self-merge to prove zero writes.
    // rusqlite::Connection::total_changes() counts all rows modified, inserted, or
    // deleted since the connection was opened. If merge_project_ids short-circuits
    // correctly, this counter must not advance.
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    for i in 0..3 {
        let emb = test_fake_embedder(&format!("row {}", i)).unwrap();
        db.insert("proj", &format!("row {}", i), &emb, None, "fact", "active")
            .unwrap();
    }

    let changes_before = db.conn().total_changes();

    db.merge_project_ids("proj", "proj").unwrap();

    let changes_after = db.conn().total_changes();
    assert_eq!(
        changes_before, changes_after,
        "self-merge must perform zero row modifications, insertions, or deletions"
    );
}

#[test]
fn test_merge_into_populated_target() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    // Seed target with existing rows
    let dst_emb = test_fake_embedder("existing target").unwrap();
    let dst_id = db
        .insert("dst", "existing target", &dst_emb, None, "fact", "active")
        .unwrap();

    // Seed source with rows to move
    let src_emb = test_fake_embedder("will move").unwrap();
    let src_id = db
        .insert("src", "will move", &src_emb, None, "fact", "active")
        .unwrap();

    assert_eq!(count_project_rows(&db, "src"), 1);
    assert_eq!(count_project_rows(&db, "dst"), 1);

    let rows_moved = db.merge_project_ids("src", "dst").unwrap();
    assert_eq!(rows_moved, 1);

    // Source is empty, target has both original + moved rows
    assert_eq!(count_project_rows(&db, "src"), 0);
    assert_eq!(count_project_rows(&db, "dst"), 2);

    // Original target row still intact
    let (dst_content, _, _, _) = get_row_fields(&db, "dst", &dst_id);
    assert_eq!(dst_content, "existing target");

    // Moved row intact
    let (src_content, _, _, _) = get_row_fields(&db, "dst", &src_id);
    assert_eq!(src_content, "will move");
}

#[test]
fn test_fts_search_returns_merged_rows() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    let emb = test_fake_embedder("findable content").unwrap();
    let id = db
        .insert("src", "findable content", &emb, None, "fact", "active")
        .unwrap();

    // Before merge: row is under "src"
    let results_before = db.search_bm25("findable", "dst", 10, None, None).unwrap();
    assert!(results_before.is_empty());

    let results_before_src = db.search_bm25("findable", "src", 10, None, None).unwrap();
    assert!(!results_before_src.is_empty());

    // Merge
    db.merge_project_ids("src", "dst").unwrap();

    // After merge: row is under "dst", not "src"
    let results_after = db.search_bm25("findable", "dst", 10, None, None).unwrap();
    assert_eq!(results_after.len(), 1);
    assert_eq!(results_after[0].id, id);

    let results_after_src = db.search_bm25("findable", "src", 10, None, None).unwrap();
    assert!(results_after_src.is_empty());
}

#[test]
fn test_row_counts_preserved_across_both_projects() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    // Source: 3 rows
    for i in 0..3 {
        let emb = test_fake_embedder(&format!("src row {}", i)).unwrap();
        db.insert(
            "src",
            &format!("src row {}", i),
            &emb,
            None,
            "fact",
            "active",
        )
        .unwrap();
    }
    // Target: 2 rows
    for i in 0..2 {
        let emb = test_fake_embedder(&format!("dst row {}", i)).unwrap();
        db.insert(
            "dst",
            &format!("dst row {}", i),
            &emb,
            None,
            "fact",
            "active",
        )
        .unwrap();
    }

    let total_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    assert_eq!(total_before, 5);

    db.merge_project_ids("src", "dst").unwrap();

    let total_after: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    assert_eq!(total_after, 5);

    assert_eq!(count_project_rows(&db, "src"), 0);
    assert_eq!(count_project_rows(&db, "dst"), 5);
}

#[test]
fn test_metadata_preserved() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("meta test").unwrap();
    let metadata = r#"{"key": "value", "nested": {"a": 1}}"#;
    let id = db
        .insert("src", "meta test", &emb, Some(metadata), "fact", "active")
        .unwrap();

    db.merge_project_ids("src", "dst").unwrap();

    let stored_meta: String = db
        .conn()
        .query_row(
            "SELECT metadata FROM memories WHERE project_id = 'dst' AND id = ?",
            [&id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stored_meta, metadata);
}

#[test]
fn test_content_preserved() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("content test").unwrap();
    let id = db
        .insert("src", "content test", &emb, None, "fact", "active")
        .unwrap();

    db.merge_project_ids("src", "dst").unwrap();

    let content: String = db
        .conn()
        .query_row(
            "SELECT content FROM memories WHERE project_id = 'dst' AND id = ?",
            [&id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(content, "content test");
}

#[test]
fn test_type_and_status_preserved() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("type test").unwrap();
    let id = db
        .insert("src", "type test", &emb, None, "preference", "candidate")
        .unwrap();

    db.merge_project_ids("src", "dst").unwrap();

    let (typ, status): (String, String) = db
        .conn()
        .query_row(
            "SELECT type, status FROM memories WHERE project_id = 'dst' AND id = ?",
            [&id],
            |row| Ok((row.get(0).unwrap(), row.get(1).unwrap())),
        )
        .unwrap();
    assert_eq!(typ, "preference");
    assert_eq!(status, "candidate");
}

#[test]
fn test_merge_empty_source() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let emb = test_fake_embedder("only target").unwrap();
    db.insert("dst", "only target", &emb, None, "fact", "active")
        .unwrap();

    let rows_moved = db.merge_project_ids("nonexistent", "dst").unwrap();
    assert_eq!(rows_moved, 0);
    assert_eq!(count_project_rows(&db, "dst"), 1);
}

#[test]
fn test_mcp_restart_notice_contains_required_content() {
    let msg = build_mcp_restart_notice();

    // Must mention MCP server
    assert!(
        msg.contains("MCP server"),
        "message must mention MCP server"
    );
    // Must mention project_id is held from startup
    assert!(
        msg.contains("project_id"),
        "message must mention project_id"
    );
    assert!(msg.contains("startup"), "message must mention startup");
    // Must instruct to restart
    assert!(
        msg.contains("restart") || msg.contains("Restart"),
        "message must instruct to restart"
    );
}
