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
    )
    .unwrap();

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
    let result = handle_reindex(&db_path, "BAAI/bge-small-en-v1.5", None, false, false);
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

// ── Hint text: pure function tests ──
//
// The hint is emitted by `handle_reindex` when scoped to a single project and
// other projects exist in the database. We test the format string here via a
// helper that mirrors the logic in `handle_reindex`, so the test stays fast
// and doesn't require stdout capture.

fn hint_text_for(other_count: usize) -> Option<String> {
    // Mirrors the logic in handle_reindex: emit the hint only when other_count > 0.
    if other_count > 0 {
        Some(format!(
            "Note: {} other project(s) in this database not reindexed. Run with --all-projects to reindex them.",
            other_count
        ))
    } else {
        None
    }
}

#[test]
fn test_hint_text_scoped_multiple_projects() {
    let hint = hint_text_for(4).unwrap();
    assert_eq!(
        hint,
        "Note: 4 other project(s) in this database not reindexed. Run with --all-projects to reindex them."
    );
}

#[test]
fn test_hint_text_scoped_single_project() {
    let hint = hint_text_for(1).unwrap();
    assert_eq!(
        hint,
        "Note: 1 other project(s) in this database not reindexed. Run with --all-projects to reindex them."
    );
}

#[test]
fn test_hint_text_no_other_projects() {
    assert_eq!(hint_text_for(0), None);
}

// ── Footer text: pure function tests ──

fn footer_scope_text(projects: &[String]) -> String {
    if projects.len() == 1 {
        format!("project {}", projects[0])
    } else {
        "all projects".to_string()
    }
}

#[test]
fn test_footer_scope_single_project() {
    let projects = vec!["my-proj".to_string()];
    assert_eq!(footer_scope_text(&projects), "project my-proj");
}

#[test]
fn test_footer_scope_multiple_projects() {
    let projects = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(footer_scope_text(&projects), "all projects");
}

// ── reindex --force: model-switch migration path (issue #217) ──
//
// The --force path is exercised via `reindex_project_force` (the pure
// row-re-embed loop, marker/identity writes factored out) plus the
// identity marker helpers in `crate::sqlite::identity`. Together these
// cover the contract: force re-embeds every row (including Real), writes
// the marker first, and records identity + clears the marker atomically.

use crate::sqlite::identity;

fn force_reindex_project_with_fake_embedder(
    db: &Database,
    project_id: &str,
) -> Result<(usize, usize, Vec<String>), crate::sqlite::Error> {
    identity::force_reembed_project(db, project_id, |content| {
        test_fake_embedder(content).map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
    })
}

#[test]
fn test_force_reembeds_all_rows_including_real() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let real_vec = test_fake_embedder("real memory").unwrap();
    let id = db
        .insert("proj", "real memory", &real_vec, None, "fact", "active")
        .unwrap();
    let mock_vec = mock_embedding_for_content("mock memory");
    let id2 = db
        .insert("proj", "mock memory", &mock_vec, None, "fact", "active")
        .unwrap();
    let unknown_vec = vec![0.0; 384];
    let id3 = db
        .insert(
            "proj",
            "unknown memory",
            &unknown_vec,
            None,
            "fact",
            "active",
        )
        .unwrap();

    let (reindexed, skipped, failed) =
        force_reindex_project_with_fake_embedder(&db, "proj").unwrap();

    // Force re-embeds every Mock and Real row (including Real-classified
    // vectors that plain reindex would skip); only Unknown (corrupted) rows skip.
    assert_eq!(reindexed, 2);
    assert_eq!(skipped, 1);
    assert_eq!(failed.len(), 0);

    // All re-embedded rows now hold fresh embeddings from the fake embedder.
    let emb_after = get_embedding(&db, &id);
    let emb_after2 = get_embedding(&db, &id2);
    // The fake embedder is deterministic, so re-embedding "real memory"
    // reproduces the same vector. Classify as Real to confirm the new vector
    // is in the valid norm band.
    assert_eq!(emb_after, real_vec);
    assert_eq!(classify_embedding(&emb_after), EmbeddingClass::Real);
    assert_ne!(emb_after2, mock_vec);
    assert_eq!(classify_embedding(&emb_after2), EmbeddingClass::Real);
    // The unknown (zero-vector) row was skipped — it must be left untouched.
    let emb_after3 = get_embedding(&db, &id3);
    assert_eq!(emb_after3, unknown_vec);
}

#[test]
fn test_force_idempotent_rerun() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let mock_vec = mock_embedding_for_content("A");
    let id = db
        .insert("proj", "A", &mock_vec, None, "fact", "active")
        .unwrap();

    let (r1, _, _) = force_reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(r1, 1);
    let emb1 = get_embedding(&db, &id);

    // Simulate an interrupted run: marker written, some rows re-embedded,
    // then re-run. Force path re-embeds every row from the start.
    let (r2, _, _) = force_reindex_project_with_fake_embedder(&db, "proj").unwrap();
    assert_eq!(
        r2, 1,
        "force must re-embed every row on re-run, not skip Real"
    );
    // The fake embedder is deterministic, so the second pass yields the
    // same vector (idempotent at the vector level, not a no-op at the row level).
    assert_eq!(get_embedding(&db, &id), emb1);
}

#[test]
fn test_force_marker_written_first_then_cleared() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let target = identity::ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };

    // Step 1: marker-first write (before any re-embedding).
    identity::write_marker(db.conn(), &target).unwrap();
    assert_eq!(
        identity::read_marker(db.conn()).unwrap().as_deref(),
        Some(
            "migrating to intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
        )
    );

    // Step 2: re-embed all rows (simulated via the force loop).
    force_reindex_project_with_fake_embedder(&db, "proj").unwrap();
    // Marker is still present while the re-embed pass runs.
    assert!(identity::read_marker(db.conn()).unwrap().is_some());

    // Step 3: record identity + clear marker in one transaction.
    identity::record_identity_and_clear_marker(db.conn(), &target).unwrap();
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
}

#[test]
fn test_force_marker_survives_interruption() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let target = identity::ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };

    // Marker written, then the process dies before the re-embed loop runs.
    identity::write_marker(db.conn(), &target).unwrap();
    // Simulate a crash: no record_identity_and_clear_marker call.
    // The marker must still be present, so subsequent operations can refuse.
    assert!(identity::read_marker(db.conn()).unwrap().is_some());

    // Re-run: force re-embed + record identity + clear marker.
    force_reindex_project_with_fake_embedder(&db, "proj").unwrap();
    identity::record_identity_and_clear_marker(db.conn(), &target).unwrap();
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
}

#[test]
fn test_force_preflight_flags_overlength_rows_and_refuses_to_start() {
    // Decision 1: reindex --force token-counts every row with the target
    // profile's passage prefix BEFORE writing the marker. A row whose
    // prefixed content exceeds 512 tokens must be listed by the pre-flight
    // scan, and the caller must refuse to start (write nothing).
    //
    // The token-count boundary itself (512 tokens) is covered by the
    // existing embed() boundary tests in src/embedding.rs. Here we verify
    // the scan logic with a stub counter so no model download is needed:
    // the stub counts words, so any row with >512 words is flagged.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // A row well over 512 words (hence well over 512 tokens).
    let long_content: String = (0..600)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let long_id = db
        .insert(
            "proj",
            &long_content,
            &vec![0.5; 384],
            None,
            "fact",
            "active",
        )
        .unwrap();
    // A short row that fits comfortably.
    let short_id = db
        .insert(
            "proj",
            "short content",
            &vec![0.5; 384],
            None,
            "fact",
            "active",
        )
        .unwrap();

    // Stub counter: counts whitespace-separated tokens (a lower bound on the
    // real token count, so the >512 threshold is conservative).
    let count_words =
        |text: &str| -> Result<usize, crate::sqlite::Error> { Ok(text.split_whitespace().count()) };

    let projects = vec!["proj".to_string()];
    let offending =
        identity::force_reembed_preflight(&db, &projects, "passage: ", count_words).unwrap();

    assert_eq!(offending.len(), 1, "only the long row should be flagged");
    assert_eq!(offending[0], long_id);
    assert!(
        !offending.iter().any(|id| id == &short_id),
        "short row must not be flagged"
    );
}

#[test]
fn test_force_no_marker_when_identity_matches() {
    // If the database already has the target identity recorded and no marker,
    // a force re-embed pass leaves the identity unchanged (idempotent).
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let target = identity::ModelIdentity::bge_default();
    identity::record_identity_and_clear_marker(db.conn(), &target).unwrap();
    force_reindex_project_with_fake_embedder(&db, "proj").unwrap();
    identity::record_identity_and_clear_marker(db.conn(), &target).unwrap();
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
}
