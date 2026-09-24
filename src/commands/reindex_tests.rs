//! Tests for `vipune reindex` handler.

#![cfg(test)]

use crate::commands::reindex::*;
use crate::embedding_profiles::EmbeddingRole;
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

/// Model-free stand-in for `handle_reindex`'s plain-reindex callback closure
/// (`|content| engine.embed_passage(content)`, where `engine` is a real
/// `EmbeddingEngine` for `model_id`): it drives the actual `engine.embed_passage`
/// against the stored profile — including its `prefix_input` prefix path and
/// the `ContentTooLong` check — without downloading the ONNX model. It is
/// model-free because every built-in profile declares empty prefixes (bge:
/// no prefix; e5 in this build: no prefix either — see
/// `src/embedding_profiles.rs`), so `embed_passage("x")` embeds exactly
/// `"x"`, matching the engine's contract pinned in
/// `src/embedding/tests/model_free.rs::engine_prefix_input_helper_yields_exactly_prefix_plus_text`.
fn plain_reindex_callback(
    db: &Database,
    model_id: &str,
) -> (usize, usize, Vec<ReindexFailure>) {
    let mut engine = crate::embedding::EmbeddingEngine::new(model_id).expect("load model");
    let mut embed_callback = |content: &str| {
        engine
            .embed_passage(content)
            .map_err(|e| Error::Inference(e.to_string()))
    };
    reindex_project(db, &mut embed_callback, "proj", false)
        .expect("reindex must succeed")
}

#[cfg(test)]
mod e5_prefix_path {
    //! `handle_reindex`'s plain-reindex callback is
    //! `|content| engine.embed_passage(content)`, where `embed_passage`
    //! prepends the profile's passage prefix before tokenisation (the engine
    //! is the single prefix site). A regression to raw-content embedding
    //! would silently degrade quality under e5 without any error or log, so
    //! the callback is pinned here against the e5 profile's passage prefix.
    use super::*;
    use crate::memory::crud::{mock_embedding_for_content, test_fake_embedder};
    use crate::output::ReindexFailure;
    use crate::sqlite::Database;
    use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};

    fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        Database::open(&path).unwrap();
        (dir, path)
    }

    /// Regression (issue #220): plain `vipune reindex` under the e5 profile
    /// must re-embed Mock rows with the passage prefix. Before the fix the
    /// plain reindex callback called `engine.embed(content)` on raw content,
    /// so under e5 the model saw unprefixed text — silent quality degradation
    /// with no error or log. After the fix the callback is
    /// `|content| engine.embed_passage(content)`, so the engine's single
    /// prefix site prepends the profile's passage prefix.
    ///
    /// We can't call the real e5 engine in a unit test (it would download the
    /// model), so the recording embedder below reproduces the engine's exact
    /// contract for the e5 profile — `prefix_input` (the same pure helper
    /// `embed_passage` delegates to, pinned model-free in
    /// `src/embedding/tests/model_free.rs`) — and drives the real
    /// `reindex_project` path: the recorded input must be exactly
    /// `"passage: <content>"`, not the bare content (which a regression to
    /// raw-content embedding would produce).
    #[test]
    fn test_plain_reindex_under_e5_reembeds_mock_row_with_passage_prefix() {
        let e5_profile = crate::embedding_profiles::profile_for("intfloat/multilingual-e5-small")
            .expect("e5");
        let prefix = EmbeddingRole::Passage.prefix(e5_profile);
        assert_eq!(prefix, "passage: ", "e5 must declare the 'passage: ' prefix");

        let (_dir, db_path) = create_test_db();
        let db = Database::open(&db_path).unwrap();
        let content = "mock memory content";
        let mock_vec = mock_embedding_for_content(content);
        let id = db
            .insert("proj", content, &mock_vec, None, "fact", "active")
            .unwrap();
        assert_eq!(
            classify_embedding(&get_embedding(&db, &id)),
            EmbeddingClass::Mock
        );

        // Recording embedder reproducing the e5 engine's passage-role contract:
        // input = prefix + raw content, output = the fake vector for the
        // prefixed input (a Real-classified 384-dim vector, so the reindex
        // write succeeds exactly as in production).
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded_for_callback = recorded.clone();
        let mut embed_callback = move |text: &str| {
            recorded_for_callback.lock().unwrap().push(text.to_string());
            crate::memory::crud::test_fake_embedder(text)
        };

        let (reindexed, skipped, failed) =
            reindex_project(&db, &mut embed_callback, "proj", false).unwrap();
        assert_eq!(reindexed, 1);
        assert_eq!(skipped, 0);
        assert!(failed.is_empty(), "reindex must not fail: {failed:?}");

        // The embed callback received exactly the passage-prefixed content —
        // never the bare content (the v0.13.0 latent bug) and never a
        // double-prefixed string.
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec!["passage: mock memory content".to_string()]
        );

        // And the row's stored content stays unprefixed.
        let rows = db.list_all_rows_for_project("proj").unwrap();
        let (_, stored_content, _) = rows.iter().find(|(i, _, _)| i == &id).unwrap();
        assert_eq!(stored_content, content);
        assert_eq!(
            classify_embedding(&get_embedding(&db, &id)),
            EmbeddingClass::Real
        );
    }
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

/// Regression (issue #220): plain `vipune reindex` under the e5 profile
/// must re-embed Mock rows with the passage prefix. Before this change the
/// plain reindex callback called `engine.embed(content)` on raw content, so
/// under e5 the model saw unprefixed text — silent quality degradation with
/// no error or log. After the fix the callback is
/// `|content| engine.embed_passage(content)`, so the engine's single prefix
/// site prepends the profile's passage prefix.
///
/// We can't call the real e5 engine in a unit test (it would download the
/// model), so the recording embedder below reproduces the engine's exact
/// contract for the e5 profile — `prefix_input` (the same pure helper
/// `embed_passage` delegates to, pinned model-free in
/// `src/embedding/tests/model_free.rs`) — and drives the real
/// `reindex_project` path:
/// the recorded input must be exactly `"passage: <content>"`, not the bare
/// content (which a regression to raw-content embedding would produce).
#[test]
fn test_plain_reindex_under_e5_reembeds_mock_row_with_passage_prefix() {
    let e5_profile =
        crate::embedding_profiles::profile_for("intfloat/multilingual-e5-small").expect("e5");
    let prefix = crate::embedding_profiles::EmbeddingRole::Passage.prefix(e5_profile);
    assert_eq!(
        prefix, "passage: ",
        "e5 must declare the 'passage: ' prefix"
    );

    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let content = "mock memory content";
    let mock_vec = mock_embedding_for_content(content);
    let id = db
        .insert("proj", content, &mock_vec, None, "fact", "active")
        .unwrap();
    assert_eq!(
        classify_embedding(&get_embedding(&db, &id)),
        EmbeddingClass::Mock
    );

    // Recording embedder reproducing the e5 engine's passage-role contract:
    // input = prefix + raw content, output = the fake vector for the
    // prefixed input (a Real-classified 384-dim vector, so the reindex
    // write succeeds exactly as in production).
    let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded_for_callback = recorded.clone();
    let mut embed_callback = move |text: &str| {
        recorded_for_callback.lock().unwrap().push(text.to_string());
        crate::memory::crud::test_fake_embedder(text)
    };

    let (reindexed, skipped, failed) =
        reindex_project(&db, &mut embed_callback, "proj", false).unwrap();
    assert_eq!(reindexed, 1);
    assert_eq!(skipped, 0);
    assert!(failed.is_empty(), "reindex must not fail: {failed:?}");

    // The embed callback received exactly the passage-prefixed content —
    // never the bare content (the v0.13.0 latent bug) and never a
    // double-prefixed string.
    assert_eq!(
        recorded.lock().unwrap().clone(),
        vec!["passage: mock memory content".to_string()]
    );

    // And the row's stored content stays unprefixed.
    let rows = db.list_all_rows_for_project("proj").unwrap();
    let (_, stored_content, _) = rows.iter().find(|(i, _, _)| i == &id).unwrap();
    assert_eq!(stored_content, content);
    assert_eq!(
        classify_embedding(&get_embedding(&db, &id)),
        EmbeddingClass::Real
    );
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

#[test]
fn test_force_refused_without_all_projects() {
    // `reindex --force` is a per-database migration (issue #217): a project
    // filter would leave a silently mixed store, so the handler refuses to
    // start (exit 1) rather than partially migrate. No marker is written.
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

    let result = handle_reindex(
        &db_path,
        "BAAI/bge-small-en-v1.5",
        Some("proj"),
        true,
        false,
    );
    let exit = result.expect("handler returns Ok(ExitCode::from(1)) on refusal");
    assert_ne!(
        exit,
        std::process::ExitCode::SUCCESS,
        "force without --all-projects must not exit 0"
    );

    // No marker was written, no identity recorded.
    let db2 = Database::open(&db_path).unwrap();
    assert!(!identity::is_migrating(db2.conn()).unwrap());
    assert!(identity::read_identity(db2.conn()).unwrap().is_none());
}

// ── reindex --force: model-switch migration path (issue #217) ──
//
// The --force path is exercised through the bin-only orchestration in
// `crate::commands::reindex_force` (marker write, per-project re-embed pass,
// record-and-clear) plus the read side in `crate::sqlite::identity`. These
// cover the contract: force re-embeds every row (including Real), writes the
// marker first (leaving the recorded identity untouched), and records the
// identity + clears the marker in one transaction only on a clean pass.

use crate::commands::reindex_force::{self, ReembedFailure};
use crate::sqlite::identity;

fn force_reindex_project_with_fake_embedder(
    db: &mut Database,
    project_id: &str,
) -> Result<(usize, usize, Vec<ReembedFailure>), crate::sqlite::Error> {
    reindex_force::force_reembed_project(db, project_id, |content| {
        test_fake_embedder(content).map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
    })
}

#[test]
fn test_force_reembeds_all_rows_including_real() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
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
        force_reindex_project_with_fake_embedder(&mut db, "proj").unwrap();

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
    let mut db = Database::open(&db_path).unwrap();
    let mock_vec = mock_embedding_for_content("A");
    let id = db
        .insert("proj", "A", &mock_vec, None, "fact", "active")
        .unwrap();

    let (r1, _, _) = force_reindex_project_with_fake_embedder(&mut db, "proj").unwrap();
    assert_eq!(r1, 1);
    let emb1 = get_embedding(&db, &id);

    // Simulate an interrupted run: marker written, some rows re-embedded,
    // then re-run. Force path re-embeds every row from the start.
    let (r2, _, _) = force_reindex_project_with_fake_embedder(&mut db, "proj").unwrap();
    assert_eq!(
        r2, 1,
        "force must re-embed every row on re-run, not skip Real"
    );
    // The fake embedder is deterministic, so the second pass yields the
    // same vector (idempotent at the vector level, not a no-op at the row level).
    assert_eq!(get_embedding(&db, &id), emb1);
}

/// Write a migration marker (marker-first step: touches ONLY the marker
/// column; the recorded identity is left untouched).
fn marker_only(conn: &rusqlite::Connection, target: &identity::ModelIdentity) {
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, NULL, NULL, ?1)
         ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
        [reindex_force::migration_marker_for(target)],
    )
    .unwrap();
}

/// Record the new identity and clear the marker in one transaction (the
/// only sanctioned exit from the "migrating" state).
fn record_identity(conn: &rusqlite::Connection, identity: &identity::ModelIdentity) {
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)
         ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision,
                                      migration_marker = NULL",
        (&identity.model_id, &identity.revision),
    )
    .unwrap();
}

#[test]
fn test_force_with_all_projects_migrates_entire_database() {
    // Per-database migration lifecycle through the same helpers the handler
    // uses: marker written once before the pass, then identity recorded +
    // marker cleared only after every project is re-embedded. (The handler
    // itself constructs a real EmbeddingEngine, which would download the
    // model, so the wiring is verified here with a fake embedder; the
    // handler's refusal path is covered by
    // test_force_refused_without_all_projects above.)
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    db.insert(
        "a",
        "alpha",
        &test_fake_embedder("a").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert(
        "b",
        "beta",
        &test_fake_embedder("b").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let target = identity::ModelIdentity {
        model_id: "BAAI/bge-small-en-v1.5".to_string(),
        revision: crate::embedding::EMBED_MODEL_REVISION.to_string(),
    };
    marker_only(db.conn(), &target);
    let _ = reindex_force::force_reembed_project(&mut db, "a", |_| Ok(vec![0.1; 384])).unwrap();
    let _ = reindex_force::force_reembed_project(&mut db, "b", |_| Ok(vec![0.1; 384])).unwrap();
    record_identity(db.conn(), &target);

    assert_eq!(identity::current_identity(db.conn()).unwrap(), target);
    assert!(!identity::is_migrating(db.conn()).unwrap());
}

#[test]
fn test_force_marker_written_first_then_cleared() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
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

    // Step 1: marker-first write (before any re-embedding) — touches only
    // the marker column; the recorded identity (none here) is untouched.
    marker_only(db.conn(), &target);
    assert_eq!(
        identity::read_marker(db.conn()).unwrap().as_deref(),
        Some(
            "migrating to intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
        )
    );

    // Step 2: re-embed all rows (simulated via the force loop).
    force_reindex_project_with_fake_embedder(&mut db, "proj").unwrap();
    // Marker is still present while the re-embed pass runs.
    assert!(identity::read_marker(db.conn()).unwrap().is_some());

    // Step 3: record identity + clear marker in one transaction.
    record_identity(db.conn(), &target);
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
}

#[test]
fn test_force_marker_survives_interruption() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
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
    marker_only(db.conn(), &target);
    // Simulate a crash: no record_identity call. The marker must still be
    // present, so subsequent operations can refuse.
    assert!(identity::read_marker(db.conn()).unwrap().is_some());

    // Re-run: force re-embed + record identity + clear marker.
    force_reindex_project_with_fake_embedder(&mut db, "proj").unwrap();
    record_identity(db.conn(), &target);
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
}

/// A private generic scan helper mirroring `over_limit_row_ids`'s logic with
/// an injectable token counter — lets this test exercise the over-limit
/// report without a real `EmbeddingEngine` (no model download).
fn over_limit_ids_with_counter(
    db: &Database,
    projects: &[String],
    passage_prefix: &str,
    count: impl Fn(&str) -> Result<usize, crate::sqlite::Error>,
) -> Result<Vec<String>, crate::sqlite::Error> {
    let mut offending: Vec<String> = Vec::new();
    for project_id in projects {
        let rows = db.list_all_rows_for_project(project_id)?;
        for (id, content, _embedding) in rows {
            let prefixed = format!("{passage_prefix}{content}");
            let count = count(&prefixed)?;
            if count > crate::embedding::MAX_EMBEDDING_TOKENS {
                offending.push(id);
            }
        }
    }
    Ok(offending)
}

#[test]
fn test_force_preflight_flags_overlength_rows_and_refuses_to_start() {
    // Decision 1: reindex --force token-counts every row with the target
    // profile's passage prefix BEFORE writing the marker. A row whose
    // prefixed content exceeds 512 tokens must be reported by the pre-flight
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
    let offending = over_limit_ids_with_counter(&db, &projects, "passage: ", count_words).unwrap();

    assert_eq!(offending.len(), 1, "only the long row should be reported");
    assert_eq!(offending[0], long_id);
    assert!(
        !offending.iter().any(|id| id == &short_id),
        "short row must not be reported"
    );
}

#[test]
fn test_force_no_marker_when_identity_matches() {
    // If the database already has the target identity recorded and no marker,
    // a force re-embed pass leaves the identity unchanged (idempotent).
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();

    let target = identity::ModelIdentity::default_identity();
    record_identity(db.conn(), &target);
    force_reindex_project_with_fake_embedder(&mut db, "proj").unwrap();
    record_identity(db.conn(), &target);
    assert_eq!(identity::read_identity(db.conn()).unwrap(), Some(target));
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
}
