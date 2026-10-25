//! Integration tests for `vipune prune` (issue #194, sub-issue 5).
//!
//! These tests exercise the handler end-to-end (DB open → select → demote →
//! count rows) against a real SQLite database, verifying the never-deletes
//! invariant, the hard exclusions (guard-type, high/critical importance), and
//! the count/age threshold boundaries.
//!
//! The unit tests in `src/commands/prune.rs` cover the evaluator in isolation;
//! this file covers the handler wiring (CLI args → evaluator → response).

use crate::commands::prune::{
    DEFAULT_PRUNE_AGE_DAYS, DEFAULT_PRUNE_COUNT, PruneEvaluator, PruneResponse,
};
use crate::memory::crud::test_fake_embedder;
use crate::sqlite::Database;

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn count_rows(db: &Database) -> usize {
    db.conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap() as usize
}

fn status_of(db: &Database, id: &str) -> String {
    db.conn()
        .query_row("SELECT status FROM memories WHERE id = ?", [id], |row| {
            row.get::<_, String>(0)
        })
        .unwrap()
}

fn seed(
    db: &Database,
    project_id: &str,
    content: &str,
    created_at: &str,
    memory_type: &str,
    status: &str,
) -> String {
    let emb = test_fake_embedder(content).unwrap();
    db.insert_with_time(
        project_id,
        content,
        &emb,
        None,
        created_at,
        created_at,
        memory_type,
        status,
    )
    .unwrap()
}

fn set_retrieval(db: &Database, id: &str, count: i64) {
    db.conn()
        .execute(
            "UPDATE memories SET retrieval_count = ? WHERE id = ?",
            (count, id),
        )
        .unwrap();
}

const OLD: &str = "2020-01-01T00:00:00Z";

#[test]
fn test_prune_handler_end_to_end() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    // Seed: 2 eligible candidates + 1 guard (excluded) + 1 active (not candidate).
    let e1 = seed(&db, "p", "eligible one", OLD, "fact", "candidate");
    set_retrieval(&db, &e1, 0);
    let e2 = seed(&db, "p", "eligible two", OLD, "preference", "candidate");
    set_retrieval(&db, &e2, 1);
    let guard = seed(&db, "p", "guard row", OLD, "guard", "candidate");
    set_retrieval(&db, &guard, 0);
    let active = seed(&db, "p", "active row", OLD, "fact", "active");
    set_retrieval(&db, &active, 0);

    let before = count_rows(&db);
    assert_eq!(before, 4);

    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    let after = count_rows(&db);
    assert_eq!(
        before, after,
        "prune must never delete: row count must be unchanged"
    );
    assert_eq!(demoted, vec![e1.clone(), e2.clone()]);
    assert_eq!(status_of(&db, &e1), "deprecated");
    assert_eq!(status_of(&db, &e2), "deprecated");
    assert_eq!(
        status_of(&db, &guard),
        "candidate",
        "guard-type must never be demoted"
    );
    assert_eq!(status_of(&db, &active), "active");
}

#[test]
fn test_prune_handler_no_eligible_rows_is_noop() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    // All active (not candidate) → nothing to demote.
    let a = seed(&db, "p", "active one", OLD, "fact", "active");
    set_retrieval(&db, &a, 0);
    let b = seed(&db, "p", "active two", OLD, "fact", "active");
    set_retrieval(&db, &b, 0);

    let before = count_rows(&db);
    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    let after = count_rows(&db);
    assert_eq!(before, after);
    assert!(demoted.is_empty());
    assert_eq!(status_of(&db, &a), "active");
    assert_eq!(status_of(&db, &b), "active");
}

#[test]
fn test_prune_handler_multiple_projects() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    let a = seed(&db, "a", "stale a", OLD, "fact", "candidate");
    set_retrieval(&db, &a, 0);
    let b = seed(&db, "b", "stale b", OLD, "fact", "candidate");
    set_retrieval(&db, &b, 0);

    let before = count_rows(&db);
    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let mut all = Vec::new();
    for p in ["a", "b"] {
        all.extend(ev.run_for_project(&mut db, p).unwrap());
    }
    let after = count_rows(&db);

    assert_eq!(before, after, "row count must be unchanged");
    assert_eq!(all.len(), 2);
    assert!(all.contains(&a));
    assert!(all.contains(&b));
}

#[test]
fn test_prune_handler_respects_count_threshold() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    // N=3: retrieval_count 2 is eligible (< 3), retrieval_count 3 is not.
    let below = seed(&db, "p", "count two", OLD, "fact", "candidate");
    set_retrieval(&db, &below, 2);
    let at = seed(&db, "p", "count three", OLD, "fact", "candidate");
    set_retrieval(&db, &at, 3);

    let ev = PruneEvaluator::new(3, 7);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    assert_eq!(
        demoted,
        vec![below.clone()],
        "retrieval_count < N is strict"
    );
    assert_eq!(status_of(&db, &below), "deprecated");
    assert_eq!(status_of(&db, &at), "candidate");
}

#[test]
fn test_prune_handler_respects_age_threshold() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();

    // T = 7 days. A row created 3 days before "now" is not old enough.
    let recent = chrono::Utc::now() - chrono::Duration::days(3);
    let recent_rfc = recent.to_rfc3339();
    let id = seed(&db, "p", "recent enough", &recent_rfc, "fact", "candidate");
    set_retrieval(&db, &id, 0);

    let ev = PruneEvaluator::new(3, 7);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    assert!(
        demoted.is_empty(),
        "age must strictly exceed T to be eligible"
    );
    assert_eq!(status_of(&db, &id), "candidate");
}

#[test]
fn test_prune_defaults_are_3_and_14() {
    assert_eq!(DEFAULT_PRUNE_COUNT, 3);
    assert_eq!(DEFAULT_PRUNE_AGE_DAYS, 14);
}

#[test]
fn test_prune_response_serializes_correctly() {
    let response = PruneResponse {
        demoted: 2,
        demoted_ids: vec!["id-1".to_string(), "id-2".to_string()],
        rows_before: 10,
        rows_after: 10,
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"demoted\":2"));
    assert!(json.contains("\"rows_before\":10"));
    assert!(json.contains("\"rows_after\":10"));
    assert!(json.contains("id-1"));
    assert!(json.contains("id-2"));
}

#[test]
fn test_prune_importance_high_never_demoted_integration() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let id = seed(&db, "p", "high importance", OLD, "fact", "candidate");
    set_retrieval(&db, &id, 0);
    db.conn()
        .execute(
            "UPDATE memories SET importance = 'high' WHERE id = ?",
            [id.clone()],
        )
        .unwrap();

    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    assert!(
        demoted.is_empty(),
        "high-importance rows are a HARD exclusion — never demoted"
    );
    assert_eq!(status_of(&db, &id), "candidate");
}

#[test]
fn test_prune_importance_critical_never_demoted_integration() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let id = seed(&db, "p", "critical importance", OLD, "fact", "candidate");
    set_retrieval(&db, &id, 0);
    db.conn()
        .execute(
            "UPDATE memories SET importance = 'critical' WHERE id = ?",
            [id.clone()],
        )
        .unwrap();

    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    assert!(
        demoted.is_empty(),
        "critical-importance rows are a HARD exclusion — never demoted"
    );
    assert_eq!(status_of(&db, &id), "candidate");
}

#[test]
fn test_prune_importance_medium_low_still_eligible_integration() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    let med = seed(&db, "p", "medium", OLD, "fact", "candidate");
    set_retrieval(&db, &med, 0);
    db.conn()
        .execute(
            "UPDATE memories SET importance = 'medium' WHERE id = ?",
            [med.clone()],
        )
        .unwrap();
    let low = seed(&db, "p", "low", OLD, "fact", "candidate");
    set_retrieval(&db, &low, 0);
    db.conn()
        .execute(
            "UPDATE memories SET importance = 'low' WHERE id = ?",
            [low.clone()],
        )
        .unwrap();

    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    assert_eq!(
        demoted.len(),
        2,
        "medium and low importance remain eligible"
    );
}

/// Issue #238: the level set **at add time** has to be the one prune reads.
///
/// Every importance-exclusion test above seeds with `insert_with_time` and then
/// `UPDATE`s the column — that proves the evaluator honours the column, not
/// that `vipune add --importance critical` ever wrote it. Here the level comes
/// only from the store's add path.
#[test]
fn test_add_time_importance_protects_from_prune() {
    use crate::memory::NewMemory;
    use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
    use crate::memory::store::MemoryStore;
    use crate::memory_types::{AddResult, IngestPolicy};

    let (_dir, db_path) = create_test_db();
    let mut store = MemoryStore::from_db_with_test_embedder(Database::open(&db_path).unwrap());
    let mut added = |level: MemoryImportance, text: &str| -> String {
        match store
            .ingest_with_importance(
                &NewMemory {
                    project_id: "p",
                    content: text,
                    metadata: None,
                    memory_type: MemoryType::Fact,
                    status: MemoryStatus::Candidate,
                },
                IngestPolicy::Force,
                level,
            )
            .unwrap()
        {
            AddResult::Added { id } => id,
            other => panic!("a forced add stores the row, got {other:?}"),
        }
    };
    let critical = added(MemoryImportance::Critical, "added as critical");
    let low = added(MemoryImportance::Low, "added as low");
    drop(store);

    let mut db = Database::open(&db_path).unwrap();
    // Age both rows past the threshold, deliberately NOT touching importance:
    // had the add path dropped the level, `critical` would sit on the DEFAULT
    // ('medium') and be demoted like its low-importance neighbour.
    for id in [&critical, &low] {
        set_retrieval(&db, id, 0);
        db.conn()
            .execute(
                "UPDATE memories SET created_at = ?1, updated_at = ?1 WHERE id = ?2",
                (OLD, id.clone()),
            )
            .unwrap();
    }

    let ev = PruneEvaluator::new(DEFAULT_PRUNE_COUNT, DEFAULT_PRUNE_AGE_DAYS);
    let demoted = ev.run_for_project(&mut db, "p").unwrap();

    assert_eq!(
        demoted.len(),
        1,
        "only the row added at low is eligible; the one added critical is a hard exclusion"
    );
    assert_eq!(status_of(&db, &critical), "candidate");
    assert_eq!(status_of(&db, &low), "deprecated");
}

/// The partial-failure gap the ignored case used to "pin": a lock partway
/// through the per-row demotion loop leaves earlier rows already demoted while
/// `handle_prune` reports only the locked message. Asserting it through the
/// handler was impossible without fabricating a mid-loop lock, so that case was
/// deleted (a `#[ignore]`d test asserting the behaviour which already holds
/// discriminates nothing). This reproduces the gap against the same demotion
/// path the handler drives — one `Database::update` per row — and asserts the
/// two halves of it: rows committed before the lock stay demoted, and the
/// failed update yields no ids, which is exactly what `wrap_busy_error`
/// discards on its way to the locked message.
#[test]
fn a_lock_mid_demotion_loop_leaves_rows_demoted_with_no_report_of_them() {
    use crate::memory::lifecycle::MemoryStatus;
    use crate::sqlite::UpdateOptions;

    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let old = (chrono::Utc::now() - chrono::Duration::days(30)).to_rfc3339();
    let ids: Vec<String> = (0..3)
        .map(|i| {
            db.insert_with_time(
                "proj-a",
                &format!("stale {i}"),
                &[0.1f32; crate::embedding::EMBEDDING_DIMS],
                None,
                &old,
                &old,
                "fact",
                "candidate",
            )
            .unwrap()
        })
        .collect();

    let demote = |id: &str| {
        db.update(
            id,
            "proj-a",
            UpdateOptions {
                content: None,
                embedding: None,
                metadata: None,
                memory_type: None,
                status: Some(MemoryStatus::Deprecated.as_str()),
                importance: None,
            },
        )
    };

    // Demote one row, then have a writer take the exclusive lock: exactly the
    // mid-loop contention a running MCP server produces.
    demote(&ids[0]).unwrap();
    let lock_conn = rusqlite::Connection::open(&db_path).unwrap();
    // Half one of the gap, asserted BEFORE the writer is taken so the read is
    // unobstructed: `Database::update` commits each row, so the demotion that
    // happened before the lock is durable — and it is exactly the id the loop
    // would have reported, which `handle_prune` discards via
    // `wrap_busy_error` on its way to the locked message.
    assert_eq!(
        status_of(&db, &ids[0]),
        "deprecated",
        "the row demoted before the lock is committed"
    );

    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();
    // Zero timeout: the contended update must fail fast, not wait.
    db.set_busy_timeout(std::time::Duration::ZERO).unwrap();

    let err = demote(&ids[1]).expect_err("the contended update must fail fast");
    let err: crate::errors::Error = err.into();
    assert!(
        err.is_locked(),
        "the mid-loop failure is a lock, got {err:?}"
    );
    assert!(
        err.to_string().contains("locked"),
        "the locked message carries no partial count, got: {err}"
    );

    lock_conn.execute("ROLLBACK", []).unwrap();
}
