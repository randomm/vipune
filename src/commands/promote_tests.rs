//! Tests for `vipune promote` — candidate → active promotion (sub-issue 4).
//!
//! Covers the pure evaluator's strict `>=` boundary and status exclusions
//! directly, plus an end-to-end apply path that promotes an eligible candidate
//! via the existing status-update path and leaves ineligible rows untouched.

#![cfg(test)]

use crate::commands::promote::{
    PROMOTION_THRESHOLD_DEFAULT, PROMOTION_THRESHOLD_ENV, fetch_candidates, handle_promote,
    resolve_promotion_threshold, run_promotion, should_promote,
};
use crate::memory::crud::test_fake_embedder;
use crate::sqlite::Database;

/// Serialises the env-mutating promotion tests so the shared process
/// environment (`VIPUNE_PROMOTION_THRESHOLD`) is not raced across test
/// threads (cargo runs tests in parallel threads that share one process env).
static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("promote.db");
    Database::open(&path).unwrap();
    (dir, path)
}

/// Seed a row with the given status and retrieval_count.
fn seed_candidate(db: &Database, content: &str, retrieval_count: i64) -> String {
    let emb = test_fake_embedder(content).unwrap();
    let id = db
        .insert("proj", content, &emb, None, "fact", "candidate")
        .unwrap();
    db.conn()
        .execute(
            "UPDATE memories SET retrieval_count = ? WHERE id = ?",
            rusqlite::params![retrieval_count, id],
        )
        .unwrap();
    id
}

fn status_of(db: &Database, id: &str) -> String {
    db.conn()
        .query_row("SELECT status FROM memories WHERE id = ?", [id], |r| {
            r.get::<_, String>(0)
        })
        .unwrap()
}

fn total_rows(db: &Database) -> i64 {
    db.conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap()
}

// ─────────────────────────── pure evaluator ───────────────────────────

#[test]
fn test_should_promote_boundary_strict_greater_than_or_equal() {
    let threshold = PROMOTION_THRESHOLD_DEFAULT; // 5

    // 4 < 5 → NOT promoted (strict boundary, the exact edge the epic pins).
    assert!(!should_promote("candidate", 4, threshold));
    // 5 >= 5 → promoted.
    assert!(should_promote("candidate", 5, threshold));
    // 6 > 5 → promoted.
    assert!(should_promote("candidate", 6, threshold));
    // 0 → not promoted (a never-retrieved candidate never promotes).
    assert!(!should_promote("candidate", 0, threshold));
}

#[test]
fn test_should_promote_rejects_non_candidate_status() {
    // Even with a high retrieval count, only `candidate` rows promote.
    assert!(!should_promote("active", 100, 5));
    assert!(!should_promote("superseded", 100, 5));
    assert!(!should_promote("deprecated", 100, 5));
}

#[test]
fn test_should_promote_respects_configured_threshold() {
    // A custom threshold of 2: candidate with 1 not promoted, with 2 promoted.
    assert!(!should_promote("candidate", 1, 2));
    assert!(should_promote("candidate", 2, 2));
    // And the default 5-row below the custom threshold is still promoted
    // because it clears 2.
    assert!(should_promote("candidate", 5, 2));
}

// ─────────────────────────── threshold resolution ───────────────────────────

#[test]
fn test_resolve_threshold_defaults_when_unset() {
    let _guard = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::remove_var(PROMOTION_THRESHOLD_ENV) };
    assert_eq!(
        resolve_promotion_threshold().unwrap(),
        PROMOTION_THRESHOLD_DEFAULT
    );
}

#[test]
fn test_resolve_threshold_env_override() {
    let _guard = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "12") };
    assert_eq!(resolve_promotion_threshold().unwrap(), 12);
    unsafe { std::env::remove_var(PROMOTION_THRESHOLD_ENV) };
}

#[test]
fn test_resolve_threshold_rejects_non_integer() {
    let _guard = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "abc") };
    let err = resolve_promotion_threshold().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains(PROMOTION_THRESHOLD_ENV));
    assert!(msg.contains("abc"));
    unsafe { std::env::remove_var(PROMOTION_THRESHOLD_ENV) };
}

#[test]
fn test_resolve_threshold_rejects_negative() {
    let _guard = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "-1") };
    let err = resolve_promotion_threshold().unwrap_err();
    assert!(err.to_string().contains(">= 0"));
    unsafe { std::env::remove_var(PROMOTION_THRESHOLD_ENV) };
}

// ─────────────────────────── fetch candidates ───────────────────────────

#[test]
fn test_fetch_candidates_returns_only_candidate_rows() {
    let (_dir, _path) = create_test_db();
    let db = Database::open(_path.as_path()).unwrap();

    // One candidate, one active, one superseded, one deprecated.
    let _cand = seed_candidate(&db, "the candidate", 7);
    let emb_a = test_fake_embedder("an active").unwrap();
    let _active = db
        .insert("proj", "an active", &emb_a, None, "fact", "active")
        .unwrap();
    let emb_s = test_fake_embedder("a superseded").unwrap();
    let _sup = db
        .insert("proj", "a superseded", &emb_s, None, "fact", "superseded")
        .unwrap();
    let emb_d = test_fake_embedder("a deprecated").unwrap();
    let _dep = db
        .insert("proj", "a deprecated", &emb_d, None, "fact", "deprecated")
        .unwrap();

    let candidates = fetch_candidates(&db, "proj").unwrap();
    assert_eq!(candidates.len(), 1, "only the candidate row is fetched");
    assert_eq!(candidates[0].retrieval_count, 7);
    assert_eq!(candidates[0].status, "candidate");
}

// ─────────────────────────── apply path ───────────────────────────

#[test]
fn test_run_promotion_promotes_eligible_only() {
    let (_dir, path) = create_test_db();
    let db = Database::open(path.as_path()).unwrap();

    // Eligible: candidate with 5 (meets default threshold).
    let eligible = seed_candidate(&db, "eligible five", 5);
    // Ineligible: candidate with 4 (just under the default threshold).
    let ineligible = seed_candidate(&db, "ineligible four", 4);
    // Ineligible: superseded, even with a high count.
    let sup = {
        let emb = test_fake_embedder("sup high").unwrap();
        let id = db
            .insert("proj", "sup high", &emb, None, "fact", "superseded")
            .unwrap();
        db.conn()
            .execute(
                "UPDATE memories SET retrieval_count = 10 WHERE id = ?",
                [&id],
            )
            .unwrap();
        id
    };

    let rows_before = total_rows(&db);

    let promoted = run_promotion(&db, "proj").unwrap();
    assert_eq!(promoted, 1, "exactly one row should promote");

    assert_eq!(
        status_of(&db, &eligible),
        "active",
        "eligible candidate promoted"
    );
    assert_eq!(
        status_of(&db, &ineligible),
        "candidate",
        "candidate just under threshold stays candidate"
    );
    assert_eq!(
        status_of(&db, &sup),
        "superseded",
        "superseded row is never promoted"
    );

    // Promotion is an UPDATE, not a delete: total row count is unchanged.
    assert_eq!(total_rows(&db), rows_before, "no rows added or removed");
}

#[test]
fn test_run_promotion_zero_when_none_eligible() {
    let (_dir, path) = create_test_db();
    let db = Database::open(path.as_path()).unwrap();

    // A candidate below the threshold and an active row: nothing to promote.
    seed_candidate(&db, "low four", 4);
    let emb = test_fake_embedder("already active").unwrap();
    let _ = db
        .insert("proj", "already active", &emb, None, "fact", "active")
        .unwrap();

    let promoted = run_promotion(&db, "proj").unwrap();
    assert_eq!(promoted, 0);
}

#[test]
fn test_run_promotion_respects_env_threshold() {
    let (_dir, path) = create_test_db();
    let db = Database::open(path.as_path()).unwrap();

    // With a threshold of 2, a candidate at 2 promotes; at 1 it does not.
    let promoted_one = seed_candidate(&db, "at two", 2);
    let not_promoted = seed_candidate(&db, "at one", 1);

    let _guard = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "2") };
    let promoted = run_promotion(&db, "proj").unwrap();
    unsafe { std::env::remove_var(PROMOTION_THRESHOLD_ENV) };

    assert_eq!(promoted, 1);
    assert_eq!(status_of(&db, &promoted_one), "active");
    assert_eq!(status_of(&db, &not_promoted), "candidate");
}

#[test]
fn test_run_promotion_is_idempotent() {
    let (_dir, path) = create_test_db();
    let db = Database::open(path.as_path()).unwrap();

    let eligible = seed_candidate(&db, "eligible", 6);

    let first = run_promotion(&db, "proj").unwrap();
    assert_eq!(first, 1);

    // Second run: the row is now active (no longer a candidate), so zero
    // promote. The apply path only fetches candidates, so the already-promoted
    // row is invisible and not double-counted.
    let second = run_promotion(&db, "proj").unwrap();
    assert_eq!(second, 0);
    assert_eq!(status_of(&db, &eligible), "active");
}

// ─────────────────────────── handler ───────────────────────────

#[test]
fn test_handle_promote_json_and_success() {
    let (_dir, path) = create_test_db();
    // Seed one eligible candidate so the handler has something to promote.
    {
        let db = Database::open(path.as_path()).unwrap();
        seed_candidate(&db, "handler eligible", 5);
    }

    let exit = handle_promote(path.as_path(), "proj", true).expect("handle_promote should succeed");
    assert_eq!(exit, std::process::ExitCode::SUCCESS);
}
