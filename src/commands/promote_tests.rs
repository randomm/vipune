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
use crate::config::tests_utils::ConfigEnvGuard;
use crate::errors::Error;
use crate::memory::crud::test_fake_embedder;
use crate::sqlite::Database;

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
    let _guard = ConfigEnvGuard::acquire();
    assert_eq!(
        resolve_promotion_threshold().unwrap(),
        PROMOTION_THRESHOLD_DEFAULT
    );
}

#[test]
fn test_resolve_threshold_env_override() {
    let _guard = ConfigEnvGuard::acquire();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "12") };
    assert_eq!(resolve_promotion_threshold().unwrap(), 12);
}

#[test]
fn test_resolve_threshold_rejects_non_integer() {
    let _guard = ConfigEnvGuard::acquire();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "abc") };
    let err = resolve_promotion_threshold().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains(PROMOTION_THRESHOLD_ENV));
    assert!(msg.contains("abc"));
}

#[test]
fn test_resolve_threshold_rejects_negative() {
    let _guard = ConfigEnvGuard::acquire();
    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "-1") };
    let err = resolve_promotion_threshold().unwrap_err();
    assert!(err.to_string().contains(">= 0"));
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
    // Reads the threshold live: whole body under the shared target guard.
    let _guard = ConfigEnvGuard::acquire();
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
    // A sibling's lowered threshold would promote the row sitting at 4.
    let _guard = ConfigEnvGuard::acquire();
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
    // Taken before seeding: seed, set_var and the read share one section.
    let _guard = ConfigEnvGuard::acquire();
    let (_dir, path) = create_test_db();
    let db = Database::open(path.as_path()).unwrap();

    // With a threshold of 2, a candidate at 2 promotes; at 1 it does not.
    let promoted_one = seed_candidate(&db, "at two", 2);
    let not_promoted = seed_candidate(&db, "at one", 1);

    unsafe { std::env::set_var(PROMOTION_THRESHOLD_ENV, "2") };
    let promoted = run_promotion(&db, "proj").unwrap();

    assert_eq!(promoted, 1);
    assert_eq!(status_of(&db, &promoted_one), "active");
    assert_eq!(status_of(&db, &not_promoted), "candidate");
}

#[test]
fn test_run_promotion_is_idempotent() {
    // Reads the threshold on both runs, so the guard spans the body.
    let _guard = ConfigEnvGuard::acquire();
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
    // handle_promote resolves the threshold itself — a read under the guard.
    let _guard = ConfigEnvGuard::acquire();
    let (_dir, path) = create_test_db();
    // Seed one eligible candidate so the handler has something to promote.
    {
        let db = Database::open(path.as_path()).unwrap();
        seed_candidate(&db, "handler eligible", 5);
    }

    let exit = handle_promote(path.as_path(), "proj", true).expect("handle_promote should succeed");
    assert_eq!(exit, std::process::ExitCode::SUCCESS);
}

// ─────────────── open-path variant contract (issue #255) ───────────────

#[test]
fn a_lock_at_the_promote_open_renders_the_actionable_text_and_classifies_as_locked() {
    // ONE contract for all eight handlers (issue #255 review): every entry
    // point opens through `db_busy::open_db`, which (a) renders a lock as the
    // actionable `Config` text the operator reads and (b) never loses the
    // typed verdict on the way — the storage error it classified from is
    // `sqlite::Error::Locked`, whose public form is `Error::DatabaseLocked`.
    // The old split (five handlers flattening a lock through
    // `db_busy::open_db_error`, three preserving it) is gone, so this test no
    // longer has to name a laxer sibling; the shared mapping is pinned in
    // `commands::db_busy::tests`.
    let (_dir, path) = create_test_db();
    // The handler resolves the promotion threshold from the environment before
    // it opens the store, so hold the config guard for the whole test body to
    // keep that read from racing an env-mutating sibling.
    let _guard = ConfigEnvGuard::acquire();
    let lock_conn = rusqlite::Connection::open(&path).unwrap();
    lock_conn.execute("BEGIN EXCLUSIVE", []).unwrap();

    // `Database::open` blocks for SQLite's default busy timeout before it
    // fails (it runs schema creation and migrations before a zero timeout can
    // be applied), so this call takes ~5s; what is under test is the variant,
    // not the latency.
    // The CLI handler renders a lock into the actionable `Config` text — that
    // is the presentation contract in `commands::db_busy`. The variant
    // contract that handler depends on is the one the storage boundary gives
    // the error *before* that rendering, so assert both: the operator text
    // and the verdict it was classified from.
    let err = handle_promote(path.as_path(), "proj", false)
        .expect_err("a locked store must fail the open");
    match &err {
        Error::Config(msg) => assert!(
            msg.contains("Stop the MCP server and retry"),
            "the operator must be told what to stop, got: {msg}"
        ),
        other => panic!("expected the actionable locked message, got {other:?}"),
    }
    let storage = match Database::open(path.as_path()) {
        Ok(_) => panic!("the store is still locked, so the open must fail"),
        Err(e) => e,
    };
    assert!(
        matches!(storage, crate::sqlite::Error::Locked),
        "promote's open classifies the contention as a lock, got {storage:?}"
    );
    let public: Error = storage.into();
    assert!(
        matches!(public, Error::DatabaseLocked) && public.is_locked(),
        "the verdict promote's open relies on survives the public conversion, got {public:?}"
    );

    lock_conn.execute("ROLLBACK", []).unwrap();
}
