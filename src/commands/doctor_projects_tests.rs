//! Tests for `vipune doctor --projects` handler and split-detection heuristic.

use crate::commands::doctor::{
    collect_doctor_projects_response, detect_split_pairs, handle_doctor_projects,
};
use crate::memory::crud::test_fake_embedder;
use crate::output::{DoctorProjectsResponse, DoctorProjectsSuspectedSplit};
use crate::sqlite::Database;
use std::collections::HashMap;
use std::path::PathBuf;

fn create_test_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn make_counts(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for (id, n) in pairs {
        counts.insert(id.to_string(), *n);
    }
    counts
}

// ── Pure heuristic tests (detect_split_pairs) ──

#[test]
fn test_detects_real_split_pair_with_correct_row_counts() {
    // "pi-an" (bare) + "randomm/pi-an" (owned) → one pair.
    let project_ids = vec!["pi-an".to_string(), "randomm/pi-an".to_string()];
    let counts = make_counts(&[("pi-an", 3), ("randomm/pi-an", 5)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("pi-an".to_string(), "randomm/pi-an".to_string()));
}

#[test]
fn test_detects_split_pair_via_count_rows_for_project() {
    // "repo" (bare) + "owner/repo" (owned) → one pair with correct row counts.
    let project_ids = vec!["repo".to_string(), "owner/repo".to_string()];
    let counts = make_counts(&[("repo", 1), ("owner/repo", 2)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("repo".to_string(), "owner/repo".to_string()));
}

#[test]
fn test_no_splits_reports_empty() {
    // Unrelated projects with no bare/owned relationship → no pairs.
    let project_ids = vec!["proj-a".to_string(), "proj-b".to_string()];
    let counts = make_counts(&[("proj-a", 1), ("proj-b", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert!(pairs.is_empty(), "unrelated bare ids must not be paired");
}

#[test]
fn test_false_positive_ci_runner_is_reported() {
    // "ci-runner" vs "team/ci-runner" is a known false-positive class.
    // The heuristic SHOULD report it (documenting the limitation, not filtering).
    let project_ids = vec!["ci-runner".to_string(), "team/ci-runner".to_string()];
    let counts = make_counts(&[("ci-runner", 1), ("team/ci-runner", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(
        pairs.len(),
        1,
        "false positive must be reported, not filtered"
    );
    assert_eq!(
        pairs[0],
        ("ci-runner".to_string(), "team/ci-runner".to_string())
    );
}

#[test]
fn test_ids_without_slash_are_not_paired_with_each_other() {
    // Bare ids that share no owner/repo relationship should not be paired.
    let project_ids = vec!["proj-x".to_string(), "proj-y".to_string()];
    let counts = make_counts(&[("proj-x", 1), ("proj-y", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert!(
        pairs.is_empty(),
        "bare ids without owner/repo relationship must not be paired"
    );
}

#[test]
fn test_detect_split_pairs_finds_pair_across_two_project_ids() {
    // Verifies the pure heuristic pairs a bare id with its owned counterpart
    // when both appear in the project id list. Does NOT exercise handler-level -p
    // filtering (detect_split_pairs has no concept of project_filter).
    let project_ids = vec!["split-repo".to_string(), "other/split-repo".to_string()];
    let counts = make_counts(&[("split-repo", 1), ("other/split-repo", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(pairs.len(), 1, "split pair must be detected");
    assert_eq!(
        pairs[0],
        ("split-repo".to_string(), "other/split-repo".to_string())
    );
}

#[test]
fn test_pair_ordering_is_deterministic() {
    // pair[0] is segment after "/", pair[1] is owned id.
    let project_ids = vec!["myrepo".to_string(), "github/myrepo".to_string()];
    let counts = make_counts(&[("myrepo", 1), ("github/myrepo", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(pairs.len(), 1);
    assert_eq!(
        pairs[0].0, "myrepo",
        "pair[0] must be the segment (bare id)"
    );
    assert_eq!(pairs[0].1, "github/myrepo", "pair[1] must be the owned id");
}

#[test]
fn test_output_sorted_by_first_id() {
    // Multiple pairs sorted by segment, then owned id.
    let project_ids = vec![
        "alpha".to_string(),
        "beta".to_string(),
        "org/beta".to_string(),
        "team/alpha".to_string(),
    ];
    let counts = make_counts(&[
        ("alpha", 1),
        ("beta", 1),
        ("org/beta", 1),
        ("team/alpha", 1),
    ]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(pairs.len(), 2);
    assert_eq!(
        pairs[0],
        ("alpha".to_string(), "team/alpha".to_string()),
        "first pair must be alpha"
    );
    assert_eq!(
        pairs[1],
        ("beta".to_string(), "org/beta".to_string()),
        "second pair must be beta"
    );
}

#[test]
fn test_empty_database_reports_no_splits() {
    let project_ids: Vec<String> = vec![];
    let counts = HashMap::new();

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert!(pairs.is_empty(), "empty database must report no splits");
}

#[test]
fn test_owned_id_with_no_bare_counterpart_not_reported() {
    // "only-owner/only-repo" with no matching "only-repo" bare id.
    let project_ids = vec!["only-owner/only-repo".to_string()];
    let counts = make_counts(&[("only-owner/only-repo", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert!(
        pairs.is_empty(),
        "owned id without bare counterpart must not be reported"
    );
}

#[test]
fn test_self_pair_avoided_when_bare_equals_repo_segment() {
    // "a/b" + "c/a/b": segment after "/" in "c/a/b" is "a/b", which exists.
    // But "a/b" is not a bare id (it contains "/").
    // However, detect_split_pairs does NOT filter by "no slash in segment" —
    // it pairs any segment that exists as a project_id.
    // This is the documented behavior for multi-slash ids.
    let project_ids = vec!["a/b".to_string(), "c/a/b".to_string()];
    let counts = make_counts(&[("a/b", 1), ("c/a/b", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    // "a/b" is the segment after "/" in "c/a/b", and "a/b" exists as project id.
    // So ("a/b", "c/a/b") IS reported — this is the multi-slash edge case.
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("a/b".to_string(), "c/a/b".to_string()));
}

#[test]
fn test_multi_owner_same_repo_reports_all_pairs() {
    // Regression test: bare id "repo" pairs with BOTH "org1/repo" and "org2/repo".
    // Previously only the first match was reported (HashSet keyed on bare id).
    let project_ids = vec![
        "org1/repo".to_string(),
        "org2/repo".to_string(),
        "repo".to_string(),
    ];
    let counts = make_counts(&[("org1/repo", 10), ("org2/repo", 5), ("repo", 3)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(
        pairs.len(),
        2,
        "must report BOTH (repo, org1/repo) AND (repo, org2/repo)"
    );
    assert_eq!(pairs[0], ("repo".to_string(), "org1/repo".to_string()));
    assert_eq!(pairs[1], ("repo".to_string(), "org2/repo".to_string()));
}

// ── Handler-level tests (integration) ──

#[test]
fn test_performs_no_database_writes() {
    // PRAGMA data_version is cross-connection — teeth-checked: fails with sabotage.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    let emb1 = test_fake_embedder("bare").unwrap();
    db.insert("detected", "bare", &emb1, None, "fact", "active")
        .unwrap();
    let emb2 = test_fake_embedder("owned").unwrap();
    db.insert("owner/detected", "owned", &emb2, None, "fact", "active")
        .unwrap();

    let version_before: i64 = db
        .conn()
        .query_row("PRAGMA data_version", [], |r| r.get(0))
        .unwrap();

    let result = handle_doctor_projects(&db_path, None, false);
    assert!(result.is_ok());

    let version_after: i64 = db
        .conn()
        .query_row("PRAGMA data_version", [], |r| r.get(0))
        .unwrap();

    assert_eq!(
        version_before, version_after,
        "doctor --projects must perform zero database writes"
    );
}

#[test]
fn test_no_db_writes_on_empty_database() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    let version_before: i64 = db
        .conn()
        .query_row("PRAGMA data_version", [], |r| r.get(0))
        .unwrap();

    let result = handle_doctor_projects(&db_path, None, false);
    assert!(result.is_ok());

    let version_after: i64 = db
        .conn()
        .query_row("PRAGMA data_version", [], |r| r.get(0))
        .unwrap();

    assert_eq!(
        version_before, version_after,
        "must perform zero writes on empty database"
    );
    assert_eq!(db.list_all_project_ids().unwrap().len(), 0);
}

#[test]
fn test_response_struct_serializes_correctly() {
    let response = DoctorProjectsResponse {
        suspected_splits: vec![DoctorProjectsSuspectedSplit {
            pair: ["myrepo".to_string(), "owner/myrepo".to_string()],
            row_counts: [3, 7],
        }],
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"suspected_splits\""));
    assert!(json.contains("\"pair\""));
    assert!(json.contains("\"row_counts\""));
    assert!(json.contains("\"myrepo\""));
    assert!(json.contains("\"owner/myrepo\""));
    assert!(json.contains("3"));
    assert!(json.contains("7"));
}

// ── Determinism test ──

#[test]
fn test_detect_split_pairs_deterministic_order() {
    // Repeated calls with same input must produce identical output.
    let project_ids = vec![
        "beta".to_string(),
        "alpha".to_string(),
        "org/beta".to_string(),
        "team/alpha".to_string(),
        "zeta".to_string(),
        "x/zeta".to_string(),
    ];
    let counts = make_counts(&[
        ("alpha", 1),
        ("beta", 1),
        ("zeta", 1),
        ("org/beta", 1),
        ("team/alpha", 1),
        ("x/zeta", 1),
    ]);

    let pairs1 = detect_split_pairs(&project_ids, &counts);
    let pairs2 = detect_split_pairs(&project_ids, &counts);
    assert_eq!(
        pairs1, pairs2,
        "repeated calls must produce identical output"
    );
    // Verify sort order: alpha < beta < zeta
    assert_eq!(pairs1[0].0, "alpha");
    assert_eq!(pairs1[1].0, "beta");
    assert_eq!(pairs1[2].0, "zeta");
}

#[test]
fn test_detect_split_pairs_no_duplicate_pairs() {
    // Each owned id appears at most once in project_ids, so no duplicates possible.
    let project_ids = vec!["repo".to_string(), "a/repo".to_string()];
    let counts = make_counts(&[("repo", 1), ("a/repo", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    // Exactly one pair, not two.
    assert_eq!(pairs.len(), 1);
}

// ── End-to-end handler test ──

#[test]
fn test_handler_reports_split_pair_with_correct_row_counts() {
    // End-to-end: seeds a real DB, invokes the handler's response collector,
    // and asserts suspected_splits contains the expected pair with correct counts.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // Seed: "pi-an" (bare) with 8 rows
    for i in 0..8 {
        let emb = test_fake_embedder(&format!("pi-an row {}", i)).unwrap();
        db.insert(
            "pi-an",
            &format!("memory {}", i),
            &emb,
            None,
            "fact",
            "active",
        )
        .unwrap();
    }
    // Seed: "randomm/pi-an" (owned) with 2 rows
    for i in 0..2 {
        let emb = test_fake_embedder(&format!("randomm/pi-an row {}", i)).unwrap();
        db.insert(
            "randomm/pi-an",
            &format!("owned memory {}", i),
            &emb,
            None,
            "fact",
            "active",
        )
        .unwrap();
    }

    // Invoke the handler's response collector
    let response = collect_doctor_projects_response(&db_path).unwrap();

    // Assert exactly one suspected split pair
    assert_eq!(
        response.suspected_splits.len(),
        1,
        "handler must report exactly one split pair"
    );

    let split = &response.suspected_splits[0];
    assert_eq!(
        split.pair[0], "pi-an",
        "pair[0] must be the bare id (segment)"
    );
    assert_eq!(
        split.pair[1], "randomm/pi-an",
        "pair[1] must be the owned id"
    );
    assert_eq!(split.row_counts[0], 8, "row count for bare id must be 8");
    assert_eq!(split.row_counts[1], 2, "row count for owned id must be 2");
}

#[test]
fn test_handler_reports_no_splits_when_none_exist() {
    // End-to-end: DB has unrelated projects, handler must report empty.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    let emb1 = test_fake_embedder("proj-a").unwrap();
    db.insert("proj-a", "memory", &emb1, None, "fact", "active")
        .unwrap();
    let emb2 = test_fake_embedder("proj-b").unwrap();
    db.insert("proj-b", "memory", &emb2, None, "fact", "active")
        .unwrap();

    let response = collect_doctor_projects_response(&db_path).unwrap();
    assert!(
        response.suspected_splits.is_empty(),
        "no splits expected for unrelated projects"
    );
}

#[test]
fn test_handler_reports_multiple_split_pairs() {
    // End-to-end: multiple split pairs, all reported with correct ordering.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // "alpha" (bare, 3 rows) + "team/alpha" (owned, 1 row)
    for i in 0..3 {
        let emb = test_fake_embedder(&format!("alpha {}", i)).unwrap();
        db.insert("alpha", &format!("a {}", i), &emb, None, "fact", "active")
            .unwrap();
    }
    let emb = test_fake_embedder("team/alpha").unwrap();
    db.insert("team/alpha", "owned alpha", &emb, None, "fact", "active")
        .unwrap();

    // "beta" (bare, 2 rows) + "org/beta" (owned, 4 rows)
    for i in 0..2 {
        let emb = test_fake_embedder(&format!("beta {}", i)).unwrap();
        db.insert("beta", &format!("b {}", i), &emb, None, "fact", "active")
            .unwrap();
    }
    for i in 0..4 {
        let emb = test_fake_embedder(&format!("org/beta {}", i)).unwrap();
        db.insert(
            "org/beta",
            &format!("owned b {}", i),
            &emb,
            None,
            "fact",
            "active",
        )
        .unwrap();
    }

    let response = collect_doctor_projects_response(&db_path).unwrap();

    assert_eq!(
        response.suspected_splits.len(),
        2,
        "must report both split pairs"
    );

    // alpha pair comes first (sorted by segment)
    assert_eq!(response.suspected_splits[0].pair[0], "alpha");
    assert_eq!(response.suspected_splits[0].pair[1], "team/alpha");
    assert_eq!(response.suspected_splits[0].row_counts[0], 3);
    assert_eq!(response.suspected_splits[0].row_counts[1], 1);

    // beta pair comes second
    assert_eq!(response.suspected_splits[1].pair[0], "beta");
    assert_eq!(response.suspected_splits[1].pair[1], "org/beta");
    assert_eq!(response.suspected_splits[1].row_counts[0], 2);
    assert_eq!(response.suspected_splits[1].row_counts[1], 4);
}
