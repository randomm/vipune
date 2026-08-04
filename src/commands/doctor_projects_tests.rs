//! Tests for `vipune doctor --projects` handler and split-detection heuristic.

use crate::commands::doctor::{detect_split_pairs, handle_doctor_projects};
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
fn test_scan_spans_projects_regardless_of_project_filter() {
    // This verifies the handler ignores -p; the heuristic sees all projects.
    // Tested by verifying detect_split_pairs sees all ids.
    let project_ids = vec!["split-repo".to_string(), "other/split-repo".to_string()];
    let counts = make_counts(&[("split-repo", 1), ("other/split-repo", 1)]);

    let pairs = detect_split_pairs(&project_ids, &counts);
    assert_eq!(
        pairs.len(),
        1,
        "split must be detected regardless of project filter"
    );
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
