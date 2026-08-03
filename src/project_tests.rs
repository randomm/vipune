//! Tests for project detection.
//!
//! All tests use [`detect_project_at_test`] — a hermetic wrapper that injects
//! an empty `env_project` sentinel into [`detect_project_at_internal`] — so the
//! real `VIPUNE_PROJECT` env var is never consulted. This prevents test failures
//! when developers run the suite with that variable set in their shell.
//! No process-global state is mutated (`set_current_dir`, `set_var`).

use std::path::PathBuf;
use std::process::Command;

use super::*;
use tempfile::TempDir;

/// Hermetic test wrapper that never reads the real VIPUNE_PROJECT env var.
///
/// Passes an empty string as `env_project` so the control flow enters the
/// `if let Some(project) = env_project` branch in `detect_project_at_internal`,
/// finds the value empty after trimming, and falls through to git detection.
///
/// This is the canonical way to call project detection from tests — production
/// behaviour of `detect_project()` and `detect_project_at()` is unchanged.
fn detect_project_at_test(root: &Path, explicit: Option<&str>) -> String {
    detect_project_at_internal(root, explicit, Some(String::new()))
}

// ── Git fixture harness ──────────────────────────────────────────────────────

/// Create a bare git repo in a temp directory.
///
/// Configures local user.name/user.email and uses an explicit initial branch
/// so it works on any machine and in CI regardless of global git config.
fn create_git_repo() -> TempDir {
    let dir = TempDir::new().expect("create temp dir for git repo");
    init_git_repo(dir.path());
    dir
}

/// Initialize a git repo at the given path.
fn init_git_repo(path: &Path) {
    Command::new("git")
        .args(["-C", path.to_str().unwrap(), "init", "-b", "main"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .expect("git init failed");

    Command::new("git")
        .args([
            "-C",
            path.to_str().unwrap(),
            "config",
            "user.name",
            "Test User",
        ])
        .output()
        .expect("git config user.name failed");

    Command::new("git")
        .args([
            "-C",
            path.to_str().unwrap(),
            "config",
            "user.email",
            "test@example.com",
        ])
        .output()
        .expect("git config user.email failed");
}

/// Add a git remote to the repo at the given path.
fn add_remote(path: &Path, name: &str, url: &str) {
    Command::new("git")
        .args(["-C", path.to_str().unwrap(), "remote", "add", name, url])
        .output()
        .expect("git remote add failed");
}

/// Create a nested subdirectory inside a git repo and return its path.
fn create_subdirectory(repo_path: &Path) -> PathBuf {
    let sub = repo_path.join("src").join("deep");
    std::fs::create_dir_all(&sub).expect("create subdirectory");
    sub
}

// ── parse_git_remote golden table ─────────────────────────────────────────────

#[test]
fn test_parse_ssh_remote() {
    assert_eq!(
        parse_git_remote("git@github.com:owner/repo.git"),
        "owner/repo"
    );
    assert_eq!(parse_git_remote("git@github.com:owner/repo"), "owner/repo");
}

#[test]
fn test_parse_https_remote() {
    assert_eq!(
        parse_git_remote("https://github.com/owner/repo.git"),
        "owner/repo"
    );
    assert_eq!(
        parse_git_remote("https://github.com/owner/repo"),
        "owner/repo"
    );
}

#[test]
fn test_parse_ssh_url_with_protocol() {
    assert_eq!(
        parse_git_remote("ssh://git@github.com/owner/repo.git"),
        "owner/repo"
    );
}

#[test]
fn test_git_suffix_stripping() {
    assert_eq!(parse_git_remote("owner/repo.git"), "owner/repo");
}

#[test]
fn test_fallback_when_no_domain() {
    assert_eq!(parse_git_remote("just-name"), "just-name");
}

#[test]
fn test_parse_mixed_case_host_and_owner() {
    // Case is preserved — Owner/Repo and owner/repo are distinct ids.
    assert_eq!(
        parse_git_remote("git@GitHub.com:Owner/Repo.git"),
        "Owner/Repo"
    );
    assert_eq!(
        parse_git_remote("https://GitHub.com/Owner/Repo.git"),
        "Owner/Repo"
    );
}

// ── explicit override tests ──────────────────────────────────────────────────

#[test]
fn test_explicit_override() {
    let dir = TempDir::new().expect("temp dir");
    assert_eq!(
        detect_project_at(dir.path(), Some("my-project")),
        "my-project"
    );
}

#[test]
fn test_explicit_override_empty() {
    // Empty explicit string should fall through to automatic detection.
    // Using a temp dir with no git repo, the fallback is the dir name.
    let dir = create_git_repo();
    let result = detect_project_at_test(dir.path(), Some(""));
    // Falls through to git root dir name since there's no origin.
    assert_eq!(result, dir.path().file_name().unwrap().to_str().unwrap());
}

#[test]
fn test_explicit_override_whitespace() {
    // Whitespace-only explicit string should fall through to automatic detection.
    let dir = create_git_repo();
    let result = detect_project_at_test(dir.path(), Some("   \t  "));
    // Falls through to git root dir name since there's no origin.
    assert_eq!(result, dir.path().file_name().unwrap().to_str().unwrap());
}

// ── env var tests ─────────────────────────────────────────────────────────────

#[test]
fn test_env_var_whitespace() {
    // Whitespace-only VIPUNE_PROJECT should fall through to git detection.
    // Inject the value as a parameter rather than mutating process-global env.
    let dir = create_git_repo();
    let result = detect_project_at_internal(dir.path(), None, Some("   ".to_string()));
    // Falls through to git root dir name since there's no origin.
    assert_eq!(result, dir.path().file_name().unwrap().to_str().unwrap());
}

#[test]
fn test_env_var_override_with_no_git() {
    // VIPUNE_PROJECT set with no git repo — should use env var value.
    let dir = TempDir::new().expect("temp dir");
    let result = detect_project_at_internal(dir.path(), None, Some("env-project".to_string()));
    assert_eq!(result, "env-project");
}

#[test]
fn test_env_var_trimmed() {
    // VIPUNE_PROJECT with leading/trailing whitespace is trimmed.
    let dir = TempDir::new().expect("temp dir");
    let result = detect_project_at_internal(dir.path(), None, Some("  trimmed  ".to_string()));
    assert_eq!(result, "trimmed");
}

// ── detection with git remotes ───────────────────────────────────────────────

#[test]
fn test_detect_https_remote() {
    let dir = create_git_repo();
    add_remote(
        dir.path(),
        "origin",
        "https://github.com/randomm/vipune.git",
    );
    assert_eq!(detect_project_at_test(dir.path(), None), "randomm/vipune");
}

#[test]
fn test_detect_ssh_remote() {
    let dir = create_git_repo();
    add_remote(dir.path(), "origin", "git@github.com:randomm/vipune.git");
    assert_eq!(detect_project_at_test(dir.path(), None), "randomm/vipune");
}

#[test]
fn test_detect_ssh_url_with_protocol() {
    let dir = create_git_repo();
    add_remote(
        dir.path(),
        "origin",
        "ssh://git@github.com/randomm/vipune.git",
    );
    assert_eq!(detect_project_at_test(dir.path(), None), "randomm/vipune");
}

#[test]
fn test_detect_remote_without_git_suffix() {
    let dir = create_git_repo();
    add_remote(dir.path(), "origin", "https://github.com/randomm/vipune");
    assert_eq!(detect_project_at_test(dir.path(), None), "randomm/vipune");
}

#[test]
fn test_detect_only_upstream_remote_uses_dir_name() {
    // When only an 'upstream' remote exists (no 'origin'), should NOT adopt
    // the upstream remote — fall back to directory name.
    let dir = create_git_repo();
    add_remote(
        dir.path(),
        "upstream",
        "https://github.com/canonical/project.git",
    );
    let result = detect_project_at_test(dir.path(), None);
    assert_eq!(result, dir.path().file_name().unwrap().to_str().unwrap());
    // Ensure we did NOT pick up the upstream remote.
    assert_ne!(result, "canonical/project");
}

#[test]
fn test_detect_no_git_repo() {
    // No git repo at all — fallback to directory name.
    let dir = TempDir::new().expect("temp dir");
    let result = detect_project_at_test(dir.path(), None);
    // Falls back to dir name (or "unknown" if dir has no file_name).
    assert!(!result.is_empty());
}

// ── determinism: root vs subdirectory ────────────────────────────────────────

#[test]
fn test_detect_same_id_from_root_and_subdirectory() {
    let dir = create_git_repo();
    add_remote(dir.path(), "origin", "https://github.com/owner/repo.git");

    let sub = create_subdirectory(dir.path());

    // Both root and subdirectory must yield the same project_id.
    let from_root = detect_project_at_test(dir.path(), None);
    let from_sub = detect_project_at_test(&sub, None);

    assert_eq!(from_root, "owner/repo");
    assert_eq!(from_sub, "owner/repo");
    assert_eq!(from_root, from_sub);
}

#[test]
fn test_detect_fallback_same_from_root_and_subdirectory() {
    // No remotes: both root and subdirectory yield the git root dir name.
    let dir = create_git_repo();
    let sub = create_subdirectory(dir.path());

    let from_root = detect_project_at_test(dir.path(), None);
    let from_sub = detect_project_at_test(&sub, None);

    let expected = dir.path().file_name().unwrap().to_str().unwrap();
    assert_eq!(from_root, expected);
    assert_eq!(from_sub, expected);
}

// ── fallback warning tests ───────────────────────────────────────────────────

#[test]
fn test_fallback_no_remotes_yields_dir_name() {
    let dir = create_git_repo();
    let result = detect_project_at_test(dir.path(), None);
    assert_eq!(result, dir.path().file_name().unwrap().to_str().unwrap());
}

#[test]
fn test_fallback_warning_message_includes_project_id() {
    let dir = create_git_repo();
    let msg = build_fallback_warning_message(
        dir.path().file_name().unwrap().to_str().unwrap(),
        dir.path(),
    );
    assert!(msg.contains("using directory name as project_id"));
    assert!(msg.contains(dir.path().file_name().unwrap().to_str().unwrap()));
}

#[test]
fn test_fallback_warning_message_lists_other_remotes() {
    let dir = create_git_repo();
    add_remote(
        dir.path(),
        "upstream",
        "https://github.com/canonical/project.git",
    );
    let msg = build_fallback_warning_message(
        dir.path().file_name().unwrap().to_str().unwrap(),
        dir.path(),
    );
    assert!(msg.contains("other remotes"));
    assert!(msg.contains("upstream"));
}

#[test]
fn test_fallback_warning_message_no_other_remotes() {
    let dir = create_git_repo();
    let msg = build_fallback_warning_message(
        dir.path().file_name().unwrap().to_str().unwrap(),
        dir.path(),
    );
    // No remotes at all — message should NOT mention other remotes.
    assert!(!msg.contains("other remotes"));
    assert!(msg.contains("This project_id may differ"));
}

// ── integration: remote-derived ids unchanged ────────────────────────────────

#[test]
fn test_remote_derived_ids_unchanged() {
    // Verify that repos resolving via remote produce the same ids as before.
    // This is the key invariant: the fix must not change existing project_ids.
    let cases = [
        ("https://github.com/randomm/vipune.git", "randomm/vipune"),
        ("https://github.com/randomm/vipune", "randomm/vipune"),
        ("git@github.com:randomm/vipune.git", "randomm/vipune"),
        ("git@github.com:randomm/vipune", "randomm/vipune"),
        ("ssh://git@github.com/randomm/vipune.git", "randomm/vipune"),
        (
            "https://gitlab.example.com/group/subgroup/project.git",
            "subgroup/project",
        ),
        (
            // SSH shorthand returns full path after colon (existing behavior).
            "git@gitlab.example.com:group/subgroup/project.git",
            "group/subgroup/project",
        ),
    ];

    for (remote_url, expected_id) in cases {
        let dir = create_git_repo();
        add_remote(dir.path(), "origin", remote_url);
        let result = detect_project_at_test(dir.path(), None);
        assert_eq!(
            result, expected_id,
            "remote '{}' should produce project_id '{}'",
            remote_url, expected_id
        );
    }
}

// ── detect_project backward compatibility ────────────────────────────────────

#[test]
fn test_detect_project_delegates_to_current_dir() {
    // detect_project(None) must return a non-empty string when called from
    // the current directory (the vipune repo itself has a git remote).
    let project = detect_project(None);
    assert!(!project.is_empty());
}

#[test]
fn test_detect_project_explicit_override() {
    assert_eq!(detect_project(Some("custom-id")), "custom-id");
}
