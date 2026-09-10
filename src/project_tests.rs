//! Tests for project detection.
//!
//! All tests use [`detect_project_at_test`] — a hermetic wrapper that injects
//! an empty `env_project` sentinel into [`detect_project_at_internal`] — so the
//! real `VIPUNE_PROJECT` env var is never consulted. This prevents test failures
//! when developers run the suite with that variable set in their shell.
//! No process-global state is mutated (`set_current_dir`, `set_var`).

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

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
    let msg = build_fallback_warning_message(dir.path().file_name().unwrap().to_str().unwrap());
    assert!(msg.contains("using directory name as project_id"));
    assert!(msg.contains(dir.path().file_name().unwrap().to_str().unwrap()));
}

#[test]
fn test_fallback_warning_message_no_other_remotes() {
    let dir = create_git_repo();
    let msg = build_fallback_warning_message(dir.path().file_name().unwrap().to_str().unwrap());
    // The other-remotes lookup was removed (issue #163 finding 2): the warning
    // must not spawn a third git subprocess on the degraded path, so it never
    // mentions remotes regardless of what remotes exist.
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
        // #164 (fixed): these two entries are the SAME repo referenced via
        // HTTPS and SSH. Previously `parse_git_remote` produced DIFFERENT
        // project_ids for the two forms (`subgroup/project` vs
        // `group/subgroup/project`). The fix applies the last-two-segment
        // rule uniformly to the SSH-shorthand branch as well, so both forms
        // now resolve to `subgroup/project` — one repo, one id.
        (
            "https://gitlab.example.com/group/subgroup/project.git",
            "subgroup/project",
        ),
        (
            "git@gitlab.example.com:group/subgroup/project.git",
            "subgroup/project",
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

// ── run_git: timeout, typed errors, stdout capture ───────────────────────────

/// A stub `git` that sleeps on any invocation — stands in for a wedged real
/// git process (frozen network fs, hung credential helper) without needing an
/// actual hung git (issue #163 acceptance: "without relying on a real hung
/// git process"). Writes a sleep script at `dir/git` and marks it executable.
fn make_git_sleep_stub(dir: &Path) {
    let stub = dir.join("git");
    // Use /bin/sleep so the stub does not depend on `sleep` being on the CI
    // runner's PATH (Linux CI pools have shipped minimal sh where a bare
    // `sleep` resolves to exit 127). `/bin/sleep` exists on macOS and Linux.
    std::fs::write(&stub, "#!/bin/sh\n/bin/sleep 5\n").expect("write sleep stub");
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&stub).expect("stat stub").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&stub, perms).expect("chmod stub");
}

/// Like `run_git` but with an injectable PATH dir and timeout so tests can
/// use a stub `git` and a short deadline without touching process env.
fn run_git_in_env(
    root: &Path,
    args: &[&str],
    timeout: Duration,
    path_dir: Option<&Path>,
) -> Result<String, GitError> {
    let root_str = root
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| GitError::Spawn(format!("non-UTF-8 path: {:?}", root)))?;

    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(&root_str);
    cmd.args(args);
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    if let Some(dir) = path_dir {
        cmd.env("PATH", dir);
    }

    let mut child = cmd.spawn().map_err(|e| GitError::Spawn(e.to_string()))?;

    let (tx, rx) = mpsc::channel();
    {
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        thread::spawn(move || {
            let stdout_buf = read_pipe(stdout);
            let stderr_buf = read_pipe(stderr);
            let _ = tx.send((stdout_buf, stderr_buf));
        });
    }

    match rx.recv_timeout(timeout) {
        Ok((stdout_buf, stderr_buf)) => match child.wait() {
            Ok(status) => {
                if status.success() {
                    Ok(String::from_utf8_lossy(&stdout_buf).trim().to_string())
                } else {
                    Err(GitError::NonZeroExit {
                        code: status.code(),
                        stderr: String::from_utf8_lossy(&stderr_buf).to_string(),
                    })
                }
            }
            Err(e) => Err(GitError::Spawn(format!("wait failure: {e}"))),
        },
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            Err(GitError::Timeout(timeout))
        }
    }
}

#[test]
fn test_run_git_timeout_kills_stub() {
    // A stub `git` that sleeps 5 s stands in for a wedged real git. With a
    // 100 ms deadline the kill path must fire and return the typed Timeout
    // error — no real hung git process needed (issue #163 acceptance).
    let stub_dir = TempDir::new().expect("stub bin dir");
    make_git_sleep_stub(stub_dir.path());

    let result = run_git_in_env(
        Path::new("/"),
        &["--version"],
        Duration::from_millis(100),
        Some(stub_dir.path()),
    );
    match result {
        Err(GitError::Timeout(d)) => {
            // The timeout must be the requested 100 ms, not the stub's 5 s.
            assert!(d.as_millis() <= 100);
        }
        other => panic!("expected Err(GitError::Timeout), got {:?}", other),
    }
}

#[test]
fn test_run_git_captures_stdout_on_success() {
    // `git -C <repo> rev-parse --show-toplevel` exits 0 with the repo root on
    // stdout. The captured value must match, proving stdout piping works
    // end-to-end through run_git (not just the empty case).
    let dir = create_git_repo();
    let out = run_git(dir.path(), &["rev-parse", "--show-toplevel"], GIT_TIMEOUT);
    let root = out.expect("expected Ok");
    // git prints the canonical (symlinks-resolved) path; resolve the temp dir
    // the same way so the comparison holds on macOS where /tmp is a symlink.
    let canonical = std::fs::canonicalize(dir.path()).expect("canonicalize temp dir");
    let expected = canonical.to_str().expect("path is valid UTF-8");
    assert_eq!(root, expected);
}

#[test]
fn test_run_git_nonzero_exit_is_typed() {
    // `git remote get-url origin` in a repo with no origin remote exits non-zero.
    // The error must carry the failure cause, not collapse to None.
    let dir = create_git_repo();
    let result = run_git(dir.path(), &["remote", "get-url", "origin"], GIT_TIMEOUT);
    match result {
        Err(GitError::NonZeroExit { code, stderr }) => {
            // git exits 2 (not 1) when the remote does not exist; the point of
            // the test is that the non-zero exit is typed and diagnosable.
            assert!(
                code == Some(1) || code == Some(2),
                "expected non-zero exit, got {code:?}"
            );
            assert!(
                !stderr.is_empty(),
                "stderr should be captured for diagnostics"
            );
        }
        other => panic!("expected Err(GitError::NonZeroExit), got {:?}", other),
    }
}

#[test]
fn test_run_git_spawn_failure_is_typed() {
    // A PATH that contains no `git` makes spawn fail; the error must name the
    // cause (issue #163 finding 3) rather than collapse to None.
    let empty_bin = TempDir::new().expect("empty bin dir");
    let result = run_git_in_env(
        Path::new("/"),
        &["--version"],
        GIT_TIMEOUT,
        Some(empty_bin.path()),
    );
    match result {
        Err(GitError::Spawn(msg)) => {
            assert!(!msg.is_empty(), "spawn error should carry a reason");
        }
        other => panic!("expected Err(GitError::Spawn), got {:?}", other),
    }
}
