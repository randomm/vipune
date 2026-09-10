//! Project auto-detection from git repository.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Maximum time a git subprocess may run before being killed.
///
/// Generous for local `git remote` / `git rev-parse` reads; bounds the
/// startup cost if git wedges (credential helper, frozen network fs, hung
/// process) — see issue #163.
const GIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Detect project identifier from the current working directory.
///
/// Delegates to [`detect_project_at`] using `std::env::current_dir()`.
///
/// # Example
/// ```no_run
/// use vipune::project::detect_project;
///
/// let project = detect_project(None);
/// println!("Detected project: {}", project);
/// ```
pub fn detect_project(explicit: Option<&str>) -> String {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    detect_project_at(&cwd, explicit)
}

/// Detect project identifier from a specific directory path.
///
/// Detection priority (checked in order):
/// 1. Explicit override parameter (if provided and non-whitespace)
/// 2. `VIPUNE_PROJECT` environment variable (if set and non-whitespace)
/// 3. Git remote origin URL (parsed to owner/repo format)
/// 4. Git repository root directory name (emits stderr warning)
/// 5. Given root directory name
///
/// Always returns a non-empty string. Falls back to "unknown" if all detection
/// methods fail.
///
/// # Arguments
/// * `root` - The directory path to detect the project from.
/// * `explicit` - Optional explicit project identifier that overrides all other
///   detection methods. If provided but empty/whitespace, falls back to automatic
///   detection.
///
/// # Returns
/// A project identifier string (never empty).
pub fn detect_project_at(root: &Path, explicit: Option<&str>) -> String {
    detect_project_at_internal(root, explicit, None)
}

/// Internal: detection chain with optional env var injection for testing.
///
/// The `env_project` parameter simulates the value of `VIPUNE_PROJECT` without
/// mutating process-global state. Pass `None` to read the real environment.
pub(crate) fn detect_project_at_internal(
    root: &Path,
    explicit: Option<&str>,
    env_project: Option<String>,
) -> String {
    // 1. Explicit override takes priority (must be non-empty)
    if let Some(project) = explicit {
        if !project.trim().is_empty() {
            return project.trim().to_string();
        }
    }

    // 2. Check environment variable (or test override)
    if let Some(project) = env_project {
        let trimmed = project.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    } else if let Ok(project) = env::var("VIPUNE_PROJECT") {
        let trimmed = project.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // 3. Try git remote origin
    match get_git_remote_url_at(root) {
        Ok(remote) => {
            let project = parse_git_remote(&remote);
            if !project.is_empty() {
                return project;
            }
        }
        Err(e) => debug_log_git_failure("remote get-url origin", &e),
    }

    // 4. Try git root directory name
    match find_git_root_at(root) {
        Ok(git_root) => {
            if let Some(name) = git_root.file_name() {
                if let Some(s) = name.to_str() {
                    emit_fallback_warning(s);
                    return s.to_string();
                }
            }
        }
        Err(e) => {
            debug_log_git_failure("rev-parse --show-toplevel", &e);
            return root
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string());
        }
    }

    // 5. Fallback to given root directory name
    root.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Typed failure for a bounded git subprocess call.
///
/// Keeps the failure cause distinguishable so it is diagnosable (issue #163
/// finding 3): spawn failure, non-zero exit with stderr, and timeout are
/// distinct variants instead of collapsing to `None`.
#[derive(Debug)]
enum GitError {
    /// `git` could not be spawned (e.g. missing from PATH, permission denied).
    Spawn(String),
    /// The command ran but exited non-zero, or its stdout was not valid UTF-8.
    NonZeroExit { code: Option<i32>, stderr: String },
    /// The command exceeded [`GIT_TIMEOUT`] and was killed.
    Timeout(Duration),
}

impl std::fmt::Display for GitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitError::Spawn(e) => write!(f, "spawn failure: {e}"),
            GitError::NonZeroExit { code, stderr } => {
                let code = code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".to_string());
                write!(f, "non-zero exit ({code}): {}", stderr.trim())
            }
            GitError::Timeout(d) => write!(f, "timed out after {d:?}"),
        }
    }
}

/// Run a git subprocess with a bounded wait and a typed error.
///
/// Spawns the child with stdout/stderr pipes, waits on a receiver with a
/// timeout, and kills the child if the deadline elapses — `Command::output()`
/// has no timeout in std, so an unbound wait would hang the CLI at startup
/// if git wedges (issue #163 finding 1). Stdlib only: spawn + channel +
/// `recv_timeout` + `kill`.
///
/// Returns the trimmed stdout on success; `Err` carries the distinguishable
/// failure cause so callers can log why detection degraded.
fn run_git(root: &Path, args: &[&str], timeout: Duration) -> Result<String, GitError> {
    let root_str = root
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| GitError::Spawn(format!("non-UTF-8 path: {:?}", root)))?;

    let mut child = Command::new("git")
        .args(["-C", root_str.as_str()])
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| GitError::Spawn(e.to_string()))?;

    let (tx, rx) = mpsc::channel();
    {
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        thread::spawn(move || {
            let stdout_buf = read_pipe(stdout);
            let stderr_buf = read_pipe(stderr);
            // The pipes are dropped at the end of this block: if the caller
            // kills the child after the timeout, readers blocked in
            // read_to_end must not be kept pinned by live pipe handles.
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

/// Read a piped stdout/stderr handle to EOF, discarding I/O errors.
///
/// The child's exit status (checked via `child.wait()`) is the authoritative
/// success signal; a truncated read after a kill is irrelevant because the
/// caller only inspects the buffers on the success path.
fn read_pipe<R: std::io::Read + Send>(mut pipe: R) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
    buf
}

/// Get git remote URL for the 'origin' remote at the given path.
///
/// Only queries the `origin` remote. `Ok(url)` on a non-empty origin URL,
/// `Ok("")` when no origin exists (distinguishable from a real failure),
/// `Err` when the git call itself fails (bounded by [`GIT_TIMEOUT`]).
fn get_git_remote_url_at(root: &Path) -> Result<String, GitError> {
    let out = run_git(root, &["remote", "get-url", "origin"], GIT_TIMEOUT)?;
    if out.is_empty() {
        Ok(String::new())
    } else {
        Ok(out)
    }
}

/// Find git repository root from the given path.
///
/// `Err` when the git call fails — including when `root` is not inside a git
/// repository (git exits non-zero with "not a git repository" on stderr).
fn find_git_root_at(root: &Path) -> Result<PathBuf, GitError> {
    let out = run_git(root, &["rev-parse", "--show-toplevel"], GIT_TIMEOUT)?;
    if out.is_empty() {
        return Err(GitError::NonZeroExit {
            code: None,
            stderr: "rev-parse --show-toplevel produced no output".to_string(),
        });
    }
    Ok(PathBuf::from(out))
}

/// Log a git subprocess failure at debug level so degraded detection is
/// diagnosable (issue #163 finding 3).
///
/// Enabled by setting `VIPUNE_DEBUG` to any non-empty value that is not
/// `"false"` or `"0"` — the project has no logging framework, so a
/// stderr line prefixed `debug:` is the debug-level log.
fn debug_log_git_failure(step: &str, err: &GitError) {
    if debug_enabled() {
        eprintln!("debug: git step '{step}' failed: {err}");
    }
}

/// Whether debug-level logging is enabled via the `VIPUNE_DEBUG` env var.
fn debug_enabled() -> bool {
    match env::var("VIPUNE_DEBUG") {
        Ok(v) => !v.is_empty() && v != "0" && v != "false",
        Err(_) => false,
    }
}

/// Build the warning message emitted when falling back to directory name.
///
/// Pure function so tests can verify the message content without capturing
/// stderr. The other-remotes lookup was removed (issue #163 finding 2): it
/// spawned a third git subprocess on the already-degraded fallback path, so a
/// hang there was failure-on-top-of-failure.
pub(crate) fn build_fallback_warning_message(project_id: &str) -> String {
    format!(
        "Warning: no git remote 'origin' found, using directory name as project_id: '{}'. This project_id may differ from the remote-derived one.",
        project_id
    )
}

/// Emit a stderr warning when falling back to directory name for project_id.
fn emit_fallback_warning(project_id: &str) {
    eprintln!("{}", build_fallback_warning_message(project_id));
}

/// Parse git remote URL to owner/repo format.
///
/// The canonical rule is the **last two path segments** of the repository
/// path, applied uniformly to both the SSH-shorthand form and the `://` forms.
/// This keeps the same repository resolving to the same project_id regardless
/// of whether it is referenced via `git@host:owner/repo` or
/// `https://host/owner/repo` — including nested namespaces such as
/// `git@gitlab.example.com:group/subgroup/project` and
/// `https://gitlab.example.com/group/subgroup/project`, which both yield
/// `subgroup/project` (issue #164).
///
/// Supported formats:
/// - SSH shorthand: `git@host:owner/repo.git` → `owner/repo`
/// - SSH shorthand, nested: `git@host:group/subgroup/project.git` → `subgroup/project`
/// - HTTPS: `https://host/owner/repo.git` → `owner/repo`
/// - SSH URL: `ssh://git@host/owner/repo.git` → `owner/repo`
/// - Generic `://` URLs are handled by splitting on `://` and taking the last
///   two path segments.
///
/// Only normalization is `trim()` and stripping trailing `.git`. Case is
/// preserved.
fn parse_git_remote(url: &str) -> String {
    let url = url.trim().trim_end_matches(".git");

    // SSH format: git@github.com:owner/repo
    if let Some(rest) = url.strip_prefix("git@") {
        if let Some(colon_pos) = rest.find(':') {
            let path = &rest[colon_pos + 1..];
            let segments: Vec<&str> = path.split('/').collect();
            // Canonical rule: last two path segments (matches the :// branch
            // so SSH and HTTPS forms of the same repo yield the same id).
            if segments.len() >= 2 {
                return format!(
                    "{}/{}",
                    segments[segments.len() - 2],
                    segments[segments.len() - 1]
                );
            }
            return path.to_string();
        }
    }

    // HTTPS / SSH URL / generic :// format
    if let Some(rest) = url.split("://").nth(1) {
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() >= 3 {
            return format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]);
        }
    }

    // Fallback: return URL as-is
    url.to_string()
}

#[cfg(test)]
#[path = "project_tests.rs"]
mod project_tests;
