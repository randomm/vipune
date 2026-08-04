//! Project auto-detection from git repository.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    if let Some(remote) = get_git_remote_url_at(root) {
        let project = parse_git_remote(&remote);
        if !project.is_empty() {
            return project;
        }
    }

    // 4. Try git root directory name
    if let Some(git_root) = find_git_root_at(root) {
        if let Some(name) = git_root.file_name() {
            if let Some(s) = name.to_str() {
                emit_fallback_warning(s, &git_root);
                return s.to_string();
            }
        }
    }

    // 5. Fallback to given root directory name
    root.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get git remote URL for the 'origin' remote at the given path.
///
/// Only queries the `origin` remote. If `origin` does not exist or the command
/// fails, returns `None`.
fn get_git_remote_url_at(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["-C", root.to_str()?, "remote", "get-url", "origin"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .ok()?;

    if output.status.success() {
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !url.is_empty() {
            return Some(url);
        }
    }
    None
}

/// Find git repository root from the given path.
fn find_git_root_at(root: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["-C", root.to_str()?, "rev-parse", "--show-toplevel"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .ok()?;

    if output.status.success() {
        let path_str = String::from_utf8_lossy(&output.stdout);
        let path = path_str.trim();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    None
}

/// Get list of git remotes at the given path.
fn get_other_remotes_at(root: &Path) -> Vec<String> {
    let root_str = match root.to_str() {
        Some(s) => s,
        None => return Vec::new(),
    };
    let output = match Command::new("git")
        .args(["-C", root_str, "remote"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
    {
        Ok(out) => out,
        Err(_) => return Vec::new(),
    };

    if output.status.success() {
        return String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
    }
    Vec::new()
}

/// Build the warning message emitted when falling back to directory name.
///
/// This is a pure function so tests can verify the message content without
/// capturing stderr.
pub(crate) fn build_fallback_warning_message(project_id: &str, git_root: &Path) -> String {
    let remotes = get_other_remotes_at(git_root);
    let mut msg = format!(
        "Warning: no git remote 'origin' found, using directory name as project_id: '{}'",
        project_id
    );
    if !remotes.is_empty() {
        msg.push_str(&format!(" (other remotes: {})", remotes.join(", ")));
    }
    msg.push_str(". This project_id may differ from the remote-derived one.");
    msg
}

/// Emit a stderr warning when falling back to directory name for project_id.
fn emit_fallback_warning(project_id: &str, git_root: &Path) {
    eprintln!("{}", build_fallback_warning_message(project_id, git_root));
}

/// Parse git remote URL to owner/repo format.
///
/// Supported formats:
/// - SSH shorthand: `git@host:owner/repo.git` → `owner/repo`
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
            return rest[colon_pos + 1..].to_string();
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
