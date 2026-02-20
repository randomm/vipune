//! Project auto-detection from git repository

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Detect project identifier from environment.
///
/// Detection priority (checked in order):
/// 1. Explicit override parameter (if provided and non-whitespace)
/// 2. `VIPUNE_PROJECT` environment variable (if set and non-whitespace)
/// 3. Git remote origin URL (parsed to owner/repo format)
/// 4. Git repository root directory name
/// 5. Current working directory name
///
/// Always returns a non-empty string. Falls back to "unknown" if all detection methods fail.
///
/// # Arguments
/// * `explicit` - Optional explicit project identifier that overrides all other detection methods.
///   If provided but empty/whitespace, falls back to automatic detection.
///
/// # Returns
/// A project identifier string (never empty)
///
/// # Example
/// ```no_run
/// use vipune::project::detect_project;
///
/// // Use explicit override
/// let project = detect_project(Some("my-project"));
/// assert_eq!(project, "my-project");
///
/// // Auto-detect from git
/// let project = detect_project(None);
/// println!("Detected project: {}", project);
/// ```
pub fn detect_project(explicit: Option<&str>) -> String {
    // 1. Explicit override takes priority (must be non-empty)
    if let Some(project) = explicit {
        if !project.trim().is_empty() {
            return project.trim().to_string();
        }
        // If explicit is empty/whitespace, proceed to fallback methods
    }

    // 2. Check environment variable
    if let Ok(project) = env::var("VIPUNE_PROJECT") {
        let trimmed = project.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // 3. Try git remote origin
    if let Some(remote) = get_git_remote_origin() {
        let project = parse_git_remote(&remote);
        if !project.is_empty() {
            return project;
        }
    }

    // 4. Try git root directory name
    if let Some(root) = find_git_root() {
        if let Some(name) = root.file_name() {
            if let Some(s) = name.to_str() {
                return s.to_string();
            }
        }
    }

    // 5. Fallback to current directory name
    env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get git remote origin URL
fn get_git_remote_origin() -> Option<String> {
    let output = Command::new("git")
        .args(["remote", "get-url", "origin"])
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

/// Find git repository root
fn find_git_root() -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
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

/// Parse git remote URL to owner/repo format
fn parse_git_remote(url: &str) -> String {
    let url = url.trim().trim_end_matches(".git");

    // SSH format: git@github.com:owner/repo
    if let Some(rest) = url.strip_prefix("git@") {
        if let Some(colon_pos) = rest.find(':') {
            return rest[colon_pos + 1..].to_string();
        }
    }

    // HTTPS format: https://github.com/owner/repo
    if let Some(rest) = url.split("://").nth(1) {
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() >= 3 {
            return format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]);
        }
    }

    // Fallback: return original URL
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_explicit_override() {
        assert_eq!(detect_project(Some("my-project")), "my-project");
    }

    #[test]
    fn test_explicit_override_empty() {
        // Empty explicit string should fallback to other detection methods
        let project = detect_project(Some(""));
        assert!(!project.is_empty());
    }

    #[test]
    fn test_explicit_override_whitespace() {
        // Whitespace-only explicit string should fallback to other detection methods
        let project = detect_project(Some("   \t  "));
        assert!(!project.is_empty());
    }

    #[test]
    fn test_detect_fallback_to_current_dir() {
        let project = detect_project(None);
        assert!(!project.is_empty());
    }

    #[test]
    fn test_env_var_whitespace() {
        // This test runs in isolation, safe to set env var
        std::env::set_var("VIPUNE_PROJECT", "   ");
        let project = detect_project(None);
        assert!(!project.is_empty()); // Should ignore whitespace and use fallback
        std::env::remove_var("VIPUNE_PROJECT");
    }
}
