//! Path expansion utilities for tilde (`~`) substitution.

use std::path::{Path, PathBuf};

/// The home directory for default paths: `dirs::home_dir()`, then `$HOME`,
/// then `"."`.
///
/// This is the chain every default-path computation in the crate uses **except
/// two deliberate exceptions**, which keep their own semantics because routing
/// them through the `"."` fallback would silently change behaviour:
///
/// - [`crate::commands::hook_install::home_dir`] errors with
///   `Error::Config("Could not determine home directory")` instead of falling
///   back — hooks must never be installed into `./.claude`.
/// - [`expand_tilde`] / [`expand_tilde_path`] leave a leading `~` unexpanded
///   when no home can be determined, rather than expanding `~/x` to `./x`.
pub(crate) fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    })
}

/// Expand `~` to home directory in a PathBuf (in-place).
pub fn expand_tilde(path: &mut PathBuf) {
    if path.starts_with("~")
        && let Some(home) = dirs::home_dir()
    {
        let rest = path.strip_prefix("~").unwrap_or(Path::new(""));
        *path = home.join(rest);
    }
}

/// Expand `~` to home directory in a PathBuf (returns new PathBuf).
pub fn expand_tilde_path(path: &Path) -> PathBuf {
    if path.starts_with("~")
        && let Some(home) = dirs::home_dir()
    {
        let rest = path.strip_prefix("~").unwrap_or(Path::new(""));
        return home.join(rest);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests_utils;

    #[test]
    fn test_home_dir_resolves_home() {
        // Mutates the process env, so it takes the same guard as every other
        // env-touching test in this target: `dirs::home_dir()` reads `HOME`
        // through `getenv` on whatever thread a sibling test runs on.
        let _guard = tests_utils::lock_env_mutex();
        let saved = std::env::var("HOME").ok();
        unsafe {
            std::env::set_var("HOME", "/tmp/vipune-home-probe");
        }
        let resolved = home_dir();
        unsafe {
            match saved {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }
        assert_eq!(resolved, PathBuf::from("/tmp/vipune-home-probe"));
    }

    #[test]
    fn test_expand_tilde() {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from(""));
        if home.as_os_str().is_empty() {
            return;
        }
        let mut path = PathBuf::from("~/test/path");
        expand_tilde(&mut path);

        assert!(!path.starts_with("~"));
        assert!(path.starts_with(&home));
        assert!(path.ends_with("test/path"));
    }

    #[test]
    fn test_expand_tilde_no_tilde() {
        let mut path = PathBuf::from("/absolute/path");
        let original = path.clone();

        expand_tilde(&mut path);

        assert_eq!(path, original);
    }
}
