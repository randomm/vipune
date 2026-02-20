//! Path expansion utilities for tilde (`~`) substitution.

use std::path::{Path, PathBuf};

/// Expand `~` to home directory in a PathBuf (in-place).
#[allow(dead_code)] // Dead code justified: used in Config::load()
pub fn expand_tilde(path: &mut PathBuf) {
    if path.starts_with("~") {
        if let Some(home) = dirs::home_dir() {
            let rest = path.strip_prefix("~").unwrap_or(Path::new(""));
            *path = home.join(rest);
        }
    }
}

/// Expand `~` to home directory in a PathBuf (returns new PathBuf).
#[allow(dead_code)] // Dead code justified: used in Config::load()
pub fn expand_tilde_path(path: &Path) -> PathBuf {
    if path.starts_with("~") {
        if let Some(home) = dirs::home_dir() {
            let rest = path.strip_prefix("~").unwrap_or(Path::new(""));
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

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
