//! Backward compatibility for pre-v0.2 database and model cache paths.

use std::path::PathBuf;

/// Find old database path for backward compatibility (pre-v0.2 migrations).
pub fn find_legacy_database_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;

    // Check macOS Application Support
    let macos_path = home.join("Library/Application Support/vipune/memories.db");
    if macos_path.exists() {
        return Some(macos_path);
    }

    // Check Linux XDG data directory
    let linux_path = home.join(".local/share/vipune/memories.db");
    if linux_path.exists() {
        return Some(linux_path);
    }

    None
}

/// Find old model cache path for backward compatibility (pre-v0.2 migrations).
pub fn find_legacy_model_cache_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;

    // Check macOS Caches directory
    let macos_cache = home.join("Library/Caches/vipune/models");
    if macos_cache.exists() {
        return Some(macos_cache);
    }

    // Check Linux XDG cache directory
    let linux_cache = home.join(".cache/vipune/models");
    if linux_cache.exists() {
        return Some(linux_cache);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_legacy_database_path_nonexistent() {
        // Nonexistent paths should return None
        let result = find_legacy_database_path();
        // Result may be Some or None depending on what exists on the system
        // This test verifies the function doesn't panic when paths don't exist
        let _ = result;
    }

    #[test]
    fn test_find_legacy_model_cache_path_nonexistent() {
        // Nonexistent paths should return None
        let result = find_legacy_model_cache_path();
        // Result may be Some or None depending on what exists on the system
        // This test verifies the function doesn't panic when paths don't exist
        let _ = result;
    }
}
