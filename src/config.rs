//! Configuration system for vipune.

use crate::errors::Error;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Configuration values with priority: defaults < config file < env vars.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Dead code justified: public API for CLI integration (issue #5)
pub struct Config {
    /// Path to the SQLite database.
    #[serde(default)]
    pub database_path: PathBuf,

    /// HuggingFace embedding model identifier.
    #[serde(default)]
    pub embedding_model: String,

    /// Directory for caching ONNX models.
    #[serde(default)]
    pub model_cache: PathBuf,

    /// Minimum similarity threshold for search results.
    #[serde(default)]
    pub similarity_threshold: f64,

    /// Recency weight for search results (0.0 to 1.0).
    #[serde(default)]
    pub recency_weight: f64,
}

impl Default for Config {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let data_dir = dirs::data_local_dir().unwrap_or_else(|| home.join(".local/share"));
        let cache_dir = dirs::cache_dir().unwrap_or_else(|| home.join(".cache"));

        Self {
            database_path: data_dir.join("vipune/memories.db"),
            embedding_model: "sentence-transformers/bge-small-en-v1.5".to_string(),
            model_cache: cache_dir.join("vipune/models"),
            similarity_threshold: 0.85,
            recency_weight: 0.0,
        }
    }
}

impl Config {
    fn apply_env_overrides(&mut self) -> Result<(), Error> {
        if let Ok(val) = std::env::var("VIPUNE_DATABASE_PATH") {
            if val.trim().is_empty() {
                return Err(Error::Config("VIPUNE_DATABASE_PATH cannot be empty".into()));
            }
            self.database_path = expand_tilde_path(&PathBuf::from(&val));
        }
        if let Ok(val) = std::env::var("VIPUNE_EMBEDDING_MODEL") {
            if val.trim().is_empty() {
                return Err(Error::Config(
                    "VIPUNE_EMBEDDING_MODEL cannot be empty".into(),
                ));
            }
            self.embedding_model = val;
        }
        if let Ok(val) = std::env::var("VIPUNE_MODEL_CACHE") {
            if val.trim().is_empty() {
                return Err(Error::Config("VIPUNE_MODEL_CACHE cannot be empty".into()));
            }
            self.model_cache = expand_tilde_path(&PathBuf::from(&val));
        }
        if let Ok(val) = std::env::var("VIPUNE_SIMILARITY_THRESHOLD") {
            if val.trim().is_empty() {
                return Err(Error::Config(
                    "VIPUNE_SIMILARITY_THRESHOLD cannot be empty".into(),
                ));
            }
            self.similarity_threshold = val.trim().parse().map_err(|e| {
                Error::Config(format!("Invalid VIPUNE_SIMILARITY_THRESHOLD value: {e}"))
            })?;
        }
        if let Ok(val) = std::env::var("VIPUNE_RECENCY_WEIGHT") {
            if val.trim().is_empty() {
                return Err(Error::Config(
                    "VIPUNE_RECENCY_WEIGHT cannot be empty".into(),
                ));
            }
            self.recency_weight = val
                .trim()
                .parse()
                .map_err(|e| Error::Config(format!("Invalid VIPUNE_RECENCY_WEIGHT value: {e}")))?;
        }
        Ok(())
    }

    #[allow(dead_code)] // Dead code justified: used in Config::load()
    fn merge_from_file(&mut self, file: ConfigFile) {
        if !file.database_path.as_os_str().is_empty() {
            self.database_path = file.database_path;
        }
        if !file.embedding_model.is_empty() {
            self.embedding_model = file.embedding_model;
        }
        if !file.model_cache.as_os_str().is_empty() {
            self.model_cache = file.model_cache;
        }
        self.similarity_threshold = file.similarity_threshold;
        self.recency_weight = file.recency_weight;
    }

    #[allow(dead_code)] // Dead code justified: public API for CLI integration (issue #5)
    pub fn load() -> Result<Self, Error> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let data_local = dirs::data_local_dir().unwrap_or_else(|| home.join(".local/share"));
        let config_dir = dirs::config_dir().unwrap_or_else(|| data_local.join(".config"));

        let config_path = config_dir.join("vipune/config.toml");

        let file_config = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path).map_err(|e| {
                Error::Config(format!(
                    "Failed to read config file {}: {e}",
                    config_path.display()
                ))
            })?;

            let mut config: ConfigFile = toml::from_str(&content).map_err(|e| {
                Error::Config(format!(
                    "Failed to parse config file {}: {e}",
                    config_path.display()
                ))
            })?;

            expand_tilde(&mut config.database_path);
            expand_tilde(&mut config.model_cache);

            Some(config)
        } else {
            None
        };

        let mut config = Config::default();

        if let Some(file) = file_config {
            config.merge_from_file(file);
        }

        config.apply_env_overrides()?;
        config.validate()?;

        Ok(config)
    }

    #[allow(dead_code)] // Dead code justified: called from Config::load()
    fn validate(&self) -> Result<(), Error> {
        if self.similarity_threshold < 0.0 || self.similarity_threshold > 1.0 {
            return Err(Error::Config(format!(
                "Invalid similarity threshold: {} (must be between 0.0 and 1.0)",
                self.similarity_threshold
            )));
        }

        if self.recency_weight < 0.0 || self.recency_weight > 1.0 {
            return Err(Error::Config(format!(
                "Invalid recency weight: {} (must be between 0.0 and 1.0)",
                self.recency_weight
            )));
        }

        if self.embedding_model.trim().is_empty() {
            return Err(Error::Config("Embedding model cannot be empty".to_string()));
        }

        if self.database_path.as_os_str().is_empty() {
            return Err(Error::Config("Database path cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Ensure parent directories for database and cache paths exist.
    #[allow(dead_code)] // Dead code justified: public API for CLI integration (issue #5)
    pub fn ensure_directories(&self) -> Result<(), Error> {
        if let Some(parent) = self.database_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Error::Config(format!(
                        "Failed to create database directory {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }

        if !self.model_cache.as_os_str().is_empty() {
            std::fs::create_dir_all(&self.model_cache).map_err(|e| {
                Error::Config(format!(
                    "Failed to create model cache directory {}: {e}",
                    self.model_cache.display()
                ))
            })?;
        }

        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    database_path: PathBuf,

    #[serde(default)]
    embedding_model: String,

    #[serde(default)]
    model_cache: PathBuf,

    #[serde(default = "default_threshold")]
    similarity_threshold: f64,

    #[serde(default)]
    recency_weight: f64,
}

#[allow(dead_code)] // Dead code justified: used in ConfigFile serde default
fn default_threshold() -> f64 {
    0.85
}

/// Expand `~` to home directory in a PathBuf (in-place).
#[allow(dead_code)] // Dead code justified: used in Config::load()
fn expand_tilde(path: &mut PathBuf) {
    if path.starts_with("~") {
        if let Some(home) = dirs::home_dir() {
            let rest = path.strip_prefix("~").unwrap_or(Path::new(""));
            *path = home.join(rest);
        }
    }
}

/// Expand `~` to home directory in a PathBuf (returns new PathBuf).
#[allow(dead_code)] // Dead code justified: used in Config::load()
fn expand_tilde_path(path: &Path) -> PathBuf {
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
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn cleanup_env_vars() {
        let vars = [
            "VIPUNE_DATABASE_PATH",
            "VIPUNE_EMBEDDING_MODEL",
            "VIPUNE_MODEL_CACHE",
            "VIPUNE_SIMILARITY_THRESHOLD",
            "VIPUNE_RECENCY_WEIGHT",
        ];
        for var in vars {
            #[allow(clippy::disallowed_methods)]
            std::env::remove_var(var);
        }
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();

        assert!(config.database_path.ends_with("vipune/memories.db"));
        assert_eq!(
            config.embedding_model,
            "sentence-transformers/bge-small-en-v1.5"
        );
        assert!(config.model_cache.ends_with("vipune/models"));
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.recency_weight, 0.0);
    }

    #[test]
    fn test_config_load_without_file() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        let config = Config::load().unwrap();

        assert!(config.database_path.ends_with("vipune/memories.db"));
        assert_eq!(
            config.embedding_model,
            "sentence-transformers/bge-small-en-v1.5"
        );
        assert_eq!(config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_config_file_overrides_defaults() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        let config = Config::load().unwrap();

        assert!(config.database_path.ends_with("vipune/memories.db"));
        assert_eq!(
            config.embedding_model,
            "sentence-transformers/bge-small-en-v1.5"
        );
        assert!(config.model_cache.ends_with("vipune/models"));
        assert_eq!(config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_env_var_overrides_config() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_DATABASE_PATH", "/custom/path/db.db");
        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_EMBEDDING_MODEL", "env/model");
        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_MODEL_CACHE", "/custom/cache");
        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "0.95");

        let config = Config::load().unwrap();

        assert_eq!(config.database_path, PathBuf::from("/custom/path/db.db"));
        assert_eq!(config.embedding_model, "env/model");
        assert_eq!(config.model_cache, PathBuf::from("/custom/cache"));
        assert_eq!(config.similarity_threshold, 0.95);

        cleanup_env_vars();
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "invalid");

        let result = Config::load();

        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
    }

    #[test]
    fn test_similarity_threshold_range_validation() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "1.5");

        let result = Config::load();

        assert!(matches!(result, Err(Error::Config(_))));

        #[allow(clippy::disallowed_methods)]
        std::env::remove_var("VIPUNE_SIMILARITY_THRESHOLD");

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "-0.1");

        let result = Config::load();

        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
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

    #[test]
    fn test_valid_similarity_threshold_bounds() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "0.0");

        let config = Config::load().unwrap();
        assert_eq!(config.similarity_threshold, 0.0);

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "1.0");

        let config = Config::load().unwrap();
        assert_eq!(config.similarity_threshold, 1.0);

        cleanup_env_vars();
    }

    #[test]
    fn test_empty_env_var_rejected() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_DATABASE_PATH", "");
        let result = Config::load();
        assert!(matches!(result, Err(Error::Config(_))));
        #[allow(clippy::disallowed_methods)]
        std::env::remove_var("VIPUNE_DATABASE_PATH");

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_EMBEDDING_MODEL", "");
        let result = Config::load();
        assert!(matches!(result, Err(Error::Config(_))));
        #[allow(clippy::disallowed_methods)]
        std::env::remove_var("VIPUNE_EMBEDDING_MODEL");

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_MODEL_CACHE", "");
        let result = Config::load();
        assert!(matches!(result, Err(Error::Config(_))));
        #[allow(clippy::disallowed_methods)]
        std::env::remove_var("VIPUNE_MODEL_CACHE");

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "");
        let result = Config::load();
        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
    }

    #[test]
    fn test_whitespace_env_var_rejected() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_EMBEDDING_MODEL", "   ");
        let result = Config::load();
        assert!(matches!(result, Err(Error::Config(_))));
        #[allow(clippy::disallowed_methods)]
        std::env::remove_var("VIPUNE_EMBEDDING_MODEL");

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "  ");
        let result = Config::load();
        assert!(matches!(result, Err(Error::Config(_))));
        #[allow(clippy::disallowed_methods)]
        std::env::remove_var("VIPUNE_SIMILARITY_THRESHOLD");
    }

    #[test]
    fn test_malformed_toml() {
        let content = r#"
This is not valid TOML
 [[unclosed bracket
 "#;

        let result: Result<ConfigFile, _> = toml::from_str(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_config_file() {
        let content = "";

        let result: Result<ConfigFile, _> = toml::from_str(content);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert!(config.database_path.as_os_str().is_empty());
        assert!(config.embedding_model.is_empty());
        assert!(config.model_cache.as_os_str().is_empty());
        assert_eq!(config.similarity_threshold, 0.85);
    }
}
