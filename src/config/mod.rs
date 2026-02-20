//! Configuration system for vipune.

mod env_parser;
mod loader;
mod overrides;
mod paths;
mod validation;

#[cfg(test)]
mod tests_utils;
#[cfg(test)]
use tests_utils::ENV_MUTEX;

use crate::errors::Error;
use serde::Deserialize;
use std::path::PathBuf;

pub use loader::ConfigFile;

/// Configuration values with priority: defaults < config file < env vars.
#[derive(Debug, Clone, Deserialize)]
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
        // Use home directory with sensible fallback for systems without HOME
        let home = dirs::home_dir().unwrap_or_else(|| {
            std::env::var("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
        });
        let vipune_dir = home.join(".vipune");

        Self {
            database_path: vipune_dir.join("memories.db"),
            embedding_model: "BAAI/bge-small-en-v1.5".to_string(),
            model_cache: vipune_dir.join("models"),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
        }
    }
}

impl Config {
    /// Load configuration with defaults, file values, and environment overrides.
    pub fn load() -> Result<Self, Error> {
        let file_config = loader::load_from_file()?;

        let mut config = Config::default();

        if let Some(mut file) = file_config {
            paths::expand_tilde(&mut file.database_path);
            paths::expand_tilde(&mut file.model_cache);
            config.merge_from_file(file);
        }

        overrides::apply_env_overrides(
            &mut config.database_path,
            &mut config.embedding_model,
            &mut config.model_cache,
            &mut config.similarity_threshold,
            &mut config.recency_weight,
        )?;

        config.validate()?;

        Ok(config)
    }

    /// Merge configuration from a file into this config.
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

    /// Validate configuration values.
    fn validate(&self) -> Result<(), Error> {
        let validator = validation::ConfigValidator {
            database_path: self.database_path.clone(),
            embedding_model: self.embedding_model.clone(),
            similarity_threshold: self.similarity_threshold,
            recency_weight: self.recency_weight,
        };

        validator.validate()
    }

    /// Ensure parent directories for database and cache paths exist.
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

#[cfg(test)]
mod tests {
    use super::*;

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

        assert!(config.database_path.ends_with(".vipune/memories.db"));
        assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
        assert!(config.model_cache.ends_with(".vipune/models"));
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.recency_weight, 0.3);
    }

    #[test]
    fn test_config_load_without_file() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        let config = Config::load().unwrap();

        assert!(config.database_path.ends_with(".vipune/memories.db"));
        assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_config_file_overrides_defaults() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        let config = Config::load().unwrap();

        assert!(config.database_path.ends_with(".vipune/memories.db"));
        assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
        assert!(config.model_cache.ends_with(".vipune/models"));
        assert_eq!(config.similarity_threshold, 0.85);
    }
}
