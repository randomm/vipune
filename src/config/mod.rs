//! Configuration system for vipune.

mod env_parser;
mod loader;
mod overrides;
mod paths;
mod validation;

#[cfg(test)]
mod tests_utils;
#[cfg(test)]
use tests_utils::{ENV_MUTEX, cleanup_env_vars};

use crate::embedding::EMBED_MODEL_ID;
use crate::errors::Error;
use serde::Deserialize;
use std::path::PathBuf;

pub use loader::ConfigFile;

/// Configuration values with priority: defaults < config file < env vars.
///
/// Provides configuration options for the memory store, including database paths,
/// embedding model selection, and search parameters.
///
/// # Example
///
/// ```
/// use vipune::Config;
///
/// let config = Config::default();
/// println!("Database path: {:?}", config.database_path);
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Path to the SQLite database file (e.g., `~/.vipune/memories.db`).
    #[serde(default)]
    pub database_path: PathBuf,

    /// HuggingFace embedding model identifier (e.g., `BAAI/bge-small-en-v1.5`).
    #[serde(default)]
    pub embedding_model: String,

    /// Minimum similarity score (0.0-1.0) required for search results to be returned.
    #[serde(default)]
    pub similarity_threshold: f64,

    /// Weight applied to recency in search ranking (0.0 = ignore time, 1.0 = prioritize recent).
    #[serde(default)]
    pub recency_weight: f64,

    /// Whether to use hybrid search (semantic + BM25) by default.
    #[serde(default)]
    pub hybrid: bool,
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
            embedding_model: EMBED_MODEL_ID.to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
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
            config.merge_from_file(file);
        }

        overrides::apply_env_overrides(
            &mut config.database_path,
            &mut config.embedding_model,
            &mut config.similarity_threshold,
            &mut config.recency_weight,
            &mut config.hybrid,
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

    /// Ensure the parent directory of the database path exists.
    ///
    /// Model files are cached in the HuggingFace Hub cache
    /// (`~/.cache/huggingface/hub/` by default), which `hf-hub` creates on
    /// demand — vipune does not pre-create any model directory.
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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();

        assert!(config.database_path.ends_with(".vipune/memories.db"));
        assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.recency_weight, 0.3);
        assert!(!config.hybrid);
    }

    #[test]
    fn test_config_load_without_file() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars(&[
            "VIPUNE_DATABASE_PATH",
            "VIPUNE_EMBEDDING_MODEL",
            "VIPUNE_SIMILARITY_THRESHOLD",
            "VIPUNE_RECENCY_WEIGHT",
        ]);

        let config = Config::load().unwrap();

        assert!(config.database_path.ends_with(".vipune/memories.db"));
        assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_config_file_overrides_defaults() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars(&[
            "VIPUNE_DATABASE_PATH",
            "VIPUNE_EMBEDDING_MODEL",
            "VIPUNE_SIMILARITY_THRESHOLD",
            "VIPUNE_RECENCY_WEIGHT",
        ]);

        let config = Config::load().unwrap();

        assert!(config.database_path.ends_with(".vipune/memories.db"));
        assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_default_model_matches_embed_constant() {
        let config = Config::default();
        assert_eq!(
            config.embedding_model,
            crate::embedding::EMBED_MODEL_ID,
            "Config::default() model must match EMBED_MODEL_ID — update both when changing the default model"
        );
    }
}
