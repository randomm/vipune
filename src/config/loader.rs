//! Configuration file loading and parsing.

use crate::errors::Error;
use serde::Deserialize;
use std::path::PathBuf;

/// Configuration loaded from TOML file.
#[derive(Debug, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub database_path: PathBuf,

    #[serde(default)]
    pub embedding_model: String,

    #[serde(default)]
    pub model_cache: PathBuf,

    #[serde(default = "default_threshold")]
    pub similarity_threshold: f64,

    #[serde(default = "default_recency_weight")]
    pub recency_weight: f64,
}

#[allow(dead_code)]
fn default_threshold() -> f64 {
    0.85
}

#[allow(dead_code)]
fn default_recency_weight() -> f64 {
    0.3
}

/// Load configuration from TOML file.
pub fn load_from_file() -> Result<Option<ConfigFile>, Error> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let config_dir = dirs::config_dir().unwrap_or_else(|| home.join(".config"));

    let config_path = config_dir.join("vipune/config.toml");

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path).map_err(|e| {
            Error::Config(format!(
                "Failed to read config file {}: {e}",
                config_path.display()
            ))
        })?;

        let config: ConfigFile = toml::from_str(&content).map_err(|e| {
            Error::Config(format!(
                "Failed to parse config file {}: {e}",
                config_path.display()
            ))
        })?;

        Ok(Some(config))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_config_file_missing_recency_weight() {
        let content = ""; // No recency_weight field

        let result: Result<ConfigFile, _> = toml::from_str(content);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.recency_weight, 0.3); // Should use default, not f64::default() (0.0)
    }

    #[test]
    fn test_config_file_partial_toml() {
        let content = r#"
            database_path = "/test/db.db"
        "#;

        let result: Result<ConfigFile, _> = toml::from_str(content);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.database_path, PathBuf::from("/test/db.db"));
        assert_eq!(config.recency_weight, 0.3); // Missing field uses default 0.3
        assert_eq!(config.similarity_threshold, 0.85); // Missing field uses default 0.85
    }
}
