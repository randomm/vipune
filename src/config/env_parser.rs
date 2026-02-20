//! Environment variable parsing utilities for configuration.

use crate::errors::Error;
use std::path::PathBuf;

use super::paths;

/// Parse environment variable value or return error if empty/whitespace.
fn parse_env_string(name: &str, value: &str) -> Result<String, Error> {
    if value.trim().is_empty() {
        return Err(Error::Config(format!("{name} cannot be empty")));
    }
    Ok(value.to_string())
}

/// Parse environment variable as a path, expanding tilde.
fn parse_env_path(name: &str, value: &str) -> Result<PathBuf, Error> {
    if value.trim().is_empty() {
        return Err(Error::Config(format!("{name} cannot be empty")));
    }
    Ok(paths::expand_tilde_path(&PathBuf::from(value)))
}

/// Parse environment variable as a f64 with range validation after parsing.
fn parse_env_float(name: &str, value: &str) -> Result<f64, Error> {
    if value.trim().is_empty() {
        return Err(Error::Config(format!("{name} cannot be empty")));
    }
    value
        .trim()
        .parse()
        .map_err(|e| Error::Config(format!("Invalid {name} value: {e}")))
}

/// Apply VIPUNE_DATABASE_PATH environment variable override.
pub fn apply_database_path_override(database_path: &mut PathBuf) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_DATABASE_PATH") {
        *database_path = parse_env_path("VIPUNE_DATABASE_PATH", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_EMBEDDING_MODEL environment variable override.
pub fn apply_embedding_model_override(embedding_model: &mut String) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_EMBEDDING_MODEL") {
        *embedding_model = parse_env_string("VIPUNE_EMBEDDING_MODEL", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_MODEL_CACHE environment variable override.
pub fn apply_model_cache_override(model_cache: &mut PathBuf) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_MODEL_CACHE") {
        *model_cache = parse_env_path("VIPUNE_MODEL_CACHE", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_SIMILARITY_THRESHOLD environment variable override.
pub fn apply_similarity_threshold_override(similarity_threshold: &mut f64) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_SIMILARITY_THRESHOLD") {
        *similarity_threshold = parse_env_float("VIPUNE_SIMILARITY_THRESHOLD", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_RECENCY_WEIGHT environment variable override.
pub fn apply_recency_weight_override(recency_weight: &mut f64) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_RECENCY_WEIGHT") {
        *recency_weight = parse_env_float("VIPUNE_RECENCY_WEIGHT", &val)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_env_string_empty() {
        let result = parse_env_string("TEST_VAR", "");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_parse_env_string_whitespace() {
        let result = parse_env_string("TEST_VAR", "   ");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_parse_env_string_valid() {
        let result = parse_env_string("TEST_VAR", "valid");
        assert_eq!(result.unwrap(), "valid");
    }

    #[test]
    fn test_parse_env_float_invalid() {
        let result = parse_env_float("TEST_FLOAT", "invalid");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_parse_env_float_valid() {
        let result = parse_env_float("TEST_FLOAT", "0.5");
        assert_eq!(result.unwrap(), 0.5);
    }
}
