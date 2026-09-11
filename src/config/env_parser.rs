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

/// Parse an environment variable as a non-negative f64.
fn parse_env_non_negative_float(name: &str, value: &str) -> Result<f64, Error> {
    let parsed = parse_env_float(name, value)?;
    if parsed < 0.0 {
        return Err(Error::Config(format!(
            "Invalid {name} value: {parsed} (must be >= 0)"
        )));
    }
    Ok(parsed)
}

/// Parse an environment variable as a positive integer (>= 1).
fn parse_env_positive_int(name: &str, value: &str) -> Result<i64, Error> {
    if value.trim().is_empty() {
        return Err(Error::Config(format!("{name} cannot be empty")));
    }
    let parsed = value
        .trim()
        .parse()
        .map_err(|e| Error::Config(format!("Invalid {name} value: {e}")))?;
    if parsed < 1 {
        return Err(Error::Config(format!(
            "Invalid {name} value: {parsed} (must be >= 1)"
        )));
    }
    Ok(parsed)
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

/// Apply VIPUNE_HYBRID environment variable override.
pub fn apply_hybrid_override(hybrid: &mut bool) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_HYBRID") {
        match val.to_ascii_lowercase().as_str() {
            "true" | "1" => *hybrid = true,
            "false" | "0" | "" => *hybrid = false,
            _ => {
                eprintln!(
                    "warning: VIPUNE_HYBRID='{}' is not a valid boolean (expected true/false/1/0), keeping default",
                    val
                );
            }
        }
    }
    Ok(())
}

/// Apply VIPUNE_DECAY_REFRESH_DAYS environment variable override.
pub fn apply_decay_refresh_days_override(decay_refresh_days: &mut f64) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_DECAY_REFRESH_DAYS") {
        *decay_refresh_days = parse_env_non_negative_float("VIPUNE_DECAY_REFRESH_DAYS", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_PROMOTION_THRESHOLD environment variable override.
pub fn apply_promotion_threshold_override(promotion_threshold: &mut i64) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_PROMOTION_THRESHOLD") {
        *promotion_threshold = parse_env_positive_int("VIPUNE_PROMOTION_THRESHOLD", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_PRUNE_RETRIEVAL_LIMIT environment variable override.
pub fn apply_prune_retrieval_limit_override(prune_retrieval_limit: &mut i64) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_PRUNE_RETRIEVAL_LIMIT") {
        *prune_retrieval_limit = parse_env_positive_int("VIPUNE_PRUNE_RETRIEVAL_LIMIT", &val)?;
    }
    Ok(())
}

/// Apply VIPUNE_PRUNE_MIN_AGE_DAYS environment variable override.
pub fn apply_prune_min_age_days_override(prune_min_age_days: &mut f64) -> Result<(), Error> {
    if let Ok(val) = std::env::var("VIPUNE_PRUNE_MIN_AGE_DAYS") {
        *prune_min_age_days = parse_env_non_negative_float("VIPUNE_PRUNE_MIN_AGE_DAYS", &val)?;
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

    #[test]
    fn test_parse_env_non_negative_float_negative_rejected() {
        let result = parse_env_non_negative_float("TEST_VAR", "-1.5");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_parse_env_non_negative_float_valid() {
        let result = parse_env_non_negative_float("TEST_VAR", "14");
        assert_eq!(result.unwrap(), 14.0);
    }

    #[test]
    fn test_parse_env_positive_int_zero_rejected() {
        let result = parse_env_positive_int("TEST_VAR", "0");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_parse_env_positive_int_negative_rejected() {
        let result = parse_env_positive_int("TEST_VAR", "-1");
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_parse_env_positive_int_valid() {
        let result = parse_env_positive_int("TEST_VAR", "10");
        assert_eq!(result.unwrap(), 10);
    }
}
