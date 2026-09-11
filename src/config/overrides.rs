//! Environment variable overrides for configuration.

use crate::config::Config;
use crate::errors::Error;

#[cfg(test)]
use crate::config::tests_utils::{ENV_MUTEX, cleanup_env_vars};
#[cfg(test)]
use std::path::PathBuf;

use super::env_parser;

/// Apply environment variable overrides to configuration.
pub fn apply_env_overrides(config: &mut Config) -> Result<(), Error> {
    env_parser::apply_database_path_override(&mut config.database_path)?;
    env_parser::apply_embedding_model_override(&mut config.embedding_model)?;
    env_parser::apply_similarity_threshold_override(&mut config.similarity_threshold)?;
    env_parser::apply_recency_weight_override(&mut config.recency_weight)?;
    env_parser::apply_hybrid_override(&mut config.hybrid)?;
    env_parser::apply_decay_refresh_days_override(&mut config.decay_refresh_days)?;
    env_parser::apply_promotion_threshold_override(&mut config.promotion_threshold)?;
    env_parser::apply_prune_retrieval_limit_override(&mut config.prune_retrieval_limit)?;
    env_parser::apply_prune_min_age_days_override(&mut config.prune_min_age_days)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_env() {
        cleanup_env_vars(&[
            "VIPUNE_DATABASE_PATH",
            "VIPUNE_EMBEDDING_MODEL",
            "VIPUNE_SIMILARITY_THRESHOLD",
            "VIPUNE_RECENCY_WEIGHT",
            "VIPUNE_HYBRID",
            "VIPUNE_DECAY_REFRESH_DAYS",
            "VIPUNE_PROMOTION_THRESHOLD",
            "VIPUNE_PRUNE_RETRIEVAL_LIMIT",
            "VIPUNE_PRUNE_MIN_AGE_DAYS",
        ]);
    }

    #[test]
    fn test_env_var_overrides_config() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_DATABASE_PATH", "/custom/path/db.db");
            std::env::set_var("VIPUNE_EMBEDDING_MODEL", "env/model");
            std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "0.95");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        apply_env_overrides(&mut config).unwrap();

        assert_eq!(config.database_path, PathBuf::from("/custom/path/db.db"));
        assert_eq!(config.embedding_model, "env/model");
        assert_eq!(config.similarity_threshold, 0.95);

        clean_env();
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "invalid");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        let result = apply_env_overrides(&mut config);

        assert!(matches!(result, Err(Error::Config(_))));

        clean_env();
    }

    #[test]
    fn test_empty_env_var_rejected() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_DATABASE_PATH", "");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        let result = apply_env_overrides(&mut config);

        assert!(matches!(result, Err(Error::Config(_))));

        clean_env();
    }

    #[test]
    fn test_whitespace_env_var_rejected() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_EMBEDDING_MODEL", "   ");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        let result = apply_env_overrides(&mut config);

        assert!(matches!(result, Err(Error::Config(_))));

        clean_env();
    }

    #[test]
    fn test_recency_weight_env_var_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_RECENCY_WEIGHT", "0.5");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        apply_env_overrides(&mut config).unwrap();

        assert_eq!(config.recency_weight, 0.5);

        clean_env();
    }

    #[test]
    fn test_hybrid_env_var_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_HYBRID", "true");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        apply_env_overrides(&mut config).unwrap();

        assert!(config.hybrid);

        clean_env();
    }

    #[test]
    fn test_hybrid_env_var_override_false() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_HYBRID", "false");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: true,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        apply_env_overrides(&mut config).unwrap();

        assert!(!config.hybrid);

        clean_env();
    }

    #[test]
    fn test_invalid_recency_weight_format() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_RECENCY_WEIGHT", "invalid");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        let result = apply_env_overrides(&mut config);

        assert!(matches!(result, Err(Error::Config(_))));

        clean_env();
    }

    #[test]
    fn test_lifecycle_knob_env_overrides() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_DECAY_REFRESH_DAYS", "14.0");
            std::env::set_var("VIPUNE_PROMOTION_THRESHOLD", "10");
            std::env::set_var("VIPUNE_PRUNE_RETRIEVAL_LIMIT", "3");
            std::env::set_var("VIPUNE_PRUNE_MIN_AGE_DAYS", "7");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        apply_env_overrides(&mut config).unwrap();

        assert_eq!(config.decay_refresh_days, 14.0);
        assert_eq!(config.promotion_threshold, 10);
        assert_eq!(config.prune_retrieval_limit, 3);
        assert_eq!(config.prune_min_age_days, 7.0);

        clean_env();
    }

    #[test]
    fn test_invalid_lifecycle_knob_values() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_PROMOTION_THRESHOLD", "-1");
            std::env::set_var("VIPUNE_PRUNE_MIN_AGE_DAYS", "invalid");
        }

        let mut config = Config {
            database_path: PathBuf::from("/default"),
            embedding_model: "default/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.3,
            hybrid: false,
            decay_refresh_days: 30.0,
            promotion_threshold: 5,
            prune_retrieval_limit: 5,
            prune_min_age_days: 30.0,
        };

        let result = apply_env_overrides(&mut config);

        assert!(matches!(result, Err(Error::Config(_))));

        clean_env();
    }
}
