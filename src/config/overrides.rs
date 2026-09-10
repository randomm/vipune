//! Environment variable overrides for configuration.

use crate::errors::Error;
use std::path::PathBuf;

use super::env_parser;

#[cfg(test)]
use super::tests_utils::{ENV_MUTEX, cleanup_env_vars};

/// Apply environment variable overrides to configuration.
pub fn apply_env_overrides(
    database_path: &mut PathBuf,
    embedding_model: &mut String,
    similarity_threshold: &mut f64,
    recency_weight: &mut f64,
    hybrid: &mut bool,
) -> Result<(), Error> {
    env_parser::apply_database_path_override(database_path)?;
    env_parser::apply_embedding_model_override(embedding_model)?;
    env_parser::apply_similarity_threshold_override(similarity_threshold)?;
    env_parser::apply_recency_weight_override(recency_weight)?;
    env_parser::apply_hybrid_override(hybrid)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overrides(
        database_path: &mut PathBuf,
        embedding_model: &mut String,
        similarity_threshold: &mut f64,
        recency_weight: &mut f64,
        hybrid: &mut bool,
    ) -> Result<(), Error> {
        apply_env_overrides(
            database_path,
            embedding_model,
            similarity_threshold,
            recency_weight,
            hybrid,
        )
    }

    fn clean_env() {
        cleanup_env_vars(&[
            "VIPUNE_DATABASE_PATH",
            "VIPUNE_EMBEDDING_MODEL",
            "VIPUNE_SIMILARITY_THRESHOLD",
            "VIPUNE_RECENCY_WEIGHT",
            "VIPUNE_HYBRID",
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

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        )
        .unwrap();

        assert_eq!(database_path, PathBuf::from("/custom/path/db.db"));
        assert_eq!(embedding_model, "env/model");
        assert_eq!(similarity_threshold, 0.95);

        clean_env();
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "invalid");
        }

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        let result = overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        );

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

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        let result = overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        );

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

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        let result = overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        );

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

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        )
        .unwrap();

        assert_eq!(recency_weight, 0.5);

        clean_env();
    }

    #[test]
    fn test_hybrid_env_var_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_HYBRID", "true");
        }

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        )
        .unwrap();

        assert!(hybrid);

        clean_env();
    }

    #[test]
    fn test_hybrid_env_var_override_false() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_HYBRID", "false");
        }

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = true;

        overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        )
        .unwrap();

        assert!(!hybrid);

        clean_env();
    }

    #[test]
    fn test_invalid_recency_weight_format() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clean_env();

        unsafe {
            std::env::set_var("VIPUNE_RECENCY_WEIGHT", "invalid");
        }

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;
        let mut hybrid = false;

        let result = overrides(
            &mut database_path,
            &mut embedding_model,
            &mut similarity_threshold,
            &mut recency_weight,
            &mut hybrid,
        );

        assert!(matches!(result, Err(Error::Config(_))));

        clean_env();
    }
}
