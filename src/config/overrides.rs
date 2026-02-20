//! Environment variable overrides for configuration.

use crate::errors::Error;
use std::path::PathBuf;

use super::env_parser;

#[cfg(test)]
use super::tests_utils::ENV_MUTEX;

/// Apply environment variable overrides to configuration.
pub fn apply_env_overrides(
    database_path: &mut PathBuf,
    embedding_model: &mut String,
    model_cache: &mut PathBuf,
    similarity_threshold: &mut f64,
    recency_weight: &mut f64,
) -> Result<(), Error> {
    env_parser::apply_database_path_override(database_path)?;
    env_parser::apply_embedding_model_override(embedding_model)?;
    env_parser::apply_model_cache_override(model_cache)?;
    env_parser::apply_similarity_threshold_override(similarity_threshold)?;
    env_parser::apply_recency_weight_override(recency_weight)?;
    Ok(())
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

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut model_cache = PathBuf::from("/default/cache");
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;

        apply_env_overrides(
            &mut database_path,
            &mut embedding_model,
            &mut model_cache,
            &mut similarity_threshold,
            &mut recency_weight,
        )
        .unwrap();

        assert_eq!(database_path, PathBuf::from("/custom/path/db.db"));
        assert_eq!(embedding_model, "env/model");
        assert_eq!(model_cache, PathBuf::from("/custom/cache"));
        assert_eq!(similarity_threshold, 0.95);

        cleanup_env_vars();
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_SIMILARITY_THRESHOLD", "invalid");

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut model_cache = PathBuf::from("/default/cache");
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;

        let result = apply_env_overrides(
            &mut database_path,
            &mut embedding_model,
            &mut model_cache,
            &mut similarity_threshold,
            &mut recency_weight,
        );

        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
    }

    #[test]
    fn test_empty_env_var_rejected() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_DATABASE_PATH", "");

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut model_cache = PathBuf::from("/default/cache");
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;

        let result = apply_env_overrides(
            &mut database_path,
            &mut embedding_model,
            &mut model_cache,
            &mut similarity_threshold,
            &mut recency_weight,
        );

        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
    }

    #[test]
    fn test_whitespace_env_var_rejected() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_EMBEDDING_MODEL", "   ");

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut model_cache = PathBuf::from("/default/cache");
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;

        let result = apply_env_overrides(
            &mut database_path,
            &mut embedding_model,
            &mut model_cache,
            &mut similarity_threshold,
            &mut recency_weight,
        );

        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
    }

    #[test]
    fn test_recency_weight_env_var_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_RECENCY_WEIGHT", "0.5");

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut model_cache = PathBuf::from("/default/cache");
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;

        apply_env_overrides(
            &mut database_path,
            &mut embedding_model,
            &mut model_cache,
            &mut similarity_threshold,
            &mut recency_weight,
        )
        .unwrap();

        assert_eq!(recency_weight, 0.5);

        cleanup_env_vars();
    }

    #[test]
    fn test_invalid_recency_weight_format() {
        let _guard = ENV_MUTEX.lock().unwrap();
        cleanup_env_vars();

        #[allow(clippy::disallowed_methods)]
        std::env::set_var("VIPUNE_RECENCY_WEIGHT", "invalid");

        let mut database_path = PathBuf::from("/default");
        let mut embedding_model = "default/model".to_string();
        let mut model_cache = PathBuf::from("/default/cache");
        let mut similarity_threshold = 0.85;
        let mut recency_weight = 0.3;

        let result = apply_env_overrides(
            &mut database_path,
            &mut embedding_model,
            &mut model_cache,
            &mut similarity_threshold,
            &mut recency_weight,
        );

        assert!(matches!(result, Err(Error::Config(_))));

        cleanup_env_vars();
    }
}
