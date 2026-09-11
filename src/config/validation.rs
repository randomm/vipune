//! Configuration validation logic.

use crate::errors::Error;
use std::path::PathBuf;

/// Validates configuration values.
pub struct ConfigValidator {
    /// Path to the SQLite database file.
    pub database_path: PathBuf,
    /// HuggingFace embedding model identifier.
    pub embedding_model: String,
    /// Minimum similarity threshold for search results.
    pub similarity_threshold: f64,
    /// Recency weight for search ranking.
    pub recency_weight: f64,
    /// Cap (in days) on the recency refresh used in decay scoring.
    pub decay_refresh_days: f64,
    /// Retrieval-count threshold (>=) for promoting a candidate to active.
    pub promotion_threshold: i64,
    /// Prune eligibility: a candidate with retrieval_count < N is prunable.
    pub prune_retrieval_limit: i64,
    /// Prune eligibility: a candidate older than T days is prunable.
    pub prune_min_age_days: f64,
}

impl ConfigValidator {
    /// Validate all configuration values for correctness and constraints.
    ///
    /// Checks that:
    /// - Similarity threshold is between 0.0 and 1.0
    /// - Recency weight is between 0.0 and 1.0
    /// - Embedding model is not empty
    /// - Database path is not empty
    /// - No NaN or infinite values
    ///
    /// # Errors
    ///
    /// Returns `Error::Config` if any validation check fails.
    pub fn validate(&self) -> Result<(), Error> {
        self.validate_similarity_threshold()?;
        self.validate_recency_weight()?;
        self.validate_embedding_model()?;
        self.validate_database_path()?;
        self.validate_decay_refresh_days()?;
        self.validate_promotion_threshold()?;
        self.validate_prune_retrieval_limit()?;
        self.validate_prune_min_age_days()?;

        Ok(())
    }

    fn validate_similarity_threshold(&self) -> Result<(), Error> {
        if self.similarity_threshold.is_nan() || self.similarity_threshold.is_infinite() {
            return Err(Error::Config(
                "Invalid similarity threshold: NaN and infinity are not allowed".into(),
            ));
        }

        if self.similarity_threshold < 0.0 || self.similarity_threshold > 1.0 {
            return Err(Error::Config(format!(
                "Invalid similarity threshold: {} (must be between 0.0 and 1.0)",
                self.similarity_threshold
            )));
        }

        Ok(())
    }

    fn validate_recency_weight(&self) -> Result<(), Error> {
        if self.recency_weight.is_nan() || self.recency_weight.is_infinite() {
            return Err(Error::Config(
                "Invalid recency weight: NaN and infinity are not allowed".into(),
            ));
        }

        if self.recency_weight < 0.0 || self.recency_weight > 1.0 {
            return Err(Error::Config(format!(
                "Invalid recency weight: {} (must be between 0.0 and 1.0)",
                self.recency_weight
            )));
        }

        Ok(())
    }

    fn validate_embedding_model(&self) -> Result<(), Error> {
        if self.embedding_model.trim().is_empty() {
            return Err(Error::Config("Embedding model cannot be empty".to_string()));
        }

        Ok(())
    }

    fn validate_database_path(&self) -> Result<(), Error> {
        if self.database_path.as_os_str().is_empty() {
            return Err(Error::Config("Database path cannot be empty".to_string()));
        }

        Ok(())
    }

    fn validate_decay_refresh_days(&self) -> Result<(), Error> {
        if self.decay_refresh_days.is_nan() || self.decay_refresh_days.is_infinite() {
            return Err(Error::Config(
                "Invalid decay refresh days: NaN and infinity are not allowed".into(),
            ));
        }

        if self.decay_refresh_days < 0.0 {
            return Err(Error::Config(format!(
                "Invalid decay refresh days: {} (must be >= 0)",
                self.decay_refresh_days
            )));
        }

        Ok(())
    }

    fn validate_promotion_threshold(&self) -> Result<(), Error> {
        if self.promotion_threshold < 1 {
            return Err(Error::Config(format!(
                "Invalid promotion threshold: {} (must be >= 1)",
                self.promotion_threshold
            )));
        }

        Ok(())
    }

    fn validate_prune_retrieval_limit(&self) -> Result<(), Error> {
        if self.prune_retrieval_limit < 1 {
            return Err(Error::Config(format!(
                "Invalid prune retrieval limit: {} (must be >= 1)",
                self.prune_retrieval_limit
            )));
        }

        Ok(())
    }

    fn validate_prune_min_age_days(&self) -> Result<(), Error> {
        if self.prune_min_age_days.is_nan() || self.prune_min_age_days.is_infinite() {
            return Err(Error::Config(
                "Invalid prune min age days: NaN and infinity are not allowed".into(),
            ));
        }

        if self.prune_min_age_days < 0.0 {
            return Err(Error::Config(format!(
                "Invalid prune min age days: {} (must be >= 0)",
                self.prune_min_age_days
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIFECYCLE_DEFAULTS: (f64, i64, i64, f64) = (30.0, 5, 5, 30.0);

    fn build_validator(
        similarity_threshold: f64,
        recency_weight: f64,
        (decay_refresh_days, promotion_threshold, prune_retrieval_limit, prune_min_age_days): (
            f64,
            i64,
            i64,
            f64,
        ),
    ) -> ConfigValidator {
        ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold,
            recency_weight,
            decay_refresh_days,
            promotion_threshold,
            prune_retrieval_limit,
            prune_min_age_days,
        }
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let validator = build_validator(1.5, 0.3, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_similarity_threshold_range_validation() {
        let validator = build_validator(1.5, 0.3, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_valid_similarity_threshold_bounds() {
        let mut validator = build_validator(0.0, 0.3, LIFECYCLE_DEFAULTS);
        assert!(validator.validate().is_ok());

        validator.similarity_threshold = 1.0;
        assert!(validator.validate().is_ok());
    }

    #[test]
    fn test_similarity_threshold_nan_rejected() {
        let validator = build_validator(f64::NAN, 0.3, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_similarity_threshold_infinity_rejected() {
        let validator = build_validator(f64::INFINITY, 0.3, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_recency_weight_range_validation() {
        let validator = build_validator(0.85, 1.5, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_valid_recency_weight_bounds() {
        let mut validator = build_validator(0.85, 0.0, LIFECYCLE_DEFAULTS);
        assert!(validator.validate().is_ok());

        validator.recency_weight = 1.0;
        assert!(validator.validate().is_ok());

        validator.recency_weight = 0.3;
        assert!(validator.validate().is_ok());
    }

    #[test]
    fn test_recency_weight_nan_rejected() {
        let validator = build_validator(0.85, f64::NAN, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_recency_weight_infinity_rejected() {
        let validator = build_validator(0.85, f64::INFINITY, LIFECYCLE_DEFAULTS);

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_decay_refresh_days_negative_rejected() {
        let validator = build_validator(0.85, 0.3, (-1.0, 5, 5, 30.0));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_decay_refresh_days_nan_rejected() {
        let validator = build_validator(0.85, 0.3, (f64::NAN, 5, 5, 30.0));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_promotion_threshold_zero_rejected() {
        let validator = build_validator(0.85, 0.3, (30.0, 0, 5, 30.0));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_promotion_threshold_negative_rejected() {
        let validator = build_validator(0.85, 0.3, (30.0, -3, 5, 30.0));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_prune_retrieval_limit_zero_rejected() {
        let validator = build_validator(0.85, 0.3, (30.0, 5, 0, 30.0));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_prune_min_age_days_negative_rejected() {
        let validator = build_validator(0.85, 0.3, (30.0, 5, 5, -1.0));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_prune_min_age_days_nan_rejected() {
        let validator = build_validator(0.85, 0.3, (30.0, 5, 5, f64::NAN));

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_valid_lifecycle_defaults() {
        let validator = build_validator(0.85, 0.3, LIFECYCLE_DEFAULTS);

        assert!(validator.validate().is_ok());
    }
}
