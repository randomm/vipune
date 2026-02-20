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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_similarity_threshold() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 1.5,
            recency_weight: 0.3,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_similarity_threshold_range_validation() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 1.5,
            recency_weight: 0.3,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_valid_similarity_threshold_bounds() {
        let mut validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 0.0,
            recency_weight: 0.3,
        };
        assert!(validator.validate().is_ok());

        validator.similarity_threshold = 1.0;
        assert!(validator.validate().is_ok());
    }

    #[test]
    fn test_similarity_threshold_nan_rejected() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: f64::NAN,
            recency_weight: 0.3,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_similarity_threshold_infinity_rejected() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: f64::INFINITY,
            recency_weight: 0.3,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_recency_weight_range_validation() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 1.5,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_valid_recency_weight_bounds() {
        let mut validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: 0.0,
        };
        assert!(validator.validate().is_ok());

        validator.recency_weight = 1.0;
        assert!(validator.validate().is_ok());

        validator.recency_weight = 0.3;
        assert!(validator.validate().is_ok());
    }

    #[test]
    fn test_recency_weight_nan_rejected() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: f64::NAN,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }

    #[test]
    fn test_recency_weight_infinity_rejected() {
        let validator = ConfigValidator {
            database_path: PathBuf::from("/test"),
            embedding_model: "test/model".to_string(),
            similarity_threshold: 0.85,
            recency_weight: f64::INFINITY,
        };

        assert!(matches!(validator.validate(), Err(Error::Config(_))));
    }
}
