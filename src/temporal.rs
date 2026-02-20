//! Temporal decay scoring for search result recency weighting.

use chrono::{DateTime, Utc};

#[cfg(test)]
use chrono::Duration; // Only import in tests

/// Decay function type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecayFunction {
    /// Exponential decay: e^(-λ × age_seconds)
    Exponential,
    /// Linear decay: 1 - λ × age_days (scaled to [0,1])
    Linear,
}

/// Configuration for temporal decay calculation.
#[derive(Debug, Clone, Copy)]
pub struct DecayConfig {
    /// Decay function to use.
    pub function: DecayFunction,
    /// Decay rate.
    ///
    /// **IMPORTANT:** Lambda ranges are function-specific:
    /// - Exponential: λ in per-second (1e-10 to 1e-3, default: 1e-6 ~50% decay at 8 days)
    /// - Linear: λ in per-day (1e-6 to 100.0)
    ///
    /// **WARNING:** If you change `function` from Exponential to Linear, you **must** also adjust `lambda`.
    /// Default lambda=1e-6 is appropriate for Exponential but produces negligible decay for Linear.
    /// For Linear decay, use lambda≥0.01 (1% decay per day minimum).
    pub lambda: f64,
    /// Grace period with no decay in days (default: 0.0).
    pub offset_days: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            function: DecayFunction::Exponential,
            lambda: 1e-6,
            offset_days: 0.0,
        }
    }
}

impl DecayFunction {
    /// Get all available decay functions.
    ///
    /// Returns an iterator over all decay function variants.
    #[allow(dead_code)]
    pub fn all() -> impl Iterator<Item = Self> {
        [DecayFunction::Exponential, DecayFunction::Linear].into_iter()
    }
}

impl DecayConfig {
    /// Validate decay configuration parameters.
    ///
    /// Returns error if parameters are mathematically invalid (e.g., negative lambda).
    pub fn new() -> Result<Self, String> {
        let config = Self::default();
        config.validate()?;
        Ok(config)
    }

    /// Validate decay configuration parameters.
    fn validate(&self) -> Result<(), String> {
        if self.lambda <= 0.0 {
            return Err(format!(
                "Invalid lambda: {} (must be positive)",
                self.lambda
            ));
        }

        // Function-specific validation
        match self.function {
            DecayFunction::Exponential => {
                if self.lambda > 1e-3 {
                    return Err(format!(
                        "Exponential decay lambda {} is too large (max: 1e-3)",
                        self.lambda
                    ));
                }
                if self.lambda < 1e-10 {
                    return Err(format!(
                        "Exponential decay lambda {} is too small (min: 1e-10)",
                        self.lambda
                    ));
                }
            }
            DecayFunction::Linear => {
                if self.lambda > 100.0 {
                    return Err(format!(
                        "Linear decay lambda {} is too large (max: 100.0)",
                        self.lambda
                    ));
                }
                if self.lambda < 1e-6 {
                    return Err(format!(
                        "Linear decay lambda {} is too small to be useful (min: 1e-6)",
                        self.lambda
                    ));
                }
            }
        }

        if self.offset_days < 0.0 {
            return Err(format!(
                "Invalid offset_days: {} (must be >= 0)",
                self.offset_days
            ));
        }
        Ok(())
    }

    /// Calculate decay factor for a memory created at `created_at`.
    ///
    /// Returns 1.0 for brand new, approaches 0.0 for very old.
    ///
    /// # Invariant
    ///
    /// This method assumes the configuration is valid. Validity is guaranteed by
    /// `DecayConfig::new()` which validates all parameters at construction time.
    /// Direct struct construction (only used in tests) bypassing validation may
    /// produce mathematically incorrect results.
    pub fn calculate_decay(&self, created_at: &DateTime<Utc>) -> f64 {
        let now = Utc::now();
        let age = now.signed_duration_since(*created_at);
        let age_seconds = age.num_seconds().max(0) as f64;

        // Guard against extreme values (should not occur with i64 age)
        if age_seconds.is_nan() || age_seconds.is_infinite() {
            return 0.0;
        }

        // Apply offset (grace period)
        let offset_seconds = self.offset_days * 86400.0;
        let effective_age = (age_seconds - offset_seconds).max(0.0);

        match self.function {
            DecayFunction::Exponential => {
                let exponent = -self.lambda * effective_age;
                // Guard against underflow/overflow in exp()
                if exponent < -700.0 {
                    return 0.0;
                }
                if exponent > 700.0 {
                    return 1.0;
                }
                exponent.exp()
            }
            DecayFunction::Linear => {
                let decay_rate = self.lambda * effective_age / 86400.0;
                (1.0 - decay_rate).clamp(0.0, 1.0)
            }
        }
    }
}

/// Apply recency weighting to search results.
///
/// Formula: final_score = (1 - α) × similarity + α × decay
///
/// # Arguments
///
/// * `similarity` - Original semantic similarity score
/// * `created_at` - Timestamp when the memory was created
/// * `recency_weight` - Weight parameter α (0.0 to 1.0)
/// * `config` - Decay configuration
///
/// # Returns
///
/// Combined score incorporating both semantic similarity and temporal decay.
pub fn apply_recency_weight(
    similarity: f64,
    created_at: &DateTime<Utc>,
    recency_weight: f64,
    config: &DecayConfig,
) -> f64 {
    if recency_weight <= 0.0 {
        return similarity;
    }
    let decay = config.calculate_decay(created_at);
    (1.0 - recency_weight) * similarity + recency_weight * decay
}

/// Validate recency weight is in valid range [0.0, 1.0].
pub fn validate_recency_weight(recency_weight: f64) -> Result<(), String> {
    if !(0.0..=1.0).contains(&recency_weight) {
        return Err(format!(
            "Invalid recency weight: {} (must be between 0.0 and 1.0)",
            recency_weight
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay_brand_new() {
        let config = DecayConfig::default();
        let now = Utc::now();
        let decay = config.calculate_decay(&now);
        assert!(
            (decay - 1.0).abs() < 1e-10,
            "Brand new should have decay ≈ 1.0"
        );
    }

    #[test]
    fn test_exponential_decay_8_days() {
        let config = DecayConfig::default();
        let created_at = Utc::now() - Duration::days(8);
        let decay = config.calculate_decay(&created_at);
        // With lambda=1e-6, 8 days = 8 * 86400 seconds = 691200
        // e^(-1e-6 * 691200) = e^(-0.6912) ≈ 0.50
        assert!(
            (decay - 0.5).abs() < 0.1,
            "8 days should have ~50% decay, got {}",
            decay
        );
    }

    #[test]
    fn test_exponential_decay_very_old() {
        let config = DecayConfig::default();
        let created_at = Utc::now() - Duration::days(365);
        let decay = config.calculate_decay(&created_at);
        // 1 year = 365 * 86400 seconds, should be very close to 0
        assert!(
            decay < 0.1,
            "1 year old should approach 0 decay, got {}",
            decay
        );
    }

    #[test]
    fn test_decay_with_offset() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: 1e-6,
            offset_days: 7.0,
        };
        let created_at = Utc::now() - Duration::days(3);
        let decay = config.calculate_decay(&created_at);
        // Within offset period, should be 1.0
        assert!(
            (decay - 1.0).abs() < 1e-10,
            "Within offset should have no decay"
        );
    }

    #[test]
    fn test_decay_after_offset() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: 1e-6,
            offset_days: 7.0,
        };
        let created_at = Utc::now() - Duration::days(15);
        let decay = config.calculate_decay(&created_at);
        // 15 days - 7 days offset = 8 days effective age
        // Should have ~50% decay from effective age
        assert!(
            (decay - 0.5).abs() < 0.1,
            "After offset should decay from effective age, got {}",
            decay
        );
    }

    #[test]
    fn test_apply_recency_weight_zero() {
        let config = DecayConfig::default();
        let now = Utc::now();
        let result = apply_recency_weight(0.9, &now, 0.0, &config);
        assert!(
            (result - 0.9).abs() < 1e-10,
            "α=0 should return pure similarity"
        );
    }

    #[test]
    fn test_apply_recency_weight_one() {
        let config = DecayConfig::default();
        let now = Utc::now();
        let result = apply_recency_weight(0.9, &now, 1.0, &config);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "α=1 with brand new should return decay=1.0"
        );
    }

    #[test]
    fn test_apply_recency_weight_half() {
        let config = DecayConfig::default();
        let now = Utc::now();
        let similarity = 0.8;
        let result = apply_recency_weight(similarity, &now, 0.5, &config);
        // 0.5 * 0.8 + 0.5 * 1.0 = 0.9
        assert!(
            (result - 0.9).abs() < 1e-10,
            "α=0.5 should average similarity and decay"
        );
    }

    #[test]
    fn test_recency_weight_negative_clamped() {
        let config = DecayConfig::default();
        let now = Utc::now();
        let result = apply_recency_weight(0.9, &now, -0.5, &config);
        assert!(
            (result - 0.9).abs() < 1e-10,
            "Negative recency weight should behave like 0.0"
        );
    }

    #[test]
    fn test_validate_recency_weight_valid() {
        assert!(validate_recency_weight(0.0).is_ok());
        assert!(validate_recency_weight(0.5).is_ok());
        assert!(validate_recency_weight(1.0).is_ok());
    }

    #[test]
    fn test_validate_recency_weight_negative() {
        let result = validate_recency_weight(-0.1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be between 0.0 and 1.0"));
    }

    #[test]
    fn test_validate_recency_weight_exceeds_one() {
        let result = validate_recency_weight(1.1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be between 0.0 and 1.0"));
    }

    #[test]
    fn test_decay_config_default() {
        let config = DecayConfig::default();
        assert!(matches!(config.function, DecayFunction::Exponential));
        assert_eq!(config.lambda, 1e-6);
        assert_eq!(config.offset_days, 0.0);

        // Also verify Linear variant exists
        let linear_config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0 / 86400.0,
            offset_days: 0.0,
        };
        assert!(matches!(linear_config.function, DecayFunction::Linear));
    }

    #[test]
    fn test_decay_config_new_valid() {
        let result = DecayConfig::new();
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.lambda, 1e-6);
    }

    #[test]
    fn test_decay_config_validate_negative_lambda() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: -1e-6,
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));
    }

    #[test]
    fn test_decay_config_validate_zero_lambda() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: 0.0,
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));
    }

    #[test]
    fn test_decay_config_validate_large_lambda() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: 1e-2,
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[test]
    fn test_decay_config_validate_negative_offset() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: 1e-6,
            offset_days: -7.0,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be >= 0"));
    }

    #[test]
    fn test_decay_config_validate_valid_offset() {
        let config = DecayConfig {
            function: DecayFunction::Exponential,
            lambda: 1e-6,
            offset_days: 7.0,
        };
        let result = config.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_recency_weight_with_old_memory() {
        let config = DecayConfig::default();
        let old_date = Utc::now() - Duration::days(365);
        let similarity = 0.9;
        let result = apply_recency_weight(similarity, &old_date, 0.5, &config);
        // Old memory has decay close to 0, so result should be ~0.45
        assert!(
            result < 0.6,
            "Old memory should be penalized, got {}",
            result
        );
        assert!(result > 0.3, "But still has some similarity contribution");
    }

    #[test]
    fn test_linear_decay_brand_new() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0 / 86400.0, //decay 1 per day
            offset_days: 0.0,
        };
        let now = Utc::now();
        let decay = config.calculate_decay(&now);
        assert!(
            (decay - 1.0).abs() < 1e-10,
            "Brand new should have decay ≈ 1.0"
        );
    }

    #[test]
    fn test_linear_decay_half_day() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0, // decay 1 per day
            offset_days: 0.0,
        };
        let created_at = Utc::now() - Duration::seconds(43200); // 12 hours
        let decay = config.calculate_decay(&created_at);
        // 12 hours = 0.5 days, decay = 1 - 1 * 0.5 = 0.5
        assert!(
            (decay - 0.5).abs() < 1e-10,
            "12 hours should have 50% decay, got {}",
            decay
        );
    }

    #[test]
    fn test_linear_decay_full_day() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0, // decay 1 per day
            offset_days: 0.0,
        };
        let created_at = Utc::now() - Duration::days(1);
        let decay = config.calculate_decay(&created_at);
        // 1 day, decay = 1 - 1 * 1 = 0
        assert!(
            (decay - 0.0).abs() < 1e-10,
            "1 day should have 0% decay, got {}",
            decay
        );
    }

    #[test]
    fn test_linear_decay_clamped() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0, // decay 1 per day
            offset_days: 0.0,
        };
        let created_at = Utc::now() - Duration::days(5);
        let decay = config.calculate_decay(&created_at);
        // 5 days, decay would be 1 - 5 = -4, but clamped to 0
        assert!(
            (decay - 0.0).abs() < 1e-10,
            "5 days should be clamped to 0 decay, got {}",
            decay
        );
    }

    #[test]
    fn test_linear_decay_with_offset() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0,      // decay 1 per day
            offset_days: 7.0, // no decay for 7 days
        };
        let created_at = Utc::now() - Duration::days(3);
        let decay = config.calculate_decay(&created_at);
        // Within offset period, should be 1.0
        assert!(
            (decay - 1.0).abs() < 1e-10,
            "Within offset should have no decay"
        );
    }

    #[test]
    fn test_linear_decay_after_offset() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0, // decay 1 per day
            offset_days: 7.0,
        };
        let created_at = Utc::now() - Duration::days(10); // 10 days total
        let decay = config.calculate_decay(&created_at);
        // 10 days - 7 days offset = 3 days effective age
        // decay = 1 - 1 * 3 = -2, clamped to 0
        assert!(
            (decay - 0.0).abs() < 1e-10,
            "After offset with excessive age should clamp to 0"
        );
    }

    #[test]
    fn test_decay_function_all() {
        let functions: Vec<_> = DecayFunction::all().collect();
        assert_eq!(functions.len(), 2);
        assert!(functions.contains(&DecayFunction::Exponential));
        assert!(functions.contains(&DecayFunction::Linear));
    }

    #[test]
    fn test_linear_decay_validation_too_small_lambda() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1e-7, // Too small for Linear
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small to be useful"));
    }

    #[test]
    fn test_linear_decay_validation_too_large_lambda() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 200.0, // Too large for Linear
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[test]
    fn test_linear_decay_validation_valid_min() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1e-6, // Valid minimum
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_ok(), "Linear lambda 1e-6 should be valid");
    }

    #[test]
    fn test_linear_decay_validation_valid_max() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 100.0, // Valid maximum
            offset_days: 0.0,
        };
        let result = config.validate();
        assert!(result.is_ok(), "Linear lambda 100.0 should be valid");
    }

    #[test]
    fn test_linear_decay_actually_decays() {
        let config = DecayConfig {
            function: DecayFunction::Linear,
            lambda: 1.0, // 1 per day (reasonable value)
            offset_days: 0.0,
        };
        let now = Utc::now();
        let decay_now = config.calculate_decay(&now);
        let decay_half_day = config.calculate_decay(&(now - Duration::seconds(43200)));
        let decay_one_day = config.calculate_decay(&(now - Duration::days(1)));

        assert!(
            decay_now > decay_half_day,
            "Linear decay should decrease over time"
        );
        assert!(
            decay_half_day > decay_one_day,
            "Linear decay should decrease over time"
        );
        assert!(
            (decay_now - 1.0).abs() < 1e-10 && (decay_half_day - 0.5).abs() < 1e-1,
            "Linear decay with lambda=1.0 should produce meaningful values"
        );
    }
}
