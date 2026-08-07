//! Temporal decay scoring for search result recency weighting.

use chrono::{DateTime, Utc};

/// Decay function type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecayFunction {
    /// Exponential decay: e^(-λ × age_seconds)
    Exponential,
    /// Linear decay: 1 - λ × age_days (scaled to [0,1])
    /// Implemented and thoroughly tested. Currently unused in production paths,
    /// but available for future use or external library consumers.
    #[cfg_attr(not(test), allow(dead_code))]
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
    /// This method documents all available decay function types.
    #[cfg(test)]
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
#[path = "temporal_tests.rs"]
mod temporal_tests;
