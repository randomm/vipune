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
    /// Cap (in days) on the recency refresh from retrieval telemetry (default: 30.0).
    ///
    /// An unbounded refresh would drive effective_age to 0 for every retrieved row
    /// and collapse the score to raw similarity; the cap keeps old rows decaying.
    pub refresh_cap_days: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            function: DecayFunction::Exponential,
            lambda: 1e-6,
            offset_days: 0.0,
            refresh_cap_days: 30.0,
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

        if self.refresh_cap_days < 0.0 {
            return Err(format!(
                "Invalid refresh_cap_days: {} (must be >= 0)",
                self.refresh_cap_days
            ));
        }

        Ok(())
    }

    /// Calculate decay factor for a memory created at `created_at`.
    ///
    /// Legacy path: no recency refresh, no importance scaling (equivalent
    /// to `calculate_decay_with_telemetry` with zero telemetry and
    /// `ImportanceLevel::Low`). Returns 1.0 for brand new, approaches 0.0 for very old.
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

        self.apply_function(effective_age)
    }

    /// Calculate decay factor with recency refresh and importance scaling.
    ///
    /// `effective_age = max(0, age_seconds - offset_seconds - recency_refresh)`
    /// where `recency_refresh` is capped by K days (default 30) and never
    /// exceeds the raw age, per the settled bound from issue #194.
    ///
    /// Then the decay is `f(-λ × importance_scale × effective_age)` where
    /// `importance_scale` is 0.0 (critical) → 1.0 (low), so low decays fastest
    /// and critical does not decay.
    pub fn calculate_decay_with_telemetry(
        &self,
        created_at: &DateTime<Utc>,
        importance: ImportanceLevel,
        telemetry: &RetrievalTelemetry,
    ) -> f64 {
        let now = Utc::now();
        let age = now.signed_duration_since(*created_at);
        let age_seconds = age.num_seconds().max(0) as f64;

        // Guard against extreme values (should not occur with i64 age)
        if age_seconds.is_nan() || age_seconds.is_infinite() {
            return 0.0;
        }

        // Recency refresh: zero when never-retrieved; capped at K days and at age_seconds
        let refresh = recency_refresh(
            telemetry.retrieval_count,
            telemetry.last_retrieved_at,
            created_at,
            self.refresh_cap_days,
        );

        // Apply offset (grace period) and recency refresh; floor at 0
        let offset_seconds = self.offset_days * 86400.0;
        let effective_age = (age_seconds - offset_seconds - refresh).max(0.0);

        // Importance scales the decay rate: critical=0.0 (no decay), low=1.0 (full decay)
        let scale = importance.scale();
        self.apply_function_scaled(effective_age, scale)
    }

    /// Apply the decay function to an already-computed effective age (no offset/refresh).
    fn apply_function(&self, effective_age: f64) -> f64 {
        self.apply_function_scaled(effective_age, 1.0)
    }

    /// Apply the decay function with an importance scale factor.
    fn apply_function_scaled(&self, effective_age: f64, scale: f64) -> f64 {
        match self.function {
            DecayFunction::Exponential => {
                // When scale == 0.0, the exponent is 0 and decay is 1.0 (no decay).
                let exponent = -self.lambda * scale * effective_age;
                if exponent < -700.0 {
                    return 0.0;
                }
                if exponent > 700.0 {
                    return 1.0;
                }
                exponent.exp()
            }
            DecayFunction::Linear => {
                let decay_rate = self.lambda * scale * effective_age / 86400.0;
                (1.0 - decay_rate).clamp(0.0, 1.0)
            }
        }
    }
}

/// Importance level of a memory, used to scale temporal decay.
///
/// Higher importance → smaller decay scale (slower forgetting).
/// This is a local helper to the decay pipeline; the operator-facing
/// `MemoryImportance` enum (with parsing and storage) lives in
/// `src/memory/lifecycle.rs` (issue #194, sub-issue 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceLevel {
    /// Decays fastest.
    Low,
    /// Default importance; half-speed decay.
    #[default]
    Medium,
    /// Slow decay (1/4 rate).
    High,
    /// No decay — pinned fresh for recency purposes.
    Critical,
}

impl ImportanceLevel {
    /// Parse an importance string.
    ///
    /// Accepts the four canonical levels (lowercase, case-insensitive).
    /// Any other value (e.g. `"normal"`) is rejected.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "critical" => Ok(Self::Critical),
            _ => Err(format!(
                "Invalid importance '{}'. Must be one of: low, medium, high, critical",
                s
            )),
        }
    }

    /// Get the string representation of the importance level.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    /// Decay scale factor: 0.0 for critical, 0.25 high, 0.5 medium, 1.0 low.
    ///
    /// Multiplied against λ in the decay exponent: `exp(-λ × scale × effective_age)`,
    /// so critical never decays and low decays fastest.
    pub fn scale(self) -> f64 {
        match self {
            Self::Low => 1.0,
            Self::Medium => 0.5,
            Self::High => 0.25,
            Self::Critical => 0.0,
        }
    }
}

/// Retrieval telemetry used to compute recency refresh.
///
/// Encapsulates the two telemetry fields written by `touch_memories` after
/// every search/get, and read here so that frequently-retrieved memories
/// are not penalized purely by age.
#[derive(Debug, Clone, Copy, Default)]
pub struct RetrievalTelemetry {
    /// Total number of times the memory has been retrieved (0 = never).
    pub retrieval_count: i64,
    /// Timestamp of the most recent retrieval (None = never retrieved).
    pub last_retrieved_at: Option<DateTime<Utc>>,
}

impl RetrievalTelemetry {
    /// Build telemetry from raw stored values (RFC3339 string).
    ///
    /// A `None` or unparseable `last_retrieved_at` yields `last_retrieved_at: None`.
    /// Callers that want strict parsing should parse the timestamp separately
    /// (as the search path already does for `created_at`).
    #[must_use]
    pub fn from_stored(
        retrieval_count: i64,
        last_retrieved_at: Option<&str>,
    ) -> Self {
        let last_retrieved_at = last_retrieved_at
            .and_then(|ts| ts.parse::<DateTime<Utc>>().ok());
        Self {
            retrieval_count,
            last_retrieved_at,
        }
    }
}

/// Compute the recency refresh (in seconds) that "refreshes" a memory's
/// effective age, based on how recently it was retrieved.
///
/// Semantics (settled bound from issue #194):
/// - Returns **zero** when `retrieval_count == 0` or `last_retrieved_at` is `None`.
/// - Otherwise returns `min(elapsed_since_last_retrieval, k_days, age_seconds)`,
///   where `age_seconds` is the elapsed time from `created_at` to `now`.
///
/// The upper bound at `age_seconds` prevents a recently-retrieved-but-
/// old-created memory from having its effective age driven to 0 (which would
/// collapse its score to raw similarity).
///
/// # Invariants
///
/// - Non-negative
/// - Non-decreasing in both `retrieval_count` and `last_retrieved_at`
/// - Zero when `retrieval_count == 0` or `last_retrieved_at` is `None`
pub fn recency_refresh(
    retrieval_count: i64,
    last_retrieved_at: Option<DateTime<Utc>>,
    created_at: &DateTime<Utc>,
    k_days: f64,
) -> f64 {
    if retrieval_count <= 0 {
        return 0.0;
    }
    let Some(last) = last_retrieved_at else {
        return 0.0;
    };
    let now = Utc::now();
    let age_seconds = now.signed_duration_since(*created_at).num_seconds().max(0) as f64;
    let since_last = now.signed_duration_since(last).num_seconds().max(0) as f64;
    let cap_seconds = (k_days.max(0.0) * 86400.0);
    since_last.min(cap_seconds).min(age_seconds)
}

/// Apply recency weighting to search results.
///
/// Formula: `final_score = (1 - α) × similarity + α × decay`
///
/// The decay term incorporates importance scaling and retrieval-telemetry-
/// based recency refresh (see `calculate_decay_with_telemetry`). The outer
/// composition formula and the raw similarity term are unchanged from the
/// pre-epic behavior (the only exact identity invariant: α=0 or age=0).
///
/// # Arguments
///
/// * `similarity` - Original semantic similarity score
/// * `created_at` - Timestamp when the memory was created
/// * `recency_weight` - Weight parameter α (0.0 to 1.0)
/// * `config` - Decay configuration
/// * `importance` - Importance level (scales the decay rate)
/// * `telemetry` - Retrieval telemetry (drives recency refresh)
///
/// # Returns
///
/// Combined score incorporating both semantic similarity and temporal decay.
pub fn apply_recency_weight(
    similarity: f64,
    created_at: &DateTime<Utc>,
    recency_weight: f64,
    config: &DecayConfig,
    importance: ImportanceLevel,
    telemetry: &RetrievalTelemetry,
) -> f64 {
    if recency_weight <= 0.0 {
        return similarity;
    }
    let decay = config.calculate_decay_with_telemetry(created_at, importance, telemetry);
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
