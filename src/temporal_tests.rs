//! Temporal decay tests.
#[cfg(test)]
mod tests {
    use super::super::{DecayConfig, DecayFunction, apply_recency_weight, validate_recency_weight};
    use chrono::Duration;
    use chrono::Utc;

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
