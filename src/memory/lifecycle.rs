//! Memory type and lifecycle status definitions.
//!
//! Provides validation and constants for memory types and lifecycle status values.

use crate::errors::Error;

/// Valid memory types.
///
/// Each type serves a specific role in memory retrieval and routing:
/// - `fact`: General factual information (default)
/// - `preference`: User or system preferences
/// - `procedure`: Step-by-step instructions or workflows
/// - `guard`: Safety rules or constraints that should be checked before actions
/// - `observation`: Temporary or observational data that may become stale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryType {
    #[default]
    Fact,
    Preference,
    Procedure,
    Guard,
    Observation,
}

impl MemoryType {
    /// Get the string representation of the memory type.
    #[allow(dead_code)] // Public API for library consumers
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Fact => "fact",
            MemoryType::Preference => "preference",
            MemoryType::Procedure => "procedure",
            MemoryType::Guard => "guard",
            MemoryType::Observation => "observation",
        }
    }

    /// Parse a string into a MemoryType.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the string is not a valid memory type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, Error> {
        match s.to_lowercase().as_str() {
            "fact" => Ok(MemoryType::Fact),
            "preference" => Ok(MemoryType::Preference),
            "procedure" => Ok(MemoryType::Procedure),
            "guard" => Ok(MemoryType::Guard),
            "observation" => Ok(MemoryType::Observation),
            _ => Err(Error::InvalidInput(format!(
                "Invalid memory type '{}'. Must be one of: fact, preference, procedure, guard, observation",
                s
            ))),
        }
    }
}

/// Memory lifecycle status.
///
/// Tracks the state of a memory through its lifecycle:
/// - `active`: Currently in use and returned by default in searches (default)
/// - `candidate`: Proposed memory awaiting activation (not returned by default)
/// - `superseded`: Replaced by a newer memory (not returned by default)
/// - `deprecated`: Marked as obsolete or incorrect (not returned by default)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryStatus {
    #[default]
    Active,
    Candidate,
    Superseded,
    Deprecated,
}

impl MemoryStatus {
    /// Get the string representation of the memory status.
    #[allow(dead_code)] // Public API for library consumers
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryStatus::Active => "active",
            MemoryStatus::Candidate => "candidate",
            MemoryStatus::Superseded => "superseded",
            MemoryStatus::Deprecated => "deprecated",
        }
    }

    /// Parse a string into a MemoryStatus.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the string is not a valid memory status.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, Error> {
        match s.to_lowercase().as_str() {
            "active" => Ok(MemoryStatus::Active),
            "candidate" => Ok(MemoryStatus::Candidate),
            "superseded" => Ok(MemoryStatus::Superseded),
            "deprecated" => Ok(MemoryStatus::Deprecated),
            _ => Err(Error::InvalidInput(format!(
                "Invalid memory status '{}'. Must be one of: active, candidate, superseded, deprecated",
                s
            ))),
        }
    }

    /// Check if this status is allowed for new memory insertion.
    pub fn is_valid_for_insert(self) -> bool {
        matches!(self, MemoryStatus::Active | MemoryStatus::Candidate)
    }
}

/// Operator-assigned importance of a memory.
///
/// Higher importance rows are protected from lifecycle pruning (the prune
/// command never demotes `high`/`critical` rows) and decay more slowly in the
/// recency weighting pipeline (see `src/temporal.rs`). `medium` is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryImportance {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

impl MemoryImportance {
    /// Get the string representation of the importance.
    #[allow(dead_code)] // Public API for library consumers
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryImportance::Low => "low",
            MemoryImportance::Medium => "medium",
            MemoryImportance::High => "high",
            MemoryImportance::Critical => "critical",
        }
    }

    /// Parse a string into a MemoryImportance.
    ///
    /// Case-insensitive. Any other value (including the historically-common
    /// `"normal"`) is rejected.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the string is not a valid importance.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, Error> {
        match s.to_lowercase().as_str() {
            "low" => Ok(MemoryImportance::Low),
            "medium" => Ok(MemoryImportance::Medium),
            "high" => Ok(MemoryImportance::High),
            "critical" => Ok(MemoryImportance::Critical),
            _ => Err(Error::InvalidInput(format!(
                "Invalid importance '{}'. Must be one of: low, medium, high, critical",
                s
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_roundtrip() {
        for (s, expected) in [
            ("fact", MemoryType::Fact),
            ("preference", MemoryType::Preference),
            ("procedure", MemoryType::Procedure),
            ("guard", MemoryType::Guard),
            ("observation", MemoryType::Observation),
        ] {
            let parsed = MemoryType::from_str(s).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), s);
        }
    }

    #[test]
    fn test_memory_type_case_insensitive() {
        assert_eq!(MemoryType::from_str("FACT").unwrap(), MemoryType::Fact);
        assert_eq!(MemoryType::from_str("Guard").unwrap(), MemoryType::Guard);
    }

    #[test]
    fn test_memory_type_invalid() {
        assert!(MemoryType::from_str("invalid").is_err());
        assert!(MemoryType::from_str("").is_err());
    }

    #[test]
    fn test_memory_status_roundtrip() {
        for (s, expected) in [
            ("active", MemoryStatus::Active),
            ("candidate", MemoryStatus::Candidate),
            ("superseded", MemoryStatus::Superseded),
            ("deprecated", MemoryStatus::Deprecated),
        ] {
            let parsed = MemoryStatus::from_str(s).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), s);
        }
    }

    #[test]
    fn test_memory_status_invalid() {
        assert!(MemoryStatus::from_str("invalid").is_err());
    }

    #[test]
    fn test_memory_importance_roundtrip() {
        for (s, expected) in [
            ("low", MemoryImportance::Low),
            ("medium", MemoryImportance::Medium),
            ("high", MemoryImportance::High),
            ("critical", MemoryImportance::Critical),
        ] {
            let parsed = MemoryImportance::from_str(s).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), s);
        }
        assert_eq!(MemoryImportance::default(), MemoryImportance::Medium);
    }

    #[test]
    fn test_memory_importance_case_insensitive() {
        assert_eq!(
            MemoryImportance::from_str("LOW").unwrap(),
            MemoryImportance::Low
        );
        assert_eq!(
            MemoryImportance::from_str("Critical").unwrap(),
            MemoryImportance::Critical
        );
    }

    #[test]
    fn test_memory_importance_invalid() {
        assert!(MemoryImportance::from_str("invalid").is_err());
        assert!(MemoryImportance::from_str("").is_err());
        // "normal" is not a canonical level and must be rejected.
        assert!(MemoryImportance::from_str("normal").is_err());
    }

    #[test]
    fn test_is_valid_for_insert() {
        assert!(MemoryStatus::Active.is_valid_for_insert());
        assert!(MemoryStatus::Candidate.is_valid_for_insert());
        assert!(!MemoryStatus::Superseded.is_valid_for_insert());
        assert!(!MemoryStatus::Deprecated.is_valid_for_insert());
    }
}
