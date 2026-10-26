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

    /// The one owner of the "this status cannot be inserted" verdict.
    ///
    /// A row being created is either `active` or `candidate`. `superseded` and
    /// `deprecated` are lifecycle states an existing row is *moved into* — by
    /// the supersede transaction's own UPDATE, by `prune`, or by
    /// `vipune update --status` — never states a fresh INSERT may carry, and the
    /// column has no CHECK constraint, so a store method is the only thing that
    /// can enforce it. The rule has exactly one call site per store write path,
    /// in the body that owns that path's vocabulary, so no entry point above it
    /// can admit a status the INSERT must not name:
    ///
    /// - [`MemoryStore::supersede_with`](crate::memory::MemoryStore::supersede_with)
    ///   — the only supersede body: a superseding row is a *new* row, and the
    ///   old row's `superseded` UPDATE is issued by that same body. Every
    ///   supersede caller, including the `supersede` defaulting wrapper, goes
    ///   through it.
    /// - the store's `add_impl` body — the one both
    ///   [`MemoryStore::add_with_conflict`](crate::memory::MemoryStore::add_with_conflict)
    ///   (and through it `ingest`, `ingest_with_type_status` and `batch_ingest`)
    ///   and
    ///   [`MemoryStore::ingest_with_importance`](crate::memory::MemoryStore::ingest_with_importance)
    ///   — the params-based add the CLI drives — forward to, so one check there
    ///   covers every add entry point whatever default it binds.
    ///
    /// [`MemoryStore::update`](crate::memory::MemoryStore::update) is the
    /// deliberate exception: it changes an *existing* row, `deprecated` is a
    /// legal target there (it is what `prune` writes), and it rejects only
    /// `superseded`, which no caller may set outside the supersede transaction.
    ///
    /// The CLI (`handle_add`) and MCP (`store_memory`) layers call this method
    /// too, before reaching the store, only so the rejection reaches their client
    /// in their own error shape; hand-written variants of this message are how the
    /// two wordings drifted apart (issue #258). They are prechecks, not the
    /// enforcement: a library caller that skips them is stopped by the store copy.
    ///
    /// The predicate form of [`Self::ensure_valid_for_insert`], kept for the
    /// released public API (shipped in v0.15.4 — issue #258 must not remove it).
    ///
    /// The store's write paths call the `Result`-returning method above, so the
    /// wording of the rejection lives in exactly one place; this copy exists so
    /// a library caller that only wants the verdict keeps the method it
    /// compiled against. Delegation is the only body: the two cannot disagree.
    // Released in v0.15.4 and part of the additive-only public API; the store's
    // write paths all call `ensure_valid_for_insert`, so nothing in this crate
    // reaches it but the test below.
    #[cfg_attr(not(test), allow(dead_code))] // Public API for library consumers
    #[must_use = "the verdict is the reason to call this"]
    pub fn is_valid_for_insert(self) -> bool {
        self.ensure_valid_for_insert().is_ok()
    }

    /// The two lifecycle states an existing row is moved *into* and a fresh row
    /// may never be born as — the vocabulary both the insert rule
    /// ([`Self::ensure_valid_for_insert`]) and the re-supersede guard on the one
    /// supersede INSERT
    /// ([`crate::sqlite::Database::supersede_with_importance`]) are written
    /// against, so a fifth variant moves both rules together instead of silently
    /// widening only one of them.
    const RETIRED: [MemoryStatus; 2] = [MemoryStatus::Superseded, MemoryStatus::Deprecated];

    /// The verdict behind [`Self::ensure_valid_for_insert`]: `true` for the two
    /// [`Self::RETIRED`] statuses. The storage guard compares the fetched column
    /// against `RETIRED`'s own [`Self::as_str`] spellings rather than calling
    /// this, because the storage layer keeps comparing strings (its convention
    /// everywhere else) and must not pretend a stored row was ever validated.
    #[must_use = "the verdict is the reason to call this"]
    pub(crate) fn is_retired(self) -> bool {
        Self::RETIRED.contains(&self)
    }

    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `self` is not insertable.
    pub fn ensure_valid_for_insert(self) -> Result<(), Error> {
        if self.is_retired() {
            return Err(Error::InvalidInput(format!(
                "Status '{}' is not valid for new memory insertion. Must be 'active' or 'candidate'.",
                self.as_str()
            )));
        }
        Ok(())
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

    /// The two importance vocabularies in this crate are maintained apart: this
    /// one gates what may be written (the `--importance` flag and the import
    /// path), while `temporal::ImportanceLevel::parse` reads the stored column
    /// back for decay and `prune` matches the raw `'high'`/`'critical'` SQL
    /// literals. Nothing structurally enforces that they accept the same words,
    /// so a level added to one enum and not the other would write rows the read
    /// side rejects (or vice versa) — pinned here instead.
    #[test]
    fn test_importance_vocabularies_agree_with_the_read_side() {
        use crate::temporal::ImportanceLevel;

        for word in [
            "low", "medium", "high", "critical", "LOW", "Medium", "CRITICAL", "normal", "urgent",
            "", " high",
        ] {
            assert_eq!(
                MemoryImportance::from_str(word).is_ok(),
                ImportanceLevel::parse(word).is_ok(),
                "write gate and read side disagree on {word:?}"
            );
        }
        // Whatever this enum can name, the decay pipeline must be able to read.
        for level in [
            MemoryImportance::Low,
            MemoryImportance::Medium,
            MemoryImportance::High,
            MemoryImportance::Critical,
        ] {
            assert!(
                ImportanceLevel::parse(level.as_str()).is_ok(),
                "the read side must understand {}",
                level.as_str()
            );
        }
    }

    #[test]
    fn test_ensure_valid_for_insert() {
        // `is_valid_for_insert` (shipped in v0.15.4, restored by #258) delegates
        // to this method, so one table pins both spellings of the rule rather
        // than letting a predicate that always returns `false` pass unnoticed.
        assert!(MemoryStatus::Active.ensure_valid_for_insert().is_ok());
        assert!(MemoryStatus::Candidate.ensure_valid_for_insert().is_ok());
        for accepted in [MemoryStatus::Active, MemoryStatus::Candidate] {
            assert!(accepted.is_valid_for_insert(), "{accepted:?} is insertable");
        }
        // The two states below belong to an existing row's UPDATE, never to an
        // INSERT: if either ever became insertable the supersede path would
        // accept a replacement row that is born already replaced.
        for rejected in [MemoryStatus::Superseded, MemoryStatus::Deprecated] {
            let err = rejected
                .ensure_valid_for_insert()
                .expect_err("not insertable");
            assert!(
                err.to_string()
                    .contains("is not valid for new memory insertion"),
                "unexpected error for {rejected:?}: {err}"
            );
            assert!(
                !rejected.is_valid_for_insert(),
                "the predicate must refuse {rejected:?} for the same reason"
            );
        }
    }
}
