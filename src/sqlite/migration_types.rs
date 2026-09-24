//! Migration report types (issue #221).
//!
//! Shared between the lib and bin targets (the bin target's `reindex --force`
//! path will eventually consume these; for now they are constructed by the
//! lib's `crate::migration` module).

/// The result of a completed model migration pass.
///
/// `reindexed` is the number of rows whose embedding BLOB was replaced;
/// `skipped` is the number of Unknown-classified (corrupted) rows that were
/// left as-is; `failures` lists every row whose embed call failed (always
/// empty on a clean pass; populated inside `Error::MigrationIncomplete`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// Number of rows whose embedding was replaced by the pass.
    pub reindexed: usize,
    /// Number of corrupted (Unknown-classified) rows skipped by the pass.
    pub skipped: usize,
    /// Per-row embed failures (empty when the pass is clean).
    pub failures: Vec<MigrationRowFailure>,
}

/// A single row whose re-embedding failed during a migration pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationRowFailure {
    /// The memory id of the failed row.
    pub id: String,
    /// The embed error for that row.
    pub error: String,
}

#[cfg(test)]
mod migration_error_tests {
    use super::*;

    /// Construct the migration variants to keep them alive in both the lib
    /// and bin targets (the production construction sites live in
    /// `crate::migration`, which is only compiled in the lib target).
    #[test]
    fn migration_variants_are_constructible() {
        let _refused: super::super::Error = super::super::Error::MigrationRefused {
            offending: vec!["id".to_string()],
        };
        let _incomplete: super::super::Error = super::super::Error::MigrationIncomplete {
            report: MigrationReport {
                reindexed: 0,
                skipped: 0,
                failures: vec![MigrationRowFailure {
                    id: "id".to_string(),
                    error: "e".to_string(),
                }],
            },
        };
    }
}
