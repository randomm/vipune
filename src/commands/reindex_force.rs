//! `reindex --force` model-switch migration (issue #217) — thin wrapper over
//! the library's `crate::migration` module (issue #221).
//!
//! This module exists for backward compatibility with the existing tests in
//! `reindex_tests.rs` that reference the old `reindex_force` API. The actual
//! migration logic lives in `crate::migration` (the library-level module).
//!
//! All functions are `#[cfg(test)]` because the CLI now uses the library's
//! `migrate_model` directly; only the tests reference this wrapper.

pub struct ReembedFailure {
    /// The memory id whose embed failed.
    pub id: String,
    /// The embed error for that row.
    pub error: String,
}

pub struct OverLimitRow {
    /// The memory id of the offending row.
    pub id: String,
}

pub fn over_limit_row_ids(
    db: &crate::sqlite::Database,
    engine: &crate::embedding::EmbeddingEngine,
    projects: &[String],
) -> Result<Vec<OverLimitRow>, crate::sqlite::Error> {
    let offending = crate::migration::preflight_over_limit(db, projects, engine)?;
    Ok(offending
        .into_iter()
        .map(|id| OverLimitRow { id })
        .collect())
}

pub(crate) fn force_reembed_project<F>(
    db: &mut crate::sqlite::Database,
    project_id: &str,
    mut embed: F,
) -> Result<(usize, usize, Vec<ReembedFailure>), crate::sqlite::Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, crate::sqlite::Error>,
{
    let (reindexed, skipped, failures) =
        crate::migration::reembed_project(db, project_id, |content| {
            embed(content).map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
        })?;
    let failed: Vec<ReembedFailure> = failures
        .into_iter()
        .map(|f| ReembedFailure {
            id: f.id,
            error: f.error,
        })
        .collect();
    Ok((reindexed, skipped, failed))
}

pub fn force_migrate_database<F>(
    db: &mut crate::sqlite::Database,
    target: &crate::sqlite::identity::ModelIdentity,
    projects: &[String],
    mut embed: F,
) -> Result<(usize, usize, Vec<ReembedFailure>), crate::sqlite::Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, crate::sqlite::Error>,
{
    let config = crate::config::Config {
        embedding_model: target.model_id.clone(),
        ..Default::default()
    };
    let result = crate::migration::migrate_model_with_embedder(db, &config, &mut |content| {
        embed(content).map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
    });
    match result {
        Ok(report) => Ok((
            report.reindexed,
            report.skipped,
            report
                .failures
                .into_iter()
                .map(|f| ReembedFailure {
                    id: f.id,
                    error: f.error,
                })
                .collect(),
        )),
        Err(crate::sqlite::Error::MigrationIncomplete { report }) => Ok((
            report.reindexed,
            report.skipped,
            report
                .failures
                .into_iter()
                .map(|f| ReembedFailure {
                    id: f.id,
                    error: f.error,
                })
                .collect(),
        )),
        Err(e) => Err(e),
    }
}

pub(crate) fn write_marker(
    db: &crate::sqlite::Database,
    target: &crate::sqlite::identity::ModelIdentity,
) -> Result<(), crate::sqlite::Error> {
    crate::migration::write_migration_marker(db, target)
}

pub(crate) fn record_identity_and_clear_marker(
    db: &mut crate::sqlite::Database,
    identity: &crate::sqlite::identity::ModelIdentity,
) -> Result<(), crate::sqlite::Error> {
    crate::migration::record_identity_and_clear_marker(db, identity)
}

pub(crate) fn migration_marker_for(identity: &crate::sqlite::identity::ModelIdentity) -> String {
    crate::migration::migration_marker_for(identity)
}
