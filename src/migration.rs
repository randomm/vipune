//! Library-level model-switch migration (issue #221).
//!
//! The same crash-safe lifecycle the `reindex --force` CLI path uses,
//! exposed for library consumers:
//!
//! 1. Resolve the configured model profile (unknown id → error, no writes).
//! 2. **Pre-flight**: token-count every stored row's content with the target
//!    profile's *passage* role. Any row over the 512-token limit →
//!    `Error::MigrationRefused { offending }` listing every offending id.
//!    Nothing is written; no rows are changed.
//! 3. **Marker first**: the migration marker naming the target identity is
//!    (re)written before any row is touched. A crash mid-pass leaves the
//!    marker behind, so all embedding operations refuse until a re-run
//!    completes.
//! 4. **Re-embed pass**: every row of every project is re-embedded (prefix
//!    applied exactly once, by the embed closure). Unknown/corrupted rows are
//!    skipped and counted in `skipped`.
//! 5. **Final commit**: only when the pass is fully clean, the new identity
//!    is recorded and the marker cleared in ONE transaction →
//!    `Ok(MigrationReport)`.
//! 6. Any per-row failure → `Err(Error::MigrationIncomplete { report })`
//!    where `report.failures` lists them; the marker stays and the old
//!    recorded identity is kept.
//! 7. Re-running after an interruption performs a FULL pass.
//!
//! The embedding source is injectable: `migrate_model_with_embedder` takes
//! an `embed` closure and a `token_count` closure, so tests run with fake
//! implementations and no model download. The production path in
//! `migrate_model` wraps `EmbeddingEngine::embed_passage` /
//! `EmbeddingEngine::token_count`.
//!
//! The `embed` closure receives the **raw stored content** (never prefixed —
//! prefixes live only at embed time, never in the DB) and must return a
//! 384-dim f32 vector.

use crate::embedding::EmbeddingEngine;
use crate::embedding::MAX_EMBEDDING_TOKENS;
use crate::embedding_profiles::{EmbeddingRole, profile_for};
use crate::errors::Error;
use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};
use crate::sqlite::identity::ModelIdentity;
use crate::sqlite::{self, Database, Error as SqliteError};
use rusqlite::Transaction;
use std::path::Path;

/// The result of a completed model migration pass.
///
/// Re-exported from `crate::sqlite::migration_types::MigrationReport`.
pub use crate::sqlite::migration_types::MigrationReport;

/// A single row whose re-embedding failed during a migration pass.
///
/// Re-exported from `crate::sqlite::migration_types::MigrationRowFailure`.
pub use crate::sqlite::migration_types::MigrationRowFailure;

/// Migrate the whole database (all projects) at `db_path` to the model
/// profile named by `config.embedding_model`.
///
/// Opens its own `Database` handle with the busy timeout set to zero (fast
/// fail on a locked database — callers must close any other handle before
/// calling this; see the library README section on in-process cutover).
///
/// The outcome contract is the same as `reindex --force` (see the module
/// docs): pre-flight refusal → `Error::MigrationRefused`; marker-first
/// write; per-row failures → `Error::MigrationIncomplete` with the marker
/// left in place and the old identity kept; a clean pass →
/// `Ok(MigrationReport)` with the marker cleared and the new identity
/// recorded in one transaction.
///
/// # Errors
///
/// - `Error::Config` — unknown / unresolvable `config.embedding_model`
///   (before any write).
/// - `Error::MigrationRefused { offending }` — a pre-flight row exceeded the
///   512-token limit once the target profile's passage prefix is applied.
/// - `Error::MigrationIncomplete { report }` — the pass ran but at least one
///   row failed to embed; the marker stays, the old identity is kept.
/// - `Error::SqliteModule(_)` — database-level failure (open, marker write,
///   row listing, final commit).
pub fn migrate_model(
    db_path: &Path,
    config: &crate::config::Config,
) -> Result<MigrationReport, Error> {
    // Resolve the profile first (before any DB open) so an unknown model id
    // fails fast with no side effects.
    let _profile = profile_for(&config.embedding_model)?;

    let mut db = Database::open(db_path)?;
    // Fast-fail on a locked database (the CLI wraps this with an MCP-server
    // hint; the library caller gets the raw locked error).
    db.set_busy_timeout(std::time::Duration::ZERO)?;

    let mut engine = EmbeddingEngine::new(&config.embedding_model)?;

    // Pre-flight: token-count every row with the target profile's passage
    // prefix. The engine's immutable borrow ends here, before the mutable
    // borrow for the re-embed pass begins.
    let projects = db.list_all_project_ids().map_err(Error::from)?;
    let offending =
        preflight_over_limit_with_engine(&db, &projects, &engine).map_err(Error::from)?;
    if !offending.is_empty() {
        return Err(Error::MigrationRefused { offending });
    }

    // The embed closure owns the engine (mutable borrow); token_count is no
    // longer needed here (pre-flight already ran).
    let mut embed_closure = |content: &str| {
        engine
            .embed_passage(content)
            .map_err(|e| SqliteError::Sqlite(e.to_string()))
    };

    // token_count closure — not called by the core (pre-flight already ran);
    // the core's internal pre-flight check uses a no-op counter that always
    // returns 0, since the actual pre-flight was already performed above.
    let mut noop_token_count = |_content: &str| Ok(0);

    let result =
        migrate_model_with_embedder(&mut db, config, &mut embed_closure, &mut noop_token_count);

    result.map_err(Error::from)
}

/// The engine-injectable core of `migrate_model`.
///
/// Resolves the target profile from `config.embedding_model`, runs the
/// pre-flight token scan, and — if it passes — executes the marker-first
/// lifecycle over every project in the database.
///
/// The `embed` closure receives the **raw stored content** (the prefix is
/// applied by the closure itself, exactly once — the production path
/// delegates to `EmbeddingEngine::embed_passage`). The `token_count`
/// closure counts the passage-prefixed token count for a row's content
/// (used by the pre-flight scan).
///
/// This function is crate-private; library tests call it with fake closures.
pub(crate) fn migrate_model_with_embedder<E, T>(
    db: &mut Database,
    config: &crate::config::Config,
    embed: &mut E,
    token_count: &mut T,
) -> Result<MigrationReport, SqliteError>
where
    E: FnMut(&str) -> Result<Vec<f32>, SqliteError>,
    T: FnMut(&str) -> Result<usize, SqliteError>,
{
    let profile = profile_for(&config.embedding_model)
        .map_err(|e| SqliteError::InvalidInput(e.to_string()))?;
    let target = ModelIdentity {
        model_id: profile.model_id.to_string(),
        revision: profile.revision.to_string(),
    };

    let projects = db.list_all_project_ids()?;

    // Pre-flight: nothing is written before this returns Ok.
    let offending = preflight_over_limit(db, &projects, token_count)?;
    if !offending.is_empty() {
        return Err(SqliteError::MigrationRefused { offending });
    }

    // Marker first (durable before any row is touched).
    write_migration_marker(db, &target)?;

    let mut reindexed: usize = 0;
    let mut skipped: usize = 0;
    let mut failures: Vec<MigrationRowFailure> = Vec::new();

    for project_id in &projects {
        let (r, s, f) = reembed_project(db, project_id, &mut *embed)?;
        reindexed += r;
        skipped += s;
        failures.extend(f);
    }

    if !failures.is_empty() {
        // Marker stays; old identity is kept (never recorded).
        return Err(SqliteError::MigrationIncomplete {
            report: MigrationReport {
                reindexed,
                skipped,
                failures,
            },
        });
    }

    record_identity_and_clear_marker(db, &target)?;

    Ok(MigrationReport {
        reindexed,
        skipped,
        failures: Vec::new(),
    })
}

/// Token-count every stored row in `projects` with the target profile's
/// passage prefix and return the ids of rows that exceed the limit.
///
/// Runs BEFORE the marker is written (pre-flight, no writes). A row at
/// exactly `MAX_EMBEDDING_TOKENS` is allowed (the check is strict `>`).
fn preflight_over_limit<T>(
    db: &Database,
    projects: &[String],
    token_count: &mut T,
) -> Result<Vec<String>, SqliteError>
where
    T: FnMut(&str) -> Result<usize, SqliteError>,
{
    let mut offending: Vec<String> = Vec::new();
    for project_id in projects {
        let rows = db.list_all_rows_for_project(project_id)?;
        for (id, content, _embedding) in rows {
            let count = token_count(&content)?;
            if count > MAX_EMBEDDING_TOKENS {
                offending.push(id);
            }
        }
    }
    Ok(offending)
}

/// Pre-flight using the engine's real tokenizer (production path in
/// `migrate_model`). Runs BEFORE any mutable borrow of the engine.
fn preflight_over_limit_with_engine(
    db: &Database,
    projects: &[String],
    engine: &EmbeddingEngine,
) -> Result<Vec<String>, SqliteError> {
    let mut offending: Vec<String> = Vec::new();
    for project_id in projects {
        let rows = db.list_all_rows_for_project(project_id)?;
        for (id, content, _embedding) in rows {
            let count = engine
                .token_count(EmbeddingRole::Passage, &content)
                .map_err(|e| SqliteError::Sqlite(e.to_string()))?;
            if count > MAX_EMBEDDING_TOKENS {
                offending.push(id);
            }
        }
    }
    Ok(offending)
}

/// Re-embed every row in a project (bypassing Mock/Real classification).
///
/// Unknown-classified (corrupted) rows are skipped and counted in the
/// returned `skipped` value. Returns `(reindexed, skipped, failures)`.
fn reembed_project<F>(
    db: &mut Database,
    project_id: &str,
    mut embed: F,
) -> Result<(usize, usize, Vec<MigrationRowFailure>), SqliteError>
where
    F: FnMut(&str) -> Result<Vec<f32>, SqliteError>,
{
    let rows = db.list_all_rows_for_project(project_id)?;

    let mut reindexed: usize = 0;
    let mut skipped: usize = 0;
    let mut failed: Vec<MigrationRowFailure> = Vec::new();

    let tx = db.begin_transaction()?;
    for (id, content, embedding) in rows {
        if classify_embedding(&embedding) == EmbeddingClass::Unknown {
            skipped += 1;
            continue;
        }
        match embed(&content) {
            Ok(new_vec) => {
                update_embedding_in(&tx, &id, &new_vec)?;
                reindexed += 1;
            }
            Err(e) => {
                failed.push(MigrationRowFailure {
                    id,
                    error: e.to_string(),
                });
            }
        }
    }
    commit_tx(tx)?;
    Ok((reindexed, skipped, failed))
}

/// Update one row's embedding BLOB inside an open transaction.
fn update_embedding_in(
    tx: &Transaction<'_>,
    id: &str,
    embedding: &[f32],
) -> Result<(), SqliteError> {
    let blob = sqlite::vec_to_blob(embedding)?;
    tx.execute(
        "UPDATE memories SET embedding = ?1 WHERE id = ?2",
        rusqlite::params![&blob, id],
    )
    .map_err(|e| SqliteError::Sqlite(e.to_string()))?;
    Ok(())
}

/// Write the migration marker for `target` (marker-first crash safety).
///
/// The write touches ONLY the marker column; the recorded identity is left
/// as-is. If no identity row exists yet, one is inserted with NULL
/// `model_id` + the marker.
fn write_migration_marker(db: &Database, target: &ModelIdentity) -> Result<(), SqliteError> {
    let marker = migration_marker_for(target);
    db.conn()
        .execute(
            "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
             VALUES (1, NULL, NULL, ?1)
             ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
            [marker],
        )
        .map_err(|e| SqliteError::Sqlite(e.to_string()))?;
    Ok(())
}

/// Record the new identity and clear the migration marker in ONE transaction.
///
/// The only sanctioned exit from the "migrating" state; the write is rolled
/// back automatically if it fails, so identity and marker can never be
/// half-updated.
fn record_identity_and_clear_marker(
    db: &mut Database,
    identity: &ModelIdentity,
) -> Result<(), SqliteError> {
    let tx = db.begin_transaction()?;
    tx.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)
         ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision,
                                      migration_marker = NULL",
        (&identity.model_id, &identity.revision),
    )
    .map_err(|e| SqliteError::Sqlite(e.to_string()))?;
    commit_tx(tx).map_err(|e| SqliteError::Sqlite(e.to_string()))?;
    Ok(())
}

/// Format the migration marker text for `identity`.
fn migration_marker_for(identity: &ModelIdentity) -> String {
    format!("migrating to {}", identity.display())
}

/// Commit a rusqlite transaction.
fn commit_tx(tx: Transaction<'_>) -> rusqlite::Result<()> {
    tx.commit()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
//
// All tests use a fake embedder (no model download). The fake is
// deterministic: it returns a 384-dim vector derived from the content hash,
// so re-running the pass on the same content is idempotent at the vector
// level. Fakes for token_count return the word count (a lower bound on the
// real token count, so the >512 threshold is conservative).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "migration_tests.rs"]
mod migration_tests;
