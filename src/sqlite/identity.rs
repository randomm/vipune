//! Model identity tracking for the `model_identity` table (migration v6,
//! issue #217).
//!
//! A database records which embedding model (id + pinned revision) produced
//! its vectors. A database with no identity row is treated as the built-in
//! default — bge at its pinned revision
//! (`crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION}`).
//!
//! An optional migration marker (`"migrating to <id>@<revision>"`) signals
//! that a `reindex --force` model switch was started but not finished. The
//! marker is written once, BEFORE any row of the re-embed pass, and cleared in
//! the same transaction that records the new identity, so a crash mid-run
//! leaves the marker behind and every embedding operation (add / update /
//! search / hook insert) refuses until `reindex --force` completes.
//!
//! The identity lifecycle is per-DATABASE, not per-project: the marker is
//! written once and the identity is recorded once after every row of every
//! project in the database has been re-embedded. That is what keeps a
//! multi-project database from ending up as a silently mixed store after a
//! partial run.

use crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION};
use crate::embedding_profiles::profile_for;
use crate::sqlite::Database;
use crate::sqlite::Error;
use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};
use rusqlite::{Connection, OptionalExtension};

/// The (model id, pinned revision) pair a database's vectors were produced
/// with, or that a migration is switching to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_id: String,
    pub revision: String,
}

impl ModelIdentity {
    /// The default identity: bge at its pinned revision. A database with no
    /// recorded identity row is treated as this identity.
    pub fn default_identity() -> Self {
        Self {
            model_id: EMBED_MODEL_ID.to_string(),
            revision: EMBED_MODEL_REVISION.to_string(),
        }
    }

    /// Render as `<id>@<revision>` (used in markers and error messages).
    pub fn display(&self) -> String {
        format!("{}@{}", self.model_id, self.revision)
    }
}

/// Read the recorded identity. `None` row ⇒ the bge default.
#[allow(dead_code)]
pub fn read_identity(conn: &Connection) -> Result<Option<ModelIdentity>, Error> {
    match conn.query_row(
        "SELECT model_id, model_revision FROM model_identity WHERE id = 1",
        [],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
    ) {
        Ok((model_id, revision)) => {
            let revision = revision.unwrap_or_else(|| EMBED_MODEL_REVISION.to_string());
            Ok(Some(ModelIdentity { model_id, revision }))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(Error::Sqlite(e.to_string())),
    }
}

/// Read the migration marker, if one is in flight.
#[allow(dead_code)]
pub fn read_marker(conn: &Connection) -> Result<Option<String>, Error> {
    match conn.query_row(
        "SELECT migration_marker FROM model_identity WHERE id = 1 AND migration_marker IS NOT NULL",
        [],
        |row| row.get::<_, String>(0),
    ) {
        Ok(marker) => Ok(Some(marker)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(Error::Sqlite(e.to_string())),
    }
}

/// Format the migration marker for a target identity.
pub(crate) fn migration_marker_for(identity: &ModelIdentity) -> String {
    format!("migrating to {}", identity.display())
}

/// Write the migration marker for a target identity (marker-first
/// crash-safety).
///
/// The marker is written in its own transaction BEFORE the re-embed pass so
/// that an interruption (crash, kill, locked database) leaves the marker
/// behind and all embedding operations refuse until `reindex --force`
/// finishes. The upsert also stages the target identity in the same row so a
/// marker-present database always names the interrupted target.
#[allow(dead_code)]
pub fn write_marker(conn: &Connection, target: &ModelIdentity) -> Result<(), Error> {
    let marker = migration_marker_for(target);
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision,
                                      migration_marker = excluded.migration_marker",
        (&target.model_id, &target.revision, marker),
    )
    .map_err(|e| Error::Sqlite(e.to_string()))?;
    Ok(())
}

/// Record the new identity and clear the migration marker in ONE transaction.
///
/// This is the only sanctioned exit from the "migrating" state. Identity and
/// marker updates are atomic: a crash before the commit leaves the previous
/// state intact (either the old identity with no marker, or the marker still
/// set with the old identity), never a half-migrated state.
#[allow(dead_code)]
pub fn record_identity_and_clear_marker(
    conn: &Connection,
    identity: &ModelIdentity,
) -> Result<(), Error> {
    conn.execute_batch("BEGIN;")
        .map_err(|e| Error::Sqlite(e.to_string()))?;
    let write = conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)
         ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision,
                                      migration_marker = NULL",
        (&identity.model_id, &identity.revision),
    );
    match write {
        Ok(_) => match conn.execute_batch("COMMIT;") {
            Ok(_) => Ok(()),
            Err(e) => Err(Error::Sqlite(e.to_string())),
        },
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK;");
            Err(Error::Sqlite(e.to_string()))
        }
    }
}

/// True while a migration marker is present (an interrupted `reindex --force`).
#[allow(dead_code)]
pub fn is_migrating(conn: &Connection) -> Result<bool, Error> {
    Ok(read_marker(conn)?.is_some())
}

/// The identity the store currently has: the recorded identity, or the bge
/// default when no row is recorded.
#[allow(dead_code)]
pub fn current_identity(conn: &Connection) -> Result<ModelIdentity, Error> {
    Ok(read_identity(conn)?.unwrap_or_else(ModelIdentity::default_identity))
}

/// The identity the currently configured model resolves to.
///
/// A built-in profile id resolves to its pinned revision (via the profile
/// registry in `crate::embedding_profiles`); any configured id that is not a
/// built-in profile resolves to the id itself with no revision recorded,
/// which makes any store with a recorded row mismatch (refused) rather than
/// silently "matching". Config validation rejects unknown ids anyway.
#[allow(dead_code)]
pub fn configured_identity(configured_model_id: &str) -> ModelIdentity {
    match profile_for(configured_model_id) {
        Ok(profile) => ModelIdentity {
            model_id: profile.model_id.to_string(),
            revision: profile.revision.to_string(),
        },
        Err(_) => ModelIdentity {
            model_id: configured_model_id.to_string(),
            revision: String::new(),
        },
    }
}

/// Read the recorded identity and any migration marker in one query.
///
/// Returns `(identity, migration_marker)` where `identity` is `None` when no
/// row exists (callers compare against [`ModelIdentity::default_identity`])
/// and `migration_marker` is the "migrating to ..." text if one is present.
#[allow(dead_code)]
pub fn read_identity_and_marker(
    conn: &Connection,
) -> Result<(Option<ModelIdentity>, Option<String>), Error> {
    let row: Option<(String, Option<String>, Option<String>)> = conn
        .query_row(
            "SELECT model_id, model_revision, migration_marker FROM model_identity WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()
        .map_err(|e| Error::Sqlite(e.to_string()))?;

    let Some((model_id, revision, marker)) = row else {
        return Ok((None, None));
    };
    // A row written by `reindex --force`'s marker-first step stages the
    // target identity alongside the marker; treat a partial row as a
    // recorded identity too.
    let identity = ModelIdentity {
        model_id,
        revision: revision.unwrap_or_default(),
    };
    Ok((Some(identity), marker))
}

/// Mismatch / in-flight-migration refusal for the embedding chokepoints
/// (issue #217: add, update, search, hook inserts, and anything else that
/// embeds).
///
/// Refuses when a migration marker is present (refusal names the interrupted
/// target), or when the effective identity (recorded row, or bge default
/// when unrecorded) differs from the configured model's identity (id OR
/// revision). The error always points at `vipune reindex --force`.
///
/// # Errors
///
/// `Error::Config` with the refusal message; `Error::SqliteModule` if the
/// identity table cannot be read.
#[allow(dead_code)]
pub fn assert_identity_ok(
    conn: &Connection,
    configured_model_id: &str,
) -> Result<(), crate::errors::Error> {
    let (recorded, marker) = read_identity_and_marker(conn)
        .map_err(|e| crate::errors::Error::SqliteModule(e.to_string()))?;
    if let Some(marker) = marker {
        return Err(crate::errors::Error::Config(format!(
            "model migration in progress: database is migrating to {marker} — add/update/search are refused until the migration completes. Run `vipune reindex --force` to complete it."
        )));
    }
    let effective = recorded.unwrap_or_else(ModelIdentity::default_identity);
    let configured = configured_identity(configured_model_id);
    if effective != configured {
        return Err(crate::errors::Error::Config(format!(
            "model identity mismatch: database was last embedded with {} but the configured model is {}. Re-embed the store with `vipune reindex --force`.",
            effective.display(),
            configured.display()
        )));
    }
    Ok(())
}

/// Re-embed every row in a project, bypassing Mock/Real classification.
///
/// Used by [`force_migrate_database`] and by tests. Only Unknown (corrupted)
/// rows are skipped. The `embed` closure receives the raw stored content
/// (unprefixed) and must return a 384-dim f32 vector.
///
/// # Returns
///
/// `(reindexed, skipped, failed)` where `failed` is a `Vec<String>` of
/// `"<id>: <error>"` entries.
pub(crate) fn force_reembed_project<F>(
    db: &Database,
    project_id: &str,
    mut embed: F,
) -> Result<(usize, usize, Vec<String>), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    let rows = db.list_all_rows_for_project(project_id)?;
    let mut reindexed: usize = 0;
    let mut skipped: usize = 0;
    let mut failed: Vec<String> = vec![];

    for (id, content, embedding) in rows {
        if classify_embedding(&embedding) == EmbeddingClass::Unknown {
            skipped += 1;
            continue;
        }
        match embed(&content) {
            Ok(new_vec) => {
                db.update_embedding(&id, &new_vec)?;
                reindexed += 1;
            }
            Err(e) => {
                failed.push(format!("{}: {}", id, e));
            }
        }
    }

    Ok((reindexed, skipped, failed))
}

/// Run the `reindex --force` model-switch migration for the ENTIRE database
/// (issue #217).
///
/// The caller performs the pre-flight token check on every project first
/// (see `crate::sqlite::force_preflight`) so the marker is only written when
/// the pass can complete. This function is the SINGLE owner of the
/// marker/identity lifecycle: it writes the migration marker ONCE before any
/// row is touched, re-embeds every row of every project (bypassing Mock/Real
/// classification), then — only if every row succeeded — records the new
/// identity and clears the marker in ONE transaction. If any row failed, the
/// marker stays and the new identity is NOT recorded (decision 1); the
/// recovery is a clean re-run of `reindex --force`.
///
/// The `embed` closure receives the raw stored content (unprefixed —
/// prefixes live only at embed time, never in the DB; the caller's closure
/// applies the target profile's passage prefix).
///
/// # Returns
///
/// `(reindexed, skipped, failed)` counts summed across all projects.
///
/// # Errors
///
/// Error if the marker write fails, the re-embed pass has any failures, or
/// the final identity commit fails.
#[allow(dead_code)]
pub fn force_migrate_database<F>(
    db: &Database,
    target: &ModelIdentity,
    projects: &[String],
    mut embed: F,
) -> Result<(usize, usize, Vec<String>), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    // Marker-first (durable before any row is touched).
    write_marker(db.conn(), target)?;

    // Re-embed every row of every project, then — only on a fully clean pass
    // — record the new identity and clear the marker in one transaction.
    let mut totals: (usize, usize, Vec<String>) = (0, 0, vec![]);
    for project_id in projects {
        let (reindexed, skipped, failed) = force_reembed_project(db, project_id, &mut embed)?;
        totals.0 += reindexed;
        totals.1 += skipped;
        totals.2.extend(failed);
    }

    if !totals.2.is_empty() {
        // Marker stays — the re-embed pass did not complete, and the new
        // identity is deliberately NOT recorded (issue #217 decision 1).
        return Err(Error::InvalidInput(format!(
            "force reindex failed on {} row(s) across {} project(s): the migration marker is left in place and the new model identity was not recorded. Fix the errors and re-run `vipune reindex --force`.",
            totals.2.len(),
            projects.len()
        )));
    }

    record_identity_and_clear_marker(db.conn(), target)?;

    Ok(totals)
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod identity_tests;
