//! `reindex --force` model-switch migration (issue #217) — the per-database
//! orchestration behind the `--force` flag.
//!
//! The lifecycle: pre-flight token check → write the migration marker once
//! (marker-first crash safety) → re-embed every row of every project →
//! record the new identity and clear the marker in ONE transaction (only on
//! a fully clean pass). This module lives in the binary (the library never
//! orchestrates a migration); the `model_identity` read/write primitives it
//! uses stay in `crate::sqlite::identity`.

use crate::embedding::EmbeddingEngine;
use crate::embedding::MAX_EMBEDDING_TOKENS;
use crate::sqlite::Database;
use crate::sqlite::Error;
use crate::sqlite::embedding::classify_embedding;
use crate::sqlite::identity::ModelIdentity;
use rusqlite::Transaction;

/// Update one row's embedding BLOB inside an open transaction (the force
/// re-embed path commits per-project, not per-row).
fn update_embedding_in(tx: &Transaction<'_>, id: &str, embedding: &[f32]) -> Result<(), Error> {
    let blob = crate::sqlite::vec_to_blob(embedding)?;
    tx.execute(
        "UPDATE memories SET embedding = ?1 WHERE id = ?2",
        rusqlite::params![&blob, id],
    )
    .map_err(|e| Error::Sqlite(e.to_string()))?;
    Ok(())
}

/// A row the pre-flight scan found to exceed the token limit once prefixed.
#[derive(Debug, PartialEq, Eq)]
pub struct OverLimitRow {
    /// The memory id of the offending row.
    pub id: String,
}

/// A row whose re-embedding failed during the force pass.
///
/// Replaces the previous `"{id}: {error}"` string, which callers parsed back
/// with `splitn` (an error message containing `": "` would mis-split).
#[derive(Debug, PartialEq, Eq)]
pub struct ReembedFailure {
    /// The memory id whose embed failed.
    pub id: String,
    /// The embed error for that row.
    pub error: String,
}

/// Token-count every row of every project with the target profile's passage
/// prefix and report the ids of rows that would exceed the 512-token limit
/// once embedded.
///
/// The check runs BEFORE the migration marker is written (issue #217,
/// decision 1): the marker must only ever exist in a state the re-embed pass
/// can complete, so an over-limit row reports its id and the caller refuses
/// to start (writes nothing).
///
/// Rows are listed per project via `list_all_rows_for_project`; a row with a
/// corrupted embedding is NOT skipped here — the pre-flight is about token
/// count, not embedding validity — and a token-count error on any row is
/// propagated to the caller.
///
/// # Errors
///
/// `Error::Sqlite` if a row's content cannot be listed or token-counted.
pub fn over_limit_row_ids(
    db: &Database,
    engine: &EmbeddingEngine,
    projects: &[String],
    passage_prefix: &str,
) -> Result<Vec<OverLimitRow>, Error> {
    let mut offending: Vec<OverLimitRow> = Vec::new();
    for project_id in projects {
        let rows = db.list_all_rows_for_project(project_id)?;
        for (id, content, _embedding) in rows {
            let prefixed = format!("{passage_prefix}{content}");
            let count = engine
                .token_count(&prefixed)
                .map_err(|e| Error::InvalidInput(format!("token count failed: {e}")))?;
            if count > MAX_EMBEDDING_TOKENS {
                offending.push(OverLimitRow { id });
            }
        }
    }
    Ok(offending)
}

/// Re-embed every row in a project, bypassing Mock/Real classification.
///
/// Only Unknown (corrupted) rows are skipped. Each project's re-embed updates
/// run in ONE transaction (one commit, one fsync) instead of per-row
/// autocommits. The `embed` closure receives the raw stored content
/// (unprefixed) and must return a 384-dim f32 vector.
///
/// # Returns
///
/// `(reindexed, skipped, failed)` where `failed` is a `Vec<ReembedFailure>`.
pub(crate) fn force_reembed_project<F>(
    db: &mut Database,
    project_id: &str,
    mut embed: F,
) -> Result<(usize, usize, Vec<ReembedFailure>), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    let rows = db.list_all_rows_for_project(project_id)?;

    let mut reindexed: usize = 0;
    let mut skipped: usize = 0;
    let mut failed: Vec<ReembedFailure> = Vec::new();

    let tx = db.begin_transaction()?;
    for (id, content, embedding) in rows {
        if classify_embedding(&embedding) == crate::sqlite::embedding::EmbeddingClass::Unknown {
            skipped += 1;
            continue;
        }
        match embed(&content) {
            Ok(new_vec) => {
                update_embedding_in(&tx, &id, &new_vec)?;
                reindexed += 1;
            }
            Err(e) => {
                failed.push(ReembedFailure {
                    id,
                    error: e.to_string(),
                });
            }
        }
    }
    commit_tx(tx)?;
    Ok((reindexed, skipped, failed))
}

/// Run the `reindex --force` model-switch migration for the ENTIRE database
/// (issue #217).
///
/// The caller runs the pre-flight (`over_limit_row_ids`) on every project
/// first so the marker is only written when the pass can complete. This
/// function is the SINGLE owner of the marker/identity lifecycle: it writes
/// the migration marker ONCE before any row is touched, re-embeds every row
/// of every project (bypassing Mock/Real classification), then — only if
/// every row succeeded — records the new identity and clears the marker in
/// ONE transaction. If any row failed, the marker stays and the new identity
/// is NOT recorded (decision 1); the recovery is a clean re-run of
/// `reindex --force`.
///
/// # Crash safety
///
/// The marker write is one atomic write; the final identity commit is one
/// atomic transaction. A crash mid-pass leaves the marker plus partially
/// re-embedded rows (no half-written marker, no half-committed identity) —
/// the recovery is a full re-run, which re-embeds every row from the start.
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
pub fn force_migrate_database<F>(
    db: &mut Database,
    target: &ModelIdentity,
    projects: &[String],
    mut embed: F,
) -> Result<(usize, usize, Vec<ReembedFailure>), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    // Marker-first (durable before any row is touched).
    write_marker(db, target)?;
    // Re-embed every row of every project, then — only on a fully clean pass
    // — record the new identity and clear the marker in one transaction.
    let mut totals: (usize, usize, Vec<ReembedFailure>) = (0, 0, Vec::new());
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

    record_identity_and_clear_marker(db, target)?;

    Ok(totals)
}

/// Write the migration marker for a target identity (marker-first
/// crash-safety).
///
/// The marker is written BEFORE the re-embed pass so an interruption (crash,
/// kill, locked database) leaves the marker behind and all embedding
/// operations refuse until `reindex --force` finishes.
///
/// The write touches ONLY the marker: the recorded identity is left exactly
/// as it is (old identity, or no row at all = the bge default). Recording the
/// new identity is `record_identity_and_clear_marker`'s sole job — staging
/// the target here would make an interrupted migration report the NEW model
/// while most vectors are still the old model's.
///
/// If no identity row exists yet, the row is inserted with a NULL
/// `model_id` (and NULL `model_revision`) plus the marker; reads treat a
/// NULL model id as "no recorded identity" (the bge default).
/// Record the new identity and clear the migration marker in ONE
/// transaction.
///
/// This is the only sanctioned exit from the "migrating" state, and the
/// only step that replaces the recorded identity (the marker write leaves it
/// untouched). Uses rusqlite's `Transaction` API: the write is rolled back
/// automatically if it fails or the transaction is dropped before commit, so
/// identity and marker can never be half-updated.
///
/// Takes `&mut Database` (not `&Database`) because rusqlite 0.38's
/// `Connection::transaction` takes `&mut self`; the caller (the `--force`
/// handler) owns its `Database` for the migration's duration.
pub fn record_identity_and_clear_marker(
    db: &mut Database,
    identity: &ModelIdentity,
) -> Result<(), Error> {
    let tx = db.begin_transaction()?;
    tx.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, ?1, ?2, NULL)
         ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id,
                                      model_revision = excluded.model_revision,
                                      migration_marker = NULL",
        (&identity.model_id, &identity.revision),
    )
    .map_err(|e| Error::Sqlite(e.to_string()))?;
    commit_tx(tx).map_err(|e| Error::Sqlite(e.to_string()))?;
    Ok(())
}

/// Write the migration marker for a target identity (marker-first
/// crash-safety).
///
/// The marker is written BEFORE the re-embed pass so an interruption (crash,
/// kill, locked database) leaves the marker behind and all embedding
/// operations refuse until `reindex --force` finishes.
///
/// The write touches ONLY the marker: the recorded identity is left exactly
/// as it is (old identity, or no row at all = the bge default). Recording the
/// new identity is `record_identity_and_clear_marker`'s sole job — staging
/// the target here would make an interrupted migration report the NEW model
/// while most vectors are still the old model's.
///
/// If no identity row exists yet, the row is inserted with a NULL
/// `model_id` (and NULL `model_revision`) plus the marker; reads treat a
/// NULL model id as "no recorded identity" (the bge default).
pub fn write_marker(db: &Database, target: &ModelIdentity) -> Result<(), Error> {
    let marker = migration_marker_for(target);
    db.conn()
        .execute(
            "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
             VALUES (1, NULL, NULL, ?1)
             ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
            [marker],
        )
        .map_err(|e| Error::Sqlite(e.to_string()))?;
    Ok(())
}

/// Format the migration marker for a target identity.
pub(crate) fn migration_marker_for(identity: &ModelIdentity) -> String {
    format!("migrating to {}", identity.display())
}

/// Commit a rusqlite transaction (used by the per-project re-embed pass and
/// the final identity commit).
fn commit_tx(tx: Transaction<'_>) -> rusqlite::Result<()> {
    tx.commit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_marker_names_target() {
        let target = ModelIdentity {
            model_id: "m".to_string(),
            revision: "r".to_string(),
        };
        assert_eq!(migration_marker_for(&target), "migrating to m@r");
    }
}
