//! Model identity tracking for the `model_identity` table (migration v6).
//!
//! A database records which embedding model (id + revision) produced its
//! vectors. A database with no identity row is treated as the built-in
//! default — bge at its pinned revision (`crate::embedding::{EMBED_MODEL_ID,
//! EMBED_MODEL_REVISION}`).
//!
//! An optional migration marker (`"migrating to <id>@<revision>"`) signals
//! that a `reindex --force` model switch was started but not finished; while
//! the marker is present, operations that embed must refuse (issue #217).
//! The marker is written before the re-embed loop and cleared in the same
//! transaction that records the new identity, so a crash mid-run leaves the
//! marker behind and the database is unambiguous about its state.

use crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION};
use crate::sqlite::Database;
use crate::sqlite::Error;
use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};
use rusqlite::Connection;

/// The (model id, revision) pair a database's vectors were produced with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_id: String,
    pub revision: String,
}

impl ModelIdentity {
    /// The default identity: bge at its pinned revision.
    pub fn bge_default() -> Self {
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
///
/// # Errors
///
/// Returns error if the read fails.
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

/// Write the migration marker for a target identity (marker-first crash-safety).
///
/// The marker is written in its own transaction BEFORE the re-embed loop so
/// that an interruption (crash, kill, locked database) leaves the marker
/// behind: all subsequent embedding operations refuse until `reindex --force`
/// finishes. The upsert also stages the target identity in the same row so a
/// marker-present database always names the interrupted target.
///
/// # Errors
///
/// Returns error if the write fails.
pub(crate) fn write_marker(conn: &Connection, target: &ModelIdentity) -> Result<(), Error> {
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
///
/// # Errors
///
/// Returns error if the write or the commit fails; the transaction is rolled
/// back on any error before this function returns.
pub(crate) fn record_identity_and_clear_marker(
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

/// Begin a model-switch migration: write the migration marker for the target
/// identity. Call this BEFORE the re-embed loop so a crash mid-run leaves the
/// marker behind and all embedding operations refuse.
///
/// After the re-embed pass completes cleanly, call [`complete_migration`]
/// to record the new identity and clear the marker atomically.
///
/// # Errors
///
/// Returns error if the marker write fails.
pub(crate) fn begin_migration(conn: &Connection, target: &ModelIdentity) -> Result<(), Error> {
    write_marker(conn, target)
}

pub(crate) fn complete_migration(conn: &Connection, identity: &ModelIdentity) -> Result<(), Error> {
    record_identity_and_clear_marker(conn, identity)
}

pub fn is_migrating(conn: &Connection) -> Result<bool, Error> {
    Ok(read_marker(conn)?.is_some())
}

pub fn current_identity(conn: &Connection) -> Result<ModelIdentity, Error> {
    Ok(read_identity(conn)?.unwrap_or_else(ModelIdentity::bge_default))
}

/// Re-embed every row in a project, bypassing Mock/Real classification.
///
/// Used by [`force_migrate_project`] and by tests. Only Unknown (corrupted)
/// rows are skipped; Real and Mock rows are all re-embedded. The `embed`
/// closure receives the raw stored content (unprefixed) and must return a
/// 384-dim f32 vector.
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

/// Run the full `reindex --force` model-switch migration on a single project.
///
/// This is the lib-level function that the binary's `reindex --force` path
/// calls. It performs the crash-safe dance in order:
/// 1. Write the migration marker (via [`begin_migration`])
/// 2. Re-embed every row via the provided `embed` closure (bypassing
///    Mock/Real classification; only Unknown rows are skipped)
/// 3. If all rows succeed: record the new identity and clear the marker
///    (via [`complete_migration`]) in ONE transaction
/// 4. If any row fails: the marker stays, and the function returns an error
///
/// The `embed` closure receives the raw stored content (unprefixed — prefixes
/// live only at embed time, never in the DB) and must return a 384-dim f32
/// vector. The closure is called once per row, in the order returned by
/// `list_all_rows_for_project`.
///
/// # Arguments
///
/// * `db` - The database to migrate
/// * `target` - The target model identity (id + revision) to record
/// * `project_id` - The project to re-embed (one project per call)
/// * `embed` - The embedding function (profile's passage embedding)
///
/// # Returns
///
/// `(reindexed, skipped, failed)` counts for this project.
///
/// # Errors
///
/// Returns an error if the marker write fails, the re-embed pass has any
/// failures, or the final identity commit fails.
pub fn force_migrate_project<F>(
    db: &Database,
    target: &ModelIdentity,
    project_id: &str,
    mut embed: F,
) -> Result<(usize, usize, Vec<String>), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    // Step 1: marker-first (must be durable before any row is touched).
    begin_migration(db.conn(), target)?;

    // Step 2: re-embed every row (bypassing Mock/Real classification).
    let (reindexed, skipped, failed) =
        force_reembed_project(db, project_id, |content| embed(content))?;

    if !failed.is_empty() {
        // Marker stays — the re-embed pass did not complete.
        return Err(Error::InvalidInput(format!(
            "force reindex failed on {} row(s) for project {}: the migration marker is left in place. Fix the errors and re-run `vipune reindex --force`.",
            failed.len(),
            project_id
        )));
    }

    // Step 3: record identity + clear marker in ONE transaction.
    complete_migration(db.conn(), target)?;

    Ok((reindexed, skipped, failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn migrated_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE model_identity (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_id TEXT,
                model_revision TEXT,
                migration_marker TEXT
            );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn test_no_row_means_none() {
        let conn = migrated_conn();
        assert_eq!(read_identity(&conn).unwrap(), None);
        assert_eq!(read_marker(&conn).unwrap(), None);
    }

    #[test]
    fn test_read_identity_roundtrip() {
        let conn = migrated_conn();
        let identity = ModelIdentity {
            model_id: "intfloat/multilingual-e5-small".to_string(),
            revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
        };
        record_identity_and_clear_marker(&conn, &identity).unwrap();
        assert_eq!(read_identity(&conn).unwrap(), Some(identity));
        assert_eq!(read_marker(&conn).unwrap(), None);
    }

    #[test]
    fn test_write_marker_persists_marker_and_target() {
        let conn = migrated_conn();
        let target = ModelIdentity {
            model_id: "intfloat/multilingual-e5-small".to_string(),
            revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
        };
        write_marker(&conn, &target).unwrap();
        let marker = read_marker(&conn).unwrap();
        assert_eq!(
            marker.as_deref(),
            Some(
                "migrating to intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3"
            )
        );
        let target_recorded = read_identity(&conn).unwrap();
        assert_eq!(target_recorded, Some(target));
    }

    #[test]
    fn test_record_identity_clears_marker() {
        let conn = migrated_conn();
        let target = ModelIdentity {
            model_id: "intfloat/multilingual-e5-small".to_string(),
            revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
        };
        write_marker(&conn, &target).unwrap();
        // Simulate a crash mid-run: marker present, some rows re-embedded.
        // Re-run: record identity + clear marker in one transaction.
        record_identity_and_clear_marker(&conn, &target).unwrap();
        assert_eq!(read_identity(&conn).unwrap(), Some(target));
        assert_eq!(read_marker(&conn).unwrap(), None);
    }

    #[test]
    fn test_marker_overwrites_existing_marker() {
        let conn = migrated_conn();
        let first = ModelIdentity {
            model_id: "intfloat/multilingual-e5-small".to_string(),
            revision: "rev-a".to_string(),
        };
        let second = ModelIdentity {
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
            revision: "rev-b".to_string(),
        };
        write_marker(&conn, &first).unwrap();
        write_marker(&conn, &second).unwrap();
        let marker = read_marker(&conn).unwrap();
        assert_eq!(
            marker.as_deref(),
            Some("migrating to BAAI/bge-small-en-v1.5@rev-b")
        );
        // Still a single row.
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM model_identity", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_bge_default_identity_matches_embed_constants() {
        let default = ModelIdentity::bge_default();
        assert_eq!(default.model_id, EMBED_MODEL_ID);
        assert_eq!(default.revision, EMBED_MODEL_REVISION);
        assert_eq!(
            default.display(),
            "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        );
    }

    #[test]
    fn test_record_identity_overwrites_previous_identity() {
        let conn = migrated_conn();
        let first = ModelIdentity::bge_default();
        record_identity_and_clear_marker(&conn, &first).unwrap();
        let second = ModelIdentity {
            model_id: "intfloat/multilingual-e5-small".to_string(),
            revision: "rev-x".to_string(),
        };
        record_identity_and_clear_marker(&conn, &second).unwrap();
        assert_eq!(read_identity(&conn).unwrap(), Some(second));
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM model_identity", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    // Integration test: exercise the full force_migrate_project path so the
    // lib target's dead-code analysis sees all intermediate functions as used.
    #[test]
    fn test_force_migrate_project_full_cycle() {
        use crate::sqlite::Database;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();

        let fake_embed = |content: &str| -> Result<Vec<f32>, Error> {
            // Deterministic fake: hash content into a 384-dim vector with norm ≈ 1.0
            let mut v = vec![0.0f32; 384];
            for (i, b) in content.bytes().enumerate() {
                v[i % 384] += (b as f32) / 255.0;
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            Ok(v)
        };

        db.insert(
            "proj",
            "test content",
            &vec![0.5; 384],
            None,
            "fact",
            "active",
        )
        .unwrap();

        let target = ModelIdentity {
            model_id: "test-model".to_string(),
            revision: "test-rev".to_string(),
        };

        let (reindexed, skipped, failed) =
            force_migrate_project(&db, &target, "proj", |c| fake_embed(c)).unwrap();
        assert_eq!(reindexed, 1);
        assert_eq!(skipped, 0);
        assert!(failed.is_empty());

        // Identity recorded, marker cleared.
        assert_eq!(current_identity(db.conn()).unwrap(), target);
        assert!(!is_migrating(db.conn()).unwrap());
    }
}
