//! Model identity tracking for the `model_identity` table (migration v6,
//! issue #217).
//!
//! A database records which embedding model (id + pinned revision) produced
//! its vectors. A database with no identity row — or a row with a NULL
//! `model_id` — is treated as the built-in default: bge at its pinned
//! revision (`crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION}`).
//!
//! An optional migration marker (`"migrating to <id>@<revision>"`) signals
//! that a `reindex --force` model switch was started but not finished. The
//! marker is written once, BEFORE any row of the re-embed pass, and cleared
//! in the same transaction that records the new identity, so a crash
//! mid-run leaves the marker behind and every embedding operation (add /
//! update / search / hook insert) refuses until `reindex --force` completes.
//!
//! The marker write touches ONLY the marker: the recorded identity stays as
//! it was (old identity, or no row = the bge default) until the final
//! record-and-clear step. That is what keeps an interrupted migration from
//! reporting the NEW model while most vectors are still the old model's.
//!
//! The identity lifecycle is per-DATABASE, not per-project: the marker is
//! written once and the identity is recorded once after every row of every
//! project in the database has been re-embedded. That is what keeps a
//! multi-project database from ending up as a silently mixed store after a
//! partial run. The marker/identity writes themselves live in
//! `crate::migration`; this module is the read side plus the mismatch/marker
//! refusal the library chokepoints share.

use crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION};
use crate::embedding_profiles::profile_for;
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
    /// recorded identity is treated as this identity.
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

/// Read the singleton identity row, if one exists.
///
/// Returns `(model_id, model_revision, migration_marker)` as stored. A `NULL`
/// `model_id` (a row written by the marker-first step, which never writes an
/// identity) means "no recorded identity" — the caller folds it into the
/// `None` arm, which is the single place the bge-default rule is applied:
/// one read path, one rule.
type IdentityRow = Option<(Option<String>, Option<String>, Option<String>)>;

fn read_identity_row(conn: &Connection) -> crate::sqlite::Result<IdentityRow> {
    conn.query_row(
        "SELECT model_id, model_revision, migration_marker FROM model_identity WHERE id = 1",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )
    .optional()
    .map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
}

/// Read the recorded identity, if any.
///
/// `None` — the bge default — when there is no row at all, or when the row
/// has a NULL `model_id` (a marker-only row written by an in-flight
/// `reindex --force` on a store that had never recorded an identity).
pub fn read_identity(conn: &Connection) -> crate::sqlite::Result<Option<ModelIdentity>> {
    let Some(row) = read_identity_row(conn)? else {
        return Ok(None);
    };
    let (model_id, revision, _marker) = row;
    match model_id.filter(|id| !id.is_empty()) {
        Some(model_id) => Ok(Some(ModelIdentity {
            model_id,
            revision: revision.unwrap_or_default(),
        })),
        None => Ok(None),
    }
}

/// Read the migration marker, if one is in flight.
pub fn read_marker(conn: &Connection) -> crate::sqlite::Result<Option<String>> {
    Ok(read_identity_row(conn)?.and_then(|row| row.2))
}

/// True while a migration marker is present (an interrupted `reindex --force`).
pub fn is_migrating(conn: &Connection) -> crate::sqlite::Result<bool> {
    Ok(read_marker(conn)?.is_some())
}

/// The identity the store currently has: the recorded identity, or the bge
/// default when no row (or only a marker-only row) is recorded.
pub fn current_identity(conn: &Connection) -> crate::sqlite::Result<ModelIdentity> {
    Ok(read_identity(conn)?.unwrap_or_else(ModelIdentity::default_identity))
}

/// The identity the currently configured model resolves to.
///
/// A built-in profile id resolves to its pinned revision (via the profile
/// registry in `crate::embedding_profiles`); any configured id that is not a
/// built-in profile resolves to the id itself with no revision recorded,
/// which makes any store with a recorded row mismatch (refused) rather than
/// silently "matching". Config validation rejects unknown ids anyway.
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
/// Returns `(identity, migration_marker)` where `identity` is `None` when
/// no identity is recorded (no row, or a marker-only row with NULL
/// `model_id` — callers compare against
/// [`ModelIdentity::default_identity`]) and `migration_marker` is the
/// "migrating to ..." text if one is present.
pub fn read_identity_and_marker(
    conn: &Connection,
) -> crate::sqlite::Result<(Option<ModelIdentity>, Option<String>)> {
    let Some(row) = read_identity_row(conn)? else {
        return Ok((None, None));
    };
    let (model_id, revision, marker) = row;
    // A NULL model_id means "no recorded identity" (a marker-only row): the
    // bge-default rule is the single place `None` gets folded, exactly as in
    // `read_identity`.
    let identity = model_id
        .filter(|id| !id.is_empty())
        .map(|model_id| ModelIdentity {
            model_id,
            revision: revision.unwrap_or_default(),
        });
    Ok((identity, marker))
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

#[cfg(test)]
#[path = "identity_tests.rs"]
mod identity_tests;
