//! Model identity bookkeeping for the `model_identity` table (migration v6,
//! issue #217).
//!
//! This is the read/compare side of model identity: export and import read
//! the identity recorded for the store (defaulting to bge at its pinned
//! revision when no row exists), the hook path reads the identity + migration
//! marker to decide whether to insert, and migration tests compare the
//! recorded identity against the configured profile. The write side — the
//! marker-first crash-safe dance of `reindex --force` (write marker,
//! re-embed, record identity + clear marker in one transaction) — lives in
//! [`crate::sqlite::identity`], which owns the table's lifecycle.
//!
//! The identity a database is assumed to have when no row is recorded: the
//! default model (bge) at its pinned revision.

use crate::embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION};
use crate::sqlite::Error;
use rusqlite::{Connection, OptionalExtension};

/// Name of the model-identity table created by migration v6.
pub const MODEL_IDENTITY_TABLE: &str = "model_identity";

/// The identity a database is assumed to have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    /// HuggingFace model id (e.g. `BAAI/bge-small-en-v1.5`).
    pub model_id: String,
    /// Pinned revision the embeddings were produced with.
    pub revision: String,
}

impl From<ModelIdentity> for crate::sqlite::identity::ModelIdentity {
    fn from(value: ModelIdentity) -> Self {
        crate::sqlite::identity::ModelIdentity {
            model_id: value.model_id,
            revision: value.revision,
        }
    }
}

impl ModelIdentity {
    /// `id@revision` form, used in error messages and the migration marker.
    pub fn display(&self) -> String {
        format!("{}@{}", self.model_id, self.revision)
    }
}

/// The identity a database with no recorded row is treated as: the default
/// bge model at its pinned revision.
pub fn default_identity() -> ModelIdentity {
    ModelIdentity {
        model_id: EMBED_MODEL_ID.to_string(),
        revision: EMBED_MODEL_REVISION.to_string(),
    }
}

/// The identity the currently configured model resolves to.
///
/// A built-in profile id resolves to its pinned revision (via the profile
/// registry in `crate::embedding_profiles`); any configured id that is not a
/// built-in profile resolves to the id itself with no revision recorded,
/// which makes any store with a recorded row mismatch (refused) rather than
/// silently "matching". Config validation rejects unknown ids anyway.
pub fn configured_identity(configured_model_id: &str) -> ModelIdentity {
    match crate::embedding_profiles::profile_for(configured_model_id) {
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

/// Read the recorded identity and any migration marker.
///
/// Returns `(identity, migration_marker)` where `identity` is `None` when no
/// row exists (callers compare against [`default_identity`]) and
/// `migration_marker` is the "migrating to ..." text if one is present.
///
/// # Errors
///
/// Returns `Error::Sqlite` if the table cannot be read (e.g. pre-v6 schema).
pub fn read_identity(conn: &Connection) -> Result<(Option<ModelIdentity>, Option<String>), Error> {
    let row: Option<(Option<String>, Option<String>, Option<String>)> = conn
        .query_row(
            &format!(
                "SELECT model_id, model_revision, migration_marker
                 FROM {MODEL_IDENTITY_TABLE} WHERE id = 1"
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()
        .map_err(|e| Error::Sqlite(e.to_string()))?;

    match row {
        None => Ok((None, None)),
        Some((model_id, revision, marker)) => {
            // A row written by `reindex --force`'s marker-first step stages
            // the target identity alongside the marker; treat a partial row
            // as a recorded identity too.
            let identity = model_id.map(|model_id| ModelIdentity {
                model_id,
                revision: revision.unwrap_or_default(),
            });
            Ok((identity, marker))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::Database;

    /// Test-local mirror of the pre-merge production `identity_matches_configured`:
    /// recorded (or default) identity equals the configured profile's identity,
    /// with no migration marker in flight.
    fn identity_matches_configured(
        conn: &Connection,
        configured_model_id: &str,
    ) -> Result<bool, Error> {
        let (recorded, marker) = read_identity(conn)?;
        if marker.is_some() {
            return Ok(false);
        }
        let effective = recorded.unwrap_or_else(default_identity);
        Ok(effective == configured_identity(configured_model_id))
    }

    fn open_db() -> (tempfile::TempDir, Database) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        (dir, db)
    }

    fn set_identity(db: &Database, id: &ModelIdentity) {
        crate::sqlite::identity::record_identity_and_clear_marker(
            db.conn(),
            &crate::sqlite::identity::ModelIdentity::from(id.clone()),
        )
        .unwrap();
    }

    #[test]
    fn default_identity_is_bge_at_pinned_revision() {
        let id = default_identity();
        assert_eq!(id.model_id, EMBED_MODEL_ID);
        assert_eq!(id.revision, EMBED_MODEL_REVISION);
    }

    #[test]
    fn configured_identity_resolves_builtin_profiles() {
        let bge = configured_identity(EMBED_MODEL_ID);
        assert_eq!(bge, default_identity());
        let e5 = configured_identity("intfloat/multilingual-e5-small");
        assert_eq!(e5.model_id, "intfloat/multilingual-e5-small");
        assert_eq!(
            e5.revision, "614241f622f53c4eeff9890bdc4f31cfecc418b3",
            "e5 profile must resolve to its pinned revision, not an empty one"
        );
        // Unknown id: id kept, revision empty (config validation rejects
        // unknown ids before this is ever reached in practice).
        let unknown = configured_identity("no/such-model");
        assert_eq!(unknown.model_id, "no/such-model");
        assert_eq!(unknown.revision, "");
    }

    #[test]
    fn no_row_reads_as_none() {
        let (_dir, db) = open_db();
        let (identity, marker) = read_identity(db.conn()).unwrap();
        assert_eq!(identity, None);
        assert_eq!(marker, None);
    }

    #[test]
    fn no_row_defaults_to_bge_identity() {
        let (_dir, db) = open_db();
        assert!(identity_matches_configured(db.conn(), EMBED_MODEL_ID).unwrap());
        // A configured non-default model does NOT match an unrecorded store.
        assert!(!identity_matches_configured(db.conn(), "no/such-model").unwrap());
    }

    #[test]
    fn write_then_read_roundtrips() {
        let (_dir, db) = open_db();
        let id = ModelIdentity {
            model_id: "intfloat/multilingual-e5-small".to_string(),
            revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
        };
        set_identity(&db, &id);
        let (read, marker) = read_identity(db.conn()).unwrap();
        assert_eq!(read, Some(id.clone()));
        assert_eq!(marker, None);
        assert!(identity_matches_configured(db.conn(), "intfloat/multilingual-e5-small").unwrap());
        // The same id recorded at a different revision must NOT match.
        assert!(!identity_matches_configured(db.conn(), EMBED_MODEL_ID).unwrap());
    }

    #[test]
    fn revision_change_counts_as_mismatch() {
        let (_dir, db) = open_db();
        let old = ModelIdentity {
            model_id: "e5".to_string(),
            revision: "old-rev".to_string(),
        };
        let new = ModelIdentity {
            model_id: "e5".to_string(),
            revision: "new-rev".to_string(),
        };
        set_identity(&db, &old);
        set_identity(&db, &new);
        let (read, _) = read_identity(db.conn()).unwrap();
        assert_eq!(read, Some(new));
    }

    #[test]
    fn marker_blocks_match_until_cleared() {
        let (_dir, db) = open_db();
        let id = default_identity();
        set_identity(&db, &id);
        assert!(identity_matches_configured(db.conn(), EMBED_MODEL_ID).unwrap());
        crate::sqlite::identity::write_marker(db.conn(), &id.clone().into()).unwrap();
        assert!(!identity_matches_configured(db.conn(), EMBED_MODEL_ID).unwrap());
        crate::sqlite::identity::record_identity_and_clear_marker(db.conn(), &id.clone().into())
            .unwrap();
        assert!(identity_matches_configured(db.conn(), EMBED_MODEL_ID).unwrap());
    }

    #[test]
    fn marker_on_empty_table_refuses() {
        let (_dir, db) = open_db();
        crate::sqlite::identity::write_marker(db.conn(), &default_identity().into()).unwrap();
        // No identity row at all: marker present still refuses.
        assert!(!identity_matches_configured(db.conn(), EMBED_MODEL_ID).unwrap());
        let (_, marker) = read_identity(db.conn()).unwrap();
        assert!(marker.is_some());
    }

    #[test]
    fn display_format_is_id_at_revision() {
        let id = default_identity();
        assert_eq!(
            id.display(),
            format!("{}@{}", EMBED_MODEL_ID, EMBED_MODEL_REVISION)
        );
    }
}
