//! Placeholder embedding for hook-inserted rows (issue #213).
//!
//! The hook path cannot use the `#[cfg(test)]`-gated
//! `mock_embedding_for_content` helper from `src/memory/crud.rs`, and it must
//! not run the real embedder. This module provides a release-compiled
//! deterministic generator that produces a 384-dim vector whose L2 norm is
//! strictly greater than 2.0, so `classify_embedding` buckets it as `Mock`
//! and a later `reindex` run will backfill it with a real vector.
//!
//! The generator is deterministic per-content (same content → same vector),
//! so re-runs of the hook on the same candidate are byte-stable.

use crate::config::Config;
use crate::errors::Error;
use crate::sqlite::identity::assert_identity_ok;

use crate::embedding::EMBEDDING_DIMS;

/// Generate a deterministic placeholder embedding for the hook path.
///
/// Produces a 384-dim f32 vector in the range [-1, 1] per component. Because
/// the values are pseudo-randomised over 384 dims the L2 norm is typically
/// ≈ 11.3, well above the 2.0 Mock threshold in `classify_embedding`.
///
/// # Invariant
///
/// The generated vector MUST classify as `Mock`. If a future change to the
/// generator or the classifier breaks this, the hook will silently ship
/// placeholder vectors that `reindex` treats as Real (or Unknown) and never
/// backfills, corrupting search results. The test below pins this invariant.
pub fn placeholder_embedding(content: &str) -> Vec<f32> {
    // Seed with a simple content hash so identical candidates produce
    // identical vectors (byte-stable on re-run).
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    for byte in content.bytes() {
        seed = seed.wrapping_mul(31).wrapping_add(byte as u64);
    }

    let mut vec = Vec::with_capacity(EMBEDDING_DIMS);
    for i in 0..EMBEDDING_DIMS {
        // Use hash + index to generate deterministic pseudo-random values.
        // Same algorithmic shape as `mock_embedding_for_content` in crud.rs
        // so the placeholder sits in the same norm band as the test mock.
        let mut dim_hash = seed.wrapping_add(i as u64);
        dim_hash ^= dim_hash >> 33;
        dim_hash = dim_hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        dim_hash ^= dim_hash >> 33;
        dim_hash = dim_hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);

        // Map to [-1.0, 1.0].
        let value = ((dim_hash % 2000) as f32 - 1000.0) / 1000.0;
        vec.push(value);
    }
    vec
}

/// Check the hook path's model-identity source (issue #217, task-b).
///
/// The hook uses the same profile source as the main CLI path: a recorded
/// identity (or the bge@pinned-revision default when no row exists) must
/// match the configured `embedding_model`, and no interrupted model
/// migration (`migration_marker`) may be in flight.
///
/// The caller (the hook pipeline) treats any `Err` as a refusal: it prints
/// the error to stderr and exits 0 — the hook contract is exit-0-always (the
/// hook must never surface an error to the agent mid-session), but the
/// refusal message is never silent. A mismatch/migrating store must not
/// accumulate placeholder rows that no future plain `reindex` would ever
/// backfill (real vectors classify as Real, so only `reindex --force`
/// re-embeds a switched store).
///
/// # Errors
///
/// Returns an `Error::Config` message naming the recorded and configured
/// identities (or the interrupted migration target) and pointing at
/// `vipune reindex --force`, or `Error::SqliteModule` if the identity table
/// cannot be read.
pub fn ensure_hook_identity_ok(db: &crate::sqlite::Database, config: &Config) -> Result<(), Error> {
    assert_identity_ok(db.conn(), config.embedding_model.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};

    fn test_config(model: &str) -> Config {
        Config {
            embedding_model: model.to_string(),
            ..Config::default()
        }
    }

    fn open_db(dir: &tempfile::TempDir) -> crate::sqlite::Database {
        let path = dir.path().join("test.db");
        crate::sqlite::Database::open(&path).unwrap()
    }

    fn test_identity(model_id: &str, revision: &str) -> crate::sqlite::identity::ModelIdentity {
        crate::sqlite::identity::ModelIdentity {
            model_id: model_id.to_string(),
            revision: revision.to_string(),
        }
    }

    #[test]
    fn placeholder_classifies_as_mock() {
        let v = placeholder_embedding("hello world");
        assert_eq!(v.len(), EMBEDDING_DIMS);
        assert_eq!(
            classify_embedding(&v),
            EmbeddingClass::Mock,
            "placeholder must classify as Mock (L2 norm strictly > 2.0) so reindex backfills it"
        );
    }

    #[test]
    fn placeholder_is_deterministic_per_content() {
        let a = placeholder_embedding("same input");
        let b = placeholder_embedding("same input");
        assert_eq!(a, b, "same content must produce the same placeholder");
    }

    #[test]
    fn placeholder_differs_for_different_content() {
        let a = placeholder_embedding("foo");
        let b = placeholder_embedding("bar");
        assert_ne!(
            a, b,
            "different content should produce different placeholders"
        );
    }

    #[test]
    fn hook_identity_ok_when_db_default_and_config_default() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = open_db(&dir);
        assert!(
            ensure_hook_identity_ok(&db, &test_config(crate::embedding::EMBED_MODEL_ID)).is_ok()
        );
    }

    #[test]
    fn hook_identity_ok_when_recorded_matches_configured() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut db = open_db(&dir);
        let id = test_identity(
            crate::embedding::EMBED_MODEL_ID,
            crate::embedding::EMBED_MODEL_REVISION,
        );
        crate::commands::reindex_force::record_identity_and_clear_marker(&mut db, &id).unwrap();
        let config = test_config(crate::embedding::EMBED_MODEL_ID);
        assert!(ensure_hook_identity_ok(&db, &config).is_ok());
    }

    #[test]
    fn hook_identity_refuses_on_mismatch() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut db = open_db(&dir);
        let id = test_identity("other-model", "rev-1");
        crate::commands::reindex_force::record_identity_and_clear_marker(&mut db, &id).unwrap();
        let config = test_config(crate::embedding::EMBED_MODEL_ID);
        let err = ensure_hook_identity_ok(&db, &config).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("other-model"),
            "mismatch error names recorded id: {msg}"
        );
        assert!(
            msg.contains("vipune reindex --force"),
            "mismatch error points at --force: {msg}"
        );
    }

    #[test]
    fn hook_identity_refuses_while_migration_marker_present() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = open_db(&dir);
        let target = test_identity("e5-model", "rev-2");
        crate::commands::reindex_force::write_marker(&db, &target).unwrap();
        let config = test_config(crate::embedding::EMBED_MODEL_ID);
        let err = ensure_hook_identity_ok(&db, &config).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("migration"),
            "marker error names the migration: {msg}"
        );
        assert!(
            msg.contains("e5-model@rev-2"),
            "marker error names the target: {msg}"
        );
        assert!(
            msg.contains("vipune reindex --force"),
            "marker error points at --force: {msg}"
        );
    }
}
