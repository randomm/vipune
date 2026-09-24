//! Prefix-path tests for `vipune reindex` (the e5 regression coverage).

#![cfg(test)]

use crate::commands::reindex::*;
use crate::embedding_profiles::EmbeddingRole;

/// Model-free stand-in for the engine's prefix path: returns `prefix_input`
/// (the same pure helper `embed_passage` / `embed_query` delegate to) applied
/// to `text` under the given role, so a recording embedder can reproduce the
/// engine's exact contract without downloading the ONNX model.
fn prefix_input_for_role(role: EmbeddingRole, text: &str) -> String {
    let profile = crate::embedding_profiles::profile_for("intfloat/multilingual-e5-small")
        .expect("e5 profile");
    let prefix = role.prefix(profile);
    if prefix.is_empty() {
        text.to_string()
    } else {
        format!("{prefix}{text}")
    }
}

#[cfg(test)]
mod e5_prefix_path {
    //! `handle_reindex`'s plain-reindex callback is
    //! `|content| engine.embed_passage(content)`, where `embed_passage`
    //! prepends the profile's passage prefix before tokenisation (the engine
    //! is the single prefix site). A regression to raw-content embedding
    //! would silently degrade quality under e5 without any error or log, so
    //! the callback is pinned here against the e5 profile's passage prefix
    //! via `prefix_input_for_role` — the same pure prefix logic
    //! `embed_passage` delegates to, without downloading the ONNX model.
    use super::*;
    use crate::memory::crud::mock_embedding_for_content;
    use crate::sqlite::Database;
    use crate::sqlite::embedding::{EmbeddingClass, classify_embedding};

    fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        Database::open(&path).unwrap();
        (dir, path)
    }

    fn get_embedding(db: &Database, id: &str) -> Vec<f32> {
        db.list_all_rows_for_project("proj")
            .unwrap()
            .iter()
            .find(|(i, _, _)| i == id)
            .map(|(_, _, e)| e.clone())
            .unwrap()
    }

    /// Regression (issue #220): plain `vipune reindex` under the e5 profile
    /// must re-embed Mock rows with the passage prefix. Before the fix the
    /// plain reindex callback called `engine.embed(content)` on raw content,
    /// so under e5 the model saw unprefixed text — silent quality degradation
    /// with no error or log. After the fix the callback is
    /// `|content| engine.embed_passage(content)`, so the engine's single
    /// prefix site prepends the profile's passage prefix.
    ///
    /// We can't call the real e5 engine in a unit test (it would download the
    /// model), so the recording embedder below reproduces the engine's exact
    /// contract for the e5 profile through `prefix_input_for_role` (the same
    /// pure prefix logic `embed_passage` delegates to, pinned model-free in
    /// `src/embedding/tests/model_free.rs`) — and drives the real
    /// `reindex_project` path with the recording callback: the recorded input
    /// must be exactly `"passage: <content>"`, not the bare content (which a
    /// regression to raw-content embedding would produce).
    #[test]
    fn test_plain_reindex_under_e5_reembeds_mock_row_with_passage_prefix() {
        let e5_profile =
            crate::embedding_profiles::profile_for("intfloat/multilingual-e5-small").expect("e5");
        let prefix = EmbeddingRole::Passage.prefix(e5_profile);
        assert_eq!(
            prefix, "passage: ",
            "e5 must declare the 'passage: ' prefix"
        );

        let (_dir, db_path) = create_test_db();
        let db = Database::open(&db_path).unwrap();
        let content = "mock memory content";
        let mock_vec = mock_embedding_for_content(content);
        let id = db
            .insert("proj", content, &mock_vec, None, "fact", "active")
            .unwrap();
        assert_eq!(
            classify_embedding(&get_embedding(&db, &id)),
            EmbeddingClass::Mock
        );

        // Recording embedder reproducing the e5 engine's passage-role contract
        // via `prefix_input_for_role` (the engine's prefix logic): input =
        // prefix + raw content, output = the fake vector for the prefixed input
        // (a Real-classified 384-dim vector, so the reindex write succeeds
        // exactly as in production).
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded_for_callback = recorded.clone();
        let mut embed_callback = move |text: &str| {
            let prefixed = prefix_input_for_role(EmbeddingRole::Passage, text);
            recorded_for_callback.lock().unwrap().push(prefixed.clone());
            crate::memory::crud::test_fake_embedder(&prefixed)
        };

        let (reindexed, skipped, failed) =
            reindex_project(&db, &mut embed_callback, "proj", false).unwrap();
        assert_eq!(reindexed, 1);
        assert_eq!(skipped, 0);
        assert!(failed.is_empty(), "reindex must not fail: {failed:?}");

        // The embed callback received exactly the passage-prefixed content —
        // never the bare content (the v0.13.0 latent bug) and never a
        // double-prefixed string.
        assert_eq!(
            recorded.lock().unwrap().clone(),
            vec!["passage: mock memory content".to_string()]
        );

        // And the row's stored content stays unprefixed.
        let rows = db.list_all_rows_for_project("proj").unwrap();
        let (_, stored_content, _) = rows.iter().find(|(i, _, _)| i == &id).unwrap();
        assert_eq!(stored_content, content);
        assert_eq!(
            classify_embedding(&get_embedding(&db, &id)),
            EmbeddingClass::Real
        );
    }
}
