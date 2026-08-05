//! SQLite CRUD tests.
#[cfg(test)]
mod crud_tests {
    use crate::embedding::EMBEDDING_DIMS;
    use crate::sqlite::{Database, UpdateOptions};
    use tempfile::TempDir;

    fn create_test_db() -> Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    #[test]
    fn test_insert_and_get() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id, "proj1").unwrap();
        assert!(memory.is_some());
        let m = memory.unwrap();
        assert_eq!(m.content, "test content");
        assert_eq!(m.project_id, "proj1");
    }

    #[test]
    fn test_insert_with_metadata() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert(
                "proj1",
                "test content",
                &embedding,
                Some(r#"{"key": "value"}"#),
                "fact",
                "active",
            )
            .unwrap();

        let m = db.get(&id, "proj1").unwrap().unwrap();
        assert_eq!(m.metadata, Some(r#"{"key": "value"}"#.to_string()));
    }

    #[test]
    fn test_insert_invalid_embedding() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 256];
        let result = db.insert("proj1", "test", &embedding, None, "fact", "active");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_nonexistent() {
        let db = create_test_db();
        let memory = db.get("nonexistent", "proj1").unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_update() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original", &embedding, None, "fact", "active")
            .unwrap();

        db.update(
            &id,
            "proj1",
            UpdateOptions {
                content: Some("updated"),
                embedding: Some(&embedding),
                metadata: None,
                memory_type: None,
                status: None,
            },
        )
        .unwrap();

        let m = db.get(&id, "proj1").unwrap().unwrap();
        assert_eq!(m.content, "updated");
    }

    #[test]
    fn test_update_nonexistent() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let result = db.update(
            "nonexistent",
            "proj1",
            UpdateOptions {
                content: Some("content"),
                embedding: Some(&embedding),
                metadata: None,
                memory_type: None,
                status: None,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "content", &embedding, None, "fact", "active")
            .unwrap();

        let deleted = db.delete(&id, "proj1").unwrap();
        assert!(deleted);

        let memory = db.get(&id, "proj1").unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let db = create_test_db();
        let deleted = db.delete("nonexistent", "proj1").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_project_isolation() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "proj1 content", &embedding, None, "fact", "active")
            .unwrap();
        db.insert("proj2", "proj2 content", &embedding, None, "fact", "active")
            .unwrap();

        let list1 = db.list("proj1", 10, None, None).unwrap();
        let list2 = db.list("proj2", 10, None, None).unwrap();

        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].project_id, "proj1");
        assert_eq!(list2[0].project_id, "proj2");
    }

    /// Security: get() must not return memories belonging to other projects.
    #[test]
    fn test_get_cross_project_isolation() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "secret content", &embedding, None, "fact", "active")
            .unwrap();

        // proj1 can access its own memory
        let found = db.get(&id, "proj1").unwrap();
        assert!(found.is_some(), "proj1 should access its own memory");

        // proj2 must NOT access proj1's memory
        let not_found = db.get(&id, "proj2").unwrap();
        assert!(
            not_found.is_none(),
            "proj2 must not access proj1's memory (cross-project isolation)"
        );
    }

    /// Security: delete() must not delete memories belonging to other projects.
    #[test]
    fn test_delete_cross_project_isolation() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "content to protect", &embedding, None, "fact", "active")
            .unwrap();

        // proj2 must NOT delete proj1's memory
        let deleted = db.delete(&id, "proj2").unwrap();
        assert!(
            !deleted,
            "proj2 must not delete proj1's memory (cross-project isolation)"
        );

        // Verify memory still exists in proj1
        let still_exists = db.get(&id, "proj1").unwrap();
        assert!(
            still_exists.is_some(),
            "proj1 memory should survive a cross-project delete attempt"
        );

        // proj1 can delete its own memory
        let deleted = db.delete(&id, "proj1").unwrap();
        assert!(deleted, "proj1 should be able to delete its own memory");
    }

    #[test]
    fn test_get_includes_embedding() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id, "proj1").unwrap().unwrap();
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        for (i, &val) in embedding.iter().enumerate() {
            assert!((memory.embedding[i] - val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_list_includes_embeddings() {
        let db = create_test_db();
        let embedding1 = vec![0.1f32; EMBEDDING_DIMS];
        let embedding2 = vec![0.2f32; EMBEDDING_DIMS];

        db.insert("proj1", "first", &embedding1, None, "fact", "active")
            .unwrap();
        db.insert("proj1", "second", &embedding2, None, "fact", "active")
            .unwrap();

        let memories = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(memories.len(), 2);

        for memory in &memories {
            assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        }
    }

    #[test]
    fn test_embedding_roundtrip() {
        let db = create_test_db();
        let original = vec![0.123f32, 0.456f32, 0.789f32];
        let mut full_embedding = vec![0.1f32; EMBEDDING_DIMS];
        full_embedding[0] = original[0];
        full_embedding[1] = original[1];
        full_embedding[EMBEDDING_DIMS - 1] = original[2];

        let id = db
            .insert("proj1", "test", &full_embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id, "proj1").unwrap().unwrap();
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        assert!((memory.embedding[0] - original[0]).abs() < 1e-6);
        assert!((memory.embedding[1] - original[1]).abs() < 1e-6);
        assert!((memory.embedding[EMBEDDING_DIMS - 1] - original[2]).abs() < 1e-6);
    }
}