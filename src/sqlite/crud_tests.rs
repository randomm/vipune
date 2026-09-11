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
                importance: None,
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
                importance: None,
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
            .insert(
                "proj1",
                "secret content",
                &embedding,
                None,
                "fact",
                "active",
            )
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
            .insert(
                "proj1",
                "content to protect",
                &embedding,
                None,
                "fact",
                "active",
            )
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
    fn test_update_no_fields_supplied_returns_error() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original", &embedding, None, "fact", "active")
            .unwrap();

        // All optional fields are None — only updated_at would be set, which is rejected
        let result = db.update(
            &id,
            "proj1",
            UpdateOptions {
                content: None,
                embedding: None,
                metadata: None,
                memory_type: None,
                status: None,
                importance: None,
            },
        );
        assert!(
            result.is_err(),
            "update with no fields supplied must return an error"
        );
        match result.unwrap_err() {
            crate::sqlite::Error::InvalidInput(msg) => {
                assert!(
                    msg.contains("At least one field"),
                    "expected 'At least one field' message, got: {}",
                    msg
                );
            }
            other => panic!("Expected InvalidInput error, got: {:?}", other),
        }
    }

    /// `insert_with_id` must restore all 12 columns verbatim under the
    /// caller-supplied id, including the three columns the regular `insert()`
    /// does not accept (`superseded_by`, `retrieval_count`, `last_retrieved_at`),
    /// and must return `true` on a fresh insert.
    #[test]
    fn test_insert_with_id_restores_all_columns() {
        let db = create_test_db();
        let blob: Vec<u8> = (0..384)
            .flat_map(|i| (i as f32 * 0.001f32).to_le_bytes())
            .collect();
        let inserted = db
            .insert_with_id(
                "fixed-id-1",
                "proj",
                "restored content",
                &blob,
                Some(r#"{"k":"v"}"#),
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "guard",
                "superseded",
                Some("superseding-id"),
                42,
                Some("2024-01-03T00:00:00Z"),
            )
            .unwrap();
        assert!(inserted, "fresh id must insert and return true");

        // Raw BLOB byte-identity (compared as raw bytes, not decoded).
        let conn = db.conn();
        let stored_blob: Vec<u8> = conn
            .query_row(
                "SELECT embedding FROM memories WHERE id = ?",
                ["fixed-id-1"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored_blob, blob);

        // All other columns verbatim.
        let row = conn
            .query_row(
                "SELECT project_id, content, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at FROM memories WHERE id = 'fixed-id-1'",
                [],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, String>(5)?,
                        r.get::<_, String>(6)?,
                        r.get::<_, Option<String>>(7)?,
                        r.get::<_, i64>(8)?,
                        r.get::<_, Option<String>>(9)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(row.0, "proj");
        assert_eq!(row.1, "restored content");
        assert_eq!(row.2, Some(r#"{"k":"v"}"#.to_string()));
        assert_eq!(row.3, "2024-01-01T00:00:00Z");
        assert_eq!(row.4, "2024-01-02T00:00:00Z");
        assert_eq!(row.5, "guard");
        assert_eq!(row.6, "superseded");
        assert_eq!(row.7, Some("superseding-id".to_string()));
        assert_eq!(row.8, 42);
        assert_eq!(row.9, Some("2024-01-03T00:00:00Z".to_string()));
    }

    /// `insert_with_id` with an id that already exists must NOT upsert:
    /// it surfaces as a PK-constraint error, and the original row (including
    /// its raw blob and counters) is left byte-for-byte untouched. The import
    /// handler pre-filters against `existing_ids()`, so this error is the
    /// defensive guard for a duplicate that slips through the skip set.
    #[test]
    fn test_insert_with_id_existing_id_is_not_upserted() {
        let db = create_test_db();
        let blob: Vec<u8> = vec![1u8; 1536];
        let inserted = db
            .insert_with_id(
                "dup-id",
                "proj",
                "original content",
                &blob,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
                None,
                7,
                None,
            )
            .unwrap();
        assert!(inserted);

        // Second call with the same id but different data: must fail with a
        // PK-constraint error (not a silent `false`, not an upsert).
        let other_blob: Vec<u8> = vec![2u8; 1536];
        let result = db.insert_with_id(
            "dup-id",
            "other-proj",
            "replaced content",
            &other_blob,
            None,
            "2024-02-01T00:00:00Z",
            "2024-02-01T00:00:00Z",
            "fact",
            "active",
            None,
            99,
            None,
        );
        assert!(
            result.is_err(),
            "duplicate id must error (PK constraint), not upsert"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("UNIQUE constraint") || err_msg.contains("constraint"),
            "expected a PK-constraint error, got: {}",
            err_msg
        );

        let conn = db.conn();
        let (content, stored_blob, retrieval_count) = conn
            .query_row(
                "SELECT content, embedding, retrieval_count FROM memories WHERE id = 'dup-id'",
                [],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Vec<u8>>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            content, "original content",
            "original row must be untouched"
        );
        assert_eq!(stored_blob, blob, "original blob must be byte-identical");
        assert_eq!(retrieval_count, 7);
    }

    /// `existing_ids` returns the set of all ids present across all projects,
    /// and is empty for a fresh database.
    #[test]
    fn test_existing_ids() {
        let db = create_test_db();
        assert!(db.existing_ids().unwrap().is_empty(), "fresh DB has no ids");

        let blob: Vec<u8> = vec![0u8; 1536];
        db.insert_with_id(
            "a",
            "proj-a",
            "x",
            &blob,
            None,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        .unwrap();
        db.insert_with_id(
            "b",
            "proj-b",
            "y",
            &blob,
            None,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        .unwrap();

        let mut ids = db.existing_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"], "skip set spans all projects");
    }

    #[test]
    fn test_embedding_roundtrip() {
        let db = create_test_db();
        let original = [0.123f32, 0.456f32, 0.789f32];
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
