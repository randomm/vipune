#[cfg(test)]
mod tests {
    use crate::embedding::EMBEDDING_DIMS;
    use crate::sqlite::Database;
    use tempfile::TempDir;

    fn create_test_db() -> Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    #[test]
    fn test_list_since_basic() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let _id1 = db
            .insert_with_time(
                "proj1",
                "old",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();
        let _id2 = db
            .insert_with_time(
                "proj1",
                "new",
                &embedding,
                None,
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();

        // List since 2024-01-01T12:00:00Z should only return "new"
        let results = db.list_since("proj1", "2024-01-01T12:00:00Z", 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "new");
    }

    #[test]
    fn test_list_since_exclusive_boundary() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id = db
            .insert_with_time(
                "proj1",
                "boundary",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();

        // List since exact timestamp should NOT return the memory (exclusive)
        let results = db.list_since("proj1", "2024-01-01T00:00:00Z", 10, None, None).unwrap();
        assert_eq!(results.len(), 0);

        // But list since 1ms before SHOULD return it
        let results = db.list_since("proj1", "2023-12-31T23:59:59Z", 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn test_list_since_invalid_timestamp() {
        let db = create_test_db();
        let result = db.list_since("proj1", "invalid-timestamp", 10, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid RFC3339"));
    }

    #[test]
    fn test_list_since_limit() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        for i in 0..5 {
            db.insert_with_time(
                "proj1",
                &format!("content {}", i),
                &embedding,
                None,
                &format!("2024-01-{:02}T00:00:00Z", i + 1),
                &format!("2024-01-{:02}T00:00:00Z", i + 1),
                "fact",
                "active",
            )
            .unwrap();
        }

        // Should only return 2 most recent
        let results = db.list_since("proj1", "2024-01-01T00:00:00Z", 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_list_since_ordering() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db
            .insert_with_time(
                "proj1",
                "first",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();
        let id2 = db
            .insert_with_time(
                "proj1",
                "second",
                &embedding,
                None,
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();
        let id3 = db
            .insert_with_time(
                "proj1",
                "third",
                &embedding,
                None,
                "2024-01-03T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();

        // Should return newest first
        let results = db.list_since("proj1", "2023-12-31T00:00:00Z", 10, None, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, id3);
        assert_eq!(results[1].id, id2);
        assert_eq!(results[2].id, id1);
    }

    #[test]
    fn test_get_many_basic() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db.insert("proj1", "content 1", &embedding, None, "fact", "active").unwrap();
        let id2 = db.insert("proj1", "content 2", &embedding, None, "fact", "active").unwrap();

        let results = db.get_many(&[&id1, &id2]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert_eq!(results[0].as_ref().unwrap().content, "content 1");
        assert_eq!(results[1].as_ref().unwrap().content, "content 2");
    }

    #[test]
    fn test_get_many_preserves_ordering() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db.insert("proj1", "first", &embedding, None, "fact", "active").unwrap();
        let id2 = db.insert("proj1", "second", &embedding, None, "fact", "active").unwrap();
        let id3 = db.insert("proj1", "third", &embedding, None, "fact", "active").unwrap();

        // Query in reverse order
        let results = db.get_many(&[&id3, &id1, &id2]).unwrap();
        assert_eq!(results[0].as_ref().unwrap().id, id3);
        assert_eq!(results[1].as_ref().unwrap().id, id1);
        assert_eq!(results[2].as_ref().unwrap().id, id2);
    }

    #[test]
    fn test_get_many_with_missing_ids() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db.insert("proj1", "content 1", &embedding, None, "fact", "active").unwrap();
        let id2 = db.insert("proj1", "content 2", &embedding, None, "fact", "active").unwrap();

        // Mix of valid and invalid IDs
        let results = db
            .get_many(&[&id1, "nonexistent-id", &id2, "another-missing"])
            .unwrap();
        assert_eq!(results.len(), 4);
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_some());
        assert!(results[3].is_none());
        assert_eq!(results[0].as_ref().unwrap().id, id1);
        assert_eq!(results[2].as_ref().unwrap().id, id2);
    }

    #[test]
    fn test_get_many_all_missing() {
        let db = create_test_db();

        let results = db.get_many(&["missing1", "missing2", "missing3"]).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_none()));
    }

    #[test]
    fn test_get_many_empty_input() {
        let db = create_test_db();

        let results = db.get_many(&[]).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_get_many_single_id() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id = db.insert("proj1", "content", &embedding, None, "fact", "active").unwrap();

        let results = db.get_many(&[&id]).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_some());
        assert_eq!(results[0].as_ref().unwrap().content, "content");
    }

    #[test]
    fn test_get_many_with_duplicate_ids() {
        // Test that duplicate IDs return duplicate results (stable behavior)
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db.insert("proj1", "content 1", &embedding, None, "fact", "active").unwrap();
        let id2 = db.insert("proj1", "content 2", &embedding, None, "fact", "active").unwrap();

        // Query with duplicate IDs
        let results = db.get_many(&[&id1, &id2, &id1, &id2]).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].as_ref().unwrap().id, id1);
        assert_eq!(results[1].as_ref().unwrap().id, id2);
        assert_eq!(results[2].as_ref().unwrap().id, id1);
        assert_eq!(results[3].as_ref().unwrap().id, id2);
    }

    #[test]
    fn test_list_since_with_timezone_offset() {
        // Test that RFC3339 with timezone offsets is accepted.
        // Note: SQLite does string comparison, not timezone-aware comparison.
        // This test documents the actual behavior: timestamps are compared as strings.
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        // Insert memories at specific UTC times
        let _id1 = db
            .insert_with_time(
                "proj1",
                "old",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();
        let _id2 = db
            .insert_with_time(
                "proj1",
                "new",
                &embedding,
                None,
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();

        // Query with UTC+01:00 offset works (RFC3339 parsing succeeds)
        // SQLite compares as strings: "2024-01-02T00:00:00Z" > "2024-01-01T11:00:00+01:00"
        let results = db
            .list_since("proj1", "2024-01-01T11:00:00+01:00", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "new");

        // Query with UTC-05:00 offset also works
        // SQLite compares as strings: "2024-01-02T00:00:00Z" > "2024-01-01T19:00:00-05:00"
        let results = db
            .list_since("proj1", "2024-01-01T19:00:00-05:00", 10, None, None)
            .unwrap();
        // String comparison: "2024-01-02..." is lexicographically greater than "2024-01-01..."
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "new");

        // Query after the stored timestamp returns nothing (Exclusive bound)
        // SQLite string comparison: "2024-01-02T00:00:00Z" is NOT > "2024-01-02T00:00:00Z"
        let results = db.list_since("proj1", "2024-01-02T00:00:00Z", 10, None, None).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_list_since_timestamp_precision_equivalence() {
        // Test that different timestamp precisions with same instant behave identically
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let _id = db
            .insert_with_time(
                "proj1",
                "test",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();

        // Query with and without fractional seconds - should behave identically
        let results1 = db.list_since("proj1", "2023-12-31T23:59:59Z", 10, None, None).unwrap();
        let results2 = db
            .list_since("proj1", "2023-12-31T23:59:59.000Z", 10, None, None)
            .unwrap();

        assert_eq!(results1.len(), results2.len());
        if results1.len() > 0 {
            assert_eq!(results1[0].id, results2[0].id);
        }

        // Precision equivalence with milliseconds
        let results3 = db
            .list_since("proj1", "2023-12-31T23:59:59.999Z", 10, None, None)
            .unwrap();
        // Should include the memory since it's before the timestamp
        assert_eq!(results3.len(), 1);
    }

    #[test]
    fn test_list_regression_coverage() {
        // Verify list() regression coverage for existing behavior
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        // Insert multiple memories
        let id1 = db.insert("proj1", "first", &embedding, None, "fact", "active").unwrap();
        let id2 = db.insert("proj1", "second", &embedding, None, "fact", "active").unwrap();
        let id3 = db.insert("proj1", "third", &embedding, None, "fact", "active").unwrap();

        // Test ordering (newest first)
        let results = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, id3);
        assert_eq!(results[1].id, id2);
        assert_eq!(results[2].id, id1);

        // Test limit
        let results = db.list("proj1", 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id3);

        // Test project isolation
        db.insert("proj2", "other", &embedding, None, "fact", "active").unwrap();
        let proj1_results = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(proj1_results.len(), 3);

        // Test empty project
        let empty_results = db.list("nonexistent", 10, None, None).unwrap();
        assert_eq!(empty_results.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id).unwrap();
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

        let m = db.get(&id).unwrap().unwrap();
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
        let memory = db.get("nonexistent").unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_list_ordering() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id1 = db
            .insert_with_time(
                "proj1",
                "first",
                &embedding,
                None,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();
        let id2 = db
            .insert_with_time(
                "proj1",
                "second",
                &embedding,
                None,
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "fact",
                "active",
            )
            .unwrap();

        let memories = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].id, id2); // Newest first
        assert_eq!(memories[1].id, id1);
    }

    #[test]
    fn test_list_limit() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        for i in 0..5 {
            db.insert("proj1", &format!("content {}", i), &embedding, None, "fact", "active")
                .unwrap();
        }

        let memories = db.list("proj1", 2, None, None).unwrap();
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_update() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db.insert("proj1", "original", &embedding, None, "fact", "active").unwrap();

        db.update(&id, Some("updated"), Some(&embedding), None, None, None).unwrap();

        let m = db.get(&id).unwrap().unwrap();
        assert_eq!(m.content, "updated");
    }

    #[test]
    fn test_update_nonexistent() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let result = db.update("nonexistent", Some("content"), Some(&embedding), None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db.insert("proj1", "content", &embedding, None, "fact", "active").unwrap();

        let deleted = db.delete(&id).unwrap();
        assert!(deleted);

        let memory = db.get(&id).unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let db = create_test_db();
        let deleted = db.delete("nonexistent").unwrap();
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

    #[test]
    fn test_get_includes_embedding() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let memory = db.get(&id).unwrap().unwrap();
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

        db.insert("proj1", "first", &embedding1, None, "fact", "active").unwrap();
        db.insert("proj1", "second", &embedding2, None, "fact", "active").unwrap();

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

        let memory = db.get(&id).unwrap().unwrap();
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        assert!((memory.embedding[0] - original[0]).abs() < 1e-6);
        assert!((memory.embedding[1] - original[1]).abs() < 1e-6);
        assert!((memory.embedding[EMBEDDING_DIMS - 1] - original[2]).abs() < 1e-6);
    }
}
