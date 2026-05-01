//! SQLite migration and advanced query tests.
#[cfg(test)]
mod migration_tests {
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

        let results = db
            .list_since("proj1", "2024-01-01T12:00:00Z", 10, None, None)
            .unwrap();
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

        let results = db
            .list_since("proj1", "2024-01-01T00:00:00Z", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 0);

        let results = db
            .list_since("proj1", "2023-12-31T23:59:59Z", 10, None, None)
            .unwrap();
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

        let results = db
            .list_since("proj1", "2024-01-01T00:00:00Z", 2, None, None)
            .unwrap();
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

        let results = db
            .list_since("proj1", "2023-12-31T00:00:00Z", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, id3);
        assert_eq!(results[1].id, id2);
        assert_eq!(results[2].id, id1);
    }

    #[test]
    fn test_get_many_basic() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db
            .insert("proj1", "content 1", &embedding, None, "fact", "active")
            .unwrap();
        let id2 = db
            .insert("proj1", "content 2", &embedding, None, "fact", "active")
            .unwrap();

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

        let id1 = db
            .insert("proj1", "first", &embedding, None, "fact", "active")
            .unwrap();
        let id2 = db
            .insert("proj1", "second", &embedding, None, "fact", "active")
            .unwrap();
        let id3 = db
            .insert("proj1", "third", &embedding, None, "fact", "active")
            .unwrap();

        let results = db.get_many(&[&id3, &id1, &id2]).unwrap();
        assert_eq!(results[0].as_ref().unwrap().id, id3);
        assert_eq!(results[1].as_ref().unwrap().id, id1);
        assert_eq!(results[2].as_ref().unwrap().id, id2);
    }

    #[test]
    fn test_get_many_with_missing_ids() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db
            .insert("proj1", "content 1", &embedding, None, "fact", "active")
            .unwrap();
        let id2 = db
            .insert("proj1", "content 2", &embedding, None, "fact", "active")
            .unwrap();

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

        let id = db
            .insert("proj1", "content", &embedding, None, "fact", "active")
            .unwrap();

        let results = db.get_many(&[&id]).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_some());
        assert_eq!(results[0].as_ref().unwrap().content, "content");
    }

    #[test]
    fn test_get_many_with_duplicate_ids() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db
            .insert("proj1", "content 1", &embedding, None, "fact", "active")
            .unwrap();
        let id2 = db
            .insert("proj1", "content 2", &embedding, None, "fact", "active")
            .unwrap();

        let results = db.get_many(&[&id1, &id2, &id1, &id2]).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].as_ref().unwrap().id, id1);
        assert_eq!(results[1].as_ref().unwrap().id, id2);
        assert_eq!(results[2].as_ref().unwrap().id, id1);
        assert_eq!(results[3].as_ref().unwrap().id, id2);
    }

    #[test]
    fn test_list_since_with_timezone_offset() {
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

        let results = db
            .list_since("proj1", "2024-01-01T11:00:00+01:00", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "new");

        let results = db
            .list_since("proj1", "2024-01-01T19:00:00-05:00", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "new");

        let results = db
            .list_since("proj1", "2024-01-02T00:00:00Z", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_list_since_timestamp_precision_equivalence() {
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

        let results1 = db
            .list_since("proj1", "2023-12-31T23:59:59Z", 10, None, None)
            .unwrap();
        let results2 = db
            .list_since("proj1", "2023-12-31T23:59:59.000Z", 10, None, None)
            .unwrap();

        assert_eq!(results1.len(), results2.len());
        if results1.len() > 0 {
            assert_eq!(results1[0].id, results2[0].id);
        }

        let results3 = db
            .list_since("proj1", "2023-12-31T23:59:59.999Z", 10, None, None)
            .unwrap();
        assert_eq!(results3.len(), 1);
    }

    #[test]
    fn test_list_regression_coverage() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];

        let id1 = db
            .insert("proj1", "first", &embedding, None, "fact", "active")
            .unwrap();
        let id2 = db
            .insert("proj1", "second", &embedding, None, "fact", "active")
            .unwrap();
        let id3 = db
            .insert("proj1", "third", &embedding, None, "fact", "active")
            .unwrap();

        let results = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, id3);
        assert_eq!(results[1].id, id2);
        assert_eq!(results[2].id, id1);

        let results = db.list("proj1", 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id3);

        db.insert("proj2", "other", &embedding, None, "fact", "active")
            .unwrap();
        let proj1_results = db.list("proj1", 10, None, None).unwrap();
        assert_eq!(proj1_results.len(), 3);

        let empty_results = db.list("nonexistent", 10, None, None).unwrap();
        assert_eq!(empty_results.len(), 0);
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
        assert_eq!(memories[0].id, id2);
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
}