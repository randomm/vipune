//! FTS5 tests.
#[cfg(test)]
mod tests {
    use super::super::Database;
    use tempfile::TempDir;

    fn create_test_db() -> Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    #[test]
    fn test_fts5_search() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "rust programming",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert("proj1", "python data", &embedding, None, "fact", "active")
            .unwrap();

        let results = db.search_bm25("rust", "proj1", 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("rust"));
    }

    #[test]
    fn test_fts5_triggers() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        let id = db
            .insert("proj1", "original text", &embedding, None, "fact", "active")
            .unwrap();

        assert_eq!(
            db.search_bm25("original", "proj1", 10, None, None)
                .unwrap()
                .len(),
            1
        );

        db.update(
            &id,
            Some("updated text"),
            Some(&embedding.as_slice()),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            db.search_bm25("updated", "proj1", 10, None, None)
                .unwrap()
                .len(),
            1
        );

        db.delete(&id, "proj1").unwrap();
        assert_eq!(
            db.search_bm25("updated", "proj1", 10, None, None)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn test_fts5_limit_validation() {
        let db = create_test_db();
        assert!(db.search_bm25("test", "proj1", 0, None, None).is_err());
        assert!(
            db.search_bm25("test", "proj1", 100_000, None, None)
                .is_err()
        );
    }

    #[test]
    fn test_fts5_special_characters() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "test with \"quotes\"",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert(
            "proj1",
            "test with 'apos'",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert(
            "proj1",
            "test with\\slash",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();

        assert_eq!(
            db.search_bm25("test with \"quotes\"", "proj1", 10, None, None)
                .unwrap()
                .len(),
            1
        );

        // Test that backslash in query is properly escaped
        assert_eq!(
            db.search_bm25("test with\\slash", "proj1", 10, None, None)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_fts5_empty_query() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        // Empty query returns no results
        let results = db.search_bm25("", "proj1", 10, None, None).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fts5_phrase_search() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "rust programming",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert(
            "proj1",
            "rust error handling",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();

        // Multi-word phrase should find matching content
        let results = db
            .search_bm25("rust programming", "proj1", 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("programming"));
    }

    #[test]
    fn test_fts5_unicode_text() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "café résumé 日本語",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();

        // Test basic Unicode matching
        let results = db.search_bm25("café", "proj1", 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("café"));
    }

    #[test]
    fn test_initialize_fts_migration() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        std::mem::forget(dir);

        {
            let db = Database::open(&path).unwrap();
            db.insert(
                "proj1",
                "before migration",
                &vec![0.1f32; 384],
                None,
                "fact",
                "active",
            )
            .unwrap();
        }

        {
            let db = Database::open(&path).unwrap();
            db.initialize_fts().unwrap();
            assert_eq!(
                db.search_bm25("before", "proj1", 10, None, None)
                    .unwrap()
                    .len(),
                1
            );
        }
    }

    #[test]
    fn test_initialize_fts_consistency_handling() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        std::mem::forget(dir);

        // Initial data with 3 rows
        {
            let db = Database::open(&path).unwrap();
            db.insert("proj1", "first", &vec![0.1f32; 384], None, "fact", "active")
                .unwrap();
            db.insert(
                "proj1",
                "second",
                &vec![0.1f32; 384],
                None,
                "fact",
                "active",
            )
            .unwrap();
            db.insert("proj1", "third", &vec![0.1f32; 384], None, "fact", "active")
                .unwrap();
        }

        // FTS migration
        {
            let db = Database::open(&path).unwrap();
            db.initialize_fts().unwrap();

            let fts_count: i64 = db
                .conn()
                .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))
                .unwrap();
            assert_eq!(fts_count, 3);
        }

        // Call initialize_fts again - should handle consistent state gracefully
        {
            let db = Database::open(&path).unwrap();
            db.initialize_fts().unwrap();

            assert_eq!(
                db.search_bm25("first", "proj1", 10, None, None)
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                db.search_bm25("second", "proj1", 10, None, None)
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                db.search_bm25("third", "proj1", 10, None, None)
                    .unwrap()
                    .len(),
                1
            );
        }
    }

    #[test]
    fn test_bm25_search_filters_by_status() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();
        db.insert(
            "proj1",
            "test content",
            &embedding,
            None,
            "fact",
            "superseded",
        )
        .unwrap();
        db.insert(
            "proj1",
            "test content",
            &embedding,
            None,
            "fact",
            "candidate",
        )
        .unwrap();

        // With explicit status filter
        let results = db
            .search_bm25("test", "proj1", 10, None, Some(&["active"]))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, "active");

        let results = db
            .search_bm25("test", "proj1", 10, None, Some(&["active", "candidate"]))
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bm25_search_default_excludes_non_active() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "active memory", &embedding, None, "fact", "active")
            .unwrap();
        db.insert(
            "proj1",
            "candidate memory",
            &embedding,
            None,
            "fact",
            "candidate",
        )
        .unwrap();
        db.insert(
            "proj1",
            "superseded memory",
            &embedding,
            None,
            "fact",
            "superseded",
        )
        .unwrap();

        // With statuses=None, should default to active only
        let results = db.search_bm25("memory", "proj1", 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, "active");
    }

    #[test]
    fn test_bm25_search_filters_by_type() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert(
            "proj1",
            "a fact about rust programming",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
        db.insert(
            "proj1",
            "a preference for python data",
            &embedding,
            None,
            "preference",
            "active",
        )
        .unwrap();
        db.insert(
            "proj1",
            "a procedure for testing code",
            &embedding,
            None,
            "procedure",
            "active",
        )
        .unwrap();

        // Filter by type - search for content that matches fact memory
        let results = db
            .search_bm25("rust", "proj1", 10, Some(&["fact"]), None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_type, "fact");

        // Search for python with multiple type filter
        let results = db
            .search_bm25("python", "proj1", 10, Some(&["fact", "preference"]), None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_type, "preference");
    }
}