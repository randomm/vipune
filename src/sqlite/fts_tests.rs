//! FTS5 tests.
#[cfg(test)]
mod tests {
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
            "proj1",
            UpdateOptions {
                content: Some("updated text"),
                embedding: Some(embedding.as_slice()),
                metadata: None,
                memory_type: None,
                status: None,
            },
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

    // ====================================================================
    // FTS5 desync detection tests (Issue #193)
    // ====================================================================

    #[test]
    fn test_detect_fts_desync_fresh_zero_row_db_is_healthy() {
        let db = create_test_db();
        let report = db.detect_fts_desync().unwrap();
        assert!(report.underpopulated_by_project.is_empty());
        assert_eq!(report.underpopulated_global, 0);
        assert_eq!(report.orphans, 0);
        assert!(!report.is_desynced());
    }

    #[test]
    fn test_detect_fts_desync_healthy_trigger_synced_db_is_in_sync() {
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        for content in ["first memory", "second memory", "third memory"] {
            db.insert("proj1", content, &embedding, None, "fact", "active")
                .unwrap();
        }

        let report = db.detect_fts_desync().unwrap();
        assert!(!report.is_desynced());
        assert_eq!(report.underpopulated_global, 0);
        assert_eq!(report.orphans, 0);
        assert_eq!(report.underpopulated_by_project.len(), 0);
    }

    // NOTE: FTS5 external-content tables (content='memories') make it
    // difficult to create desync fixtures in unit tests. The triggers are
    // the primary sync mechanism, but dropping them and doing raw SQL
    // operations does NOT reliably create a detectable desync because
    // SELECT rowid FROM memories_fts reads from the content table, not
    // the FTS index. The desync detection is designed for production
    // scenarios where desync arises from trigger-bypassing code paths
    // (e.g., direct SQL manipulation, failed migrations, etc.).
    //
    // The tests below verify the detection method works correctly on
    // healthy databases and that the SQL queries are well-formed.
    // Comprehensive desync fixture tests would require either:
    // 1. Modifying the FTS5 schema to use contentless_rowid
    // 2. Using a mock FTS table
    // 3. Testing at the integration level with real desync scenarios

    #[test]
    fn test_detect_fts_desync_report_is_read_only() {
        // Verify that detection does not modify the database.
        let db = create_test_db();
        let embedding = vec![0.1f32; 384];
        db.insert("proj1", "row one", &embedding, None, "fact", "active")
            .unwrap();
        db.insert("proj1", "row two", &embedding, None, "fact", "active")
            .unwrap();

        let data_version_before: i64 = db
            .conn()
            .query_row("PRAGMA data_version", [], |row| row.get(0))
            .unwrap();

        let report = db.detect_fts_desync().unwrap();
        assert!(!report.is_desynced());

        let data_version_after: i64 = db
            .conn()
            .query_row("PRAGMA data_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(data_version_before, data_version_after);
    }
}
