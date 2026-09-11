//! Raw full-table scan for the export command (issue #195).
//!
//! `scan_all_rows` selects all 12 columns for EVERY project with no LIMIT and
//! keeps the embedding as the raw stored BLOB. It deliberately does NOT decode
//! the blob (`blob_to_vec`) and does NOT reuse `list()` (capped at
//! MAX_SEARCH_LIMIT) or `list_all_rows_for_project` (single project, decodes),
//! so a corrupt, short, or empty embedding BLOB exports faithfully and no
//! row-count cap can truncate a backup.
//!
use super::{Database, Result};

/// One row of the memories table, with the embedding kept as the raw
/// stored BLOB (undecoded). This is the shape the JSONL export consumes —
/// base64-encoding the blob directly is what makes the round-trip
/// byte-identity contract possible without touching vector semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRow {
    /// Global primary key (TEXT).
    pub id: String,
    /// Project this row belongs to.
    pub project_id: String,
    /// Memory content.
    pub content: String,
    /// User-provided JSON metadata (NULL → None).
    pub metadata: Option<String>,
    /// Raw embedding BLOB exactly as stored — no length or shape validation.
    pub embedding_blob: Vec<u8>,
    /// RFC3339 creation timestamp.
    pub created_at: String,
    /// RFC3339 last-update timestamp.
    pub updated_at: String,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    pub status: String,
    /// Id of the memory that superseded this one (NULL → None).
    pub superseded_by: Option<String>,
    /// Retrieval counter, restored verbatim on import.
    pub retrieval_count: i64,
    /// Last retrieval timestamp (NULL → None), restored verbatim on import.
    pub last_retrieved_at: Option<String>,
}

impl Database {
    /// Scan ALL rows across ALL projects, uncapped and unfiltered.
    ///
    /// - No LIMIT (unlike `list()`, which caps at MAX_SEARCH_LIMIT).
    /// - No status/type filter (superseded, candidate, deprecated all included).
    /// - Embedding kept as the raw BLOB — never decoded — so a corrupt or
    ///   short row cannot abort the scan.
    /// - Rows come back in stable rowid order.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails. Individual rows never error here.
    pub fn scan_all_rows(&self) -> Result<Vec<ExportRow>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
            FROM memories
            ORDER BY rowid
            "#,
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(ExportRow {
                id: row.get(0)?,
                project_id: row.get(1)?,
                content: row.get(2)?,
                metadata: row.get(3)?,
                embedding_blob: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
                memory_type: row.get(7)?,
                status: row.get(8)?,
                superseded_by: row.get(9)?,
                retrieval_count: row.get(10)?,
                last_retrieved_at: row.get(11)?,
            })
        })?;

        let mut results = Vec::new();
        for row_result in rows {
            results.push(row_result?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::embedding::{blob_to_vec, vec_to_blob};
    use crate::sqlite::query_mod::map_row_to_memory;
    use rusqlite::{Connection, params};
    use tempfile::TempDir;

    fn create_test_db() -> (TempDir, Database) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        (dir, db)
    }

    /// Insert a raw row with full column control, bypassing `insert()`.
    fn seed_raw(
        conn: &Connection,
        id: &str,
        project_id: &str,
        content: &str,
        embedding: Option<&[u8]>,
        retrieval_count: i64,
        last_retrieved_at: Option<&str>,
    ) {
        conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at)
            VALUES (?1, ?2, ?3, ?4, NULL, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active', NULL, ?5, ?6)
            "#,
            params![
                id,
                project_id,
                content,
                embedding,
                retrieval_count,
                last_retrieved_at,
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_scan_all_rows_returns_every_row_across_projects_uncapped() {
        let (_dir, db) = create_test_db();
        let conn = db.conn();
        let good_blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
        seed_raw(conn, "a1", "projA", "content A", Some(&good_blob), 0, None);
        seed_raw(
            conn,
            "b1",
            "projB",
            "content B",
            Some(&good_blob),
            3,
            Some("2024-02-01T00:00:00Z"),
        );
        seed_raw(conn, "b2", "projB", "content C", Some(&good_blob), 0, None);

        let rows = db.scan_all_rows().unwrap();
        assert_eq!(rows.len(), 3);
        let ids: Vec<_> = rows.iter().map(|r| r.id.clone()).collect();
        assert_eq!(
            ids,
            vec!["a1".to_string(), "b1".to_string(), "b2".to_string()]
        );

        let b1 = &rows[1];
        assert_eq!(b1.project_id, "projB");
        assert_eq!(b1.content, "content B");
        assert_eq!(b1.embedding_blob, good_blob);
        assert_eq!(b1.retrieval_count, 3);
        assert_eq!(
            b1.last_retrieved_at,
            Some("2024-02-01T00:00:00Z".to_string())
        );
        assert_eq!(b1.created_at, "2024-01-01T00:00:00Z");
        assert_eq!(b1.updated_at, "2024-01-01T00:00:00Z");
        assert_eq!(b1.memory_type, "fact");
        assert_eq!(b1.status, "active");
        assert_eq!(b1.superseded_by, None);
        assert_eq!(b1.metadata, None);
    }

    #[test]
    fn test_scan_all_rows_exports_corrupt_and_empty_blobs_without_error() {
        let (_dir, db) = create_test_db();
        let conn = db.conn();
        let good_blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
        seed_raw(conn, "good", "p", "g", Some(&good_blob), 0, None);
        // 1535-byte blob: wrong length for 384xf32 — would fail in map_row_to_memory.
        seed_raw(conn, "short", "p", "s", Some(&vec![0xABu8; 1535]), 0, None);
        // Empty blob: exported faithfully as an empty BLOB.
        seed_raw(conn, "empty", "p", "e", Some(&[]), 0, None);

        let rows = db.scan_all_rows().unwrap();
        assert_eq!(rows.len(), 3);

        let by_id = |id: &str| rows.iter().find(|r| r.id == id).unwrap();
        assert_eq!(by_id("good").embedding_blob, good_blob);
        assert_eq!(by_id("short").embedding_blob, vec![0xABu8; 1535]);
        assert!(by_id("empty").embedding_blob.is_empty());
    }

    #[test]
    fn test_scan_all_rows_does_not_use_capped_list_path() {
        // The scan must return every row with no LIMIT — pin the shape of the
        // contract by checking row count equals the table's true COUNT(*).
        let (_dir, db) = create_test_db();
        let conn = db.conn();
        let good_blob = vec_to_blob(&vec![0.1f32; 384]).unwrap();
        for i in 0..250 {
            seed_raw(
                conn,
                &format!("id-{i}"),
                "p",
                &format!("content {i}"),
                Some(&good_blob),
                0,
                None,
            );
        }

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 250);
        assert_eq!(db.scan_all_rows().unwrap().len(), 250);
    }

    #[test]
    fn test_scan_all_rows_includes_non_active_statuses() {
        let (_dir, db) = create_test_db();
        let conn = db.conn();
        let good_blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
        seed_raw(conn, "s1", "p", "c", Some(&good_blob), 0, None);
        conn.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = ?1 WHERE id = 's1'",
            ["s2"],
        )
        .unwrap();

        let rows = db.scan_all_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "superseded");
        assert_eq!(rows[0].superseded_by, Some("s2".to_string()));
    }

    #[test]
    fn test_insert_with_id_restores_all_12_columns_verbatim() {
        let (_dir, db) = create_test_db();
        let blob = vec_to_blob(&vec![0.25f32; 384]).unwrap();

        db.insert_with_id(
            "fixed-id-1",
            "projX",
            "restored content",
            &blob,
            Some(r#"{"k":"v"}"#),
            "2023-05-05T05:05:05Z",
            "2023-06-06T06:06:06Z",
            "guard",
            "candidate",
            Some("other-id"),
            42,
            Some("2023-07-07T07:07:07Z"),
        )
        .unwrap();

        // Verify every column via the canonical 12-column mapper (which decodes
        // the blob and would fail on any shape mismatch) plus a raw BLOB check.
        let conn = db.conn();
        let memory: crate::sqlite::Memory = conn
            .query_row(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
                FROM memories WHERE id = 'fixed-id-1'
                "#,
                [],
                map_row_to_memory,
            )
            .unwrap();
        assert_eq!(memory.id, "fixed-id-1");
        assert_eq!(memory.project_id, "projX");
        assert_eq!(memory.content, "restored content");
        assert_eq!(memory.metadata, Some(r#"{"k":"v"}"#.to_string()));
        assert_eq!(memory.created_at, "2023-05-05T05:05:05Z");
        assert_eq!(memory.updated_at, "2023-06-06T06:06:06Z");
        assert_eq!(memory.memory_type, "guard");
        assert_eq!(memory.status, "candidate");
        assert_eq!(memory.superseded_by, Some("other-id".to_string()));
        assert_eq!(memory.retrieval_count, 42);
        assert_eq!(
            memory.last_retrieved_at,
            Some("2023-07-07T07:07:07Z".to_string())
        );
        // Raw BLOB byte-identity: what was written is exactly what is stored.
        let stored: Vec<u8> = conn
            .query_row(
                "SELECT embedding FROM memories WHERE id = 'fixed-id-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, blob);
        assert_eq!(blob_to_vec(&stored).unwrap(), vec![0.25f32; 384]);
    }

    #[test]
    fn test_insert_with_id_duplicate_id_errors_without_upserting() {
        let (_dir, db) = create_test_db();
        let blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
        db.insert_with_id(
            "dup-id",
            "p",
            "original",
            &blob,
            None,
            "2023-01-01T00:00:00Z",
            "2023-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        .unwrap();

        let result = db.insert_with_id(
            "dup-id",
            "p",
            "replacement",
            &blob,
            None,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "guard",
            "candidate",
            None,
            9,
            None,
        );
        assert!(result.is_err());

        // Original row is untouched — no upsert.
        let (content, count): (String, i64) = db
            .conn()
            .query_row(
                "SELECT content, retrieval_count FROM memories WHERE id = 'dup-id'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(content, "original");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_insert_with_id_writes_raw_blob_undecoded() {
        // A non-1536-byte blob must still be storable raw — the shape
        // contract is enforced by the caller (import), not by this method.
        let (_dir, db) = create_test_db();
        let weird = vec![0x99u8; 1535];
        db.insert_with_id(
            "raw-1",
            "p",
            "c",
            &weird,
            None,
            "2023-01-01T00:00:00Z",
            "2023-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        .unwrap();
        let stored: Vec<u8> = db
            .conn()
            .query_row(
                "SELECT embedding FROM memories WHERE id = 'raw-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, weird);
    }
}
