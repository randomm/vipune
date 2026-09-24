//! Tests for `vipune export` — JSONL writer, handler, and base64 codec.
//!
//! Follows the reindex_tests.rs pattern: seeded temp DB via Database::open,
//! raw blob control, no ONNX model.

#![cfg(test)]

use crate::commands::export::{
    EXPORT_FORMAT_VERSION, ExportResponse, base64_decode, base64_encode, handle_export, write_jsonl,
};
use crate::sqlite::Database;
use crate::sqlite::embedding::vec_to_blob;
use crate::sqlite::export_scan::ExportRow;
use serde_json::Value;
use std::process::ExitCode;

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn seed_raw(
    conn: &rusqlite::Connection,
    id: &str,
    project_id: &str,
    content: &str,
    embedding: Option<&[u8]>,
    retrieval_count: i64,
    last_retrieved_at: Option<&str>,
) {
    // The memories table declares `embedding BLOB NOT NULL`, so a "NULL"
    // embedding is represented as an empty BLOB (the scan reads both back as
    // an empty Vec<u8>, which the export serialises as base64 "").
    conn.execute(
        r#"
        INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at)
        VALUES (?1, ?2, ?3, ?4, NULL, '2024-01-01T00:00:00Z', '2024-01-02T00:00:00Z', 'fact', 'active', NULL, ?5, ?6)
        "#,
        rusqlite::params![
            id,
            project_id,
            content,
            embedding.unwrap_or(&[]),
            retrieval_count,
            last_retrieved_at,
        ],
    )
    .unwrap();
}

/// Run `handle_export` on the given db path and parse every JSONL line back
/// into values.
fn export_lines(db_path: &std::path::Path, out_path: &std::path::Path) -> Vec<Value> {
    let code = handle_export(db_path, out_path, None, true).expect("handle_export ok");
    assert_eq!(code, ExitCode::SUCCESS);
    let text = std::fs::read_to_string(out_path).expect("read export file");
    text.lines()
        .map(serde_json::from_str)
        .collect::<Result<Vec<Value>, _>>()
        .expect("every line must be valid JSON")
}

#[test]
fn test_base64_roundtrip_full_blob() {
    let blob = vec_to_blob(&vec![0.25f32; 384]).unwrap();
    assert_eq!(blob.len(), crate::embedding::EMBEDDING_DIMS * 4);
    let encoded = base64_encode(&blob);
    assert_eq!(base64_decode(&encoded).unwrap(), blob);
    // Standard base64 of a full blob is 4/3 of the byte length, no padding.
    assert_eq!(encoded.len(), (blob.len() * 4) / 3);
    assert!(!encoded.contains('='));
}

#[test]
fn test_base64_empty_string_is_empty() {
    // base64 of an empty BLOB must be the empty string (NULL/empty blob contract)
    assert!(base64_encode(&[]).is_empty());
    assert_eq!(base64_decode("").unwrap(), Vec::<u8>::new());
}

#[test]
fn test_base64_decode_rejects_bad_input() {
    assert!(base64_decode("!!!").is_err()); // bad chars
    assert!(base64_decode("A").is_err()); // len % 4 != 0
    assert!(base64_decode("AB=CD").is_err()); // misplaced padding
}

#[test]
fn test_jsonl_header_fields_and_row_count() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
    seed_raw(conn, "r1", "projA", "one", Some(&blob), 0, None);
    seed_raw(
        conn,
        "r2",
        "projB",
        "two",
        Some(&blob),
        7,
        Some("2024-03-01T00:00:00Z"),
    );
    seed_raw(conn, "r3", "projA", "three", None, 0, None);

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);

    // Header + 3 data rows = 4 lines; header is NOT counted in `rows`.
    assert_eq!(lines.len(), 4);

    let header = &lines[0];
    assert_eq!(header["type"], "export");
    assert_eq!(header["format_version"], 1);
    assert_eq!(header["embedding_dims"], 384);
    // No identity row in a fresh store: the header records the default
    // bge identity at its pinned revision.
    assert_eq!(header["model_id"], crate::embedding::EMBED_MODEL_ID);
    assert_eq!(
        header["model_revision"],
        crate::embedding::EMBED_MODEL_REVISION
    );
    assert_eq!(header["rows"], 3);
    assert!(
        header["exported_at"].as_str().unwrap().contains("T"),
        "exported_at must be RFC3339: {}",
        header["exported_at"]
    );

    // Data rows carry all 12 fields — `memory_type`, never `type`.
    let row = &lines[1];
    assert_eq!(
        row.as_object().unwrap().len(),
        12,
        "row line must have exactly 12 fields: {row}"
    );
    assert!(!row.as_object().unwrap().contains_key("type"));
    assert_eq!(row["memory_type"], "fact");
    assert_eq!(row["id"], "r1");
    assert_eq!(row["project_id"], "projA");
}

#[test]
fn test_jsonl_embedding_is_base64_of_exact_blob() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.125f32; 384]).unwrap();
    seed_raw(conn, "emb", "p", "content", Some(&blob), 0, None);

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);
    let row = &lines[1];

    let encoded = row["embedding"]
        .as_str()
        .expect("embedding must be a string");
    // Decoded bytes must be byte-identical to the stored BLOB.
    assert_eq!(base64_decode(encoded).unwrap(), blob);
    // Sanity: standard base64 of the exact 1536-byte blob.
    assert_eq!(encoded.len(), 2048);
}

#[test]
fn test_jsonl_null_and_empty_blobs_export_as_empty_string() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    seed_raw(conn, "null-blob", "p", "a", None, 0, None);
    seed_raw(conn, "empty-blob", "p", "b", Some(&[]), 0, None);

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);

    let by_id = |id: &str| lines.iter().find(|v| v["id"] == id).unwrap();
    assert_eq!(by_id("null-blob")["embedding"], "");
    assert_eq!(by_id("empty-blob")["embedding"], "");
}

#[test]
fn test_jsonl_row_fields_restored_verbatim() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
    seed_raw(
        conn,
        "full",
        "projZ",
        "the content",
        Some(&blob),
        42,
        Some("2024-05-05T05:00:00Z"),
    );
    // Metadata + superseded_by via a direct update.
    conn.execute(
        "UPDATE memories SET metadata = ?, superseded_by = 'other' WHERE id = 'full'",
        [r#"{"k":"v"}"#],
    )
    .unwrap();

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);
    let row = lines
        .iter()
        .find(|v| v["id"] == "full")
        .expect("row present");

    assert_eq!(row["project_id"], "projZ");
    assert_eq!(row["content"], "the content");
    assert_eq!(row["metadata"], r#"{"k":"v"}"#);
    assert_eq!(row["created_at"], "2024-01-01T00:00:00Z");
    assert_eq!(row["updated_at"], "2024-01-02T00:00:00Z");
    assert_eq!(row["memory_type"], "fact");
    assert_eq!(row["status"], "active");
    assert_eq!(row["superseded_by"], "other");
    assert_eq!(row["retrieval_count"], 42);
    assert_eq!(row["last_retrieved_at"], "2024-05-05T05:00:00Z");
}

#[test]
fn test_jsonl_content_with_special_chars_roundtrips() {
    // Content containing quotes, newlines, and unicode must survive JSON
    // serialization intact (a naive string concat would break the file).
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
    let tricky = "line1 \"quoted\"\nline2 — unicode ✓ é";
    seed_raw(conn, "tricky", "p", tricky, Some(&blob), 0, None);

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);
    let row = lines.iter().find(|v| v["id"] == "tricky").unwrap();
    assert_eq!(row["content"], tricky);
}

#[test]
fn test_jsonl_lf_line_endings_and_line_count() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
    for i in 0..5 {
        seed_raw(
            conn,
            &format!("row-{i}"),
            "p",
            &format!("c{i}"),
            Some(&blob),
            0,
            None,
        );
    }

    let out = _dir.path().join("export.jsonl");
    export_lines(&db_path, &out);

    let bytes = std::fs::read(&out).expect("read bytes");
    let text = String::from_utf8(bytes).unwrap();
    // 1 header + 5 rows = 6 LF-terminated lines; no CRLF anywhere.
    assert_eq!(text.matches('\n').count(), 6);
    assert!(!text.contains('\r'), "no CRLF line endings");
    assert!(text.ends_with('\n'), "file must end with a final LF");
}

#[test]
fn test_export_uncapped_matches_true_row_count() {
    // The export must contain EVERY row — no silent truncation from a
    // MAX_SEARCH_LIMIT-style cap. 250 rows keeps this fast while proving the
    // scan does not go through the capped list() path.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.1f32; 384]).unwrap();
    for i in 0..250 {
        seed_raw(
            conn,
            &format!("id-{i}"),
            "p",
            &format!("content {i}"),
            Some(&blob),
            0,
            None,
        );
    }
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 250);

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);
    assert_eq!(lines.len(), 251, "header + all 250 rows");
    assert_eq!(lines[0]["rows"], 250);
}

#[test]
fn test_export_handler_json_response_shape() {
    // Field names of the response struct must be exactly: rows, format_version, path.
    let response = ExportResponse {
        rows: 3,
        format_version: EXPORT_FORMAT_VERSION,
        path: "/tmp/out.jsonl".to_string(),
    };
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("\"rows\":3"));
    assert!(json.contains("\"format_version\":1"));
    assert!(json.contains("\"path\":\"/tmp/out.jsonl\""));
}

#[test]
fn test_export_handler_human_mode_returns_success() {
    let (_dir, db_path) = create_test_db();
    let out = _dir.path().join("out.jsonl");
    let code = handle_export(&db_path, &out, Some("ignored-project"), false).expect("export ok");
    assert_eq!(code, ExitCode::SUCCESS);
    assert!(out.exists());
}

#[test]
fn test_write_jsonl_returns_row_count_not_header() {
    let blob = vec![1u8, 2, 3, 4];
    let rows = vec![
        ExportRow {
            id: "a".to_string(),
            project_id: "p".to_string(),
            content: "c".to_string(),
            metadata: None,
            embedding_blob: blob.clone(),
            created_at: "t".to_string(),
            updated_at: "t".to_string(),
            memory_type: "fact".to_string(),
            status: "active".to_string(),
            superseded_by: None,
            retrieval_count: 0,
            last_retrieved_at: None,
        },
        ExportRow {
            id: "b".to_string(),
            project_id: "p".to_string(),
            content: "c2".to_string(),
            metadata: Some("m".to_string()),
            embedding_blob: blob,
            created_at: "t".to_string(),
            updated_at: "t".to_string(),
            memory_type: "guard".to_string(),
            status: "candidate".to_string(),
            superseded_by: Some("x".to_string()),
            retrieval_count: 5,
            last_retrieved_at: Some("t".to_string()),
        },
    ];
    let mut buf = Vec::new();
    let count = write_jsonl(&mut buf, &rows, "2024-01-01T00:00:00Z", "model-a", "rev-a").unwrap();
    assert_eq!(count, 2);
    let text = String::from_utf8(buf).unwrap();
    assert_eq!(text.lines().count(), 3, "header + 2 rows");
    let header: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(header["model_id"], "model-a");
    assert_eq!(header["model_revision"], "rev-a");
}

#[test]
fn test_export_corrupt_blob_row_does_not_abort() {
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let good = vec_to_blob(&vec![0.5f32; 384]).unwrap();
    seed_raw(conn, "ok", "p", "fine", Some(&good), 0, None);
    // 1535-byte blob: would fail in map_row_to_memory, must NOT fail here.
    seed_raw(
        conn,
        "bad",
        "p",
        "corrupt",
        Some(&vec![0xAB; 1535]),
        0,
        None,
    );

    let out = _dir.path().join("export.jsonl");
    let code = handle_export(&db_path, &out, None, true).expect("export must not abort");
    assert_eq!(code, ExitCode::SUCCESS);

    let lines = std::fs::read_to_string(&out)
        .unwrap()
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<Vec<Value>, _>>()
        .unwrap();
    assert_eq!(lines.len(), 3);
    let bad = lines.iter().find(|v| v["id"] == "bad").unwrap();
    let decoded = base64_decode(bad["embedding"].as_str().unwrap()).unwrap();
    assert_eq!(decoded, vec![0xAB; 1535]);
}

#[test]
fn test_export_header_records_recorded_identity() {
    let (_dir, db_path) = create_test_db();
    {
        let db = Database::open(&db_path).unwrap();
        let id = crate::sqlite::identity::ModelIdentity {
            model_id: "e5-model".to_string(),
            revision: "e5-rev".to_string(),
        };
        crate::sqlite::identity::record_identity_and_clear_marker(db.conn(), &id).unwrap();
    }
    let db = Database::open(&db_path).unwrap();
    let conn = db.conn();
    let blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
    seed_raw(conn, "r1", "p", "c", Some(&blob), 0, None);

    let out = _dir.path().join("export.jsonl");
    let lines = export_lines(&db_path, &out);
    let header = &lines[0];
    assert_eq!(header["model_id"], "e5-model");
    assert_eq!(header["model_revision"], "e5-rev");
}
