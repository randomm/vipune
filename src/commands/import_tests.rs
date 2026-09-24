//! Tests for `vipune import` handler (issue #195).
//!
//! Covers the all-or-nothing transaction, skip-and-count of existing ids,
//! line-number error reporting, CRLF tolerance, and the raw-BLOB
//! byte-identity round-trip.

#![cfg(test)]

use crate::commands::{ImportResponse, import};
use base64::Engine;
use std::process::ExitCode;

/// 384 × f32 embedding whose raw little-endian BLOB we base64-encode for the
/// JSONL `embedding` field. Using a non-uniform vector (0..384 as f32) makes
/// the round-trip assertion meaningful — a constant vector would round-trip
/// trivially.
fn test_embedding() -> Vec<f32> {
    (0..384).map(|i| (i as f32) * 0.001).collect()
}

fn blob_of(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|&x| x.to_le_bytes()).collect()
}

fn b64(blob: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(blob)
}

#[allow(clippy::too_many_arguments)] // mirrors the 12 JSONL row fields
fn make_row_json(
    id: &str,
    project_id: &str,
    content: &str,
    metadata: Option<&str>,
    blob: &[u8],
    created_at: &str,
    updated_at: &str,
    memory_type: &str,
    status: &str,
    superseded_by: Option<&str>,
    retrieval_count: i64,
    last_retrieved_at: Option<&str>,
) -> String {
    let mut s = format!(
        r#"{{"id":"{}","project_id":"{}","content":"{}","#,
        id, project_id, content
    );
    // metadata is a STRING field in the JSONL schema (the JSON metadata object
    // is carried as a string, per the DB TEXT column) — serialize it as a JSON
    // string rather than inlining the raw JSON.
    match metadata {
        Some(m) => {
            s.push_str(&format!(
                "\"metadata\":{},",
                serde_json::to_string(m).unwrap()
            ));
        }
        None => s.push_str("\"metadata\":null,"),
    }
    s.push_str(&format!("\"embedding\":\"{}\",", b64(blob)));
    s.push_str(&format!(
        "\"created_at\":\"{}\",\"updated_at\":\"{}\",",
        created_at, updated_at
    ));
    s.push_str(&format!(
        "\"memory_type\":\"{}\",\"status\":\"{}\",",
        memory_type, status
    ));
    match superseded_by {
        Some(sb) => s.push_str(&format!("\"superseded_by\":\"{}\",", sb)),
        None => s.push_str("\"superseded_by\":null,"),
    }
    s.push_str(&format!("\"retrieval_count\":{},", retrieval_count));
    match last_retrieved_at {
        Some(lr) => s.push_str(&format!("\"last_retrieved_at\":\"{}\"", lr)),
        None => s.push_str("\"last_retrieved_at\":null"),
    }
    s.push('}');
    s
}

fn header(rows: usize) -> String {
    format!(
        r#"{{"type":"export","version":1,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":{}}}"#,
        rows
    )
}

fn header_with_identity(rows: usize, model_id: &str, model_revision: &str) -> String {
    format!(
        r#"{{"type":"export","version":1,"embedding_dims":384,"model_id":"{model_id}","model_revision":"{model_revision}","exported_at":"2024-01-01T00:00:00Z","rows":{rows}}}"#
    )
}

fn db_path_of(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("test.db")
}

/// Record a model identity on the destination database.
/// Record a model identity on the destination database. The caller is
/// responsible for recording a pair that the identity resolver
/// (`configured_identity`) can also produce — for the default model that is
/// its pinned revision; arbitrary pairs are only comparable to an export
/// header carrying the same pair.
fn set_identity(db_path: &std::path::Path, model_id: &str, revision: &str) {
    let db = crate::sqlite::Database::open(db_path).unwrap();
    let id = crate::sqlite::model_identity::ModelIdentity {
        model_id: model_id.to_string(),
        revision: revision.to_string(),
    };
    crate::sqlite::identity::record_identity_and_clear_marker(db.conn(), &id.into()).unwrap();
}

fn import_stdout_err(source_content: &str, db_path: &std::path::Path) -> Option<String> {
    use crate::commands::import::handle_import;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), source_content).unwrap();
    let result = handle_import(db_path, Some(tmp.path().to_str().unwrap()), None, false);
    match result {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    }
}

fn make_db() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    crate::sqlite::Database::open(&path).unwrap();
    dir
}

fn raw_row_blob(db_path: &std::path::Path, id: &str) -> Vec<u8> {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row("SELECT embedding FROM memories WHERE id = ?", [id], |r| {
        r.get::<_, Vec<u8>>(0)
    })
    .unwrap()
}

fn row_count(db_path: &std::path::Path) -> i64 {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap()
}

#[test]
fn test_import_inserts_row_byte_identical() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());

    let jsonl = format!("{}\n", header(1))
        + &make_row_json(
            "id-1",
            "proj",
            "content one",
            Some(r#"{"k":"v"}"#),
            &blob,
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "fact",
            "active",
            None,
            7,
            Some("2024-01-03T00:00:00Z"),
        )
        + "\n";
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let resp = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(resp.inserted, 1);
    assert_eq!(resp.skipped, 0);

    // Raw BLOB byte-identity (not cosine, not decoded vec).
    assert_eq!(raw_row_blob(&db_path, "id-1"), blob);

    // All other 12 columns verbatim.
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let row = conn
        .query_row(
            "SELECT project_id, content, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at FROM memories WHERE id = 'id-1'",
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
    assert_eq!(row.1, "content one");
    assert_eq!(row.2, Some(r#"{"k":"v"}"#.to_string()));
    assert_eq!(row.3, "2024-01-01T00:00:00Z");
    assert_eq!(row.4, "2024-01-02T00:00:00Z");
    assert_eq!(row.5, "fact");
    assert_eq!(row.6, "active");
    assert_eq!(row.7, None);
    assert_eq!(row.8, 7);
    assert_eq!(row.9, Some("2024-01-03T00:00:00Z".to_string()));
}

#[test]
fn test_import_skips_existing_ids_and_counts() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let db = crate::sqlite::Database::open(&db_path).unwrap();
    let existing_id = db
        .insert(
            "proj",
            "existing",
            &test_embedding(),
            None,
            "fact",
            "active",
        )
        .unwrap();

    let blob = blob_of(&test_embedding());
    let jsonl = format!("{}\n", header(2))
        + &make_row_json(
            &existing_id,
            "proj",
            "existing",
            None,
            &blob,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        + "\n"
        + &make_row_json(
            "new-id",
            "proj",
            "new",
            None,
            &blob,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        + "\n";
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let resp = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(resp.inserted, 1);
    assert_eq!(resp.skipped, 1);
    assert_eq!(row_count(&db_path), 2);
}

#[test]
fn test_import_re_run_is_idempotent() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());
    let jsonl = format!("{}\n", header(1))
        + &make_row_json(
            "id-1",
            "proj",
            "content",
            None,
            &blob,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        + "\n";
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let first = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(first.inserted, 1);
    assert_eq!(first.skipped, 0);

    let second = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(second.inserted, 0);
    assert_eq!(second.skipped, 1);
    assert_eq!(row_count(&db_path), 1);
}

#[test]
fn test_import_malformed_line_aborts_all_or_nothing() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());

    // Line 2 good, line 3 malformed JSON.
    let jsonl = format!("{}\n", header(2))
        + &make_row_json(
            "id-1",
            "proj",
            "good",
            None,
            &blob,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        + "\n"
        + "not valid json\n";
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let before = row_count(&db_path);
    let result = import::run_import(&db_path, file.to_str().unwrap());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("line 3"),
        "error must name the offending line: {}",
        msg
    );
    // All-or-nothing: zero rows written.
    assert_eq!(row_count(&db_path), before);
}

#[test]
fn test_import_wrong_dimension_embedding_aborts() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");

    // Wrong-dimension blob: 300 f32 = 1200 bytes (not 1536).
    let short_vec: Vec<f32> = vec![0.1; 300];
    let short_blob = blob_of(&short_vec);

    let jsonl = format!("{}\n", header(1))
        + &make_row_json(
            "id-1",
            "proj",
            "content",
            None,
            &short_blob,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        )
        + "\n";
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let result = import::run_import(&db_path, file.to_str().unwrap());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    let expected_bytes = crate::embedding::EMBEDDING_DIMS * 4;
    assert!(
        msg.contains("wrong-dimension") || msg.contains(&expected_bytes.to_string()),
        "got: {}",
        msg
    );
    assert_eq!(row_count(&db_path), 0);
}

#[test]
fn test_import_corrupt_base64_aborts() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");

    // "!!!" is not valid base64 at all.
    let jsonl = format!(
        "{}\n{{\"id\":\"id-1\",\"project_id\":\"proj\",\"content\":\"c\",\"metadata\":null,\"embedding\":\"!!!\",\"created_at\":\"2024-01-01T00:00:00Z\",\"updated_at\":\"2024-01-01T00:00:00Z\",\"memory_type\":\"fact\",\"status\":\"active\",\"superseded_by\":null,\"retrieval_count\":0,\"last_retrieved_at\":null}}\n",
        header(1)
    );
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let result = import::run_import(&db_path, file.to_str().unwrap());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("invalid base64") || msg.contains("line 2"),
        "expected base64/line error, got: {}",
        msg
    );
    assert_eq!(row_count(&db_path), 0);
}

#[test]
fn test_import_empty_embedding_aborts() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");

    let jsonl = format!(
        "{}\n{{\"id\":\"id-1\",\"project_id\":\"proj\",\"content\":\"c\",\"metadata\":null,\"embedding\":\"\",\"created_at\":\"2024-01-01T00:00:00Z\",\"updated_at\":\"2024-01-01T00:00:00Z\",\"memory_type\":\"fact\",\"status\":\"active\",\"superseded_by\":null,\"retrieval_count\":0,\"last_retrieved_at\":null}}\n",
        header(1)
    );
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let result = import::run_import(&db_path, file.to_str().unwrap());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("empty embedding"), "got: {}", msg);
    assert_eq!(row_count(&db_path), 0);
}

#[test]
fn test_import_tolerates_crlf_line_endings() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());

    let mut row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    // CRLF-terminate each line.
    let mut jsonl = header(1) + "\r\n";
    row.push('\r');
    row.push('\n');
    jsonl.push_str(&row);
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let resp = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(resp.inserted, 1);
    assert_eq!(resp.skipped, 0);
    assert_eq!(raw_row_blob(&db_path, "id-1"), blob);
}

#[test]
fn test_import_header_not_counted_in_rows() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());

    // 3 data rows + 1 header = 4 lines total.
    let mut jsonl = format!("{}\n", header(3));
    for i in 0..3 {
        let row = make_row_json(
            &format!("id-{}", i),
            "proj",
            "content",
            None,
            &blob,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "fact",
            "active",
            None,
            0,
            None,
        );
        jsonl.push_str(&row);
        jsonl.push('\n');
    }
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let resp = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(resp.inserted, 3, "header must not count as a data row");
    assert_eq!(row_count(&db_path), 3);
}

#[test]
fn test_import_missing_header_type_fails() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());

    // Header with no type field.
    let bad_header =
        r#"{"version":1,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#;
    let mut jsonl = bad_header.to_string() + "\n";
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    jsonl.push_str(&row);
    jsonl.push('\n');
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let result = import::run_import(&db_path, file.to_str().unwrap());
    assert!(result.is_err());
    assert_eq!(row_count(&db_path), 0);
}

#[test]
fn test_import_multiple_projects() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());

    let row1 = make_row_json(
        "id-1",
        "proj-a",
        "content a",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let row2 = make_row_json(
        "id-2",
        "proj-b",
        "content b",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!("{}\n{}\n{}\n", header(2), row1, row2);
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let resp = import::run_import(&db_path, file.to_str().unwrap()).unwrap();
    assert_eq!(resp.inserted, 2);
    assert_eq!(row_count(&db_path), 2);

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let projects: Vec<String> = conn
        .prepare("SELECT DISTINCT project_id FROM memories ORDER BY project_id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(projects, vec!["proj-a", "proj-b"]);
}

#[test]
fn test_handle_import_json_and_human_paths() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");
    let blob = blob_of(&test_embedding());
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!("{}\n{}\n", header(1), row);
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    // JSON path returns SUCCESS.
    let exit = import::handle_import(&db_path, Some(file.to_str().unwrap()), None, true).unwrap();
    assert_eq!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_handle_import_failure_returns_error() {
    let dir = make_db();
    let db_path = dir.path().join("test.db");

    // Empty file (no header) should fail.
    let file = dir.path().join("empty.jsonl");
    std::fs::write(&file, "").unwrap();

    let result = import::handle_import(&db_path, Some(file.to_str().unwrap()), None, false);
    assert!(result.is_err());
    assert_eq!(row_count(&db_path), 0);
}

#[test]
fn test_import_response_serializes_with_inserted_and_skipped() {
    let resp = ImportResponse {
        inserted: 5,
        skipped: 2,
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"inserted\":5"), "got: {}", json);
    assert!(json.contains("\"skipped\":2"), "got: {}", json);
}

// ---- Model identity (issue #217, task-b) ----

#[test]
fn test_import_refuses_when_header_identity_differs_from_db() {
    let dir = make_db();
    let db_path = db_path_of(&dir);
    set_identity(&db_path, "other-model", "other-rev");
    let blob = blob_of(&test_embedding());
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!(
        "{}\n{}\n",
        header_with_identity(
            1,
            crate::embedding::EMBED_MODEL_ID,
            crate::embedding::EMBED_MODEL_REVISION
        ),
        row
    );
    let err = import_stdout_err(&jsonl, &db_path).expect("must refuse");
    assert!(err.contains("import refused"), "got: {err}");
    assert!(
        err.contains("other-model"),
        "error names recorded identity: {err}"
    );
    assert_eq!(row_count(&db_path), 0, "nothing written on refusal");
}

#[test]
fn test_import_refuses_when_db_identity_differs_from_default_header() {
    let dir = make_db();
    let db_path = db_path_of(&dir);
    // Legacy (identity-free) header ⇒ treated as bge; db recorded as e5 ⇒ refuse.
    set_identity(&db_path, "e5-model", "e5-rev");
    let blob = blob_of(&test_embedding());
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!("{}\n{}\n", header(1), row);
    let err = import_stdout_err(&jsonl, &db_path).expect("must refuse");
    assert!(err.contains("import refused"), "got: {err}");
    assert!(
        err.contains("e5-model"),
        "error names recorded identity: {err}"
    );
}

#[test]
fn test_import_refuses_while_migration_marker_present() {
    let dir = make_db();
    let db_path = db_path_of(&dir);
    let db = crate::sqlite::Database::open(&db_path).unwrap();
    crate::sqlite::identity::write_marker(
        db.conn(),
        &crate::sqlite::model_identity::ModelIdentity {
            model_id: "e5".to_string(),
            revision: "rev".to_string(),
        }
        .into(),
    )
    .unwrap();
    let blob = blob_of(&test_embedding());
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!("{}\n{}\n", header(1), row);
    let err = import_stdout_err(&jsonl, &db_path).expect("must refuse");
    assert!(err.contains("migration"), "got: {err}");
    assert_eq!(row_count(&db_path), 0);
}

#[test]
fn test_import_accepts_matching_identity_header() {
    let dir = make_db();
    let db_path = db_path_of(&dir);
    set_identity(&db_path, "e5-model", "e5-rev");
    let blob = blob_of(&test_embedding());
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!(
        "{}\n{}\n",
        header_with_identity(1, "e5-model", "e5-rev"),
        row
    );
    assert!(
        import_stdout_err(&jsonl, &db_path).is_none(),
        "matching identity must import"
    );
    assert_eq!(row_count(&db_path), 1);
}

#[test]
fn test_import_accepts_legacy_header_into_default_store() {
    // Pre-#217 export (no identity keys) into a store with no identity row:
    // both resolve to the default bge identity → import proceeds.
    let dir = make_db();
    let db_path = db_path_of(&dir);
    let blob = blob_of(&test_embedding());
    let row = make_row_json(
        "id-1",
        "proj",
        "content",
        None,
        &blob,
        "2024-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        "fact",
        "active",
        None,
        0,
        None,
    );
    let jsonl = format!("{}\n{}\n", header(1), row);
    assert!(
        import_stdout_err(&jsonl, &db_path).is_none(),
        "legacy header into default store must import"
    );
    assert_eq!(row_count(&db_path), 1);
}
