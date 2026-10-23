//! Export → import contract tests (issue #229).
//!
//! The round-trip tests drive the PRODUCTION handlers — `handle_export` then
//! `run_import` / `handle_import` — with no hand-crafted header and no local
//! re-implementation of either direction; that omission is how the
//! `format_version` / `version` mismatch stayed hidden for six releases. The
//! header-spelling tests hand-craft headers by necessity: they exercise the
//! reader's contract, not the writer's output.

#![cfg(test)]

// Model-free by construction: rows are seeded with raw SQL and full-width
// `vec_to_blob` embeddings, so no ONNX model is downloaded. Both stores are
// freshly created — neither records a model identity nor a migration marker,
// so both resolve to the default bge identity and the identity guard in
// `run_import` passes. Every seeded row carries a complete blob:
// `run_import` is all-or-nothing and aborts on an empty or wrong-dimension
// embedding, so such a row could never satisfy "inserted == seeded count".

use crate::commands::export::{base64_encode, handle_export};
use crate::commands::import::{handle_import, run_import};
use crate::embedding::EMBEDDING_DIMS;
use crate::errors::Error;
use crate::sqlite::Database;
use crate::sqlite::embedding::vec_to_blob;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// One fixture row in both directions: what `seed` inserts and what
/// `fetch_all` reads back, so the round-trip asserts compare the seeded Vec
/// directly against what each store reports. The 13 JSONL row fields keep the
/// raw INSERT a field-by-field copy instead of a 14-argument call.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SeedRow {
    id: String,
    project_id: String,
    content: String,
    metadata: Option<String>,
    blob: Vec<u8>,
    created_at: String,
    updated_at: String,
    memory_type: String,
    status: String,
    superseded_by: Option<String>,
    retrieval_count: i64,
    last_retrieved_at: Option<String>,
    importance: String,
}

/// A distinct full-length embedding: non-uniform per row and per dimension,
/// so a byte-identity assertion cannot pass by coincidence.
fn blob_for(seed: usize) -> Vec<u8> {
    let vec: Vec<f32> = (0..EMBEDDING_DIMS)
        .map(|i| (i as f32 + seed as f32 * 7.0) * 0.001)
        .collect();
    let blob = vec_to_blob(&vec).expect("384-dim vector");
    assert_eq!(blob.len(), EMBEDDING_DIMS * 4, "384 x f32-LE");
    blob
}

/// The fixture store: 4 rows across 2 projects, exercising metadata,
/// supersession, a non-default type/status, retrieval telemetry, and a
/// non-default importance on every row (issue #228 — a fixture that left the
/// column at its default could not tell "restored" from "reset").
fn seed_fixture() -> Vec<SeedRow> {
    let specs: [(&str, &str, &str, i64, &str); 4] = [
        ("r0", "proj-a", "row a0", 0, "low"),
        ("r1", "proj-a", "row a1", 7, "critical"),
        ("r2", "proj-b", "row b1", 12, "high"),
        ("r3", "proj-b", "row b2", 1, "medium"),
    ];
    let mut rows: Vec<SeedRow> = specs
        .iter()
        .enumerate()
        .map(
            |(index, (id, project_id, content, retrieval_count, importance))| SeedRow {
                id: (*id).to_string(),
                project_id: (*project_id).to_string(),
                content: (*content).to_string(),
                metadata: None,
                blob: blob_for(index),
                created_at: "2024-01-01T00:00:00Z".to_string(),
                updated_at: "2024-01-02T00:00:00Z".to_string(),
                memory_type: "fact".to_string(),
                status: "active".to_string(),
                superseded_by: None,
                retrieval_count: *retrieval_count,
                last_retrieved_at: None,
                importance: (*importance).to_string(),
            },
        )
        .collect();
    rows[1].metadata = Some(r#"{"k":"v"}"#.to_string());
    rows[1].last_retrieved_at = Some("2024-03-20T14:30:00Z".to_string());
    rows[2].superseded_by = Some("r1".to_string());
    rows[2].last_retrieved_at = Some("2024-01-05T09:00:00Z".to_string());
    rows[3].memory_type = "guard".to_string();
    rows[3].status = "candidate".to_string();
    rows
}

fn seed(db_path: &Path, rows: &[SeedRow]) {
    let db = Database::open(db_path).expect("open source db");
    for row in rows {
        db.conn()
            .execute(
                r#"
                INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                "#,
                rusqlite::params![
                    row.id,
                    row.project_id,
                    row.content,
                    row.blob,
                    row.metadata,
                    row.created_at,
                    row.updated_at,
                    row.memory_type,
                    row.status,
                    row.superseded_by,
                    row.retrieval_count,
                    row.last_retrieved_at,
                    row.importance,
                ],
            )
            .expect("seed row");
    }
}

/// Read every row back straight from SQLite, in id order. Blob width is a
/// construction-time invariant of `blob_for`, so no re-check is needed here —
/// the byte-identity compares below are what enforce it end to end.
fn fetch_all(db_path: &Path) -> Vec<SeedRow> {
    let conn = rusqlite::Connection::open(db_path).expect("open for read");
    let mut stmt = conn
        .prepare(
            "SELECT id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance FROM memories ORDER BY id",
        )
        .expect("prepare");
    stmt.query_map([], |r| {
        Ok(SeedRow {
            id: r.get(0)?,
            project_id: r.get(1)?,
            content: r.get(2)?,
            blob: r.get(3)?,
            metadata: r.get(4)?,
            created_at: r.get(5)?,
            updated_at: r.get(6)?,
            memory_type: r.get(7)?,
            status: r.get(8)?,
            superseded_by: r.get(9)?,
            retrieval_count: r.get(10)?,
            last_retrieved_at: r.get(11)?,
            importance: r.get(12)?,
        })
    })
    .expect("query")
    .collect::<Result<Vec<_>, _>>()
    .expect("read rows")
}

/// A JSONL data row as a plain writer-side value (for the header-matrix tests,
/// where only the header line is under test).
fn row_line(id: &str, blob: &[u8]) -> Value {
    json!({
        "id": id,
        "project_id": "proj",
        "content": "content",
        "metadata": null,
        "embedding": base64_encode(blob),
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "memory_type": "fact",
        "status": "active",
        "superseded_by": null,
        "retrieval_count": 0,
        "last_retrieved_at": null,
    })
}

/// Write `header` + `rows` as a JSONL file and return its path.
fn write_jsonl_file(dir: &Path, name: &str, header: &Value, rows: &[Value]) -> PathBuf {
    let path = dir.join(name);
    let mut text = serde_json::to_string(header).expect("header json");
    text.push('\n');
    for row in rows {
        text.push_str(&serde_json::to_string(row).expect("row json"));
        text.push('\n');
    }
    std::fs::write(&path, text).expect("write jsonl");
    path
}

/// A header carrying the fields the reader needs (type, embedding_dims,
/// exported_at, rows) but no version key. Not a verbatim `write_jsonl`
/// header: it also omits the model-identity keys, which `HeaderIdentity::
/// resolved` defaults to the bge identity — fine for header-contract tests.
fn header_without_version(rows: usize) -> Value {
    json!({
        "type": "export",
        "embedding_dims": EMBEDDING_DIMS,
        "exported_at": "2024-01-01T00:00:00Z",
        "rows": rows,
    })
}

#[test]
fn test_export_then_import_roundtrip_is_byte_identical() {
    let dir = tempfile::TempDir::new().unwrap();
    let src = dir.path().join("source.db");
    let dst = dir.path().join("destination.db");
    let export = dir.path().join("export.jsonl");
    let seeded = seed_fixture();
    seed(&src, &seeded);

    let code = handle_export(&src, &export, None, false).expect("real export must succeed");
    assert_eq!(code, ExitCode::SUCCESS);

    // The header is whatever the real writer produced — that is the point.
    let text = std::fs::read_to_string(&export).expect("read export file");
    let lines: Vec<&str> = text.lines().collect();
    let header: Value = serde_json::from_str(lines[0]).expect("header is JSON");
    assert_eq!(header["type"], "export");
    assert_eq!(
        header["format_version"],
        json!(super::export::EXPORT_FORMAT_VERSION),
        "export keeps writing the canonical key"
    );
    assert!(
        header.get("version").is_none(),
        "export must not emit the legacy alias: {header}"
    );
    assert_eq!(header["rows"].as_u64(), Some(seeded.len() as u64));
    assert_eq!(lines.len(), seeded.len() + 1, "header + one line per row");

    // Fresh destination: no identity row, no migration marker.
    drop(Database::open(&dst).expect("create destination"));

    let response = run_import(&dst, export.to_str().unwrap()).expect("real import must succeed");
    assert_eq!(response.inserted, seeded.len(), "inserted == seeded count");
    assert_eq!(response.skipped, 0);

    // Byte-for-byte source → destination, all 13 columns: the seeded Vec is
    // the single expected value, compared directly with what each store says.
    // `importance` is part of that comparison (issue #228) — a destination row
    // reset to the column DEFAULT fails here, not silently.
    let source_rows = fetch_all(&src);
    let destination_rows = fetch_all(&dst);
    assert_eq!(source_rows, seeded, "fixture seeded what it declared");
    assert_eq!(
        destination_rows, source_rows,
        "round trip must restore every column verbatim, embedding BLOB included"
    );
}

/// Issue #228 acceptance criterion: rows seeded at non-default importance
/// survive `handle_export` → `handle_import` into a fresh database, and the
/// exported file itself carries the key.
///
/// The byte-identity round trip above already covers this column-wise; this
/// test names the regression so a future change that drops `importance` fails
/// with an explanation instead of a generic struct-mismatch.
#[test]
fn test_export_then_handle_import_preserves_importance() {
    let dir = tempfile::TempDir::new().unwrap();
    let src = dir.path().join("source.db");
    let dst = dir.path().join("destination.db");
    let export = dir.path().join("export.jsonl");
    let seeded = seed_fixture();
    seed(&src, &seeded);

    handle_export(&src, &export, None, false).expect("export ok");

    // The writer side: every row line carries the key, at the stored level.
    let text = std::fs::read_to_string(&export).expect("read export");
    let rows: Vec<Value> = text
        .lines()
        .skip(1)
        .map(|l| serde_json::from_str(l).expect("row json"))
        .collect();
    for (seeded_row, row) in seeded.iter().zip(&rows) {
        assert_eq!(
            row["importance"], seeded_row.importance,
            "export must write the row's own importance: {row}"
        );
    }

    drop(Database::open(&dst).expect("create destination"));
    let code = handle_import(&dst, export.to_str(), None, true).expect("import ok");
    assert_eq!(code, ExitCode::SUCCESS);

    let restored = fetch_all(&dst);
    assert_eq!(restored.len(), seeded.len());
    let levels: Vec<(&str, &str)> = restored
        .iter()
        .map(|r| (r.id.as_str(), r.importance.as_str()))
        .collect();
    assert_eq!(
        levels,
        vec![
            ("r0", "low"),
            ("r1", "critical"),
            ("r2", "high"),
            ("r3", "medium"),
        ],
        "import must restore each row's importance, not the column DEFAULT"
    );

    // And the read path reports the same thing `vipune get --json` would show.
    let dst_db = Database::open(&dst).expect("open destination");
    let memory = dst_db
        .conn()
        .query_row("SELECT importance FROM memories WHERE id = 'r1'", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap();
    assert_eq!(memory, "critical");
}

#[test]
fn test_export_then_handle_import_is_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    let src = dir.path().join("source.db");
    let dst = dir.path().join("destination.db");
    let export = dir.path().join("export.jsonl");
    let seeded = seed_fixture();
    seed(&src, &seeded);

    handle_export(&src, &export, None, true).expect("export ok");
    drop(Database::open(&dst).expect("create destination"));

    // The CLI entry point the user's repro goes through.
    let code = handle_import(&dst, export.to_str(), None, false).expect("import ok");
    assert_eq!(code, ExitCode::SUCCESS);
    let first = fetch_all(&dst);
    assert_eq!(first.len(), seeded.len());

    // Re-running the same file skips every id and writes nothing.
    let second = run_import(&dst, export.to_str().unwrap()).expect("re-import ok");
    assert_eq!(second.inserted, 0);
    assert_eq!(second.skipped, seeded.len());
    assert_eq!(
        fetch_all(&dst),
        first,
        "a skipped row must keep its telemetry and blob untouched"
    );
}

#[test]
fn test_import_rejects_header_missing_both_version_keys() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    drop(Database::open(&db_path).expect("create db"));

    let file = write_jsonl_file(
        dir.path(),
        "no-version.jsonl",
        &header_without_version(1),
        &[row_line("id-1", &blob_for(0))],
    );

    let err = run_import(&db_path, file.to_str().unwrap())
        .expect_err("a header with neither version key must be refused");
    // Asserted on the error message itself, not stderr: the same text reaches
    // stderr twice with different prefixes (handle_import's "Import failed:"
    // and main's "Error:"), and `Error::InvalidInput`'s Display adds its own
    // "Invalid input: " prefix on top of this message.
    match err {
        Error::InvalidInput(msg) => assert_eq!(
            msg,
            "line 1: header missing format_version (or the legacy alias \"version\")"
        ),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    assert!(fetch_all(&db_path).is_empty(), "all-or-nothing");
}

#[test]
fn test_import_accepts_format_version_without_numeric_gate() {
    // Presence-only: the value is never compared against
    // EXPORT_FORMAT_VERSION, so an unreadable future version still parses
    // rather than being rejected for a number this reader has not seen.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    drop(Database::open(&db_path).expect("create db"));

    let mut header = header_without_version(1);
    header["format_version"] = json!("2");
    let file = write_jsonl_file(
        dir.path(),
        "quoted-version.jsonl",
        &header,
        &[row_line("id-1", &blob_for(1))],
    );

    let response = run_import(&db_path, file.to_str().unwrap()).expect("quoted version parses");
    assert_eq!(response.inserted, 1);
    assert_eq!(response.skipped, 0);
}

#[test]
fn test_import_accepts_unseen_and_null_format_version_values() {
    // The other two presence-only shapes: an unseen future NUMBER and a
    // JSON null both satisfy `is_none()` presence, as documented on
    // validate_header. Mutated via serde_json::Value so only the version
    // field differs from the shared helper's header.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    drop(Database::open(&db_path).expect("create db"));

    let mut future = header_without_version(1);
    future["format_version"] = json!(super::export::EXPORT_FORMAT_VERSION + 1);
    let file = write_jsonl_file(
        dir.path(),
        "future.jsonl",
        &future,
        &[row_line("id-1", &blob_for(3))],
    );
    let response = run_import(&db_path, file.to_str().unwrap())
        .expect("a version this reader has not seen still imports");
    assert_eq!(response.inserted, 1);

    let mut null_version = header_without_version(1);
    null_version["format_version"] = Value::Null;
    let file = write_jsonl_file(
        dir.path(),
        "null-version.jsonl",
        &null_version,
        &[row_line("id-2", &blob_for(4))],
    );
    let response = run_import(&db_path, file.to_str().unwrap())
        .expect("a null version still counts as present");
    assert_eq!(response.inserted, 1);
}

#[test]
fn test_import_accepts_legacy_version_alias_still() {
    // The additive half of the fix: dropping this branch would break files
    // written by hand from the pre-#229 docs.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    drop(Database::open(&db_path).expect("create db"));

    let mut header = header_without_version(1);
    header["version"] = json!(1);
    let file = write_jsonl_file(
        dir.path(),
        "legacy.jsonl",
        &header,
        &[row_line("id-1", &blob_for(2))],
    );

    let response = run_import(&db_path, file.to_str().unwrap()).expect("legacy alias imports");
    assert_eq!(response.inserted, 1);
}
