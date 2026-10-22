//! Real export → import round-trip tests (issue #229).
//!
//! Issue #229 was invisible to the suite because every test on both sides
//! hand-crafted its own header: `import_tests.rs` wrote `"version":1` while
//! `write_jsonl` emits `"format_version":1`, and
//! `tests/backup_roundtrip_integration.rs` re-implemented both directions
//! locally, so neither pair ever met the production code.
//!
//! These tests close that gap: they call `handle_export` and `run_import`
//! directly (both reachable in-crate under the bin-only `commands` module),
//! with no hand-written header anywhere. Model-free — rows are seeded with
//! raw `vec_to_blob` BLOBs, so no ONNX model is downloaded.
//!
//! Contract covered here (issue #195, re-established on the real path):
//! the embedding BLOB survives the round trip byte-for-byte, and
//! `retrieval_count` / `last_retrieved_at` restore verbatim.

#![cfg(test)]

use crate::commands::export::handle_export;
use crate::commands::import::run_import;
use crate::sqlite::Database;
use crate::sqlite::embedding::vec_to_blob;
use std::path::Path;
use std::process::ExitCode;

/// A full 384-dim vector whose BLOB is exactly 1536 bytes. Non-uniform so a
/// byte-identity assertion cannot pass trivially (a constant vector would
/// round-trip even if the writer mangled the payload).
fn seed_vector(seed: f32) -> Vec<f32> {
    (0..384).map(|i| (i as f32) * 0.001 + seed).collect()
}

/// Open a fresh store and return its temp dir plus database path.
fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

/// Seed one row with an EXACT raw BLOB, bypassing the embedder.
///
/// Every row in a round-trip fixture must carry a full 1536-byte blob: the
/// production importer aborts all-or-nothing on an empty or
/// wrong-dimension embedding, so such a row would sink the whole import
/// rather than exercise the round trip.
#[allow(clippy::too_many_arguments)] // mirrors the 12 JSONL row fields
fn seed_raw(
    conn: &rusqlite::Connection,
    id: &str,
    project_id: &str,
    content: &str,
    blob: &[u8],
    metadata: Option<&str>,
    created_at: &str,
    updated_at: &str,
    memory_type: &str,
    status: &str,
    superseded_by: Option<&str>,
    retrieval_count: i64,
    last_retrieved_at: Option<&str>,
) {
    conn.execute(
        r#"
        INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
        "#,
        rusqlite::params![
            id,
            project_id,
            content,
            blob,
            metadata,
            created_at,
            updated_at,
            memory_type,
            status,
            superseded_by,
            retrieval_count,
            last_retrieved_at,
        ],
    )
    .unwrap();
}

struct Seeded {
    id: String,
    blob: Vec<u8>,
    project_id: String,
    content: String,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
    memory_type: String,
    status: String,
    superseded_by: Option<String>,
    retrieval_count: i64,
    last_retrieved_at: Option<String>,
}

/// Seed a six-row source store spanning two projects, with the full column
/// set (metadata, superseded_by, retrieval telemetry, non-default type and
/// status) so a dropped column cannot hide.
///
/// Neither store records a model identity and neither carries a migration
/// marker: both resolve to the default bge identity, which is the precondition
/// for `run_import`'s identity guard to let the file in.
fn seed_source(db_path: &Path) -> Vec<Seeded> {
    let db = Database::open(db_path).unwrap();
    let conn = db.conn();

    let specs: Vec<(usize, &str, &str, i64, Option<&str>)> = vec![
        (0, "proj-a", "alpha", 0, None),
        (1, "proj-a", "bravo", 7, Some("2024-03-20T14:30:00Z")),
        (2, "proj-a", "charlie", 3, None),
        (3, "proj-b", "delta", 12, Some("2024-01-05T09:00:00Z")),
        (4, "proj-b", "echo", 1, None),
        (5, "proj-b", "foxtrot", 42, Some("2023-12-31T23:59:59Z")),
    ];

    let mut seeded = Vec::new();
    for (idx, project, content, retrieval_count, last_retrieved_at) in specs {
        let blob = vec_to_blob(&seed_vector(idx as f32)).unwrap();
        assert_eq!(blob.len(), 1536, "fixture rows carry full 1536-byte blobs");
        let id = format!("row-{idx}");
        let metadata = if idx % 2 == 0 {
            Some(format!(r#"{{"k":{idx}}}"#))
        } else {
            None
        };
        let superseded_by = if idx == 5 { Some("row-4") } else { None };
        let memory_type = if idx == 2 { "guard" } else { "fact" };
        let status = if idx == 4 { "candidate" } else { "active" };

        seed_raw(
            conn,
            &id,
            project,
            content,
            &blob,
            metadata.as_deref(),
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            memory_type,
            status,
            superseded_by,
            retrieval_count,
            last_retrieved_at,
        );

        seeded.push(Seeded {
            id,
            blob,
            project_id: project.to_string(),
            content: content.to_string(),
            metadata,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-02T00:00:00Z".to_string(),
            memory_type: memory_type.to_string(),
            status: status.to_string(),
            superseded_by: superseded_by.map(str::to_string),
            retrieval_count,
            last_retrieved_at: last_retrieved_at.map(str::to_string),
        });
    }
    seeded
}

/// Run the real `vipune export` handler into `out_path`.
fn export(db_path: &Path, out_path: &Path) {
    let code = handle_export(db_path, out_path, None, true).expect("handle_export ok");
    assert_eq!(code, ExitCode::SUCCESS);
}

fn raw_blob(db_path: &Path, id: &str) -> Vec<u8> {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row("SELECT embedding FROM memories WHERE id = ?", [id], |r| {
        r.get::<_, Vec<u8>>(0)
    })
    .unwrap()
}

/// Every non-BLOB column of an imported row, read back from the destination.
struct RestoredRow {
    project_id: String,
    content: String,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
    memory_type: String,
    status: String,
    superseded_by: Option<String>,
    retrieval_count: i64,
    last_retrieved_at: Option<String>,
}

fn row_fields(db_path: &Path, id: &str) -> RestoredRow {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row(
        "SELECT project_id, content, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at FROM memories WHERE id = ?",
        [id],
        |r| {
            Ok(RestoredRow {
                project_id: r.get(0)?,
                content: r.get(1)?,
                metadata: r.get(2)?,
                created_at: r.get(3)?,
                updated_at: r.get(4)?,
                memory_type: r.get(5)?,
                status: r.get(6)?,
                superseded_by: r.get(7)?,
                retrieval_count: r.get(8)?,
                last_retrieved_at: r.get(9)?,
            })
        },
    )
    .unwrap()
}

fn count_rows(db_path: &Path) -> i64 {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap()
}

/// The bug itself: an export written by the real writer must be importable by
/// the real importer. Pre-#229 this aborted with
/// `line 1: header missing version` on every release from v0.10.0 to v0.15.0.
#[test]
fn test_real_export_imports_into_fresh_store() {
    let (src_dir, src_path) = create_test_db();
    let seeded = seed_source(&src_path);

    let export_path = src_dir.path().join("export.jsonl");
    export(&src_path, &export_path);

    // The real writer's header spells the key `format_version` — this is the
    // pin that would have caught #229 on the export side.
    let header_line = std::fs::read_to_string(&export_path)
        .unwrap()
        .lines()
        .next()
        .expect("header line")
        .to_string();
    let header: serde_json::Value = serde_json::from_str(&header_line).unwrap();
    assert_eq!(header["type"], "export");
    assert_eq!(header["format_version"], 1);
    assert!(
        header.get("version").is_none(),
        "the real writer emits format_version, not the legacy key: {header}"
    );
    assert_eq!(header["rows"], seeded.len() as u64);

    let (_dst_dir, dst_path) = create_test_db();
    let response = run_import(&dst_path, export_path.to_str().unwrap()).expect("real import ok");

    assert_eq!(
        response.inserted,
        seeded.len(),
        "every exported row must import"
    );
    assert_eq!(response.skipped, 0, "fresh destination skips nothing");
    assert_eq!(count_rows(&dst_path), seeded.len() as i64);
}

/// #195's contract, now actually exercised on the production path: the
/// embedding BLOB survives export → import byte-for-byte.
#[test]
fn test_real_roundtrip_embedding_blob_is_byte_identical() {
    let (src_dir, src_path) = create_test_db();
    let seeded = seed_source(&src_path);

    let export_path = src_dir.path().join("export.jsonl");
    export(&src_path, &export_path);

    let (_dst_dir, dst_path) = create_test_db();
    run_import(&dst_path, export_path.to_str().unwrap()).expect("import ok");

    for row in &seeded {
        let dst_blob = raw_blob(&dst_path, &row.id);
        assert_eq!(
            dst_blob, row.blob,
            "embedding BLOB must be byte-identical for {}",
            row.id
        );
        assert_eq!(dst_blob.len(), 1536, "384xf32-LE blob survives intact");
    }
}

/// Retrieval telemetry (and the rest of the 12 columns) restore verbatim —
/// asserted column by column, not just the BLOB.
#[test]
fn test_real_roundtrip_restores_all_columns_verbatim() {
    let (src_dir, src_path) = create_test_db();
    let seeded = seed_source(&src_path);

    let export_path = src_dir.path().join("export.jsonl");
    export(&src_path, &export_path);

    let (_dst_dir, dst_path) = create_test_db();
    run_import(&dst_path, export_path.to_str().unwrap()).expect("import ok");

    for row in &seeded {
        let restored = row_fields(&dst_path, &row.id);
        assert_eq!(
            restored.project_id, row.project_id,
            "project_id for {}",
            row.id
        );
        assert_eq!(restored.content, row.content, "content for {}", row.id);
        assert_eq!(restored.metadata, row.metadata, "metadata for {}", row.id);
        assert_eq!(
            restored.created_at, row.created_at,
            "created_at for {}",
            row.id
        );
        assert_eq!(
            restored.updated_at, row.updated_at,
            "updated_at for {}",
            row.id
        );
        assert_eq!(restored.memory_type, row.memory_type, "type for {}", row.id);
        assert_eq!(restored.status, row.status, "status for {}", row.id);
        assert_eq!(
            restored.superseded_by, row.superseded_by,
            "superseded_by for {}",
            row.id
        );
        assert_eq!(
            restored.retrieval_count, row.retrieval_count,
            "retrieval_count for {}",
            row.id
        );
        assert_eq!(
            restored.last_retrieved_at, row.last_retrieved_at,
            "last_retrieved_at for {}",
            row.id
        );
    }
}

/// Re-running the same real export into the populated destination skips and
/// counts every row, and leaves the restored telemetry untouched.
#[test]
fn test_real_roundtrip_reimport_skips_and_preserves_telemetry() {
    let (src_dir, src_path) = create_test_db();
    let seeded = seed_source(&src_path);

    let export_path = src_dir.path().join("export.jsonl");
    export(&src_path, &export_path);

    let (_dst_dir, dst_path) = create_test_db();
    run_import(&dst_path, export_path.to_str().unwrap()).expect("first import ok");

    let response = run_import(&dst_path, export_path.to_str().unwrap()).expect("second import ok");
    assert_eq!(response.inserted, 0);
    assert_eq!(response.skipped, seeded.len());
    assert_eq!(count_rows(&dst_path), seeded.len() as i64);

    for row in &seeded {
        let restored = row_fields(&dst_path, &row.id);
        assert_eq!(
            restored.retrieval_count, row.retrieval_count,
            "re-import must not reset retrieval_count for {}",
            row.id
        );
        assert_eq!(
            restored.last_retrieved_at, row.last_retrieved_at,
            "re-import must not reset last_retrieved_at for {}",
            row.id
        );
    }
}

// ── Header-key acceptance (issue #229) ────────────────────────────────────

/// Build one JSONL row with a full 1536-byte embedding, base64-encoded with
/// the crate's own encoder (the same one `write_jsonl` uses).
fn row_line(id: &str) -> String {
    let blob = vec_to_blob(&seed_vector(0.5)).unwrap();
    let embedding = crate::commands::export::base64_encode(&blob);
    format!(
        r#"{{"id":"{id}","project_id":"proj","content":"c","metadata":null,"embedding":"{embedding}","created_at":"2024-01-01T00:00:00Z","updated_at":"2024-01-02T00:00:00Z","memory_type":"fact","status":"active","superseded_by":null,"retrieval_count":2,"last_retrieved_at":"2024-02-02T00:00:00Z"}}"#
    )
}

/// Import accepts the canonical key the export writer emits.
#[test]
fn test_import_accepts_format_version_header_into_default_store() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    Database::open(&db_path).unwrap();

    let jsonl = format!(
        "{}\n{}\n",
        r#"{"type":"export","format_version":1,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        row_line("fmt-1")
    );
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let response =
        run_import(&db_path, file.to_str().unwrap()).expect("format_version header must import");
    assert_eq!(response.inserted, 1);
    assert_eq!(response.skipped, 0);
    assert_eq!(count_rows(&db_path), 1);
}

/// Import still accepts the legacy `version` alias — the additive half of the
/// #229 fix (v0.10.0–v0.15.0 backups in the wild carry one spelling or the
/// other; dropping either would break restores).
#[test]
fn test_import_accepts_legacy_version_header_alias() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    Database::open(&db_path).unwrap();

    let jsonl = format!(
        "{}\n{}\n",
        r#"{"type":"export","version":1,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        row_line("legacy-1")
    );
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let response = run_import(&db_path, file.to_str().unwrap())
        .expect("legacy version header must still import");
    assert_eq!(response.inserted, 1);
    assert_eq!(count_rows(&db_path), 1);
}

/// A header carrying NEITHER spelling fails with the exact message, pinned on
/// the `Error`'s `Display` (the CLI renders it twice with different prefixes,
/// so raw stderr is not a stable surface).
#[test]
fn test_import_rejects_header_missing_both_version_keys() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    Database::open(&db_path).unwrap();

    let jsonl = format!(
        "{}\n{}\n",
        r#"{"type":"export","embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        row_line("none-1")
    );
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let err = run_import(&db_path, file.to_str().unwrap())
        .expect_err("header with neither version key must be refused");
    // The required text is the `Error::InvalidInput` payload; `Display` adds
    // the variant prefix (`Invalid input: `), and the CLI prints it twice more
    // with different prefixes, so match the variant rather than a rendered line.
    match err {
        crate::errors::Error::InvalidInput(msg) => assert_eq!(
            msg,
            r#"line 1: header missing format_version (or the legacy alias "version")"#
        ),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    assert_eq!(count_rows(&db_path), 0, "refused import writes nothing");
}
