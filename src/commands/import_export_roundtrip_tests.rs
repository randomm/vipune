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
//!
//! # Division of labour with `tests/backup_roundtrip_integration.rs`
//!
//! Deliberate layer split, not duplication: THIS file owns the column-level
//! contracts (BLOB byte-identity, telemetry verbatim, re-import
//! skip-and-preserve) by calling `handle_export` / `run_import` directly, so
//! a schema change is edited here once. The CLI-spawn file owns the
//! process-level contracts around the same production path — exit code,
//! `Exported` stdout, the `--json` response shape, LF line endings, and the
//! export-side-only corrupt/NULL BLOB shapes.

#![cfg(test)]

use crate::commands::export::handle_export;
use crate::commands::import::run_import;
use crate::embedding::EMBEDDING_DIMS;
use crate::sqlite::Database;
use crate::sqlite::embedding::vec_to_blob;
use std::path::Path;
use std::process::ExitCode;

/// A full-width vector (one f32 per embedding dimension, so its BLOB is
/// `EMBEDDING_DIMS * 4` bytes). Non-uniform so a byte-identity assertion
/// cannot pass trivially (a constant vector would round-trip even if the
/// writer mangled the payload).
fn seed_vector(seed: f32) -> Vec<f32> {
    (0..EMBEDDING_DIMS)
        .map(|i| (i as f32) * 0.001 + seed)
        .collect()
}

/// Open a fresh store and return its temp dir plus database path.
fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

/// One fully-populated row of a round-trip fixture: every column the contract
/// covers, including the raw embedding BLOB. The SAME type is what
/// [`read_row`] reads back from the destination, so the seeded expectation and
/// the restored reality are one struct — a schema change is a single edit,
/// and `assert_eq!(restored, seeded)` compares all columns at once.
///
/// Every row carries a full-width BLOB (`EMBEDDING_DIMS * 4` bytes): the
/// production importer aborts all-or-nothing on an empty or wrong-dimension
/// embedding, so such a row would sink the whole import rather than exercise
/// the round trip.
#[derive(Debug, PartialEq, Eq)]
struct SeedRow {
    id: String,
    project_id: String,
    content: String,
    /// Raw BLOB bytes as written / read, bypassing the embedder.
    blob: Vec<u8>,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
    /// DB column `type`.
    memory_type: String,
    status: String,
    superseded_by: Option<String>,
    retrieval_count: i64,
    last_retrieved_at: Option<String>,
}

/// Seed one row through the production all-columns insert
/// (`Database::insert_with_id`, the import path's own insert) rather than a
/// fixture-local INSERT literal: a column added to the real insert cannot
/// drift past this fixture unnoticed.
fn insert_seed_row(db: &Database, row: &SeedRow) {
    db.insert_with_id(
        &row.id,
        &row.project_id,
        &row.content,
        &row.blob,
        row.metadata.as_deref(),
        &row.created_at,
        &row.updated_at,
        &row.memory_type,
        &row.status,
        row.superseded_by.as_deref(),
        row.retrieval_count,
        row.last_retrieved_at.as_deref(),
    )
    .unwrap();
}

/// Seed a six-row source store spanning two projects, with the full column
/// set (metadata, superseded_by, retrieval telemetry, non-default type and
/// status) so a dropped column cannot hide.
///
/// Neither store records a model identity and neither carries a migration
/// marker: both resolve to the default bge identity, which is the precondition
/// for `run_import`'s identity guard to let the file in.
fn seed_source(db_path: &Path) -> Vec<SeedRow> {
    let db = Database::open(db_path).unwrap();

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
        assert_eq!(
            blob.len(),
            EMBEDDING_DIMS * 4,
            "fixture rows carry full-width blobs"
        );
        let row = SeedRow {
            id: format!("row-{idx}"),
            project_id: project.to_string(),
            content: content.to_string(),
            blob,
            metadata: if idx % 2 == 0 {
                Some(format!(r#"{{"k":{idx}}}"#))
            } else {
                None
            },
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-02T00:00:00Z".to_string(),
            memory_type: if idx == 2 { "guard" } else { "fact" }.to_string(),
            status: if idx == 4 { "candidate" } else { "active" }.to_string(),
            superseded_by: if idx == 5 {
                Some("row-4".to_string())
            } else {
                None
            },
            retrieval_count,
            last_retrieved_at: last_retrieved_at.map(str::to_string),
        };
        insert_seed_row(&db, &row);
        seeded.push(row);
    }
    seeded
}

/// Run the real `vipune export` handler into `out_path`.
fn export(db_path: &Path, out_path: &Path) {
    let code = handle_export(db_path, out_path, None, true).expect("handle_export ok");
    assert_eq!(code, ExitCode::SUCCESS);
}

/// Read a row back from the destination, shaped exactly like a seeded
/// [`SeedRow`] so the two compare directly.
///
/// Columns are bound BY NAME, never by positional index: a positional `get(N)`
/// takes its type from the field it initialises, so reordering the SELECT (or
/// swapping two same-typed columns) still compiles and silently compares the
/// wrong columns — the "test that can no longer fail" class. See also issue
/// #186 (wrong-column-index FromSqlConversionFailure).
fn read_row(db_path: &Path, id: &str) -> SeedRow {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row(
        "SELECT id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at FROM memories WHERE id = ?",
        [id],
        |r| {
            Ok(SeedRow {
                id: r.get("id")?,
                project_id: r.get("project_id")?,
                content: r.get("content")?,
                blob: r.get("embedding")?,
                metadata: r.get("metadata")?,
                created_at: r.get("created_at")?,
                updated_at: r.get("updated_at")?,
                memory_type: r.get("type")?,
                status: r.get("status")?,
                superseded_by: r.get("superseded_by")?,
                retrieval_count: r.get("retrieval_count")?,
                last_retrieved_at: r.get("last_retrieved_at")?,
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
        let restored = read_row(&dst_path, &row.id);
        assert_eq!(
            restored.blob, row.blob,
            "embedding BLOB must be byte-identical for {}",
            row.id
        );
        assert_eq!(
            restored.blob.len(),
            EMBEDDING_DIMS * 4,
            "{}-byte {}xf32-LE blob survives intact",
            EMBEDDING_DIMS * 4,
            EMBEDDING_DIMS
        );
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
        // One struct comparison: every column of SeedRow (BLOB included) must
        // come back unchanged, and a field added to the struct is compared
        // automatically instead of needing a new assertion here.
        assert_eq!(
            read_row(&dst_path, &row.id),
            *row,
            "all columns must restore verbatim for {}",
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
        let restored = read_row(&dst_path, &row.id);
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

/// Build one JSONL row with a full-width embedding, base64-encoded with the
/// crate's own encoder (the same one `write_jsonl` uses).
fn row_line(id: &str) -> String {
    let blob = vec_to_blob(&seed_vector(0.5)).unwrap();
    let embedding = crate::commands::export::base64_encode(&blob);
    format!(
        r#"{{"id":"{id}","project_id":"proj","content":"c","metadata":null,"embedding":"{embedding}","created_at":"2024-01-01T00:00:00Z","updated_at":"2024-01-02T00:00:00Z","memory_type":"fact","status":"active","superseded_by":null,"retrieval_count":2,"last_retrieved_at":"2024-02-02T00:00:00Z"}}"#
    )
}

/// Import a one-row file whose header line is exactly `header`, into a fresh
/// default store, and return `((inserted, skipped), rows_written)`.
///
/// Every header-key test differs only in the header text, so the whole
/// fixture — temp dir, store, file, the row count the store was left with —
/// lives here once. The row count is returned because the all-or-nothing
/// contract ("a refused import writes nothing") must be asserted on the
/// refused cases too.
fn import_with_header(
    header: &str,
    id: &str,
) -> (Result<(usize, usize), crate::errors::Error>, i64) {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    Database::open(&db_path).unwrap();

    let jsonl = format!("{header}\n{}\n", row_line(id));
    let file = dir.path().join("input.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    let outcome = run_import(&db_path, file.to_str().unwrap()).map(|r| (r.inserted, r.skipped));
    let rows = count_rows(&db_path);
    (outcome, rows)
}

/// Read the `InvalidInput` payload of a refused import, panicking on any
/// other outcome. The CLI renders this payload twice more with different
/// prefixes, so the variant — never rendered stderr — is the stable surface.
fn refused(outcome: Result<(usize, usize), crate::errors::Error>) -> String {
    match outcome {
        Err(crate::errors::Error::InvalidInput(msg)) => msg,
        Err(other) => panic!("expected InvalidInput, got {other:?}"),
        Ok(counts) => panic!("expected a refusal, got {counts:?}"),
    }
}

/// The exact message for a header with neither version spelling (pinned, not
/// substring-matched: it is the user-facing text of the #229 fix).
const MISSING_VERSION_MSG: &str =
    r#"line 1: header missing format_version (or the legacy alias "version")"#;

/// Import accepts the canonical key the export writer emits.
#[test]
fn test_import_accepts_format_version_header_into_default_store() {
    let (outcome, rows) = import_with_header(
        r#"{"type":"export","format_version":1,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        "fmt-1",
    );
    assert_eq!(
        outcome.unwrap(),
        (1, 0),
        "format_version header must import"
    );
    assert_eq!(rows, 1);
}

/// Import still accepts the legacy `version` alias — the additive half of the
/// #229 fix (v0.10.0–v0.15.0 backups in the wild carry one spelling or the
/// other; dropping either would break restores).
#[test]
fn test_import_accepts_legacy_version_header_alias() {
    let (outcome, rows) = import_with_header(
        r#"{"type":"export","version":1,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        "legacy-1",
    );
    assert_eq!(outcome.unwrap(), (1, 0), "legacy version alias must import");
    assert_eq!(rows, 1);
}

/// A header carrying NEITHER spelling fails with the exact message.
#[test]
fn test_import_rejects_header_missing_both_version_keys() {
    let (outcome, rows) = import_with_header(
        r#"{"type":"export","embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        "none-1",
    );
    assert_eq!(refused(outcome), MISSING_VERSION_MSG);
    assert_eq!(rows, 0, "refused import writes nothing");
}

/// The version check is key PRESENCE, not shape: a header carrying the key
/// with an explicit JSON `null` is a real backup shape and must still import
/// (rejecting it would recreate the #229 failure for the files that carry
/// it). Pinned for BOTH spellings so a future "cleanup" of the check to a
/// null test cannot silently reject those backups.
#[test]
fn test_import_accepts_null_valued_version_keys() {
    for key in ["format_version", "version"] {
        let header = format!(
            r#"{{"type":"export","{key}":null,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}}"#
        );
        let (outcome, rows) = import_with_header(&header, &format!("null-{key}"));
        assert_eq!(
            outcome.unwrap_or_else(|e| panic!("a null-valued `{key}` must satisfy presence: {e}")),
            (1, 0)
        );
        assert_eq!(rows, 1);
    }
}

/// A header carrying BOTH spellings of the version key is ambiguous — no
/// released writer ever emitted two version keys, so the shared schema
/// refuses it instead of silently preferring one.
#[test]
fn test_import_refuses_header_carrying_both_version_spellings() {
    let (outcome, rows) = import_with_header(
        r#"{"type":"export","format_version":1,"version":2,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        "both-1",
    );
    let msg = refused(outcome);
    assert!(
        msg.contains("duplicate field"),
        "expected a duplicate-field refusal, got: {msg}"
    );
    assert_eq!(rows, 0, "refused import writes nothing");
}

/// A header declaring a NEWER format version than this build understands is
/// refused up front with a dedicated message, instead of failing later at
/// line 2 with a generic `missing field` error — or, for an additive change,
/// importing rows whose BLOBs were interpreted under the wrong schema.
#[test]
fn test_import_rejects_format_version_ahead_of_this_build() {
    let (outcome, rows) = import_with_header(
        r#"{"type":"export","format_version":99,"embedding_dims":384,"exported_at":"2024-01-01T00:00:00Z","rows":1}"#,
        "future-1",
    );
    let msg = refused(outcome);
    assert!(
        msg.contains("unsupported export format version 99"),
        "got: {msg}"
    );
    assert_eq!(rows, 0, "refused import writes nothing");
}
