//! Integration tests for issue #195: export → import byte-identity round trip.
//!
//! These tests exercise the contract that the embedding BLOB survives an
//! export → import round trip byte-for-byte (NOT cosine similarity), and that
//! retrieval_count / last_retrieved_at are restored verbatim.
//!
//! # Why this file spawns the binary (issue #229)
//!
//! It used to define its own `export_to_jsonl` / `import_from_jsonl` helpers
//! that re-implemented both directions locally — including a hand-written
//! `"version":1` header. That meant the file could never observe the real
//! writer or the real importer, which is exactly how the #229 header-key
//! mismatch (export wrote `format_version`, import required `version`) stayed
//! green while every release from v0.10.0 to v0.15.0 shipped a restore path
//! that aborted on its own exports. The fakes are gone: every export and
//! import here runs the compiled `vipune` binary.
//!
//! Rows with an empty or wrong-dimension BLOB are no longer imported here:
//! the real importer rejects them all-or-nothing BY CONTRACT
//! (`empty embedding (NULL/empty blobs are not restorable)` /
//! `wrong-dimension embedding`), so those two cases are asserted on the
//! export side only. The happy-path importer is covered again by
//! `src/commands/import_export_roundtrip_tests.rs`, which drives
//! `handle_export` → `run_import` in-crate.
//!
//! Model-free: rows are seeded through the public `Database::insert` API with
//! a self-contained fake embedder (no ONNX model, no `test_fake_embedder`),
//! and BLOB-level assertions go through the dev-dependency `rusqlite`
//! connection because `Database::conn()` is `pub(crate)`.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use rusqlite::{Connection, params};
use vipune::Database;

/// Standard-alphabet base64 decoder, vendored so the corrupt-BLOB assertion
/// needs no extra dependency. Only used to read the exported `embedding`
/// field back; the encoder side is exercised by the binary itself.
fn base64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|b| *b != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i + 1 < bytes.len() {
        let mut n: u32 = (val(bytes[i]) << 18) | (val(bytes[i + 1]) << 12);
        out.push(((n >> 16) & 0xFF) as u8);
        if i + 2 < bytes.len() {
            n |= val(bytes[i + 2]) << 6;
            out.push(((n >> 8) & 0xFF) as u8);
            if i + 3 < bytes.len() {
                n |= val(bytes[i + 3]);
                out.push((n & 0xFF) as u8);
            }
        }
        i += 4;
    }
    out
}

/// Self-contained deterministic fake embedder: 384-dim f32 vector derived
/// from the content, so a seeded row carries a full 1536-byte BLOB without
/// downloading the ONNX model.
fn fake_embedder(content: &str) -> Vec<f32> {
    let mut hash: u64 = 0x123456789abcdef;
    for byte in content.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    (0..384)
        .map(|i| {
            let mut dim_hash = hash.wrapping_add(i as u64);
            dim_hash ^= dim_hash >> 33;
            dim_hash = dim_hash.wrapping_mul(0xff51afd7ed558ccd);
            dim_hash ^= dim_hash >> 33;
            dim_hash = dim_hash.wrapping_mul(0xc4ceb9fe1a85ec5);
            ((dim_hash % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec.to_vec();
    }
    vec.iter().map(|x| x / norm).collect()
}

// ── Fixtures ──

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn raw_embedding_blob(db_path: &Path, id: &str) -> Option<Vec<u8>> {
    let conn = Connection::open(db_path).ok()?;
    conn.query_row("SELECT embedding FROM memories WHERE id = ?", [id], |row| {
        row.get::<_, Vec<u8>>(0)
    })
    .ok()
}

fn fetch_telemetry(db_path: &Path, id: &str) -> (i64, Option<String>) {
    let conn = Connection::open(db_path).unwrap();
    conn.query_row(
        "SELECT retrieval_count, last_retrieved_at FROM memories WHERE id = ?",
        [id],
        |row| {
            Ok((
                row.get::<_, i64>(0).unwrap(),
                row.get::<_, Option<String>>(1).unwrap(),
            ))
        },
    )
    .unwrap()
}

fn count_rows(db_path: &Path) -> usize {
    let conn = Connection::open(db_path).unwrap();
    conn.query_row("SELECT COUNT(*) FROM memories", [], |row| {
        row.get::<_, i64>(0)
    })
    .unwrap() as usize
}

/// Seed six rows across two projects with full 1536-byte BLOBs and explicit
/// retrieval telemetry. Returns `(id, blob, retrieval_count, last_retrieved_at)`.
fn seed_source_with_path(
    db: &Database,
    db_path: &Path,
) -> Vec<(String, Vec<u8>, i64, Option<String>)> {
    let mut seeded = Vec::new();
    let rows: [(&str, &str, i64, Option<&str>); 6] = [
        ("proj-a", "row a1", 0, None),
        ("proj-a", "row a2", 7, Some("2024-03-20T14:30:00Z")),
        ("proj-a", "row a3", 0, None),
        ("proj-b", "row b1", 12, Some("2024-01-05T09:00:00Z")),
        ("proj-b", "row b2", 0, None),
        ("proj-b", "row b3", 1, Some("2023-12-31T23:59:59Z")),
    ];

    for (project, content, rc, lr) in rows {
        let emb = l2_normalize(&fake_embedder(content));
        let id = db
            .insert(project, content, &emb, None, "fact", "active")
            .unwrap();

        let conn = Connection::open(db_path).unwrap();
        conn.execute(
            "UPDATE memories SET retrieval_count = ?, last_retrieved_at = ? WHERE id = ?",
            params![rc, lr, id],
        )
        .unwrap();
        drop(conn);

        let stored_blob = raw_embedding_blob(db_path, &id).expect("seeded row must exist");
        seeded.push((id, stored_blob, rc, lr.map(|s| s.to_string())));
    }
    seeded
}

// ── Real CLI driver (the code path a user actually runs) ──

/// Path to the compiled `vipune` binary (the test harness builds it).
/// Walks up from `target/debug/deps/<test>-<hash>` to `target/debug/vipune`.
fn vipune_bin() -> std::path::PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let mut path = exe;
    for _ in 0..4 {
        if let Some(parent) = path.parent() {
            let candidate = parent.join("vipune");
            if candidate.exists() {
                return candidate;
            }
            path = parent.to_path_buf();
        }
    }
    panic!("could not locate the vipune binary for subprocess capture");
}

/// Run `vipune [global flags] <args...>` and return (exit code, stdout, stderr).
///
/// Both `export` and `import` print informational warnings on stderr even on
/// success (`--project ... is ignored`), so assertions belong on the exit
/// code and stdout, never on an empty stderr.
fn run_vipune(db_path: &Path, args: &[&str]) -> (i32, String, String) {
    let mut cmd = Command::new(vipune_bin());
    cmd.arg("--db-path")
        .arg(db_path)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd.output().expect("spawn vipune");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Run the real `vipune export` and return the JSONL lines.
fn cli_export(db_path: &Path, out_path: &Path) -> Vec<String> {
    let out_arg = out_path.to_str().expect("utf-8 out path");
    let (code, stdout, stderr) = run_vipune(db_path, &["export", out_arg]);
    assert_eq!(code, 0, "vipune export failed: {stderr}");
    let text = fs::read_to_string(out_path).expect("export file must exist");
    assert!(
        stdout.contains("Exported"),
        "export must report the row count on stdout: {stdout}"
    );
    text.lines().map(str::to_string).collect()
}

/// Run the real `vipune import --json` and return `(inserted, skipped)`
/// parsed from the machine-readable response.
fn cli_import(db_path: &Path, jsonl_path: &Path) -> (usize, usize) {
    let src_arg = jsonl_path.to_str().expect("utf-8 source path");
    let (code, stdout, stderr) = run_vipune(db_path, &["import", src_arg, "--json"]);
    assert_eq!(code, 0, "vipune import failed: {stderr}");
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("import --json must print one JSON object");
    (
        value["inserted"].as_u64().expect("inserted count") as usize,
        value["skipped"].as_u64().expect("skipped count") as usize,
    )
}

/// Seed a source store, export it with the real CLI, and open a fresh
/// destination. Returns the seeded rows, the export path, and the destination.
fn export_from_seeded_source(
    src_dir: &tempfile::TempDir,
    src_path: &Path,
) -> Vec<(String, Vec<u8>, i64, Option<String>)> {
    let db = Database::open(src_path).unwrap();
    let seeded = seed_source_with_path(&db, src_path);
    assert_eq!(seeded.len(), 6);
    let export_path = src_dir.path().join("export.jsonl");
    let lines = cli_export(src_path, &export_path);
    assert_eq!(lines.len(), 7, "1 header + 6 data rows");
    seeded
}

// ── Acceptance tests ──

#[test]
fn test_roundtrip_embedding_blob_byte_identical() {
    let (src_dir, src_path) = create_test_db();
    let seeded = export_from_seeded_source(&src_dir, &src_path);
    let export_path = src_dir.path().join("export.jsonl");

    // The header the real writer produced is the thing that used to break the
    // round trip: `format_version`, never the legacy `version`.
    let header: serde_json::Value = serde_json::from_str(&lines_of(&export_path)[0]).unwrap();
    assert_eq!(header["type"], "export");
    assert_eq!(
        header["format_version"], 1,
        "export writes format_version; import must accept it (#229)"
    );
    assert_eq!(header["rows"], 6);

    let (_dst_dir, dst_path) = create_test_db();
    let (inserted, skipped) = cli_import(&dst_path, &export_path);
    assert_eq!(inserted, 6, "every exported row must import");
    assert_eq!(skipped, 0);

    for (id, src_blob, _, _) in &seeded {
        let dst_blob =
            raw_embedding_blob(&dst_path, id).expect("row must exist in destination after import");
        assert_eq!(
            *src_blob, dst_blob,
            "embedding BLOB must be byte-identical for id {}",
            id
        );
        assert_eq!(
            dst_blob.len(),
            1536,
            "must be full 1536-byte 384xf32-LE blob"
        );
    }
}

#[test]
fn test_roundtrip_retrieval_telemetry_verbatim() {
    let (src_dir, src_path) = create_test_db();
    let seeded = export_from_seeded_source(&src_dir, &src_path);
    let export_path = src_dir.path().join("export.jsonl");

    let (_dst_dir, dst_path) = create_test_db();
    cli_import(&dst_path, &export_path);

    for (id, _blob, rc, lr) in &seeded {
        let (dst_rc, dst_lr) = fetch_telemetry(&dst_path, id);
        assert_eq!(dst_rc, *rc, "retrieval_count must be verbatim for {}", id);
        assert_eq!(
            dst_lr.as_deref(),
            lr.as_deref(),
            "last_retrieved_at must be verbatim for {}",
            id
        );
    }
}

#[test]
fn test_import_rerun_skips_existing_and_counts() {
    let (src_dir, src_path) = create_test_db();
    export_from_seeded_source(&src_dir, &src_path);
    let export_path = src_dir.path().join("export.jsonl");

    let (_dst_dir, dst_path) = create_test_db();
    assert_eq!(cli_import(&dst_path, &export_path), (6, 0));

    let probe_id: String = {
        let conn = Connection::open(&dst_path).unwrap();
        conn.query_row("SELECT id FROM memories LIMIT 1", [], |r| r.get(0))
            .unwrap()
    };
    let (rc_before, lr_before) = fetch_telemetry(&dst_path, &probe_id);

    assert_eq!(cli_import(&dst_path, &export_path), (0, 6));

    let (rc_after, lr_after) = fetch_telemetry(&dst_path, &probe_id);
    assert_eq!(rc_before, rc_after, "re-import must not touch counters");
    assert_eq!(lr_before, lr_after);

    assert_eq!(count_rows(&dst_path), 6);
}

#[test]
fn test_header_line_not_counted_and_lf_endings() {
    let (src_dir, src_path) = create_test_db();
    export_from_seeded_source(&src_dir, &src_path);
    let export_path = src_dir.path().join("export.jsonl");

    let raw = fs::read_to_string(&export_path).unwrap();
    assert!(
        !raw.contains("\r\n"),
        "export must use LF line endings, not CRLF"
    );
    assert_eq!(
        raw.lines().count(),
        7,
        "expected 1 header + 6 data rows = 7 lines"
    );

    let (_dst_dir, dst_path) = create_test_db();
    let (inserted, skipped) = cli_import(&dst_path, &export_path);
    assert_eq!(
        (inserted, skipped),
        (6, 0),
        "header line must not be imported as a data row"
    );
}

// ── Export-side-only contracts ──
//
// The real importer refuses empty and wrong-dimension BLOBs all-or-nothing,
// so these two shapes are pinned on the export side only. Their import-side
// counterparts live in `src/commands/import_tests.rs`
// (`test_import_empty_embedding_aborts`, `test_import_wrong_dimension_embedding_aborts`).

#[test]
fn test_export_passes_through_corrupt_blob_without_decoding() {
    let (src_dir, src_path) = create_test_db();
    {
        let db = Database::open(&src_path).unwrap();
        seed_source_with_path(&db, &src_path);
    }

    let corrupt_blob = vec![0xABu8; 1535];
    let corrupt_id = uuid::Uuid::new_v4().to_string();
    {
        let conn = Connection::open(&src_path).unwrap();
        conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at, type, status) VALUES (?, 'proj-a', 'corrupt row', ?, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active')",
            params![corrupt_id, corrupt_blob],
        )
        .unwrap();
    }

    let export_path = src_dir.path().join("export.jsonl");
    let lines = cli_export(&src_path, &export_path);
    assert_eq!(lines.len(), 8, "corrupt row must be included in export");

    let corrupt_line = lines
        .iter()
        .find(|l| l.contains(&format!("\"id\":\"{corrupt_id}\"")))
        .expect("corrupt row must be present in export");
    let v: serde_json::Value = serde_json::from_str(corrupt_line).unwrap();
    let decoded = base64_decode(v["embedding"].as_str().unwrap());
    assert_eq!(decoded.len(), 1535, "byte length must survive verbatim");
    assert_eq!(decoded, corrupt_blob, "corrupt bytes must be preserved");
}

#[test]
fn test_export_null_blob_exports_as_empty_string() {
    let (src_dir, src_path) = create_test_db();
    {
        let db = Database::open(&src_path).unwrap();
        seed_source_with_path(&db, &src_path);
    }

    let null_id = uuid::Uuid::new_v4().to_string();
    let empty_blob: Vec<u8> = Vec::new();
    {
        let conn = Connection::open(&src_path).unwrap();
        conn.execute(
            "INSERT INTO memories (id, project_id, content, embedding, created_at, updated_at, type, status) VALUES (?, 'proj-a', 'empty blob row', ?, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active')",
            params![null_id.clone(), empty_blob],
        )
        .unwrap();
    }

    let export_path = src_dir.path().join("export.jsonl");
    let lines = cli_export(&src_path, &export_path);
    assert_eq!(lines.len(), 8);

    let null_line = lines
        .iter()
        .find(|l| l.contains(&format!("\"id\":\"{null_id}\"")))
        .expect("null-blob row must be present");
    let v: serde_json::Value = serde_json::from_str(null_line).unwrap();
    assert_eq!(
        v["embedding"].as_str().unwrap(),
        "",
        "NULL/empty blob must export as base64 empty string"
    );
}

#[test]
fn test_export_row_uses_memory_type_not_type() {
    let (src_dir, src_path) = create_test_db();
    export_from_seeded_source(&src_dir, &src_path);
    let export_path = src_dir.path().join("export.jsonl");

    let lines = lines_of(&export_path);
    let first_data_line = &lines[1];
    assert!(
        first_data_line.contains("\"memory_type\":"),
        "row line must use memory_type"
    );
    assert!(
        !first_data_line.contains("\"type\":"),
        "row line must NOT contain a bare \"type\" key"
    );
}

/// Read a JSONL file back as owned lines.
fn lines_of(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .expect("jsonl file")
        .lines()
        .map(str::to_string)
        .collect()
}
