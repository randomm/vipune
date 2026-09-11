//! Integration tests for issue #195: export → import byte-identity round trip.
//!
//! These tests exercise the contract that the embedding BLOB survives an
//! export → import round trip byte-for-byte (NOT cosine similarity), and that
//! retrieval_count / last_retrieved_at are restored verbatim.
//!
//! Uses the fake-embedder / seeded-raw-blob pattern (no ONNX model required),
//! mirroring `reindex_tests.rs`.
//!
//! This test runs in the external `vipune` crate's public API surface. It does
//! NOT use `db.conn()` (pub(crate)), `test_fake_embedder` (pub(crate),
//! test-only), or the crate's private `rusqlite` re-export. Instead it uses:
//! - A self-contained fake embedder (deterministic 384-dim f32 vector)
//! - Raw `rusqlite::Connection` (dev-dependency) for BLOB-level assertions
//! - A hand-rolled base64 codec (no `base64` crate dependency)

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use rusqlite::{Connection, params};
use vipune::Database;

// ── Self-contained base64 codec (no external dependency) ──

const B64_ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64_ALPHA[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_ALPHA[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_ALPHA[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

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

// ── Self-contained fake embedder (no ONNX model, no test_fake_embedder) ──

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

// ── Test fixtures ──

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

// ── Export / Import (mirrors task-a export + task-b import) ──

fn export_to_jsonl(db_path: &Path, out_path: &Path) -> usize {
    let conn = Connection::open(db_path).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at FROM memories",
        )
        .unwrap();

    type Row12 = (
        String,
        String,
        String,
        Option<String>,
        Vec<u8>,
        String,
        String,
        String,
        String,
        Option<String>,
        i64,
        Option<String>,
    );
    let rows: Vec<Row12> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, Option<String>>(9)?,
                row.get::<_, i64>(10)?,
                row.get::<_, Option<String>>(11)?,
            ))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let rows_total = rows.len();
    let mut out = String::new();

    out.push_str(&format!(
        "{{\"type\":\"export\",\"version\":1,\"embedding_dims\":384,\"exported_at\":\"2024-01-01T00:00:00Z\",\"rows\":{}}}\n",
        rows_total
    ));

    for r in &rows {
        let embedding_b64 = if r.4.is_empty() {
            String::new()
        } else {
            base64_encode(&r.4)
        };
        let row_json = serde_json::json!({
            "id": r.0,
            "project_id": r.1,
            "content": r.2,
            "metadata": r.3,
            "embedding": embedding_b64,
            "created_at": r.5,
            "updated_at": r.6,
            "memory_type": r.7,
            "status": r.8,
            "superseded_by": r.9,
            "retrieval_count": r.10,
            "last_retrieved_at": r.11,
        });
        out.push_str(&row_json.to_string());
        out.push('\n');
    }

    fs::write(out_path, out).unwrap();
    rows_total
}

fn import_from_jsonl(db_path: &Path, jsonl_path: &Path) -> (usize, usize, usize) {
    let content = fs::read_to_string(jsonl_path).unwrap();
    let mut lines = content.lines();

    let header = lines.next().expect("expected a header line");
    let header_json: serde_json::Value = serde_json::from_str(header).unwrap();
    let _header_rows = header_json["rows"].as_u64().unwrap() as usize;

    let mut conn = Connection::open(db_path).unwrap();

    let mut existing_ids: HashSet<String> = {
        let mut stmt = conn.prepare("SELECT id FROM memories").unwrap();
        stmt.query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .collect()
    };

    let mut inserted = 0usize;
    let mut skipped = 0usize;
    let mut rows_total = 0usize;

    let tx = conn.transaction().unwrap();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        rows_total += 1;
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let id = v["id"].as_str().unwrap().to_string();
        let project_id = v["project_id"].as_str().unwrap().to_string();
        let row_content = v["content"].as_str().unwrap().to_string();
        let metadata = v["metadata"].as_str().map(|s| s.to_string());
        let embedding_b64 = v["embedding"].as_str().unwrap().to_string();
        let created_at = v["created_at"].as_str().unwrap().to_string();
        let updated_at = v["updated_at"].as_str().unwrap().to_string();
        let memory_type = v["memory_type"].as_str().unwrap().to_string();
        let status = v["status"].as_str().unwrap().to_string();
        let superseded_by = v["superseded_by"].as_str().map(|s| s.to_string());
        let retrieval_count = v["retrieval_count"].as_i64().unwrap();
        let last_retrieved_at = v["last_retrieved_at"].as_str().map(|s| s.to_string());

        let embedding_blob: Vec<u8> = if embedding_b64.is_empty() {
            Vec::new()
        } else {
            base64_decode(&embedding_b64)
        };

        if existing_ids.contains(&id) {
            skipped += 1;
            continue;
        }

        tx.execute(
            "INSERT INTO memories (id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                id, project_id, row_content, metadata, embedding_blob,
                created_at, updated_at, memory_type, status, superseded_by,
                retrieval_count, last_retrieved_at,
            ],
        )
        .unwrap();
        existing_ids.insert(id);
        inserted += 1;
    }
    tx.commit().unwrap();

    (inserted, skipped, rows_total)
}

// ── Acceptance tests ──

#[test]
fn test_roundtrip_embedding_blob_byte_identical() {
    let (src_dir, src_path) = create_test_db();
    let src = Database::open(&src_path).unwrap();
    let seeded = seed_source_with_path(&src, &src_path);
    assert_eq!(seeded.len(), 6);

    let source_blobs: Vec<(String, Vec<u8>)> = seeded
        .iter()
        .map(|(id, blob, _, _)| (id.clone(), blob.clone()))
        .collect();

    let export_path = src_dir.path().join("export.jsonl");
    let rows_written = export_to_jsonl(&src_path, &export_path);
    assert_eq!(rows_written, 6);

    let (_dst_dir, dst_path) = create_test_db();
    {
        let _dst = Database::open(&dst_path).unwrap();
    }
    let (inserted, skipped, rows_total) = import_from_jsonl(&dst_path, &export_path);
    assert_eq!(inserted, 6);
    assert_eq!(skipped, 0);
    assert_eq!(rows_total, 6);

    for (id, src_blob) in &source_blobs {
        let dst_blob =
            raw_embedding_blob(&dst_path, id).expect("row must exist in destination after import");
        assert_eq!(
            src_blob, &dst_blob,
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
    let src = Database::open(&src_path).unwrap();
    let seeded = seed_source_with_path(&src, &src_path);

    let export_path = src_dir.path().join("export.jsonl");
    export_to_jsonl(&src_path, &export_path);

    let (_dst_dir, dst_path) = create_test_db();
    {
        let _dst = Database::open(&dst_path).unwrap();
    }
    import_from_jsonl(&dst_path, &export_path);

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
    {
        let src = Database::open(&src_path).unwrap();
        seed_source_with_path(&src, &src_path);
    }

    let export_path = src_dir.path().join("export.jsonl");
    export_to_jsonl(&src_path, &export_path);

    let (_dst_dir, dst_path) = create_test_db();
    {
        let _dst = Database::open(&dst_path).unwrap();
    }

    let (ins1, sk1, tot1) = import_from_jsonl(&dst_path, &export_path);
    assert_eq!((ins1, sk1, tot1), (6, 0, 6));

    let probe_id: String = {
        let conn = Connection::open(&dst_path).unwrap();
        conn.query_row("SELECT id FROM memories LIMIT 1", [], |r| r.get(0))
            .unwrap()
    };
    let (rc_before, lr_before) = fetch_telemetry(&dst_path, &probe_id);

    let (ins2, sk2, tot2) = import_from_jsonl(&dst_path, &export_path);
    assert_eq!((ins2, sk2, tot2), (0, 6, 6));

    let (rc_after, lr_after) = fetch_telemetry(&dst_path, &probe_id);
    assert_eq!(rc_before, rc_after);
    assert_eq!(lr_before, lr_after);

    assert_eq!(count_rows(&dst_path), 6);
}

#[test]
fn test_header_line_not_counted_and_lf_endings() {
    let (src_dir, src_path) = create_test_db();
    {
        let src = Database::open(&src_path).unwrap();
        seed_source_with_path(&src, &src_path);
    }

    let export_path = src_dir.path().join("export.jsonl");
    export_to_jsonl(&src_path, &export_path);

    let raw = fs::read_to_string(&export_path).unwrap();
    assert!(
        !raw.contains("\r\n"),
        "export must use LF line endings, not CRLF"
    );

    let line_count = raw.lines().count();
    assert_eq!(
        line_count, 7,
        "expected 1 header + 6 data rows = 7 lines, got {}",
        line_count
    );

    let (_dst_dir, dst_path) = create_test_db();
    {
        let _dst = Database::open(&dst_path).unwrap();
    }
    let (_ins, _sk, rows_total) = import_from_jsonl(&dst_path, &export_path);
    assert_eq!(
        rows_total, 6,
        "header line must not be counted as a data row"
    );
}

#[test]
fn test_export_passes_through_corrupt_blob_without_decoding() {
    let (src_dir, src_path) = create_test_db();
    {
        let src = Database::open(&src_path).unwrap();
        seed_source_with_path(&src, &src_path);
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
    let rows_written = export_to_jsonl(&src_path, &export_path);
    assert_eq!(rows_written, 7, "corrupt row must be included in export");

    let raw = fs::read_to_string(&export_path).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    let corrupt_line = lines
        .iter()
        .find(|l| l.contains(&format!("\"id\":\"{}\"", corrupt_id)))
        .expect("corrupt row must be present in export");
    let v: serde_json::Value = serde_json::from_str(corrupt_line).unwrap();
    let emb_b64 = v["embedding"].as_str().unwrap().to_string();
    let decoded = base64_decode(&emb_b64);
    assert_eq!(
        decoded.len(),
        1535,
        "corrupt blob must round-trip its exact byte length"
    );
    assert_eq!(
        decoded, corrupt_blob,
        "corrupt blob bytes must be preserved faithfully"
    );
}

#[test]
fn test_export_null_blob_exports_as_empty_string() {
    let (src_dir, src_path) = create_test_db();
    {
        let src = Database::open(&src_path).unwrap();
        seed_source_with_path(&src, &src_path);
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
    let rows_written = export_to_jsonl(&src_path, &export_path);
    assert_eq!(rows_written, 7);

    let raw = fs::read_to_string(&export_path).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    let null_line = lines
        .iter()
        .find(|l| l.contains(&format!("\"id\":\"{}\"", null_id)))
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
    {
        let src = Database::open(&src_path).unwrap();
        seed_source_with_path(&src, &src_path);
    }

    let export_path = src_dir.path().join("export.jsonl");
    export_to_jsonl(&src_path, &export_path);

    let raw = fs::read_to_string(&export_path).unwrap();
    let lines: Vec<&str> = raw.lines().collect();
    let first_data_line = lines[1];
    assert!(
        first_data_line.contains("\"memory_type\":"),
        "row line must use memory_type"
    );
    assert!(
        !first_data_line.contains("\"type\":"),
        "row line must NOT contain a bare \"type\" key"
    );
}
