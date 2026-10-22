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
//! # Division of labour with `src/commands/import_export_roundtrip_tests.rs`
//!
//! One layer split, stated once in that file's module doc: this file owns the
//! process-level contracts (exit code, `Exported` stdout, `--json` response
//! shape, LF line endings, export-side-only corrupt/NULL BLOB shapes); the
//! in-crate file owns the column-level contracts and the header-key pins.
//!
//! Model-free: rows are seeded through the public `Database::insert` API with
//! a self-contained fake embedder (no ONNX model, no `test_fake_embedder`),
//! and BLOB-level assertions go through the dev-dependency `rusqlite`
//! connection because `Database::conn()` is `pub(crate)`.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use rusqlite::{Connection, params};
use vipune::Database;

/// Longest a single `vipune` child may take before the harness declares the
/// run hung. Generous for a debug binary over local SQLite (real runs are
/// milliseconds-to-seconds); its job is only to turn a silently stalled CI
/// job into a named failure.
const CLI_TIMEOUT: Duration = Duration::from_secs(30);

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

/// Read a JSONL file whole. One reader for the whole file; every failure
/// names the path and carries the OS error so a missing or truncated export
/// reports itself.
fn read_jsonl(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read jsonl {} ({e})", path.display()))
}

/// Read a JSONL file back as owned lines.
fn lines_of(path: &Path) -> Vec<String> {
    read_jsonl(path).lines().map(str::to_string).collect()
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

/// A seeded source row: the four fields the round-trip assertions compare.
/// Named fields, never a positional tuple: four same-shaped columns are the
/// "swap two fields and the test still compiles" hazard (mirrors `SeedRow`
/// in the in-crate twin).
struct Seeded {
    id: String,
    blob: Vec<u8>,
    retrieval_count: i64,
    last_retrieved_at: Option<String>,
}

/// Seed six rows across two projects with full-width BLOBs and explicit
/// retrieval telemetry.
fn seed_source_with_path(db: &Database, db_path: &Path) -> Vec<Seeded> {
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
        seeded.push(Seeded {
            id,
            blob: stored_blob,
            retrieval_count: rc,
            last_retrieved_at: lr.map(str::to_string),
        });
    }
    seeded
}

/// Where the fixtures write the export for a given source store — defined
/// once so no caller re-spells the file name.
fn export_path_of(src_dir: &tempfile::TempDir) -> std::path::PathBuf {
    src_dir.path().join("export.jsonl")
}

// ── Real CLI driver (the code path a user actually runs) ──

/// Path to the compiled `vipune` binary (the test harness builds it).
/// Walks up from `target/debug/deps/<test>-<hash>` to `target/debug/vipune`,
/// once — the result is memoised so ~13 spawns do not re-stat the walk.
fn vipune_bin() -> &'static Path {
    static BIN: OnceLock<std::path::PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
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
    })
}

/// One captured `vipune` invocation.
struct CliRun {
    /// Rendered wait status (`exit status: N` or `signal: N (SIG..)`), so a
    /// signal-kill names itself in failure messages.
    status: String,
    /// Exit code, or `None` when the child was killed by a signal or the
    /// harness timed it out.
    code: Option<i32>,
    /// True when the child exceeded [`CLI_TIMEOUT`] and was killed.
    timed_out: bool,
    stdout: String,
    stderr: String,
}

impl CliRun {
    /// Everything a failed invocation needs to be diagnosed from CI logs.
    fn detail(&self) -> String {
        let mut out = String::new();
        if self.timed_out {
            out.push_str(&format!(
                "TIMED OUT after {}s and was killed;\n",
                CLI_TIMEOUT.as_secs()
            ));
        }
        out.push_str(&format!(
            "status: {}\nstdout: {:?}\nstderr: {:?}",
            self.status, self.stdout, self.stderr
        ));
        out
    }
}

/// Run `vipune [global flags] <args...>` and capture (exit status, stdout,
/// stderr) under a hard [`CLI_TIMEOUT`].
///
/// Both `export` and `import` print informational warnings on stderr even on
/// success (`--project ... is ignored`), so assertions belong on the exit
/// code and stdout, never on an empty stderr.
///
/// The raw wait status is carried alongside the code because
/// `ExitStatus::code()` is `None` for a signal-kill: collapsing those to a
/// sentinel like `-1` makes a SIGSEGV/SIGKILL/OOM indistinguishable from a
/// clean non-zero exit that printed nothing.
///
/// Stdout/stderr are drained on reader threads whose results arrive on
/// channels with `recv_timeout`: `Command::output()` alone would block
/// forever on a hung child (SQLite lock contention, a blocking busy_timeout
/// wait), stalling the whole integration binary silently.
fn run_vipune(db_path: &Path, args: &[&str]) -> CliRun {
    let mut cmd = Command::new(vipune_bin());
    cmd.arg("--db-path")
        .arg(db_path)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn vipune");
    let mut stdout_pipe = child.stdout.take().expect("piped stdout");
    let mut stderr_pipe = child.stderr.take().expect("piped stderr");

    let (out_tx, out_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        let _ = out_tx.send(buf);
    });
    let (err_tx, err_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        let _ = err_tx.send(buf);
    });

    let deadline = Instant::now() + CLI_TIMEOUT;
    let mut timed_out = false;
    let mut read_bounded = |rx: mpsc::Receiver<Vec<u8>>| match rx.recv_timeout(CLI_TIMEOUT) {
        Ok(bytes) => bytes,
        Err(_) => {
            timed_out = true;
            let _ = child.kill();
            let remaining = deadline.saturating_duration_since(Instant::now());
            rx.recv_timeout(remaining).unwrap_or_default()
        }
    };
    let stdout = read_bounded(out_rx);
    let stderr = read_bounded(err_rx);

    let status = child.wait().expect("wait for vipune");
    let status_text = if timed_out {
        format!("timed out after {}s; {:?}", CLI_TIMEOUT.as_secs(), status)
    } else {
        format!("{status:?}")
    };
    CliRun {
        status: status_text,
        code: status.code(),
        timed_out,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    }
}

/// Run the real `vipune export` and return the JSONL lines.
fn cli_export(db_path: &Path, out_path: &Path) -> Vec<String> {
    let out_arg = out_path.to_str().expect("utf-8 out path");
    let run = run_vipune(db_path, &["export", out_arg]);
    assert_eq!(run.code, Some(0), "vipune export failed:\n{}", run.detail());
    assert!(
        run.stdout.contains("Exported"),
        "export must report the row count on stdout: {:?}",
        run.stdout
    );
    // Read after the child exited cleanly: a missing file here means the
    // exit-0 child lied, and `read_jsonl` names the path and the OS error.
    lines_of(out_path)
}

/// Run the real `vipune import --json` and return `(inserted, skipped)`
/// parsed from the machine-readable response.
fn cli_import(db_path: &Path, jsonl_path: &Path) -> (usize, usize) {
    let src_arg = jsonl_path.to_str().expect("utf-8 source path");
    let run = run_vipune(db_path, &["import", src_arg, "--json"]);
    assert_eq!(run.code, Some(0), "vipune import failed:\n{}", run.detail());
    // The whole stdout is carried in the panic: field-name drift between the
    // writer and the reader is the #229 failure class, and a bare serde
    // message with no offending bytes cannot be diagnosed from CI logs.
    let value: serde_json::Value = serde_json::from_str(run.stdout.trim()).unwrap_or_else(|e| {
        panic!(
            "import --json must print one JSON object ({e}); stdout: {:?}",
            run.stdout
        )
    });
    (
        value["inserted"]
            .as_u64()
            .unwrap_or_else(|| panic!("import --json response missing `inserted`: {value}"))
            as usize,
        value["skipped"]
            .as_u64()
            .unwrap_or_else(|| panic!("import --json response missing `skipped`: {value}"))
            as usize,
    )
}

/// Seed a source store and export it with the real CLI.
///
/// Returns the seeded rows and the path the export was actually written to,
/// so callers read the same file the helper produced instead of re-deriving
/// its name.
fn export_from_seeded_source(
    src_dir: &tempfile::TempDir,
    src_path: &Path,
) -> (Vec<Seeded>, std::path::PathBuf) {
    let db = Database::open(src_path).unwrap();
    let seeded = seed_source_with_path(&db, src_path);
    assert_eq!(seeded.len(), 6);
    let export_path = export_path_of(src_dir);
    let lines = cli_export(src_path, &export_path);
    assert_eq!(lines.len(), 7, "1 header + 6 data rows");
    (seeded, export_path)
}

// ── Acceptance tests ──

#[test]
fn test_roundtrip_embedding_blob_byte_identical() {
    let (src_dir, src_path) = create_test_db();
    let (seeded, export_path) = export_from_seeded_source(&src_dir, &src_path);

    // The header-key contract itself is pinned once, in-crate, in
    // `src/commands/import_export_roundtrip_tests.rs`
    // (`test_writer_header_deserialises_through_the_shared_schema`). Here the
    // process-level guarantee is the pair meeting for real: the binary's own
    // export imports into the binary's own fresh store.
    let (_dst_dir, dst_path) = create_test_db();
    let (inserted, skipped) = cli_import(&dst_path, &export_path);
    assert_eq!(inserted, 6, "every exported row must import");
    assert_eq!(skipped, 0);

    for row in &seeded {
        let dst_blob = raw_embedding_blob(&dst_path, &row.id)
            .expect("row must exist in destination after import");
        assert_eq!(
            row.blob, dst_blob,
            "embedding BLOB must be byte-identical for id {}",
            row.id
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
    let (seeded, export_path) = export_from_seeded_source(&src_dir, &src_path);

    let (_dst_dir, dst_path) = create_test_db();
    cli_import(&dst_path, &export_path);

    for row in &seeded {
        let (dst_rc, dst_lr) = fetch_telemetry(&dst_path, &row.id);
        assert_eq!(
            dst_rc, row.retrieval_count,
            "retrieval_count must be verbatim for {}",
            row.id
        );
        assert_eq!(
            dst_lr.as_deref(),
            row.last_retrieved_at.as_deref(),
            "last_retrieved_at must be verbatim for {}",
            row.id
        );
    }
}

#[test]
fn test_import_rerun_skips_existing_and_counts() {
    let (src_dir, src_path) = create_test_db();
    let (_, export_path) = export_from_seeded_source(&src_dir, &src_path);

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
    let (_, export_path) = export_from_seeded_source(&src_dir, &src_path);

    let raw = read_jsonl(&export_path);
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

    let export_path = export_path_of(&src_dir);
    let lines = cli_export(&src_path, &export_path);
    assert_eq!(lines.len(), 8, "corrupt row must be included in export");

    let corrupt_line = lines
        .iter()
        .find(|l| l.contains(&format!("\"id\":\"{corrupt_id}\"")))
        .expect("corrupt row must be present in export");
    let v: serde_json::Value = serde_json::from_str(corrupt_line).unwrap();
    // Decoded with the `base64` crate — the same decoder the production
    // importer uses — so this assertion validates the export against the
    // implementation that actually has to accept it.
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(v["embedding"].as_str().unwrap())
        .expect("exported embedding must be valid base64");
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

    let export_path = export_path_of(&src_dir);
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
    let (_, export_path) = export_from_seeded_source(&src_dir, &src_path);

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
