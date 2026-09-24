//! `reindex --force` CLI output contract (issue #221).
//!
//! `reindex --force` is a thin wrapper over the library's `crate::migration`
//! module: its stdout/stderr and exit codes must stay byte-identical to
//! v0.14.0. These tests drive the wrapper end-to-end (real engine, locally
//! cached model) and pin the rendered contract — the "Migrating from … to
//! …" / "Resuming interrupted migration to …" banners, the pre-flight
//! refusal block (rendered from `MigrationRefused`), the skipped-corrupted
//! note, and the JSON summary — while the pre-flight test below additionally
//! pins the BLOB invariance.
//!
//! The capturing tests spawn the compiled `vipune` binary as a child process
//! with piped stdout/stderr, so the rendered output is captured without any
//! in-process fd manipulation (which conflicts with the test harness's own
//! output reader).

#![cfg(test)]

use crate::memory::crud::test_fake_embedder;
use crate::sqlite::Database;
use crate::sqlite::identity;

fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    Database::open(&path).unwrap();
    (dir, path)
}

fn get_embedding(db: &Database, id: &str) -> Vec<f32> {
    db.list_all_rows_for_project("proj")
        .unwrap()
        .iter()
        .find(|(i, _, _)| i == id)
        .map(|(_, _, e)| e.clone())
        .unwrap()
}

/// Write a migration marker (marker-first step: touches ONLY the marker
/// column; the recorded identity is left untouched).
fn marker_only(conn: &rusqlite::Connection, target: &identity::ModelIdentity) {
    conn.execute(
        "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
         VALUES (1, NULL, NULL, ?1)
         ON CONFLICT(id) DO UPDATE SET migration_marker = excluded.migration_marker",
        [crate::migration::migration_marker_for(target)],
    )
    .unwrap();
}

/// Path to the compiled `vipune` binary (the test harness builds it).
fn vipune_bin() -> std::path::PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let mut path = exe;
    // Walk up from target/debug/deps/<name>-<hash> to target/debug/vipune.
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

/// Spawn `vipune reindex --force` as a child process with captured stdout
/// and stderr, and return its exit code plus both streams as strings.
///
/// Subprocess capture is used (rather than in-process fd swapping) because
/// the test harness's own output reader conflicts with in-process fd swaps.
fn capture_force_output(db_path: &std::path::Path, json: bool) -> (i32, String, String) {
    let mut cmd = std::process::Command::new(vipune_bin());
    cmd.arg("reindex")
        .arg("--force")
        .arg("--all-projects")
        .arg("--db-path")
        .arg(db_path);
    if json {
        cmd.arg("--json");
    }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let output = cmd.output().expect("spawn vipune reindex --force");
    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (code, stdout, stderr)
}

#[test]
fn test_force_json_success_output() {
    // Clean pass, JSON mode: stdout is exactly the JSON summary array and
    // stderr is empty. Pins the wrapper's JSON rendering contract.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    drop(db);

    let (exit, out, err) = capture_force_output(&db_path, true);
    assert_eq!(exit, 0, "clean pass must exit 0");
    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(out.trim()).expect("stdout must be the JSON summary");
    assert_eq!(parsed.len(), 1);
    let obj = parsed[0].as_object().expect("summary must be an object");
    assert_eq!(obj["project_id"].as_str(), Some("project proj"));
    assert_eq!(obj["reindexed"].as_u64(), Some(1));
    assert_eq!(obj["skipped"].as_u64(), Some(0));
    assert_eq!(obj["failed"].as_array().map(|a| a.len()), Some(0));
    assert!(
        err.trim().is_empty(),
        "clean pass must write nothing to stderr, got: {err:?}"
    );
}

#[test]
fn test_force_migration_banner_on_mismatch() {
    // Recorded identity differs from the target → the wrapper prints the
    // "Migrating from … to …" banner first, then the success line and the
    // human-mode summary. Pins the banner + success-line rendering.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    let old = identity::ModelIdentity {
        model_id: "intfloat/multilingual-e5-small".to_string(),
        revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3".to_string(),
    };
    db.conn()
        .execute(
            "INSERT INTO model_identity (id, model_id, model_revision, migration_marker)
             VALUES (1, ?1, ?2, NULL)",
            (&old.model_id, &old.revision),
        )
        .unwrap();
    drop(db);

    let (exit, out, err) = capture_force_output(&db_path, false);
    assert_eq!(exit, 0, "clean pass must exit 0");
    assert_eq!(
        out.lines().next(),
        Some(
            "Migrating from intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 to BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a..."
        ),
        "banner must be the first stdout line, got: {out:?}"
    );
    assert!(
        out.contains(
            "Model identity updated to BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a (marker cleared)."
        ),
        "success line missing, got: {out:?}"
    );
    assert!(out.contains("Total across project proj:"));
    assert!(out.contains("  Reindexed: 1"));
    assert!(out.contains("  Skipped:   0"));
    assert!(out.contains("  Failed:    0"));
    assert!(
        err.trim().is_empty(),
        "clean pass must write nothing to stderr, got: {err:?}"
    );

    // Post-state: identity recorded, marker cleared.
    let db = Database::open(&db_path).unwrap();
    assert_eq!(
        identity::read_identity(db.conn()).unwrap(),
        Some(identity::ModelIdentity::default_identity())
    );
    assert_eq!(identity::read_marker(db.conn()).unwrap(), None);
}

#[test]
fn test_force_resuming_banner_when_marker_present() {
    // A marker is already present (interrupted migration) → the wrapper
    // prints "Resuming interrupted migration to …" instead of the banner.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    marker_only(db.conn(), &identity::ModelIdentity::default_identity());
    drop(db);

    let (exit, out, _err) = capture_force_output(&db_path, false);
    assert_eq!(exit, 0, "clean pass must exit 0");
    assert_eq!(
        out.lines().next(),
        Some(
            "Resuming interrupted migration to BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a..."
        ),
        "resuming banner must be the first stdout line, got: {out:?}"
    );
}

#[test]
fn test_force_skipped_corrupted_note_and_success_exit() {
    // A corrupted (zero-vector) row is skipped, not failed: the wrapper
    // still exits 0 and prints the skipped-corrupted note to stderr.
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    db.insert("proj", "bad", &vec![0.0; 384], None, "fact", "active")
        .unwrap();
    drop(db);

    let (exit, _out, err) = capture_force_output(&db_path, false);
    assert_eq!(exit, 0, "clean pass must exit 0");
    assert_eq!(
        err.trim(),
        "1 row(s) had corrupted embeddings and were skipped",
        "skipped-corrupted note must be byte-identical"
    );
}

/// Drive the wrapper's per-row-failure path with a failing embed closure
/// through `crate::migration::migrate_model_with_embedder` (the same
/// lifecycle the wrapper delegates to). Pins the `MigrationIncomplete`
/// error shape the wrapper renders (failure count + marker-stays note) plus
/// the post-state: marker kept, old identity untouched, failed row's BLOB
/// stale.
#[test]
fn test_force_per_row_failure_error_shape_and_post_state() {
    let (_dir, db_path) = create_test_db();
    let mut db = Database::open(&db_path).unwrap();
    db.insert(
        "proj",
        "ok content",
        &test_fake_embedder("c").unwrap(),
        None,
        "fact",
        "active",
    )
    .unwrap();
    let fail_id = db
        .insert(
            "proj",
            "bad content",
            &test_fake_embedder("d").unwrap(),
            None,
            "fact",
            "active",
        )
        .unwrap();

    // `Config::load()` resolves the full default config (including the
    // home-dir database path). The migration only reads `embedding_model`
    // and `database_path` here; the other fields are carried through as-is.
    let mut config = crate::config::Config::load().unwrap();
    config.database_path = db_path.clone();
    config.embedding_model = "BAAI/bge-small-en-v1.5".to_string();
    let mut embed = |content: &str| {
        if content == "bad content" {
            Err(crate::sqlite::Error::Sqlite(
                "simulated embed failure".to_string(),
            ))
        } else {
            test_fake_embedder(content).map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
        }
    };
    let mut count = |content: &str| -> Result<usize, crate::sqlite::Error> {
        Ok(content.split_whitespace().count())
    };
    let result =
        crate::migration::migrate_model_with_embedder(&mut db, &config, &mut embed, &mut count);
    let err = result.expect_err("a failing row must refuse the pass");
    match err {
        crate::sqlite::Error::MigrationIncomplete { report } => {
            assert_eq!(report.failures.len(), 1, "one failure in the report");
            assert_eq!(report.failures[0].id, fail_id);
        }
        other => panic!("expected MigrationIncomplete, got {other:?}"),
    }

    // Post-state: marker kept, old identity (unrecorded = default) kept,
    // failed row's BLOB stale, ok row re-embedded.
    assert!(
        identity::is_migrating(db.conn()).unwrap(),
        "marker must stay"
    );
    assert_eq!(
        identity::read_identity(db.conn()).unwrap(),
        None,
        "old identity (unrecorded default) must be kept"
    );
    assert_eq!(
        get_embedding(&db, &fail_id),
        test_fake_embedder("d").unwrap(),
        "failed row keeps its stale vector"
    );
}

#[test]
fn test_force_preflight_refusal_cli_output_and_blob_invariance() {
    // Pre-flight: a row over the 512-token limit once the target profile's
    // passage prefix is applied must refuse the start with the rendered
    // refusal block, exit 1, no marker, and byte-identical embedding BLOBs.
    // (The bin pre-flight test only checks the offending-id list; this
    // pins the CLI-level no-marker/no-row-changed invariance.)
    let (_dir, db_path) = create_test_db();
    let db = Database::open(&db_path).unwrap();

    // 600 words is safely over the 512-token limit for any prefix.
    let long_content: String = (0..600)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let long_id = db
        .insert(
            "proj",
            &long_content,
            &test_fake_embedder("long").unwrap(),
            None,
            "fact",
            "active",
        )
        .unwrap();
    let short_id = db
        .insert(
            "proj",
            "short content",
            &test_fake_embedder("short").unwrap(),
            None,
            "fact",
            "active",
        )
        .unwrap();
    let long_blob_before = get_embedding(&db, &long_id);
    let short_blob_before = get_embedding(&db, &short_id);
    drop(db);

    let (exit, _out, err) = capture_force_output(&db_path, false);
    assert_eq!(exit, 1, "pre-flight refusal must exit 1");
    // The rendered refusal block: header, offending id, footer.
    assert!(
        err.contains(
            "Error: reindex --force refused to start: 1 row(s) exceed the 512-token limit once the '' passage prefix is prepended. Fix or remove these memories, then re-run `vipune reindex --force`:"
        ),
        "refusal header missing, got: {err:?}"
    );
    assert!(
        err.contains(&format!("  {long_id}")),
        "offending id missing from refusal block"
    );
    assert!(
        !err.contains(&format!("  {short_id}")),
        "short row must not be reported"
    );
    assert!(
        err.contains("No migration marker was written and no rows were changed."),
        "refusal footer missing"
    );

    // BLOB invariance + no marker at the CLI level.
    let db = Database::open(&db_path).unwrap();
    assert_eq!(
        get_embedding(&db, &long_id),
        long_blob_before,
        "long row BLOB must be byte-identical"
    );
    assert_eq!(
        get_embedding(&db, &short_id),
        short_blob_before,
        "short row BLOB must be byte-identical"
    );
    assert_eq!(
        identity::read_marker(db.conn()).unwrap(),
        None,
        "no marker on pre-flight refusal"
    );
}
