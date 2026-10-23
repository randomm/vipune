//! The `importance` field on the import path (issue #228).
//!
//! `vipune export` wrote 12 of the 13 columns and `run_import`'s
//! in-transaction INSERT — the live write path — named the same 12, so SQLite
//! applied the `NOT NULL DEFAULT 'medium'` and every restored row silently
//! lost the level it was stored with. These pin both halves of the contract: a
//! present value is restored (and refused when it names no level), and an
//! absent key keeps restoring the `medium` default that every file written by
//! v0.10.0–v0.15.x relies on. Model-free: full-width `vec_to_blob` embeddings.

#![cfg(test)]

use crate::commands::export::base64_encode;
use crate::commands::import::run_import;
use crate::embedding::EMBEDDING_DIMS;
use crate::sqlite::Database;
use crate::sqlite::embedding::vec_to_blob;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// A distinct full-length embedding, so the import's dimension gate is happy.
fn blob_for(seed: usize) -> Vec<u8> {
    let vec: Vec<f32> = (0..EMBEDDING_DIMS)
        .map(|i| (i as f32 + seed as f32 * 7.0) * 0.001)
        .collect();
    vec_to_blob(&vec).expect("384-dim vector")
}

/// One JSONL data line: the 12 fields every file through v0.15.x carries, plus
/// the `importance` key only when the caller names a value — so `None` is the
/// genuine key-free legacy shape and `Value::Null` the explicit-null one.
fn row_line(id: &str, seed: usize, importance: Option<Value>) -> Value {
    let mut row = json!({
        "id": id,
        "project_id": "proj",
        "content": "content",
        "metadata": null,
        "embedding": base64_encode(&blob_for(seed)),
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "memory_type": "fact",
        "status": "active",
        "superseded_by": null,
        "retrieval_count": 0,
        "last_retrieved_at": null,
    });
    if let Some(level) = importance {
        row["importance"] = level;
    }
    row
}

/// Write `rows` under a real header line (line 1) and return the file path.
fn write_file(dir: &Path, name: &str, rows: &[Value]) -> PathBuf {
    let path = dir.join(name);
    let mut text = format!(
        "{{\"type\":\"export\",\"format_version\":1,\"embedding_dims\":{EMBEDDING_DIMS},\"exported_at\":\"2024-01-01T00:00:00Z\",\"rows\":{}}}\n",
        rows.len()
    );
    for row in rows {
        text.push_str(&serde_json::to_string(row).expect("row json"));
        text.push('\n');
    }
    std::fs::write(&path, text).expect("write jsonl");
    path
}

/// A freshly created (empty) database, ready to import into.
fn fresh_db(dir: &Path) -> PathBuf {
    let db_path = dir.join("test.db");
    drop(Database::open(&db_path).expect("create db"));
    db_path
}

/// Read a column straight from SQLite: what the INSERT wrote is the question,
/// not what a `Memory` read path chooses to surface.
fn query<T: rusqlite::types::FromSql>(db_path: &Path, sql: &str) -> T {
    rusqlite::Connection::open(db_path)
        .expect("open for read")
        .query_row(sql, [], |r| r.get(0))
        .unwrap_or_else(|e| panic!("query {sql}: {e}"))
}

#[test]
fn test_import_row_without_importance_key_defaults_to_medium() {
    // The back-compat acceptance criterion: no new required field. Every file
    // written by v0.10.0–v0.15.x has no such key and must keep importing.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = fresh_db(dir.path());

    let row = row_line("legacy-1", 1, None);
    assert!(
        !row.as_object().unwrap().contains_key("importance"),
        "fixture must be the key-free legacy shape: {row}"
    );
    let file = write_file(dir.path(), "legacy.jsonl", &[row]);

    let resp = run_import(&db_path, file.to_str().unwrap()).expect("legacy row imports");
    assert_eq!(resp.inserted, 1);
    assert_eq!(
        query::<String>(
            &db_path,
            "SELECT importance FROM memories WHERE id = 'legacy-1'"
        ),
        "medium"
    );
}

#[test]
fn test_import_rejects_malformed_importance_naming_the_line() {
    // The column has no CHECK constraint, so an "urgent" row would otherwise
    // import, break every read, and silently drop prune protection. Absent is
    // the legacy default; present-but-empty and an explicit null are malformed,
    // not "no opinion" — and the whole file is refused, valid rows included.
    let cases = [
        ("out-of-vocabulary", Some(json!("urgent")), "urgent"),
        ("empty", Some(json!("")), "Invalid importance"),
        ("explicit null", Some(Value::Null), "null"),
    ];

    for (name, value, needle) in cases {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = fresh_db(dir.path());
        let file = write_file(
            dir.path(),
            "bad.jsonl",
            &[row_line("ok-1", 0, None), row_line("bad-1", 1, value)],
        );

        let err = run_import(&db_path, file.to_str().unwrap()).expect_err(&format!(
            "a {name} importance must be refused, not defaulted"
        ));
        let msg = err.to_string();
        assert!(
            msg.contains("line 3") && msg.contains(needle),
            "a malformed {name} importance must name its line (the header is line 1) and value: {msg}"
        );
        assert_eq!(
            query::<i64>(&db_path, "SELECT COUNT(*) FROM memories"),
            0,
            "all-or-nothing for {name}"
        );
    }
}
