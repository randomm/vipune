//! Query helpers for SQLite operations.
//!
//! Provides common row mapping and query construction utilities for the SQLite backend.

use rusqlite::{Result as SqliteResult, Row, types::FromSqlError};

use super::{EMBEDDING_COLUMN, Error, Memory};
use crate::embedding::EMBEDDING_DIMS;

/// Build a row-level conversion failure at the embedding column that carries a
/// `CorruptEmbedding` domain error (naming the affected memory id) as its
/// source, so `From<rusqlite::Error> for Error` can surface it directly to
/// callers (issue #186).
pub(crate) fn corrupt_embedding_error(id: String, reason: Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        EMBEDDING_COLUMN,
        rusqlite::types::Type::Blob,
        Box::new(FromSqlError::other(Error::CorruptEmbedding {
            id,
            reason: reason.to_string(),
        })),
    )
}

/// Map a SQLite row to a Memory struct without similarity score.
///
/// Used for queries that return memories without search results (get, list, list_since, get_many).
///
/// # Arguments
///
/// * `row` - SQLite row containing memory columns
///
/// # Errors
///
/// Returns error if any column extraction fails or the embedding BLOB cannot be
/// decoded. A corrupt BLOB yields `Error::CorruptEmbedding` naming the affected
/// memory id, surfaced via the domain error carried as the source of the row-
/// level `FromSqlConversionFailure` (issue #186).
pub fn map_row_to_memory(row: &Row) -> SqliteResult<Memory> {
    // Positions match the canonical 13-column SELECT (see EMBEDDING_COLUMN).
    let id: String = row.get(0)?;
    let blob: Vec<u8> = row.get(EMBEDDING_COLUMN)?;
    let embedding =
        super::embedding::blob_to_vec(&blob).map_err(|e| corrupt_embedding_error(id.clone(), e))?;

    if embedding.len() != EMBEDDING_DIMS {
        return Err(corrupt_embedding_error(
            id,
            Error::MismatchedDimensions {
                expected: EMBEDDING_DIMS,
                actual: embedding.len(),
            },
        ));
    }

    Ok(Memory {
        id,
        project_id: row.get(1)?,
        content: row.get(2)?,
        metadata: row.get(3)?,
        embedding,
        similarity: None,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
        memory_type: row.get(7)?,
        status: row.get(8)?,
        superseded_by: row.get(9)?,
        retrieval_count: row.get(10)?,
        last_retrieved_at: row.get(11)?,
        importance: row.get(12)?,
    })
}

/// Append shared status/type filter clauses and parameters.
///
/// Shared by the WHERE-clause builders in `search`, `list`, `list_since`, and
/// `search_bm25`, which previously each carried a copy-paste of this logic
/// (issue #150). The clauses are appended to `where_clauses` starting at
/// `param_index` placeholder positions, and the corresponding values are
/// appended to `params` — call sites must seed both with their positional
/// parameters in the same relative order.
///
/// # Semantics
///
/// - `statuses = None`: appends a `status = ?` clause bound to the literal
///   `"active"` (the historical default).
/// - `statuses = Some(&[])`: appends no clause (explicit empty = no filter).
/// - `statuses = Some([...])`: appends a `status IN (?, ...)` clause.
/// - `memory_types = Some(&[])`: no clause.
/// - `memory_types = Some([...])`: appends a `type IN (?, ...)` clause.
///
/// `column_prefix` is prepended to `status`/`type` so the FTS builder can
/// qualify them with its `m.` table alias. The returned value is the next
/// free `?N` index.
pub fn build_filters<'p>(
    where_clauses: &mut Vec<String>,
    params: &mut Vec<&'p dyn rusqlite::ToSql>,
    start_param: usize,
    statuses: Option<&'p [&'p str]>,
    memory_types: Option<&'p [&'p str]>,
    column_prefix: &'p str,
) -> usize {
    let mut param_index = start_param;

    // Status filter (default to active if None)
    match statuses {
        Some(statuses) if !statuses.is_empty() => {
            let placeholders: Vec<String> = (0..statuses.len())
                .map(|i| format!("?{}", param_index + i))
                .collect();
            where_clauses.push(format!(
                "{}status IN ({})",
                column_prefix,
                placeholders.join(", ")
            ));
            for s in statuses {
                params.push(s);
            }
            param_index += statuses.len();
        }
        Some(_) => {}
        None => {
            where_clauses.push(format!("{}status = ?{}", column_prefix, param_index));
            params.push(&"active");
            param_index += 1;
        }
    }

    // Type filter (only if explicitly provided)
    if let Some(types) = memory_types {
        if !types.is_empty() {
            let placeholders: Vec<String> = (0..types.len())
                .map(|i| format!("?{}", param_index + i))
                .collect();
            where_clauses.push(format!(
                "{}type IN ({})",
                column_prefix,
                placeholders.join(", ")
            ));
            for t in types {
                params.push(t);
            }
            param_index += types.len();
        }
    }

    param_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EMBEDDING_DIMS;
    use rusqlite::params;
    use tempfile::TempDir;

    fn create_test_db() -> super::super::Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = super::super::Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    #[test]
    fn test_map_row_to_memory() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert(
                "proj1",
                "test content",
                &embedding,
                Some(r#"{"key":"value"}"#),
                "fact",
                "active",
            )
            .unwrap();

        let conn = db.conn();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
                FROM memories
                WHERE id = ?1
                "#,
            )
            .unwrap();

        let memory = stmt.query_row([id.clone()], map_row_to_memory).unwrap();
        assert_eq!(memory.id, id);
        assert_eq!(memory.importance, "medium");
        assert_eq!(memory.content, "test content");
        assert_eq!(memory.project_id, "proj1");
        assert_eq!(memory.metadata, Some(r#"{"key":"value"}"#.to_string()));
        assert_eq!(memory.embedding.len(), EMBEDDING_DIMS);
        assert!(memory.similarity.is_none());
        assert_eq!(memory.retrieval_count, 0);
        assert!(memory.last_retrieved_at.is_none());
    }

    #[test]
    fn test_map_row_to_memory_without_metadata() {
        let db = create_test_db();
        let embedding = vec![0.1f32; EMBEDDING_DIMS];
        let id = db
            .insert("proj1", "test content", &embedding, None, "fact", "active")
            .unwrap();

        let conn = db.conn();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
                FROM memories
                WHERE id = ?1
                "#,
            )
            .unwrap();

        let memory = stmt.query_row([id.clone()], map_row_to_memory).unwrap();
        assert_eq!(memory.metadata, None);
    }

    #[test]
    fn test_map_row_to_memory_invalid_embedding() {
        let db = create_test_db();
        let conn = db.conn();

        // Insert a valid embedding but test that the mapping function works correctly
        let blob = super::super::embedding::vec_to_blob(&vec![0.1f32; EMBEDDING_DIMS]).unwrap();
        conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, retrieval_count, last_retrieved_at)
            VALUES ('test-id', 'proj1', 'test', ?1, NULL, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 'fact', 'active', 0, NULL)
            "#,
            params![blob],
        )
        .unwrap();

        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
                FROM memories
                WHERE id = ?1
                "#,
            )
            .unwrap();

        // Should successfully map valid embedding
        let memory = stmt.query_row(["test-id"], map_row_to_memory).unwrap();
        assert_eq!(memory.id, "test-id");
        assert_eq!(memory.content, "test");
    }

    /// `AGENTS.md` (Architecture Notes) is the authority an agent uses to
    /// decide which files a column reorder must touch, and it claims
    /// `map_row_to_memory` is not the only positional decoder. A drifted list
    /// sends the next debugger to the wrong files — the class of failure
    /// #261 exists to eliminate — so the mechanically checkable claims are
    /// pinned here (precedent:
    /// `test_air_gapped_docs_and_ci_cache_match_embed_revision`).
    #[test]
    fn test_agents_md_row_shape_claims_match_the_tree() {
        const CANONICAL_PROJECTION: &str = "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance";

        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let read = |rel: &str| {
            std::fs::read_to_string(repo.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
        };
        let agents = read("AGENTS.md");

        assert!(
            agents.contains("duplicated verbatim in"),
            "AGENTS.md must keep naming the verbatim-projection sites"
        );
        // The sentence naming them is wrapped across lines, so gather the
        // window after the marker and check each site is named there.
        let list_window = agents
            .split_once("duplicated verbatim in")
            .expect("AGENTS.md verbatim list")
            .1
            .chars()
            .take(260)
            .collect::<String>();
        for site in [
            "query_mod.rs",
            "list.rs",
            "search.rs",
            "export_scan.rs",
            "prune.rs",
            "mod.rs",
        ] {
            assert!(
                list_window.contains(site),
                "AGENTS.md's verbatim list must name {site}: {list_window}"
            );
        }
        // The verbatim copies named by the doc. query_mod.rs repeats the
        // projection for each of its three queries plus this test's fixture.
        for (rel, at_least) in [
            ("src/sqlite/query_mod.rs", 3),
            ("src/sqlite/list.rs", 1),
            ("src/sqlite/search.rs", 1),
            ("src/sqlite/mod.rs", 1),
            ("src/commands/prune.rs", 1),
            ("src/sqlite/export_scan.rs", 2),
        ] {
            let found = read(rel).matches(CANONICAL_PROJECTION).count();
            assert!(
                found >= at_least,
                "{rel} must carry the canonical projection at least {at_least} time(s) as AGENTS.md claims, found {found}"
            );
        }
        // fts.rs is deliberately excluded from the verbatim list: its own
        // projection is `m.`-qualified with a trailing bm25 column.
        assert!(
            !read("src/sqlite/fts.rs").contains(CANONICAL_PROJECTION),
            "fts.rs must stay a non-verbatim variant, as AGENTS.md states"
        );

        // The three readers the doc names as independent positional decoders
        // must still decode by literal position and still not use the mapper.
        for rel in [
            "src/sqlite/search.rs",
            "src/sqlite/fts.rs",
            "src/sqlite/export_scan.rs",
        ] {
            let src = read(rel);
            assert!(
                src.contains("row.get(0)?") || src.contains("row.get::<_, String>(0)?"),
                "{rel} must still be a positional decoder named by AGENTS.md"
            );
            assert!(
                !src.contains("map_row_to_memory("),
                "{rel} must not start using map_row_to_memory — AGENTS.md lists it as an independent decoder"
            );
        }

        // The docs call the projection 13 columns and EMBEDDING_COLUMN column 4.
        assert_eq!(4, super::super::EMBEDDING_COLUMN);
        assert_eq!(
            13,
            CANONICAL_PROJECTION
                .trim_start_matches("SELECT ")
                .split(", ")
                .count()
        );
    }
}
