//! `reindex --force` pre-flight check for the model-switch migration
//! (issue #217, decision 1).
//!
//! Before the migration marker is written, every stored row's content is
//! token-counted with the target profile's passage prefix. If any row would
//! exceed the 512-token limit once prefixed, the command refuses to start,
//! writes nothing, and lists the offending ids — the marker must only ever
//! be written in a state the re-embed pass can complete.

use crate::embedding::EmbeddingEngine;
use crate::embedding::MAX_EMBEDDING_TOKENS;
use crate::sqlite::Database;
use crate::sqlite::Error;

/// Token-count every row's content in each project with the target profile's
/// passage prefix and refuse to start (returning the offending ids) if any
/// row would exceed the token limit once embedded.
///
/// Rows are listed per project via `list_all_rows_for_project`; a row with a
/// corrupted embedding is not skipped here — the pre-flight is about token
/// count, not embedding validity — and a token-count error on any row is
/// propagated as an `Error::Sqlite` so the caller can surface it.
///
/// # Arguments
///
/// * `db` - The database to scan
/// * `projects` - The project ids to scan (every project, not just the
///   current one, so a multi-project store can't smuggle an over-length row
///   past the check via a project-scoped reindex)
/// * `passage_prefix` - The target profile's passage prefix (prepended to
///   each row's content before the token count, matching how the re-embed
///   loop will embed the row)
/// * `count` - Token-count function for the target profile's tokenizer
///
/// # Returns
///
/// `Ok(Vec<offending_ids>)` — empty when every row fits the token budget
/// (the pass may proceed), or the list of memory ids whose prefixed content
/// exceeds the limit (the caller must refuse to start and print them).
///
/// # Errors
///
/// Returns `Error::Sqlite` if a row's content cannot be token-counted.
pub fn force_reembed_preflight<C>(
    db: &Database,
    projects: &[String],
    passage_prefix: &str,
    count: C,
) -> Result<Vec<String>, Error>
where
    C: Fn(&str) -> Result<usize, Error>,
{
    let mut offending: Vec<String> = Vec::new();
    for project_id in projects {
        let rows = db.list_all_rows_for_project(project_id)?;
        for (id, content, _embedding) in rows {
            let prefixed = format!("{passage_prefix}{content}");
            let count = count(&prefixed).map_err(|e| Error::Sqlite(e.to_string()))?;
            if count > MAX_EMBEDDING_TOKENS {
                offending.push(id);
            }
        }
    }
    Ok(offending)
}

/// Pre-flight convenience wrapper that token-counts via the engine's
/// tokenizer (no embedding work is done).
pub fn force_reembed_preflight_with_engine(
    db: &Database,
    engine: &EmbeddingEngine,
    projects: &[String],
    passage_prefix: &str,
) -> Result<Vec<String>, Error> {
    force_reembed_preflight(db, projects, passage_prefix, |text| {
        engine
            .token_count(text)
            .map_err(|e| Error::Sqlite(e.to_string()))
    })
}
