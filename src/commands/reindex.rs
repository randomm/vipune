//! `vipune reindex` handler.
//!
//! Re-embeds rows classified as mock, leaves real rows byte-identical, skips unknown.

use crate::embedding::EmbeddingEngine;
use crate::errors::Error;
use crate::output::{ReindexFailure, ReindexResponse, print_json};
use crate::sqlite::Database;
use crate::sqlite::embedding::classify_embedding;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

/// Progress interval: print progress every N rows in human mode.
const PROGRESS_INTERVAL: usize = 50;

/// Wrap a database error, converting SQLITE_BUSY into the actionable MCP-server message.
fn wrap_busy<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(Error::SqliteModule(msg)) if msg.contains("database is locked") => {
            Err(Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            ))
        }
        Err(e) => Err(e),
    }
}

/// Run the reindex operation on the database.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database
/// * `model_id` - HuggingFace model ID for the embedding engine
/// * `project_filter` - If Some, only reindex this project; if None, reindex all projects
/// * `json` - If true, output JSON; otherwise human-readable
///
/// # Errors
///
/// Returns error if the database is locked (we set busy_timeout=0 for fast-fail),
/// the embedder cannot be initialised, or all rows fail.
pub fn handle_reindex(
    db_path: &Path,
    model_id: &str,
    project_filter: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open database
    let db = Database::open(db_path).map_err(|e| {
        let err_msg = e.to_string();
        if err_msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            );
        }
        Error::Config(err_msg)
    })?;

    // Set busy timeout to 0ms for fast-fail behavior on database locks (reindex-specific)
    db.set_busy_timeout(Duration::ZERO)?;

    // Determine which projects to process
    let projects: Vec<String> = if let Some(filter) = project_filter {
        vec![filter.to_string()]
    } else {
        wrap_busy(db.list_all_project_ids().map_err(Error::from))?
    };

    if projects.is_empty() {
        if json {
            print_json(&[ReindexResponse {
                project_id: "(none)".to_string(),
                reindexed: 0,
                skipped: 0,
                failed: vec![],
            }]);
        } else {
            println!("No projects found in database. Nothing to reindex.");
        }
        return Ok(ExitCode::SUCCESS);
    }

    // Initialise the embedding engine (downloads model if needed)
    let mut engine = EmbeddingEngine::new(model_id)?;

    let mut total_reindexed: usize = 0;
    let mut total_skipped: usize = 0;
    let mut total_failed: usize = 0;
    let mut responses: Vec<ReindexResponse> = vec![];

    for project_id in &projects {
        let mut embed_callback = |content: &str| {
            engine
                .embed(content)
                .map_err(|e| Error::Inference(e.to_string()))
        };
        let (reindexed, skipped, failed) =
            wrap_busy(reindex_project(&db, &mut embed_callback, project_id, json))?;
        let failed_count = failed.len();
        total_reindexed += reindexed;
        total_skipped += skipped;
        total_failed += failed_count;
        responses.push(ReindexResponse {
            project_id: project_id.clone(),
            reindexed,
            skipped,
            failed,
        });

        if !json {
            println!(
                "Project {}: {} reindexed, {} skipped, {} failed",
                project_id, reindexed, skipped, failed_count
            );
        }
    }

    // Print JSON: single array of all project responses
    if json {
        print_json(&responses);
    }

    // Print failures summary to stderr
    if !json {
        let any_failures = responses.iter().any(|r| !r.failed.is_empty());
        if any_failures {
            eprintln!("Warning: {} row(s) failed during reindex", total_failed);
            for response in &responses {
                for failure in &response.failed {
                    eprintln!("  {} — {}", failure.id, failure.error);
                }
            }
            println!();
        }
    }

    // Print total summary for all projects (always shown in human mode)
    if !json {
        println!("Total across all projects:");
        println!("  Reindexed: {}", total_reindexed);
        println!("  Skipped:   {}", total_skipped);
        println!("  Failed:    {}", total_failed);
    }

    Ok(ExitCode::SUCCESS)
}

pub(crate) fn reindex_project<F>(
    db: &Database,
    embed: &mut F,
    project_id: &str,
    json: bool,
) -> Result<(usize, usize, Vec<ReindexFailure>), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    let rows = db.list_all_rows_for_project(project_id)?;

    let mut reindexed: usize = 0;
    let mut skipped: usize = 0;
    let mut failed: Vec<ReindexFailure> = vec![];
    let mut processed: usize = 0;

    for (id, content, embedding) in rows {
        processed += 1;

        // Classify
        match classify_embedding(&embedding) {
            crate::sqlite::embedding::EmbeddingClass::Real => {
                // Real row — leave byte-identical
            }
            crate::sqlite::embedding::EmbeddingClass::Unknown => {
                // Corrupted/unknown — skip
                skipped += 1;
                if !json {
                    eprintln!(
                        "  Skipping {} — unknown embedding (norm not in valid range)",
                        id
                    );
                }
                continue;
            }
            crate::sqlite::embedding::EmbeddingClass::Mock => {
                // Re-embed this row
                match reindex_row(embed, &id, &content, db) {
                    Ok(()) => {
                        reindexed += 1;
                    }
                    Err(e) => {
                        failed.push(ReindexFailure {
                            id: id.clone(),
                            error: e.to_string(),
                        });
                        if !json {
                            eprintln!("  Failed {} — {}", id, e);
                        }
                        continue;
                    }
                }
            }
        }

        if !json && processed % PROGRESS_INTERVAL == 0 {
            println!(
                "  Progress: {} processed ({} reindexed, {} skipped, {} failed)",
                processed,
                reindexed,
                skipped,
                failed.len()
            );
        }
    }

    Ok((reindexed, skipped, failed))
}

fn reindex_row<F>(embed: &mut F, id: &str, content: &str, db: &Database) -> Result<(), Error>
where
    F: FnMut(&str) -> Result<Vec<f32>, Error>,
{
    let new_embedding = embed(content)?;
    db.update_embedding(id, &new_embedding)?;
    Ok(())
}
