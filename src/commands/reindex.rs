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

/// Progress interval: print progress every N rows in human mode.
const PROGRESS_INTERVAL: usize = 50;

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
/// Returns error if the database is locked (busy_timeout=0 ensures fast-fail),
/// the embedder cannot be initialised, or all rows fail.
pub fn handle_reindex(
    db_path: &Path,
    model_id: &str,
    project_filter: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open database (busy_timeout=0 is set inside Database::open for fast-fail)
    let db = Database::open(db_path).map_err(|e| {
        // Match on rusqlite errors for structured detection of database busy state
        // Database::open returns sqlite::Error which already contains rusqlite::Error
        let err_msg = e.to_string();
        // SQLite busy errors contain "database is locked" or "database is locked (5)"
        // With busy_timeout=0, we get a fast fail instead of hanging
        if err_msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            );
        }
        Error::Config(err_msg)
    })?;

    // Determine which projects to process
    let projects: Vec<String> = if let Some(filter) = project_filter {
        vec![filter.to_string()]
    } else {
        db.list_all_project_ids()?
    };

    if projects.is_empty() {
        if json {
            print_json(&ReindexResponse {
                project_id: "(none)".to_string(),
                reindexed: 0,
                skipped: 0,
                failed: vec![],
            });
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
    let mut all_failed: Vec<ReindexFailure> = vec![];

    for project_id in &projects {
        let mut embed_callback = |content: &str| {
            engine
                .embed(content)
                .map_err(|e| Error::Inference(e.to_string()))
        };
        let (reindexed, skipped, failed) =
            reindex_project(&db, &mut embed_callback, project_id, json)?;
        let failed_count = failed.len();
        total_reindexed += reindexed;
        total_skipped += skipped;
        total_failed += failed_count;
        all_failed.extend(failed);

        if json {
            print_json(&ReindexResponse {
                project_id: project_id.clone(),
                reindexed,
                skipped,
                failed: Vec::new(), // Print per-project, failures collected separately
            });
        } else {
            println!(
                "Project {}: {} reindexed, {} skipped, {} failed",
                project_id, reindexed, skipped, failed_count
            );
        }
    }

    // Print failures summary
    if !all_failed.is_empty() && !json {
        eprintln!("Warning: {} row(s) failed during reindex", all_failed.len());
        for failure in &all_failed {
            eprintln!("  {} — {}", failure.id, failure.error);
        }
    }

    // Print total summary for all projects
    if !json {
        println!();
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

        processed += 1;
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
    db.update_embedding(id, &new_embedding)
        .map_err(|e| Error::NotFound(e.to_string()))?;
    Ok(())
}
