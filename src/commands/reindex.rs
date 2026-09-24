//! `vipune reindex` handler.
//!
//! Re-embeds rows classified as mock, leaves real rows byte-identical, skips unknown.
//!
//! `--force` (issue #217): model-switch migration path. Writes a
//! "migrating to `<id>@<revision>`" marker first, re-embeds every row
//! (bypassing Mock/Real classification) with the configured profile, then
//! in ONE transaction records the new identity and clears the marker. A
//! re-run after an interruption re-embeds every row from the start (full
//! idempotent pass). Plain reindex (no `--force`) still re-embeds only
//! Mock-classified rows.

use crate::commands::reindex_force::{self, ReembedFailure};
use crate::embedding::{EmbeddingEngine, MAX_EMBEDDING_TOKENS};
use crate::embedding_profiles::profile_for;
use crate::errors::Error;
use crate::output::{ReindexFailure, ReindexResponse, print_json};
use crate::sqlite::Database;
use crate::sqlite::embedding::classify_embedding;
use crate::sqlite::identity;
use crate::sqlite::identity::ModelIdentity;
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
/// * `force` - If true, full model-switch migration (marker → re-embed all rows → record identity)
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
    force: bool,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open database
    let mut db = Database::open(db_path).map_err(|e| {
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

    // Determine which projects to process (the full list is also needed for
    // the scoped-reindex hint below).
    let all_project_ids = wrap_busy(db.list_all_project_ids().map_err(Error::from))?;
    let projects: Vec<String> = if let Some(filter) = project_filter {
        vec![filter.to_string()]
    } else {
        all_project_ids.clone()
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

    // Human mode, scoped to a single project: nudge the user about the other
    // projects in the store so a scoped reindex doesn't quietly leave them
    // behind (mirrors the issue #156 under-reporting concern).
    if !json {
        if let Some(filter) = project_filter {
            let other_count = all_project_ids
                .iter()
                .filter(|pid| pid.as_str() != filter)
                .count();
            if other_count > 0 {
                println!(
                    "Note: {} other project(s) in this database not reindexed. Run with --all-projects to reindex them.",
                    other_count
                );
            }
        }
    }

    if force {
        // `reindex --force` is a per-DATABASE migration (issue #217): the
        // marker is written once and the new identity is recorded once, only
        // after every row of every project has been re-embedded. A project
        // filter would leave a silently mixed store, so --force requires
        // --all-projects.
        if project_filter.is_some() {
            eprintln!(
                "Error: `reindex --force` migrates the entire database: pass --all-projects so every project is re-embedded before the marker is written and the new identity is recorded."
            );
            return Ok(ExitCode::from(1));
        }
        return handle_reindex_force(&mut db, model_id, &all_project_ids, json);
    }

    // Plain reindex: mock-only classification path.
    let mut total_failed: usize = 0;
    let mut responses: Vec<ReindexResponse> = vec![];

    // Initialise the embedding engine (downloads model if needed)
    let mut engine = EmbeddingEngine::new(model_id)?;

    for project_id in &projects {
        if !json {
            println!("Project {}: reindexing...", project_id);
        }

        // The passage role applies the profile's prefix (e.g. `passage: `
        // under e5) — the engine is the single prefix site.
        let mut embed_callback = |content: &str| {
            engine
                .embed_passage(content)
                .map_err(|e| Error::Inference(e.to_string()))
        };
        let (reindexed, skipped, failed) =
            wrap_busy(reindex_project(&db, &mut embed_callback, project_id, json))?;
        let failed_count = failed.len();
        total_failed += failed_count;
        responses.push(ReindexResponse {
            project_id: project_id.clone(),
            reindexed,
            skipped,
            failed,
        });

        if !json {
            println!(
                "  Done: {} reindexed, {} skipped, {} failed",
                reindexed, skipped, failed_count
            );
        }
    }

    print_summary(
        &responses,
        &total_projects_scope(&projects),
        json,
        total_failed,
    )?;
    Ok(ExitCode::SUCCESS)
}

/// `--force` model-switch path (issue #217).
///
/// The caller performs the pre-flight token check first (decision 1): every
/// row's content is token-counted with the target profile's passage prefix
/// BEFORE the marker is written, and a single row over the token limit
/// refuses the start. This function then delegates the marker/identity
/// lifecycle to [`reindex_force::force_migrate_database`] — the single owner of
/// the per-database migration state — which writes the migration marker once
/// before any row is touched, re-embeds every row of every project (with the
/// passage prefix applied exactly once per row), and — only on a fully clean
/// pass — records the new identity and clears the marker in ONE transaction.
///
/// Re-running after an interruption re-embeds every row from the start (full
/// idempotent pass). On any per-row failure the marker stays in place and the
/// exit code is non-zero, so operations refuse until a clean re-run finishes.
fn handle_reindex_force(
    db: &mut Database,
    model_id: &str,
    projects: &[String],
    json: bool,
) -> Result<ExitCode, Error> {
    // Resolve the profile: the target identity records the profile's pinned
    // revision (not the bge default's), so the marker and the recorded
    // identity both name the model the store will actually be re-embedded with.
    let profile = profile_for(model_id)?;
    let target = ModelIdentity {
        model_id: profile.model_id.to_string(),
        revision: profile.revision.to_string(),
    };
    // The passage prefix, for the error message below.
    let passage_prefix = crate::embedding_profiles::EmbeddingRole::Passage
        .prefix(profile)
        .to_string();

    // Pre-check: if the database is already in a migrating state, the marker
    // is already set. Re-running --force is safe (full idempotent pass), so
    // proceed. But log the current identity and migration state for the
    // user's awareness.
    let current = identity::current_identity(db.conn()).map_err(Error::from)?;
    let migrating = identity::is_migrating(db.conn()).map_err(Error::from)?;
    if !json {
        if migrating {
            println!("Resuming interrupted migration to {}...", target.display());
        } else if current != target {
            println!(
                "Migrating from {} to {}...",
                current.display(),
                target.display()
            );
        }
    }

    // Initialise the embedding engine (downloads model if needed)
    let mut engine = EmbeddingEngine::new(model_id)?;

    // Pre-flight (issue #217, decision 1): token-count every row's content
    // with the target profile's passage prefix BEFORE writing the migration
    // marker. If any row would exceed the 512-token limit once prefixed,
    // report the offending ids, refuse to start, and write nothing — the
    // marker must only ever be written in a state the re-embed pass can
    // complete.
    let offending = reindex_force::over_limit_row_ids(db, &engine, projects)?;
    if !offending.is_empty() {
        eprintln!(
            "Error: reindex --force refused to start: {} row(s) exceed the {}-token limit once the '{}' passage prefix is prepended. Fix or remove these memories, then re-run `vipune reindex --force`:",
            offending.len(),
            MAX_EMBEDDING_TOKENS,
            passage_prefix.trim(),
        );
        for row in &offending {
            eprintln!("  {}", row.id);
        }
        eprintln!("No migration marker was written and no rows were changed.");
        return Ok(ExitCode::from(1));
    }

    // The embed closure delegates to the engine's passage-role method: the
    // raw stored content never carries a prefix, and the engine applies the
    // target profile's passage prefix exactly once (the single prefix site —
    // no double-prefixing, see the no-double-prefixing edge case).
    let embed = |content: &str| -> Result<Vec<f32>, crate::sqlite::Error> {
        engine
            .embed_passage(content)
            .map_err(|e| crate::sqlite::Error::Sqlite(e.to_string()))
    };

    // Single lifecycle owner (issue #217): marker once, re-embed every row
    // of every project, then identity + clear marker once — only on a fully
    // clean pass. A failed pass leaves the marker in place.
    // `force_migrate_database` only returns Err on a database-level failure
    // (marker write, row listing, final identity commit); a per-row embed
    // failure is returned as Ok with a populated `failed` list, so the
    // per-project failure reporting below stays intact.
    let (reindexed, skipped, failed) =
        reindex_force::force_migrate_database(db, &target, projects, embed)
            .map_err(|e| Error::SqliteModule(e.to_string()))?;
    let failed: Vec<ReindexFailure> = failed
        .into_iter()
        .map(|f: ReembedFailure| ReindexFailure {
            id: f.id,
            error: f.error,
        })
        .collect();
    let total_failed = failed.len();
    let responses = vec![ReindexResponse {
        project_id: total_projects_scope(projects),
        reindexed,
        skipped,
        failed,
    }];

    if total_failed > 0 {
        // Any failure means the re-embed pass did not complete; the marker
        // stays (force_migrate_database refuses to record the new identity
        // when the pass is incomplete) so operations refuse until a clean
        // re-run finishes.
        eprintln!(
            "Error: {} row(s) failed during force reindex. The migration marker is left in place; fix the errors and re-run `vipune reindex --force`.",
            total_failed
        );
        print_summary(
            &responses,
            &total_projects_scope(projects),
            json,
            total_failed,
        )?;
        return Ok(ExitCode::from(1));
    }

    if skipped > 0 {
        // Corrupted (Unknown-classified) rows cannot be re-embedded: they are
        // skipped and left as-is. Surface it so a clean-looking success does
        // not silently hide rows the pass never touched.
        eprintln!("{skipped} row(s) had corrupted embeddings and were skipped");
    }

    if !json {
        println!(
            "Model identity updated to {} (marker cleared).",
            target.display()
        );
    }

    print_summary(
        &responses,
        &total_projects_scope(projects),
        json,
        total_failed,
    )?;
    Ok(ExitCode::SUCCESS)
}

/// Print the per-project JSON / human summary shared by plain and force paths.
fn print_summary(
    responses: &[ReindexResponse],
    scope: &str,
    json: bool,
    total_failed: usize,
) -> Result<(), Error> {
    if json {
        print_json(&responses);
    } else {
        if total_failed > 0 {
            eprintln!("Warning: {} row(s) failed during reindex", total_failed);
            for response in responses {
                for failure in &response.failed {
                    eprintln!("  {} — {}", failure.id, failure.error);
                }
            }
        }
        println!("Total across {}:", scope);
        let reindexed: usize = responses.iter().map(|r| r.reindexed).sum();
        let skipped: usize = responses.iter().map(|r| r.skipped).sum();
        println!("  Reindexed: {}", reindexed);
        println!("  Skipped:   {}", skipped);
        println!("  Failed:    {}", total_failed);
    }
    Ok(())
}

/// Scope label for the summary footer (single project vs all projects).
fn total_projects_scope(projects: &[String]) -> String {
    if projects.len() == 1 {
        format!("project {}", projects[0])
    } else {
        "all projects".to_string()
    }
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
