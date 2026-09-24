//! Command handlers for vipune CLI.

mod backup;
mod doctor;
mod doctor_fts;
mod export;
mod handlers;
pub mod hook_install;
pub mod hook_run;

#[cfg(test)]
mod hook_install_tests;
mod import;
mod merge;
mod promote;
mod prune;
mod reindex;
pub mod reindex_force;

#[cfg(test)]
pub(crate) use handlers::{SearchContext, handle_get, handle_list, handle_search};

#[cfg(test)]
mod backup_tests;

#[cfg(test)]
mod doctor_fts_tests;

#[cfg(test)]
mod doctor_projects_tests;

#[cfg(test)]
mod export_tests;

#[cfg(test)]
mod import_tests;

#[cfg(test)]
mod promote_tests;

#[cfg(test)]
mod merge_tests;

#[cfg(test)]
mod prune_tests;

#[cfg(test)]
mod reindex_prefix_tests;

#[cfg(test)]
mod reindex_tests;

use crate::config;
use crate::errors::Error;
use crate::hook::HookEvent;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::{MemoryStore, UpdateParams};
use serde::Serialize;
use std::path::Path;
use std::process::ExitCode;

/// Commands supported by vipune CLI.
#[derive(clap::Subcommand)]
pub enum Commands {
    Validate {
        /// Text to validate for embedding
        text: String,
    },
    Add {
        /// Memory text content
        text: String,

        /// Optional JSON metadata
        #[arg(short = 'm', long)]
        metadata: Option<String>,

        /// Bypass conflict detection and store the memory unconditionally.
        #[arg(long)]
        force: bool,

        /// Memory type (fact, preference, procedure, guard, observation)
        #[arg(long, default_value = "fact")]
        memory_type: String,

        /// Memory status (active, candidate)
        #[arg(long, default_value = "active")]
        status: String,

        /// Operator-assigned importance (low, medium, high, critical)
        #[arg(long, default_value = "medium")]
        importance: String,

        /// Supersede an existing memory (atomic replacement)
        #[arg(long)]
        supersedes: Option<String>,
    },
    Search {
        /// Search query text
        query: String,

        /// Maximum number of results (default: 5)
        #[arg(short = 'l', long, default_value = "5")]
        limit: usize,

        /// Recency weight for search results (0.0 to 1.0)
        #[arg(long)]
        recency: Option<f64>,

        /// Use hybrid search (semantic + BM25 with RRF fusion)
        #[arg(long)]
        hybrid: bool,

        /// Disable hybrid search even when enabled in config
        #[arg(long)]
        no_hybrid: bool,

        /// Filter by memory type (comma-separated)
        #[arg(long)]
        memory_type: Option<String>,

        /// Filter by status (default: active)
        #[arg(long)]
        status: Option<String>,

        /// Include candidate memories in results
        #[arg(long)]
        include_candidates: bool,

        /// Do not update retrieval telemetry (retrieval_count, last_retrieved_at)
        #[arg(long)]
        no_touch: bool,
    },
    Get {
        /// Memory ID
        id: String,

        /// Do not update retrieval telemetry
        #[arg(long)]
        no_touch: bool,
    },
    List {
        /// Maximum number of results (default: 10)
        #[arg(short = 'l', long, default_value = "10")]
        limit: usize,

        /// Filter by memory type (comma-separated)
        #[arg(long)]
        memory_type: Option<String>,

        /// Filter by status (default: active)
        #[arg(long)]
        status: Option<String>,

        /// Include candidate memories in results
        #[arg(long)]
        include_candidates: bool,
    },
    Delete {
        /// Memory ID
        id: String,
    },
    Update {
        /// Memory ID
        id: String,

        /// New content (optional)
        #[arg(short = 't', long)]
        text: Option<String>,

        /// Optional JSON metadata (replaces existing metadata)
        #[arg(short = 'm', long)]
        metadata: Option<String>,

        /// Update memory type
        #[arg(long)]
        memory_type: Option<String>,

        /// Update memory status
        #[arg(long)]
        status: Option<String>,

        /// Update operator-assigned importance (low, medium, high, critical)
        #[arg(long)]
        importance: Option<String>,
    },
    /// Diagnose database health.
    #[command(group = clap::ArgGroup::new("doctor-mode").args(["embeddings", "projects", "fts"]).required(true).multiple(false))]
    Doctor {
        /// Check embedding quality (classifies real/mock/unknown)
        #[arg(long)]
        embeddings: bool,

        /// Scan all projects for suspected split pairs (bare id vs owner/repo)
        #[arg(long)]
        projects: bool,

        /// Check FTS index for desync against the memories table (bidirectional rowid join)
        #[arg(long)]
        fts: bool,

        /// Project identifier (only relevant for --embeddings; ignored for --projects with a warning)
        #[arg(long, short = 'p')]
        project: Option<String>,

        /// Rebuild the FTS index when desync is detected (only for --fts; global, ignores -p)
        #[arg(long)]
        repair: bool,
    },

    /// Import memories from a JSONL export file (or `-` for stdin).
    ///
    /// All-or-nothing: the file is restored in a single transaction, so a
    /// malformed line or wrong-dimension embedding aborts the whole import.
    Import {
        /// Path to the JSONL export file, or `-` for stdin
        source: Option<String>,
    },

    /// Re-embed rows with mock embeddings using the real model.
    ///
    /// `--force` performs a full model-switch migration: writes a
    /// "migrating to `<id>@<revision>`" marker, re-embeds every row
    /// (bypassing Mock/Real classification) with the configured profile,
    /// then records the new identity and clears the marker in one
    /// transaction. Re-run after an interruption to complete the migration.
    Reindex {
        /// Reindex all projects in the database instead of only the current one
        #[arg(long)]
        all_projects: bool,

        /// Force a full model-switch migration: re-embed every row with the
        /// configured embedding profile, regardless of Mock/Real classification.
        /// Before writing the migration marker, it pre-checks that no row exceeds
        /// the 512-token limit once the profile's passage prefix is prepended
        /// (refusing to start and listing the offending ids if so). After a
        /// clean re-embed pass it records the new model identity and clears the
        /// marker in one transaction. Re-run after an interruption to complete
        /// the migration with a full idempotent pass.
        #[arg(long)]
        force: bool,
    },

    /// Export all rows (all projects, uncapped) to a JSONL file.
    Export {
        /// Destination JSONL file (use "> out.jsonl" via shell if omitting)
        output_path: String,
    },

    /// Promote candidates that have been retrieved enough times to active.
    ///
    /// A candidate promotes when `status='candidate'` AND `retrieval_count >=
    /// threshold` (default 5, overridable via `VIPUNE_PROMOTION_THRESHOLD`).
    /// Superseded and deprecated rows are never promoted. The promotion issues
    /// `UPDATE status='active'` via the existing update path.
    Promote,

    /// Prune stale candidate memories by demoting them to `deprecated`.
    ///
    /// Prune **never deletes**: demotions are issued as `UPDATE status =
    /// 'deprecated'` through the existing update path, so the total row count
    /// of the database is unchanged after any run. A row is demoted when and
    /// only when `status = 'candidate' AND retrieval_count < N AND
    /// age(created_at) > T`, where N and T are configurable (TOML
    /// `prune_count` / `prune_age_days` with `VIPUNE_PRUNE_COUNT` /
    /// `VIPUNE_PRUNE_AGE_DAYS` env overrides mirroring `VIPUNE_RECENCY_WEIGHT`).
    /// Guard-type memories and rows with `importance IN ('high','critical')`
    /// are hard exclusions: they are never demoted.
    Prune {
        /// Retrieval-count threshold N: prune candidates with
        /// `retrieval_count < N` (subject to the age rule and hard
        /// exclusions). Configurable via TOML `prune_count` / env
        /// `VIPUNE_PRUNE_COUNT`.
        #[arg(long)]
        count: Option<i64>,
        /// Age threshold T (in days): prune candidates with
        /// `age(created_at) > T` (subject to the count rule and hard
        /// exclusions). Configurable via TOML `prune_age_days` / env
        /// `VIPUNE_PRUNE_AGE_DAYS`.
        #[arg(long)]
        age_days: Option<i64>,
    },

    /// Back up the database to a consistent snapshot using SQLite's Online Backup API.
    ///
    /// Produces a byte-complete, queryable copy of the memories database. The
    /// command honours `--db-path` (operates on the resolved override path)
    /// and fast-fails if the source is locked by another process.
    Backup {
        /// Optional explicit output path. Defaults to `<source>-backup.<ext>`
        /// alongside the source database.
        #[arg(short = 'o', long)]
        output: Option<std::path::PathBuf>,
    },

    /// Project management operations.
    Project {
        #[command(subcommand)]
        command: ProjectCommands,
    },

    /// Agent lifecycle hook handling (issue #191).
    ///
    /// Each event subcommand reads a Claude Code JSON payload on stdin,
    /// extracts candidate memories, and inserts them (deduped) with
    /// placeholder embeddings. `install` / `uninstall` manage the
    /// `~/.claude/settings.json` hook wiring.
    Hook {
        #[command(subcommand)]
        command: HookCommands,
    },

    Version,

    #[cfg(feature = "mcp")]
    /// Start MCP server over stdio
    Mcp,
}

/// Response for `vipune import`: how many rows were inserted and how many
/// were skipped because their id already existed in the destination.
#[derive(Debug, Serialize)]
pub struct ImportResponse {
    /// Number of rows inserted by this import.
    pub inserted: usize,
    /// Number of rows skipped because their id already existed (normal path).
    pub skipped: usize,
}

/// Subcommands under `vipune project`.
#[derive(clap::Subcommand)]
pub enum ProjectCommands {
    /// Merge all rows from one project into another.
    ///
    /// Moves every row whose project_id matches `from` to `to`.
    /// The operation is atomic — either all rows move or none do.
    /// Only the project_id column changes; all other data is preserved byte-identically.
    /// Merging from a project into itself is a no-op.
    Merge {
        /// Source project id (rows moved from this)
        from: String,
        /// Target project id (rows moved to this)
        to: String,
    },
}

/// Subcommands under `vipune hook`.
#[derive(clap::Subcommand)]
pub enum HookCommands {
    /// Handle a Claude Code SessionStart hook event (payload on stdin).
    SessionStart,
    /// Handle a Claude Code UserPromptSubmit hook event (payload on stdin).
    UserPromptSubmit,
    /// Handle a Claude Code PreToolUse hook event (payload on stdin).
    PreToolUse,
    /// Handle a Claude Code PostToolUse hook event (payload on stdin).
    PostToolUse,
    /// Handle a Claude Code PreCompact hook event (payload on stdin).
    PreCompact,
    /// Install vipune hook entries into ~/.claude/settings.json.
    Install,
    /// Remove vipune hook entries from ~/.claude/settings.json.
    Uninstall,
}

/// Execute a CLI command.
pub fn execute(
    command: &Commands,
    store: &mut MemoryStore,
    project_id: String,
    config: &config::Config,
    json: bool,
) -> Result<ExitCode, Error> {
    match command {
        Commands::Validate { text } => {
            handlers::handle_validate(text, &config.embedding_model, json)
        }
        Commands::Add {
            text,
            metadata,
            force,
            memory_type,
            status,
            importance,
            supersedes,
        } => handlers::handle_add(
            store,
            &project_id,
            text,
            metadata.as_deref(),
            *force,
            memory_type,
            status,
            importance,
            supersedes.as_deref(),
            json,
        ),
        Commands::Search {
            query,
            limit,
            recency,
            hybrid,
            no_hybrid,
            memory_type,
            status,
            include_candidates,
            no_touch,
        } => handlers::handle_search(
            store,
            &project_id,
            &handlers::SearchContext {
                query: query.clone(),
                limit: *limit,
                recency: *recency,
                hybrid: *hybrid,
                no_hybrid: *no_hybrid,
                memory_type: memory_type.clone(),
                status: status.clone(),
                include_candidates: *include_candidates,
                no_touch: *no_touch,
            },
            config,
            json,
        ),
        Commands::Get { id, no_touch } => {
            handlers::handle_get(store, id, &project_id, *no_touch, json)
        }
        Commands::List {
            limit,
            memory_type,
            status,
            include_candidates,
        } => handlers::handle_list(
            store,
            &project_id,
            *limit,
            memory_type.as_deref(),
            status.as_deref(),
            *include_candidates,
            json,
        ),
        Commands::Delete { id } => handlers::handle_delete(store, id, &project_id, json),
        Commands::Update {
            id,
            text,
            metadata,
            memory_type,
            status,
            importance,
        } => {
            let memory_type_val = memory_type
                .as_deref()
                .map(MemoryType::from_str)
                .transpose()?;
            let status_val = status.as_deref().map(MemoryStatus::from_str).transpose()?;
            let importance_val = importance
                .as_deref()
                .map(MemoryImportance::from_str)
                .transpose()?;
            handlers::handle_update(
                store,
                id,
                &project_id,
                UpdateParams {
                    text: text.as_deref(),
                    metadata: metadata.as_deref(),
                    memory_type: memory_type_val,
                    status: status_val,
                    importance: importance_val,
                },
                json,
            )
        }
        Commands::Doctor {
            embeddings: _,
            projects,
            fts,
            project: doctor_project,
            repair,
        } => {
            if *projects {
                // doctor --projects always scans all projects; -p is ignored (with warning).
                let project_filter = doctor_project.as_deref();
                doctor::handle_doctor_projects(&config.database_path, project_filter, json)
            } else if *fts {
                // doctor --fts: joins the rowid join over ALL projects (no detected-project
                // fallback, unlike --embeddings). -p scopes the under-population count but
                // is ignored for orphan rows and --repair (always global).
                let project_filter = doctor_project.as_deref();
                doctor_fts::handle_doctor_fts(&config.database_path, project_filter, *repair, json)
            } else {
                // doctor --embeddings: use explicit -p or fall back to the detected project_id.
                let project_filter = doctor_project.as_deref().or(Some(project_id.as_str()));
                doctor::handle_doctor(&config.database_path, project_filter, json)
            }
        }
        Commands::Import { source } => {
            let source = source.as_ref().map(|s| s.as_str());
            import::handle_import(
                &config.database_path,
                source,
                Some(project_id.as_str()),
                json,
            )
        }
        Commands::Reindex {
            all_projects,
            force,
        } => {
            let project_filter = if !*all_projects {
                Some(project_id.as_str())
            } else {
                None
            };
            reindex::handle_reindex(
                &config.database_path,
                &config.embedding_model,
                project_filter,
                *force,
                json,
            )
        }
        Commands::Export { output_path } => {
            export::handle_export(&config.database_path, Path::new(output_path), None, json)
        }
        Commands::Promote => promote::handle_promote(&config.database_path, &project_id, json),
        Commands::Prune { count, age_days } => {
            let n = count.unwrap_or(prune::DEFAULT_PRUNE_COUNT);
            let t = age_days.unwrap_or(prune::DEFAULT_PRUNE_AGE_DAYS);
            prune::handle_prune(&config.database_path, n, t, json)
        }
        Commands::Backup { output } => {
            backup::handle_backup(&config.database_path, output.as_deref(), json)
        }
        Commands::Project { command } => match command {
            ProjectCommands::Merge { from, to } => {
                merge::handle_merge(&config.database_path, from, to, json)
            }
        },
        Commands::Hook { command } => match command {
            HookCommands::Install => hook_install::handle_hook_install(json),
            HookCommands::Uninstall => hook_install::handle_hook_uninstall(json),
            // Event subcommands read stdin, call the hook run path, and
            // always exit 0 (the hook must never surface an error mid-session).
            // The event type is carried by the subcommand variant — the
            // Claude Code hook contract does not include an `event_type`
            // field in the JSON payload itself.
            HookCommands::SessionStart => {
                hook_run::handle_hook_event(config, json, HookEvent::SessionStart)
            }
            HookCommands::UserPromptSubmit => {
                hook_run::handle_hook_event(config, json, HookEvent::UserPromptSubmit)
            }
            HookCommands::PreToolUse => {
                hook_run::handle_hook_event(config, json, HookEvent::PreToolUse)
            }
            HookCommands::PostToolUse => {
                hook_run::handle_hook_event(config, json, HookEvent::PostToolUse)
            }
            HookCommands::PreCompact => {
                hook_run::handle_hook_event(config, json, HookEvent::PreCompact)
            }
        },
        Commands::Version => handlers::handle_version(json),
        #[cfg(feature = "mcp")]
        Commands::Mcp => unreachable!("Mcp is handled before execute"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_short_text() {
        let result =
            handlers::handle_validate("hello world", "not-a-real-model-should-fail", false);
        // Should fail because model doesn't exist, not because of token count
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_long_text() {
        let long_text = "a".repeat(1000);
        let result = handlers::handle_validate(&long_text, "not-a-real-model-should-fail", false);
        // Should fail because model doesn't exist, not because of token count
        assert!(result.is_err());
    }
}
