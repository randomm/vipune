//! Command handlers for vipune CLI.

mod doctor;
mod handlers;
mod merge;
mod reindex;

#[cfg(test)]
mod doctor_projects_tests;

#[cfg(test)]
mod merge_tests;

#[cfg(test)]
mod reindex_tests;

use crate::config;
use crate::errors::Error;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory::{MemoryStore, UpdateParams};
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
    },
    /// Diagnose database health.
    #[command(group = clap::ArgGroup::new("doctor-mode").args(["embeddings", "projects"]).required(true).multiple(false))]
    Doctor {
        /// Check embedding quality (classifies real/mock/unknown)
        #[arg(long)]
        embeddings: bool,

        /// Scan all projects for suspected split pairs (bare id vs owner/repo)
        #[arg(long)]
        projects: bool,

        /// Project identifier (only relevant for --embeddings; ignored for --projects with a warning)
        #[arg(long, short = 'p')]
        project: Option<String>,
    },

    /// Re-embed rows with mock embeddings using the real model.
    Reindex {
        /// Reindex all projects in the database instead of only the current one
        #[arg(long)]
        all_projects: bool,
    },

    /// Project management operations.
    Project {
        #[command(subcommand)]
        command: ProjectCommands,
    },

    Version,

    #[cfg(feature = "mcp")]
    /// Start MCP server over stdio
    Mcp,
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
            supersedes,
        } => handlers::handle_add(
            store,
            &project_id,
            text,
            metadata.as_deref(),
            *force,
            memory_type,
            status,
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
        } => {
            let memory_type_val = memory_type
                .as_deref()
                .map(MemoryType::from_str)
                .transpose()?;
            let status_val = status.as_deref().map(MemoryStatus::from_str).transpose()?;
            handlers::handle_update(
                store,
                id,
                &project_id,
                UpdateParams {
                    text: text.as_deref(),
                    metadata: metadata.as_deref(),
                    memory_type: memory_type_val,
                    status: status_val,
                },
                json,
            )
        }
        Commands::Doctor {
            embeddings: _,
            projects,
            project: doctor_project,
        } => {
            if *projects {
                // doctor --projects always scans all projects; -p is ignored (with warning).
                let project_filter = doctor_project.as_deref();
                doctor::handle_doctor_projects(&config.database_path, project_filter, json)
            } else {
                // doctor --embeddings: use explicit -p or fall back to the detected project_id.
                let project_filter = doctor_project.as_deref().or(Some(project_id.as_str()));
                doctor::handle_doctor(&config.database_path, project_filter, json)
            }
        }
        Commands::Reindex { all_projects } => {
            let project_filter = if !*all_projects {
                Some(project_id.as_str())
            } else {
                None
            };
            reindex::handle_reindex(
                &config.database_path,
                &config.embedding_model,
                project_filter,
                json,
            )
        }
        Commands::Project { command } => match command {
            ProjectCommands::Merge { from, to } => {
                merge::handle_merge(&config.database_path, from, to, json)
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
