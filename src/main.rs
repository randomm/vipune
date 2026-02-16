//! CLI entry point for vipune memory layer.

mod config;
mod embedding;
mod errors;
mod import;
mod memory;
mod memory_types;
mod output;
mod project;
mod sqlite;
mod temporal;

use clap::{Parser, Subcommand};
use errors::Error;
use memory::MemoryStore;
use memory_types::AddResult;
use output::*;
use project::detect_project;
use std::path::PathBuf;
use std::process::ExitCode;

/// vipune - A minimal memory layer for AI agents
#[derive(Parser)]
#[command(name = "vipune", about = "Minimal memory layer for AI agents", long_about = None)]
struct Cli {
    /// Output as JSON (default: human-readable)
    #[arg(long, global = true)]
    json: bool,

    /// Project identifier (auto-detected from git if omitted)
    #[arg(long, short = 'p', global = true)]
    project: Option<String>,

    /// Override database path
    #[arg(long, global = true)]
    db_path: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add {
        /// Memory text content
        text: String,

        /// Optional JSON metadata
        #[arg(short = 'm', long)]
        metadata: Option<String>,

        /// Force add (no-op until #6)
        #[arg(long)]
        force: bool,
    },
    Search {
        /// Search query text
        query: String,

        /// Maximum number of results (default: 5)
        #[arg(short = 'l', long, default_value = "5")]
        limit: usize,

        /// Recency weight for search results (0.0 to 1.0, default: 0.0)
        #[arg(long, default_value = "0.0")]
        recency: f64,
    },
    Get {
        /// Memory ID
        id: String,
    },
    List {
        /// Maximum number of results (default: 10)
        #[arg(short = 'l', long, default_value = "10")]
        limit: usize,
    },
    Delete {
        /// Memory ID
        id: String,
    },
    Update {
        /// Memory ID
        id: String,
        /// New content
        text: String,
    },
    Version,
    Import {
        /// Path to remory database or JSON file
        source: String,

        /// Dry run (show what would be imported)
        #[arg(long)]
        dry_run: bool,

        /// Import format (default: sqlite)
        #[arg(short = 'f', long, default_value = "sqlite")]
        format: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(&cli) {
        Ok(exit_code) => exit_code,
        Err(error) => {
            if cli.json {
                print_json(&ErrorResponse {
                    error: error.to_string(),
                });
            } else {
                eprintln!("Error: {}", error);
            }
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> Result<ExitCode, Error> {
    let mut config = config::Config::load()?;
    config.ensure_directories()?;

    if let Some(db_path) = &cli.db_path {
        config.database_path = db_path.clone().into();
    }

    let project_id = detect_project(cli.project.as_deref());

    let mut store = MemoryStore::new(
        &config.database_path,
        &config.embedding_model,
        config.clone(),
    )?;

    match &cli.command {
        Commands::Add {
            text,
            metadata,
            force,
        } => match store.add_with_conflict(&project_id, text, metadata.as_deref(), *force)? {
            AddResult::Added { id } => {
                if cli.json {
                    print_json(&AddResponse {
                        status: "added".to_string(),
                        id,
                    });
                } else {
                    println!("Added memory: {}", id);
                }
                Ok(ExitCode::SUCCESS)
            }
            AddResult::Conflicts {
                proposed,
                conflicts,
            } => {
                if cli.json {
                    let conflict_responses: Vec<ConflictMemoryResponse> = conflicts
                        .into_iter()
                        .map(|c| ConflictMemoryResponse {
                            id: c.id,
                            content: c.content,
                            similarity: c.similarity,
                        })
                        .collect();
                    print_json(&ConflictsResponse {
                        status: "conflicts".to_string(),
                        proposed,
                        conflicts: conflict_responses,
                    });
                } else {
                    println!(
                        "Conflicts detected: {} similar memory/memories found",
                        conflicts.len()
                    );
                    println!("Proposed: {}", proposed);
                    println!("Use --force to add anyway");
                    for conflict in conflicts {
                        println!("  {} (similarity: {:.3})", conflict.id, conflict.similarity);
                        println!("    {}", conflict.content);
                    }
                }
                Ok(ExitCode::from(2))
            }
        },
        Commands::Search {
            query,
            limit,
            recency,
        } => {
            let recency_weight = if *recency == 0.0 {
                config.recency_weight
            } else {
                *recency
            };
            temporal::validate_recency_weight(recency_weight)?;
            let memories = store.search(&project_id, query, *limit, recency_weight)?;
            if cli.json {
                let results: Vec<SearchResultItem> = memories
                    .into_iter()
                    .map(|m| SearchResultItem {
                        id: m.id,
                        content: m.content,
                        similarity: m.similarity.unwrap_or(0.0),
                        created_at: m.created_at,
                    })
                    .collect();
                print_json(&SearchResponse { results });
            } else {
                for memory in memories {
                    let score = memory.similarity.unwrap_or(0.0);
                    println!(
                        "{} [score: {:.2}]\n  {}\n",
                        memory.id, score, memory.content
                    );
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Commands::Get { id } => {
            let memory = store.get(id)?.ok_or_else(|| Error::NotFound(id.clone()))?;
            if cli.json {
                print_json(&GetResponse {
                    id: memory.id.clone(),
                    content: memory.content.clone(),
                    project_id: memory.project_id,
                    metadata: memory.metadata,
                    created_at: memory.created_at,
                    updated_at: memory.updated_at,
                });
            } else {
                println!("ID: {}", memory.id);
                println!("Content: {}", memory.content);
                println!("Project: {}", memory.project_id);
                if let Some(meta) = &memory.metadata {
                    println!("Metadata: {}", meta);
                }
                println!("Created: {}", memory.created_at);
                println!("Updated: {}", memory.updated_at);
            }
            Ok(ExitCode::SUCCESS)
        }
        Commands::List { limit } => {
            let memories = store.list(&project_id, *limit)?;
            if cli.json {
                let items: Vec<ListItem> = memories
                    .into_iter()
                    .map(|m| ListItem {
                        id: m.id,
                        content: m.content,
                        created_at: m.created_at,
                    })
                    .collect();
                print_json(&ListResponse { memories: items });
            } else {
                for memory in memories {
                    println!("{}: {}", memory.id, memory.content);
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Commands::Delete { id } => {
            let deleted = store.delete(id)?;
            if deleted {
                if cli.json {
                    print_json(&DeleteResponse {
                        status: "deleted".to_string(),
                        id: id.clone(),
                    });
                } else {
                    println!("Deleted memory: {}", id);
                }
                Ok(ExitCode::SUCCESS)
            } else {
                Err(Error::NotFound(id.clone()))
            }
        }
        Commands::Update { id, text } => {
            store.update(id, text)?;
            if cli.json {
                print_json(&UpdateResponse {
                    status: "updated".to_string(),
                    id: id.clone(),
                });
            } else {
                println!("Updated memory: {}", id);
            }
            Ok(ExitCode::SUCCESS)
        }
        Commands::Version => {
            if cli.json {
                print_json(&serde_json::json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "name": env!("CARGO_PKG_NAME")
                }));
            } else {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            }
            Ok(ExitCode::SUCCESS)
        }
        Commands::Import {
            source,
            dry_run,
            format,
        } => {
            let source_path = PathBuf::from(&source);

            let stats = match format.as_str() {
                "sqlite" => import::import_from_sqlite(
                    &source_path,
                    *dry_run,
                    &config.database_path,
                    &config.embedding_model,
                    config.clone(),
                )?,
                "json" => {
                    if *dry_run {
                        return Err(Error::InvalidInput(
                            "Dry run not supported for JSON import".to_string(),
                        ));
                    }
                    import::import_from_json(
                        &source_path,
                        &config.database_path,
                        &config.embedding_model,
                        config.clone(),
                    )?
                }
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "Invalid format '{}': must be 'sqlite' or 'json'",
                        format
                    )));
                }
            };

            if cli.json {
                print_json(&ImportResponse {
                    status: if *dry_run {
                        "dry_run".to_string()
                    } else {
                        "imported".to_string()
                    },
                    total_memories: stats.total_memories,
                    imported: stats.imported_memories,
                    skipped_duplicates: stats.skipped_duplicates,
                    skipped_corrupted: stats.skipped_corrupted,
                    projects: stats.projects.len(),
                });
            } else {
                if *dry_run {
                    println!("Dry run: would import from {}", source);
                } else {
                    println!("Imported from {}", source);
                }
                println!("Total memories: {}", stats.total_memories);
                println!("Imported: {}", stats.imported_memories);
                if stats.skipped_duplicates > 0 {
                    println!("Skipped duplicates: {}", stats.skipped_duplicates);
                }
                if stats.skipped_corrupted > 0 {
                    println!("Skipped corrupted: {}", stats.skipped_corrupted);
                }
                println!("Projects: {}", stats.projects.len());
                for project in &stats.projects {
                    println!("  - {}", project);
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_add() {
        let cli = Cli::parse_from(&["vipune", "add", "test content"]);
        assert_eq!(cli.json, false);
        assert!(cli.project.is_none());
        assert!(cli.db_path.is_none());
        matches!(cli.command, Commands::Add { .. });
    }

    #[test]
    fn test_cli_parse_with_json() {
        let cli = Cli::parse_from(&["vipune", "--json", "add", "test"]);
        assert_eq!(cli.json, true);
    }

    #[test]
    fn test_cli_parse_with_project() {
        let cli = Cli::parse_from(&["vipune", "-p", "my-project", "add", "test"]);
        assert_eq!(cli.project, Some("my-project".to_string()));
    }

    #[test]
    fn test_cli_parse_search() {
        let cli = Cli::parse_from(&["vipune", "search", "query", "--limit", "10"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                limit: 10,
                ..
            } if query == "query"
        );
    }

    #[test]
    fn test_cli_parse_get() {
        let cli = Cli::parse_from(&["vipune", "get", "memory-id"]);
        matches!(cli.command, Commands::Get { id } if id == "memory-id");
    }

    #[test]
    fn test_cli_parse_list() {
        let cli = Cli::parse_from(&["vipune", "list"]);
        matches!(cli.command, Commands::List { .. });
    }

    #[test]
    fn test_cli_parse_delete() {
        let cli = Cli::parse_from(&["vipune", "delete", "memory-id"]);
        matches!(cli.command, Commands::Delete { id } if id == "memory-id");
    }

    #[test]
    fn test_cli_parse_update() {
        let cli = Cli::parse_from(&["vipune", "update", "memory-id", "new content"]);
        matches!(
            cli.command,
            Commands::Update { id, text } if id == "memory-id" && text == "new content"
        );
    }

    #[test]
    fn test_cli_parse_version() {
        let cli = Cli::parse_from(&["vipune", "version"]);
        matches!(cli.command, Commands::Version);
    }

    #[test]
    fn test_cli_parse_with_db_path() {
        let cli = Cli::parse_from(&["vipune", "--db-path", "/custom/path.db", "add", "test"]);
        assert_eq!(cli.db_path, Some("/custom/path.db".to_string()));
    }

    #[test]
    fn test_cli_parse_import_sqlite() {
        let cli = Cli::parse_from(&["vipune", "import", "/path/to/db.sqlite"]);
        matches!(
            cli.command,
            Commands::Import {
                source,
                dry_run: false,
                format
            } if source == "/path/to/db.sqlite" && format == "sqlite"
        );
    }

    #[test]
    fn test_cli_parse_import_dry_run() {
        let cli = Cli::parse_from(&["vipune", "import", "/path/to/db.sqlite", "--dry-run"]);
        matches!(
            cli.command,
            Commands::Import {
                source,
                dry_run: true,
                ..
            } if source == "/path/to/db.sqlite"
        );
    }

    #[test]
    fn test_cli_parse_import_json() {
        let cli = Cli::parse_from(&["vipune", "import", "export.json", "--format", "json"]);
        matches!(
            cli.command,
            Commands::Import {
                source,
                dry_run: false,
                format
            } if source == "export.json" && format == "json"
        );
    }
}
