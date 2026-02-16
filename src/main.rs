//! CLI entry point for vipune memory layer.

mod config;
mod embedding;
mod errors;
mod memory;
mod project;
mod sqlite;

use clap::{Parser, Subcommand};
use errors::Error;
use memory::{AddResult, MemoryStore};
use project::detect_project;
use serde::Serialize;
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
}

#[derive(Serialize)]
struct AddResponse {
    status: String,
    id: String,
}

#[derive(Serialize)]
struct SearchResultItem {
    id: String,
    content: String,
    similarity: f64,
    created_at: String,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
struct GetResponse {
    id: String,
    content: String,
    project_id: String,
    metadata: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Serialize)]
struct ListItem {
    id: String,
    content: String,
    created_at: String,
}

#[derive(Serialize)]
struct ListResponse {
    memories: Vec<ListItem>,
}

#[derive(Serialize)]
struct DeleteResponse {
    status: String,
    id: String,
}

#[derive(Serialize)]
struct UpdateResponse {
    status: String,
    id: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct ConflictMemoryResponse {
    id: String,
    content: String,
    similarity: f64,
}

#[derive(Serialize)]
struct ConflictsResponse {
    status: String,
    proposed: String,
    conflicts: Vec<ConflictMemoryResponse>,
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
        Commands::Search { query, limit } => {
            let memories = store.search(&project_id, query, *limit)?;
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
    }
}

fn print_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("Failed to serialize JSON: {}", e);
            std::process::exit(1);
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
                limit: 10
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
    fn test_serialize_add_response() {
        let response = AddResponse {
            status: "added".to_string(),
            id: "test-id".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"added\""));
        assert!(json.contains("\"id\":\"test-id\""));
    }

    #[test]
    fn test_serialize_search_response() {
        let response = SearchResponse {
            results: vec![SearchResultItem {
                id: "test-id".to_string(),
                content: "test content".to_string(),
                similarity: 0.95,
                created_at: "2024-01-01T00:00:00Z".to_string(),
            }],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"results\""));
        assert!(json.contains("\"similarity\":0.95"));
    }
}
