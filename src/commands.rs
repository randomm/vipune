//! Command handlers for vipune CLI.

use crate::errors::Error;
use crate::memory::MemoryStore;
use crate::memory_types::AddResult;
use crate::output::*;
use crate::{config, temporal};
use std::process::ExitCode;

struct SearchContext {
    query: String,
    limit: usize,
    recency: Option<f64>,
    hybrid: bool,
}

/// Commands supported by vipune CLI.
#[derive(clap::Subcommand)]
pub enum Commands {
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

        /// Recency weight for search results (0.0 to 1.0)
        #[arg(long)]
        recency: Option<f64>,

        /// Use hybrid search (semantic + BM25 with RRF fusion)
        #[arg(long)]
        hybrid: bool,
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

/// Execute a CLI command.
pub fn execute(
    command: &Commands,
    store: &mut MemoryStore,
    project_id: String,
    config: &config::Config,
    json: bool,
) -> Result<ExitCode, Error> {
    match command {
        Commands::Add {
            text,
            metadata,
            force,
        } => handle_add(store, &project_id, text, metadata.as_deref(), *force, json),
        Commands::Search {
            query,
            limit,
            recency,
            hybrid,
        } => handle_search(
            store,
            &project_id,
            &SearchContext {
                query: query.clone(),
                limit: *limit,
                recency: *recency,
                hybrid: *hybrid,
            },
            config,
            json,
        ),
        Commands::Get { id } => handle_get(store, id, json),
        Commands::List { limit } => handle_list(store, &project_id, *limit, json),
        Commands::Delete { id } => handle_delete(store, id, json),
        Commands::Update { id, text } => handle_update(store, id, text, json),
        Commands::Version => handle_version(json),
    }
}

fn handle_add(
    store: &mut MemoryStore,
    project_id: &str,
    text: &str,
    metadata: Option<&str>,
    force: bool,
    json: bool,
) -> Result<ExitCode, Error> {
    match store.add_with_conflict(project_id, text, metadata, force)? {
        AddResult::Added { id } => {
            if json {
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
            if json {
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
    }
}

fn handle_search(
    store: &mut MemoryStore,
    project_id: &str,
    opts: &SearchContext,
    config: &config::Config,
    json: bool,
) -> Result<ExitCode, Error> {
    let recency_weight = opts.recency.unwrap_or(config.recency_weight);
    temporal::validate_recency_weight(recency_weight)?;
    let memories = if opts.hybrid {
        store.search_hybrid(project_id, &opts.query, opts.limit, recency_weight)?
    } else {
        store.search(project_id, &opts.query, opts.limit, recency_weight)?
    };
    if json {
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

fn handle_get(store: &mut MemoryStore, id: &str, json: bool) -> Result<ExitCode, Error> {
    let memory = store
        .get(id)?
        .ok_or_else(|| Error::NotFound(id.to_string()))?;
    if json {
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

fn handle_list(
    store: &mut MemoryStore,
    project_id: &str,
    limit: usize,
    json: bool,
) -> Result<ExitCode, Error> {
    let memories = store.list(project_id, limit)?;
    if json {
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

fn handle_delete(store: &mut MemoryStore, id: &str, json: bool) -> Result<ExitCode, Error> {
    let deleted = store.delete(id)?;
    if deleted {
        if json {
            print_json(&DeleteResponse {
                status: "deleted".to_string(),
                id: id.to_string(),
            });
        } else {
            println!("Deleted memory: {}", id);
        }
        Ok(ExitCode::SUCCESS)
    } else {
        Err(Error::NotFound(id.to_string()))
    }
}

fn handle_update(
    store: &mut MemoryStore,
    id: &str,
    text: &str,
    json: bool,
) -> Result<ExitCode, Error> {
    store.update(id, text)?;
    if json {
        print_json(&UpdateResponse {
            status: "updated".to_string(),
            id: id.to_string(),
        });
    } else {
        println!("Updated memory: {}", id);
    }
    Ok(ExitCode::SUCCESS)
}

fn handle_version(json: bool) -> Result<ExitCode, Error> {
    if json {
        print_json(&serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "name": env!("CARGO_PKG_NAME")
        }));
    } else {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    }
    Ok(ExitCode::SUCCESS)
}
