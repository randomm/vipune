//! Command handlers for vipune CLI.

use crate::errors::Error;
use crate::memory::MemoryStore;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, IngestPolicy};
use crate::output::*;
use crate::{config, embedding::EmbeddingEngine, temporal};
use std::process::ExitCode;

struct SearchContext {
    query: String,
    limit: usize,
    recency: Option<f64>,
    hybrid: bool,
    no_hybrid: bool,
    memory_type: Option<String>,
    status: Option<String>,
    include_candidates: bool,
    no_touch: bool,
}

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
    Version,

    #[cfg(feature = "mcp")]
    /// Start MCP server over stdio
    Mcp,
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
        Commands::Validate { text } => handle_validate(text, &config.embedding_model, json),
        Commands::Add {
            text,
            metadata,
            force,
            memory_type,
            status,
            supersedes,
        } => handle_add(
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
        } => handle_search(
            store,
            &project_id,
            &SearchContext {
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
        Commands::Get { id, no_touch } => handle_get(store, id, *no_touch, json),
        Commands::List {
            limit,
            memory_type,
            status,
            include_candidates,
        } => handle_list(
            store,
            &project_id,
            *limit,
            memory_type.as_deref(),
            status.as_deref(),
            *include_candidates,
            json,
        ),
        Commands::Delete { id } => handle_delete(store, id, json),
        Commands::Update {
            id,
            text,
            metadata,
            memory_type,
            status,
        } => handle_update(
            store,
            id,
            text.as_deref(),
            metadata.as_deref(),
            memory_type.as_deref(),
            status.as_deref(),
            json,
        ),
        Commands::Version => handle_version(json),
        #[cfg(feature = "mcp")]
        Commands::Mcp => unreachable!("Mcp is handled before execute"),
    }
}

fn handle_validate(text: &str, model_id: &str, json: bool) -> Result<ExitCode, Error> {
    let engine = EmbeddingEngine::new(model_id)?;
    let token_count = engine.token_count(text)?;

    if token_count > crate::embedding::MAX_EMBEDDING_TOKENS {
        return Err(Error::ContentTooLong {
            token_count,
            max_tokens: crate::embedding::MAX_EMBEDDING_TOKENS,
        });
    }

    if json {
        print_json(&ValidateResponse {
            token_count,
            max_tokens: crate::embedding::MAX_EMBEDDING_TOKENS,
            within_limit: true,
        });
    } else {
        println!(
            "Token count: {}/{} — within limit",
            token_count,
            crate::embedding::MAX_EMBEDDING_TOKENS
        );
    }

    Ok(ExitCode::SUCCESS)
}

#[allow(clippy::too_many_arguments)]
fn handle_add(
    store: &mut MemoryStore,
    project_id: &str,
    text: &str,
    metadata: Option<&str>,
    force: bool,
    memory_type: &str,
    status: &str,
    supersedes: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Validate memory_type
    let _ = MemoryType::from_str(memory_type)?;

    // Validate status
    let status_val = MemoryStatus::from_str(status)?;
    if !status_val.is_valid_for_insert() {
        return Err(Error::InvalidInput(format!(
            "Status '{}' is not valid for new memory insertion. Must be 'active' or 'candidate'.",
            status
        )));
    }

    // Check mutually exclusive flags
    if supersedes.is_some() && force {
        return Err(Error::InvalidInput(
            "Cannot use both --supersedes and --force flags together".to_string(),
        ));
    }

    // If supersedes is provided, use supersede flow
    if let Some(old_id) = supersedes {
        let new_id = store.supersede(project_id, text, metadata, memory_type, old_id)?;

        if json {
            print_json(&AddResponse {
                status: "superseded".to_string(),
                id: new_id,
            });
        } else {
            println!("Superseded memory {} with new memory", old_id);
        }
        return Ok(ExitCode::SUCCESS);
    }

    let policy = if force {
        IngestPolicy::Force
    } else {
        IngestPolicy::ConflictAware
    };

    match store.ingest_with_type_status(project_id, text, metadata, policy, memory_type, status)? {
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

    // Build filter params from CLI flags
    let type_vec: Option<Vec<String>> = opts
        .memory_type
        .as_ref()
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());
    let type_strs: Option<Vec<&str>> = type_vec
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    let type_slice: Option<&[&str]> = type_strs.as_deref();

    let status_vec: Option<Vec<String>> = if opts.include_candidates {
        Some(vec!["active".to_string(), "candidate".to_string()])
    } else {
        opts.status
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
    };
    let status_strs: Option<Vec<&str>> = status_vec
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    let status_slice: Option<&[&str]> = status_strs.as_deref();

    let use_hybrid = (opts.hybrid || config.hybrid) && !opts.no_hybrid;
    let memories = if use_hybrid {
        store.search_hybrid(
            project_id,
            &opts.query,
            opts.limit,
            recency_weight,
            type_slice,
            status_slice,
        )?
    } else {
        store.search(
            project_id,
            &opts.query,
            opts.limit,
            recency_weight,
            type_slice,
            status_slice,
        )?
    };

    // Update retrieval telemetry unless disabled
    if !opts.no_touch {
        let ids: Vec<&str> = memories.iter().map(|m| m.id.as_str()).collect();
        if !ids.is_empty() {
            store.db.touch_memories(&ids).ok();
        }
    }

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

fn handle_get(
    store: &mut MemoryStore,
    id: &str,
    no_touch: bool,
    json: bool,
) -> Result<ExitCode, Error> {
    let memory = store
        .get(id)?
        .ok_or_else(|| Error::NotFound("memory not found".to_string()))?;

    // Update retrieval telemetry unless disabled
    if !no_touch {
        store.db.touch_memories(&[id]).ok();
    }

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
    memory_type: Option<&str>,
    status: Option<&str>,
    include_candidates: bool,
    json: bool,
) -> Result<ExitCode, Error> {
    // Build filter params from CLI flags
    let type_vec: Option<Vec<String>> =
        memory_type.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());
    let type_strs: Option<Vec<&str>> = type_vec
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    let type_slice: Option<&[&str]> = type_strs.as_deref();

    let status_vec: Option<Vec<String>> = if include_candidates {
        Some(vec!["active".to_string(), "candidate".to_string()])
    } else {
        status.map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
    };
    let status_strs: Option<Vec<&str>> = status_vec
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());
    let status_slice: Option<&[&str]> = status_strs.as_deref();

    let memories = store.list(project_id, limit, type_slice, status_slice)?;
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
        Err(Error::NotFound("memory not found".to_string()))
    }
}

fn handle_update(
    store: &mut MemoryStore,
    id: &str,
    text: Option<&str>,
    metadata: Option<&str>,
    memory_type: Option<&str>,
    status: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    if text.is_none() && metadata.is_none() && memory_type.is_none() && status.is_none() {
        return Err(Error::InvalidInput(
            "At least one of text, metadata, memory_type, or status must be provided".to_string(),
        ));
    }

    // Validate metadata: reject empty strings and invalid JSON
    if let Some(meta) = metadata {
        if meta.trim().is_empty() {
            return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
        }
        // Validate that metadata is valid JSON
        serde_json::from_str::<serde_json::Value>(meta)
            .map_err(|e| Error::InvalidInput(format!("invalid metadata JSON: {}", e)))?;
    }

    store.update(id, text, metadata, memory_type, status)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_short_text() {
        let short_text = "hello world";
        let result = handle_validate(short_text, "not-a-real-model-should-fail", false);
        // Should fail because model doesn't exist, not because of token count
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_long_text() {
        let long_text = "a".repeat(1000);
        let result = handle_validate(&long_text, "not-a-real-model-should-fail", false);
        // Should fail because model doesn't exist, not because of token count
        assert!(result.is_err());
    }
}
