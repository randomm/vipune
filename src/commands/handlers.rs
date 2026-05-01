//! Command handlers for vipune CLI.

use crate::errors::Error;
use crate::memory::MemoryStore;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory_types::{AddResult, IngestPolicy};
use crate::output::*;
use crate::{config, embedding::EmbeddingEngine, temporal};
use std::process::ExitCode;

pub(crate) struct SearchContext {
    pub(crate) query: String,
    pub(crate) limit: usize,
    pub(crate) recency: Option<f64>,
    pub(crate) hybrid: bool,
    pub(crate) no_hybrid: bool,
    pub(crate) memory_type: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) include_candidates: bool,
    pub(crate) no_touch: bool,
}

pub(crate) fn handle_validate(text: &str, model_id: &str, json: bool) -> Result<ExitCode, Error> {
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
pub(crate) fn handle_add(
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
    let memory_type_val = MemoryType::from_str(memory_type)?;
    let status_val = MemoryStatus::from_str(status)?;
    if !status_val.is_valid_for_insert() {
        return Err(Error::InvalidInput(format!(
            "Status '{}' is not valid for new memory insertion. Must be 'active' or 'candidate'.",
            status
        )));
    }

    if supersedes.is_some() && force {
        return Err(Error::InvalidInput(
            "Cannot use both --supersedes and --force flags together".to_string(),
        ));
    }

    if let Some(old_id) = supersedes {
        let new_id = store.supersede(project_id, text, metadata, memory_type_val, old_id)?;

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

    match store.ingest_with_type_status(
        project_id,
        text,
        metadata,
        policy,
        memory_type_val,
        status_val,
    )? {
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

pub(crate) fn handle_search(
    store: &mut MemoryStore,
    project_id: &str,
    opts: &SearchContext,
    config: &config::Config,
    json: bool,
) -> Result<ExitCode, Error> {
    let recency_weight = opts.recency.unwrap_or(config.recency_weight);
    temporal::validate_recency_weight(recency_weight)?;

    let type_vec: Option<Vec<String>> = opts
        .memory_type
        .as_ref()
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());
    let type_strs: Option<Vec<&str>> = type_vec
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());

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

    let use_hybrid = (opts.hybrid || config.hybrid) && !opts.no_hybrid;
    let search_options = crate::memory::SearchOptions {
        memory_types: type_strs,
        statuses: status_strs,
    };
    let memories = if use_hybrid {
        store.search_hybrid(
            project_id,
            &opts.query,
            opts.limit,
            recency_weight,
            search_options,
        )?
    } else {
        store.search(
            project_id,
            &opts.query,
            opts.limit,
            recency_weight,
            search_options,
        )?
    };

    if !opts.no_touch {
        let ids: Vec<&str> = memories.iter().map(|m| m.id.as_str()).collect();
        if !ids.is_empty() {
            if let Err(e) = store.db.touch_memories(&ids) {
                eprintln!("warning: failed to update retrieval stats: {}", e);
            }
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

pub(crate) fn handle_get(
    store: &mut MemoryStore,
    id: &str,
    no_touch: bool,
    json: bool,
) -> Result<ExitCode, Error> {
    let memory = store
        .get(id)?
        .ok_or_else(|| Error::NotFound("memory not found".to_string()))?;

    if !no_touch {
        if let Err(e) = store.db.touch_memories(&[id]) {
            eprintln!("warning: failed to update retrieval stats: {}", e);
        }
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

pub(crate) fn handle_list(
    store: &mut MemoryStore,
    project_id: &str,
    limit: usize,
    memory_type: Option<&str>,
    status: Option<&str>,
    include_candidates: bool,
    json: bool,
) -> Result<ExitCode, Error> {
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

pub(crate) fn handle_delete(
    store: &mut MemoryStore,
    id: &str,
    json: bool,
) -> Result<ExitCode, Error> {
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

pub(crate) fn handle_update(
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

    if let Some(meta) = metadata {
        if meta.trim().is_empty() {
            return Err(Error::InvalidInput("metadata cannot be empty".to_string()));
        }
        serde_json::from_str::<serde_json::Value>(meta)
            .map_err(|e| Error::InvalidInput(format!("invalid metadata JSON: {}", e)))?;
    }

    let memory_type_val = memory_type.map(MemoryType::from_str).transpose()?;
    let status_val = status.map(MemoryStatus::from_str).transpose()?;

    store.update(id, text, metadata, memory_type_val, status_val)?;
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

pub(crate) fn handle_version(json: bool) -> Result<ExitCode, Error> {
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
