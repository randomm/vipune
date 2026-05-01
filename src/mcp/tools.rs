//! MCP tool implementations.

use crate::mcp::store_wrapper::{McpError, StoreWrapper};

use crate::errors::Error;
use crate::mcp::params::*;
use crate::memory::MemoryStore;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::tool;
use rmcp::tool_handler;
use rmcp::tool_router;
use std::sync::{Arc, Mutex};

/// MCP tool handler.
pub struct ToolHandler {
    tool_router: ToolRouter<Self>,
    store: StoreWrapper,
    project_id: String,
    config: crate::config::Config,
}

impl ToolHandler {
    pub fn new(
        store: Arc<Mutex<MemoryStore>>,
        project_id: String,
        config: crate::config::Config,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            store: StoreWrapper(store),
            project_id,
            config,
        }
    }
}

#[tool_router]
impl ToolHandler {
    /// Store information for later recall.
    ///
    /// Use this when you learn something worth remembering — facts, decisions,
    /// preferences, or context about a project. Memories are searchable by meaning,
    /// not just keywords.
    #[tool(
        name = "store_memory",
        description = "Store information for later recall. Use this when you learn something worth remembering — facts, decisions, preferences, or context about a project. Memories are searchable by meaning, not just keywords."
    )]
    async fn store_memory(
        &self,
        Parameters(params): Parameters<StoreMemoryParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Validate input
        if params.text.trim().is_empty() {
            return Err(McpError::invalid_input("Text cannot be empty"));
        }

        // Parse and validate memory_type (default: "fact")
        let memory_type = params.memory_type.as_deref().unwrap_or("fact");
        let memory_type_val = MemoryType::from_str(memory_type)
            .map_err(|e| McpError::invalid_input(&format!("Invalid memory type: {}", e)))?;

        // Parse and validate status (default: "active")
        let status_str = params.status.as_deref().unwrap_or("active");
        let status_val = MemoryStatus::from_str(status_str)
            .map_err(|e| McpError::invalid_input(&format!("Invalid status: {}", e)))?;
        if !status_val.is_valid_for_insert() {
            return Err(McpError::invalid_input(&format!(
                "Status '{}' is not valid for new memory. Must be 'active' or 'candidate'.",
                status_str
            )));
        }

        // Serialize metadata
        let metadata_str = match &params.metadata {
            Some(meta) => serde_json::to_string(meta)
                .map_err(|e| McpError::invalid_input(&format!("Invalid metadata: {}", e)))?,
            None => "null".to_string(),
        };

        // Handle supersedes path
        if let Some(supersedes_id) = &params.supersedes {
            let value = self.store.supersede(
                &self.project_id,
                &params.text,
                &metadata_str,
                memory_type_val,
                supersedes_id,
            )?;
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string(&value).map_err(McpError::from_serde_error)?,
            )]));
        }

        // Normal ingest path with type and status
        let value = self.store.ingest_with_type_status(
            &self.project_id,
            &params.text,
            &metadata_str,
            false,
            memory_type_val,
            status_val,
        )?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// Search memories by meaning.
    ///
    /// Describe what you are looking for in natural language. Use this when you
    /// need to recall previously stored information, check if a topic was discussed,
    /// or find related context. Start here when you need information from memory.
    #[tool(
        name = "search_memories",
        description = "Search memories by meaning. Describe what you are looking for in natural language. Use this when you need to recall previously stored information, check if a topic was discussed, or find related context."
    )]
    async fn search_memories(
        &self,
        Parameters(params): Parameters<SearchMemoriesParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Validate input
        if params.query.trim().is_empty() {
            return Err(McpError::invalid_input("Query cannot be empty"));
        }

        let limit = params.limit.unwrap_or(5);

        // Validate limit
        if limit == 0 {
            return Err(McpError::invalid_input("Limit must be greater than 0"));
        }
        if limit > 10_000 {
            return Err(McpError::invalid_input(
                "Limit exceeds maximum allowed (10000)",
            ));
        }

        // Convert filter params - move into the block where they're used
        let recency_weight = params.recency_weight.unwrap_or(self.config.recency_weight);
        let use_hybrid = params.hybrid.unwrap_or(self.config.hybrid);

        let memories = {
            let mut store = self.store.0.lock().unwrap();
            let type_strs: Option<Vec<&str>> = params
                .memory_types
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            let status_strs: Option<Vec<&str>> = params
                .statuses
                .as_ref()
                .map(|v| v.iter().map(|s| s.as_str()).collect());
            let search_options = crate::memory::SearchOptions {
                memory_types: type_strs,
                statuses: status_strs,
            };
            if use_hybrid {
                store
                    .search_hybrid(
                        &self.project_id,
                        &params.query,
                        limit,
                        recency_weight,
                        search_options,
                    )
                    .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?
            } else {
                store
                    .search(
                        &self.project_id,
                        &params.query,
                        limit,
                        recency_weight,
                        search_options,
                    )
                    .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?
            }
        };

        // Build JSON response manually (Memory doesn't derive Serialize)
        let results: Vec<serde_json::Value> = memories
            .into_iter()
            .map(|m| {
                let metadata_value = match m.metadata {
                    Some(ref meta_str) if meta_str.trim() != "null" => {
                        serde_json::from_str::<serde_json::Value>(meta_str)
                            .unwrap_or_else(|_| serde_json::Value::String(meta_str.clone()))
                    }
                    _ => serde_json::Value::Null,
                };
                serde_json::json!({
                    "id": m.id,
                    "content": m.content,
                    "similarity": m.similarity.unwrap_or(0.0),
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                    "project_id": m.project_id,
                    "metadata": metadata_value,
                    "retrieval_count": m.retrieval_count,
                    "last_retrieved_at": m.last_retrieved_at
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&results).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// List recent memories.
    ///
    /// Use this to review what was recently stored, get an overview of stored
    /// knowledge, or find memories when you are not sure what to search for.
    #[tool(
        name = "list_memories",
        description = "List recent memories. Use this to review what was recently stored, get an overview of stored knowledge, or find memories when you are not sure what to search for."
    )]
    async fn list_memories(
        &self,
        Parameters(params): Parameters<ListMemoriesParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let limit = params.limit.unwrap_or(10);

        // Validate limit
        if limit == 0 {
            return Err(McpError::invalid_input("Limit must be greater than 0"));
        }
        if limit > 10_000 {
            return Err(McpError::invalid_input(
                "Limit exceeds maximum allowed (10000)",
            ));
        }

        // Convert filter params
        let type_strs: Option<Vec<&str>> = params
            .memory_types
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let type_slice: Option<&[&str]> = type_strs.as_deref();

        let status_strs: Option<Vec<&str>> = params
            .statuses
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let status_slice: Option<&[&str]> = status_strs.as_deref();

        let value = self
            .store
            .list(&self.project_id, limit, type_slice, status_slice)
            .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// Replace an existing memory with new content.
    ///
    /// The old memory is marked as superseded and a new memory is created.
    /// Use this when information has changed and the old version should no
    /// longer appear in search results.
    #[tool(
        name = "supersede_memory",
        description = "Replace an existing memory with new content. The old memory is marked as superseded and a new memory is created. Use this when information has changed and the old version should no longer appear in search results."
    )]
    async fn supersede_memory(
        &self,
        Parameters(params): Parameters<SupersedeMemoryParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Validate input
        if params.new_text.trim().is_empty() {
            return Err(McpError::invalid_input("new_text cannot be empty"));
        }

        if params.old_memory_id.trim().is_empty() {
            return Err(McpError::invalid_input("old_memory_id cannot be empty"));
        }

        // Parse and validate memory_type (default: "fact")
        let memory_type = params.memory_type.as_deref().unwrap_or("fact");
        let memory_type_val = MemoryType::from_str(memory_type)
            .map_err(|e| McpError::invalid_input(&format!("Invalid memory type: {}", e)))?;

        // Serialize metadata
        let metadata_str = match &params.metadata {
            Some(meta) => serde_json::to_string(meta)
                .map_err(|e| McpError::invalid_input(&format!("Invalid metadata: {}", e)))?,
            None => "null".to_string(),
        };

        let value = self.store.supersede(
            &self.project_id,
            &params.new_text,
            &metadata_str,
            memory_type_val,
            &params.old_memory_id,
        )?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).map_err(McpError::from_serde_error)?,
        )]))
    }
}

#[tool_handler]
impl rmcp::ServerHandler for ToolHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}
