//! MCP tool implementations.

use crate::mcp::helpers::format_memory_json;
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
        let force = params.force.unwrap_or(false);
        let value = self.store.ingest_with_type_status(
            &self.project_id,
            &params.text,
            &metadata_str,
            force,
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
        let no_touch = params.no_touch.unwrap_or(false);

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
        let memories = if use_hybrid {
            self.store
                .search_hybrid_raw(
                    &self.project_id,
                    &params.query,
                    limit,
                    recency_weight,
                    search_options,
                )
                .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?
        } else {
            self.store
                .search_raw(
                    &self.project_id,
                    &params.query,
                    limit,
                    recency_weight,
                    search_options,
                )
                .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?
        };

        // Update retrieval telemetry unless no_touch is true
        if !no_touch && !memories.is_empty() {
            let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
            let id_refs: Vec<&str> = memory_ids.iter().map(|s| s.as_str()).collect();
            let _ = self.store.touch_memories(&id_refs);
        }

        // Build JSON response using helper
        let results: Vec<serde_json::Value> = memories.iter().map(format_memory_json).collect();

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

    /// Retrieve a specific memory by ID.
    ///
    /// Use when you know the exact memory ID.
    #[tool(
        name = "get_memory",
        description = "Retrieve a specific memory by ID. Use when you know the exact memory ID."
    )]
    async fn get_memory(
        &self,
        Parameters(params): Parameters<GetMemoryParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Validate input
        if params.id.trim().is_empty() {
            return Err(McpError::invalid_input("ID cannot be empty"));
        }

        // Get the memory
        let memory_opt = self.store.get(&params.id)?;
        let memory = match memory_opt {
            Some(m) => m,
            None => {
                return Err(McpError::invalid_input(&format!(
                    "memory not found: {}",
                    params.id
                )));
            }
        };

        // Update retrieval telemetry unless no_touch is true
        if !params.no_touch.unwrap_or(false) {
            let _ = self.store.touch_memories(&[memory.id.as_str()]);
        }

        // Format memory as JSON using helper
        let result = format_memory_json(&memory);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// Delete a memory permanently by ID.
    #[tool(
        name = "delete_memory",
        description = "Delete a memory permanently by ID."
    )]
    async fn delete_memory(
        &self,
        Parameters(params): Parameters<DeleteMemoryParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Validate input
        if params.id.trim().is_empty() {
            return Err(McpError::invalid_input("ID cannot be empty"));
        }

        let deleted = self.store.delete(&params.id)?;

        if !deleted {
            return Err(McpError::invalid_input(&format!(
                "memory not found: {}",
                params.id
            )));
        }

        let result = serde_json::json!({
            "id": params.id,
            "status": "deleted"
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// Update an existing memory's content, metadata, type, or status.
    ///
    /// At least one field must be provided.
    #[tool(
        name = "update_memory",
        description = "Update an existing memory's content, metadata, type, or status. At least one field must be provided."
    )]
    async fn update_memory(
        &self,
        Parameters(params): Parameters<UpdateMemoryParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Validate input
        if params.id.trim().is_empty() {
            return Err(McpError::invalid_input("ID cannot be empty"));
        }

        // Validate at least one optional field is provided
        if params.text.is_none()
            && params.metadata.is_none()
            && params.memory_type.is_none()
            && params.status.is_none()
        {
            return Err(McpError::invalid_input(
                "At least one of text, metadata, memory_type, or status must be provided",
            ));
        }

        // Parse memory_type if provided
        let memory_type_val = if let Some(ref mt) = params.memory_type {
            let mt_str = mt.as_str();
            Some(
                MemoryType::from_str(mt_str)
                    .map_err(|e| McpError::invalid_input(&format!("Invalid memory type: {}", e)))?,
            )
        } else {
            None
        };

        // Parse status if provided
        let status_val = if let Some(ref s) = params.status {
            let s_str = s.as_str();
            Some(
                MemoryStatus::from_str(s_str)
                    .map_err(|e| McpError::invalid_input(&format!("Invalid status: {}", e)))?,
            )
        } else {
            None
        };

        // Serialize metadata if provided
        let metadata_str = if let Some(ref meta) = params.metadata {
            Some(
                serde_json::to_string(meta)
                    .map_err(|e| McpError::invalid_input(&format!("Invalid metadata: {}", e)))?,
            )
        } else {
            None
        };

        // Convert text to &str if provided
        let text_str = params.text.as_deref();

        // Perform the update
        self.store.update(
            &params.id,
            text_str,
            metadata_str.as_deref(),
            memory_type_val,
            status_val,
        )?;

        let result = serde_json::json!({
            "id": params.id,
            "status": "updated"
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).map_err(McpError::from_serde_error)?,
        )]))
    }
}

#[tool_handler]
impl rmcp::ServerHandler for ToolHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}
