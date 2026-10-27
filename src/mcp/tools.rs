//! MCP tool implementations.

use crate::mcp::helpers::{format_memory_json, metadata_column};
use crate::mcp::store_wrapper::{McpError, StoreWrapper};

use crate::errors::Error;
use crate::mcp::params::*;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::{MemoryStore, NewMemory, UpdateParams};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerConfig};
use rmcp::tool;
use rmcp::tool_handler;
use rmcp::tool_router;
use std::sync::{Arc, Mutex};

/// MCP tool handler.
pub struct ToolHandler {
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
            store: StoreWrapper(store),
            project_id,
            config,
        }
    }

    /// Drive one tool exactly as an MCP client does, through the public
    /// `ServerHandler::call_tool` entry point the protocol server itself uses:
    /// same `CallToolRequestParams` → `Parameters<T>` deserialization, same
    /// result envelope.
    ///
    /// The `#[tool]` methods are private (rmcp routes them through the router),
    /// so a test that wants to pin the wiring *inside* a tool handler — not just
    /// the wrapper it calls — has to enter this way. Used by the #258 regression
    /// test for `store_memory`'s `supersedes` arm.
    #[cfg(test)]
    pub(crate) async fn call_tool_for_test(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
        use rmcp::ServerHandler;
        use rmcp::model::CallToolRequestParams;

        let request =
            CallToolRequestParams::new(name.to_string()).with_arguments(match arguments {
                serde_json::Value::Object(map) => map,
                _ => serde_json::Map::new(),
            });
        // The tool handlers never touch the request context (they only take
        // `Parameters<T>`), so the minimal no-op transport below is enough to
        // satisfy the signature.
        self.call_tool(request, crate::mcp::test_transport::test_request_context())
            .await
    }

    /// Test-only accessor exposing the stored config, so tests can assert
    /// that `ToolHandler::new` retains the exact config it was given rather
    /// than silently falling back to defaults for any field (see #149).
    #[cfg(test)]
    pub(crate) fn config(&self) -> &crate::config::Config {
        &self.config
    }

    /// Test-only: the tool definitions the server advertises via `tools/list`,
    /// including each tool's JSON-schema `input_schema`. The schema is generated
    /// from the params structs' `#[schemars(description = ...)]` attributes, so
    /// this is the only crate-visible path to what a client actually sees.
    #[cfg(test)]
    pub(crate) fn test_tool_definition(name: &str) -> Option<rmcp::model::Tool> {
        Self::test_tool_router().get(name).cloned()
    }

    /// `#[tool_router]` generates a **private** `tool_router()` constructor
    /// (rmcp-macros 1.8.0 src/tool_router.rs:63-67 emits `#vis fn tool_router()`
    /// and `#vis` is filled only from the macro's own `vis = "..."` attribute),
    /// so the generated item is not reachable from `mcp::tests`. This thin
    /// wrapper gives the tests a crate-visible path to the router the server
    /// actually serves; rename the generated one and this line, not the tests,
    /// is what breaks.
    #[cfg(test)]
    pub(crate) fn test_tool_router() -> rmcp::handler::server::router::tool::ToolRouter<Self> {
        Self::tool_router()
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

        // Parse and validate status (default: "active"). The store's supersede
        // body runs the same insertability check for library callers; this copy
        // exists to answer with an MCP error rather than a store error, and
        // goes through the one helper that owns the wording (issue #258).
        let status_str = params.status.as_deref().unwrap_or("active");
        let status_val = MemoryStatus::from_str(status_str)
            .map_err(|e| McpError::invalid_input(&format!("Invalid status: {}", e)))?;
        status_val.ensure_valid_for_insert()?;

        // Parse and validate importance (default: "medium"). The level is parsed
        // here, before either write branch, so an invalid value answers with an
        // MCP invalid_input and no row is inserted on either path (#256).
        let importance_str = params.importance.as_deref().unwrap_or("medium");
        let importance_val = MemoryImportance::from_str(importance_str)
            .map_err(|e| McpError::invalid_input(&format!("Invalid importance: {}", e)))?;

        // Serialize metadata
        let metadata_str = metadata_column(&params.metadata)?;

        // Handle supersedes path — the validated status and importance have to
        // travel with the new row or `status: "candidate"` is silently stored as
        // `active` (issue #258) and a named level as `medium` (issue #256), both
        // the MCP twins of the CLI drops. The row is the store's own `NewMemory`,
        // so the two add paths share one row type rather than duplicating it.
        let row = NewMemory {
            project_id: &self.project_id,
            content: &params.text,
            metadata: metadata_str.as_deref(),
            memory_type: memory_type_val,
            status: status_val,
        };

        if let Some(supersedes_id) = &params.supersedes {
            // `--supersedes` and `--force` are mutually exclusive on the CLI
            // (`handle_add`); the MCP twin has to refuse them together too, or a
            // client could ask for a supersede that skips the conflict gate the
            // supersede path never runs. Checked before any write.
            if params.force.unwrap_or(false) {
                return Err(McpError::invalid_input(
                    "Cannot use both --supersedes and --force flags together",
                ));
            }
            let value = self
                .store
                .supersede_with_row(&row, supersedes_id, importance_val)?;
            return Ok(CallToolResult::success(vec![ContentBlock::text(
                serde_json::to_string(&value).map_err(McpError::from_serde_error)?,
            )]));
        }

        // Normal ingest path with type, status and importance
        let value = self.store.ingest_with_lifecycle(
            &row,
            params.force.unwrap_or(false),
            importance_val,
        )?;

        Ok(CallToolResult::success(vec![ContentBlock::text(
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
            let id_refs: Vec<&str> = memories.iter().map(|m| m.id.as_str()).collect();
            if let Err(e) = self.store.touch_memories(&id_refs) {
                eprintln!("warning: touch_memories failed (telemetry not updated): {e}");
            }
        }

        // Build JSON response using helper
        let results: Vec<serde_json::Value> = memories.iter().map(format_memory_json).collect();

        Ok(CallToolResult::success(vec![ContentBlock::text(
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

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string(&value).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// Replace an existing memory with new content.
    ///
    /// The old memory is marked as superseded and a new memory is created.
    /// Use this when information has changed and the old version should no
    /// longer appear in search results. Omitted or `null` `metadata` is stored as SQL NULL (#305).
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

        let metadata_str = metadata_column(&params.metadata)?;

        let value = self.store.supersede(
            &self.project_id,
            &params.new_text,
            metadata_str.as_deref(),
            memory_type_val,
            &params.old_memory_id,
        )?;

        Ok(CallToolResult::success(vec![ContentBlock::text(
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

        // Get the memory (scoped to current project)
        let memory_opt = self.store.get(&params.id, &self.project_id)?;
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
        if !params.no_touch.unwrap_or(false)
            && let Err(e) = self.store.touch_memories(&[memory.id.as_str()])
        {
            eprintln!("warning: touch_memories failed (telemetry not updated): {e}");
        }

        // Format memory as JSON using helper
        let result = format_memory_json(&memory);

        Ok(CallToolResult::success(vec![ContentBlock::text(
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

        let deleted = self.store.delete(&params.id, &self.project_id)?;

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

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string(&result).map_err(McpError::from_serde_error)?,
        )]))
    }

    /// Update an existing memory's content, metadata, type, status, or importance.
    ///
    /// At least one field must be provided.
    #[tool(
        name = "update_memory",
        description = "Update an existing memory's content, metadata, type, status, or importance. At least one field must be provided."
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
            && params.importance.is_none()
        {
            return Err(McpError::invalid_input(
                "At least one of text, metadata, memory_type, status, or importance must be provided",
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

        // Parse importance if provided. Parsed before the write so an invalid
        // level answers with an MCP invalid_input and the row keeps the level it
        // had: the column has no CHECK constraint, so this frame is the only
        // thing between a client typo and an unparsable level (#280).
        let importance_val = params
            .importance
            .as_deref()
            .map(MemoryImportance::from_str)
            .transpose()
            .map_err(|e| McpError::invalid_input(&format!("Invalid importance: {e}")))?;

        let metadata_str = metadata_column(&params.metadata)?;

        // Convert text to &str if provided
        let text_str = params.text.as_deref();

        // Perform the update
        self.store.update(
            &params.id,
            &self.project_id,
            UpdateParams {
                text: text_str,
                metadata: metadata_str.as_deref(),
                memory_type: memory_type_val,
                status: status_val,
                importance: importance_val,
            },
        )?;

        let result = serde_json::json!({
            "id": params.id,
            "status": "updated"
        });

        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string(&result).map_err(McpError::from_serde_error)?,
        )]))
    }
}

#[tool_handler]
impl rmcp::ServerHandler for ToolHandler {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}
