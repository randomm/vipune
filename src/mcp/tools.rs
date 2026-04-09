//! MCP tool implementations.

use crate::errors::Error;
use crate::mcp::params::*;
use crate::memory::MemoryStore;
use crate::memory_types::ConflictMemory as InternalConflictMemory;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::tool;
use rmcp::tool_handler;
use rmcp::tool_router;
use std::sync::{Arc, Mutex};

/// Wrapper for MemoryStore to allow async-sound use.
///
/// Since MCP stdio handles requests sequentially, the Mutex will never contend.
struct StoreWrapper(Arc<Mutex<MemoryStore>>);

impl StoreWrapper {
    /// Store a memory with conflict detection.
    fn ingest(
        &self,
        project_id: &str,
        text: &str,
        metadata: &str,
        force: bool,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();
        let policy = if force {
            crate::memory_types::IngestPolicy::Force
        } else {
            crate::memory_types::IngestPolicy::ConflictAware
        };

        match store
            .ingest(project_id, text, Some(metadata), policy)
            .map_err(|e| -> rmcp::ErrorData { e.into() })?
        {
            crate::memory_types::AddResult::Added { id } => {
                Ok(serde_json::to_value(SuccessResponse {
                    id,
                    status: "added".to_string(),
                })
                .map_err(McpError::from_serde_error)?)
            }
            crate::memory_types::AddResult::Conflicts {
                proposed,
                conflicts,
            } => Err(McpError::conflict(
                &proposed,
                conflicts
                    .into_iter()
                    .map(|c: InternalConflictMemory| ConflictMemory {
                        id: c.id,
                        content: c.content,
                        similarity: c.similarity,
                    })
                    .collect(),
            )),
        }
    }

    /// Search memories by semantic meaning.
    fn search(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<serde_json::Value, Error> {
        let mut store = self.0.lock().unwrap();
        let memories = store.search(project_id, query, limit, 0.0)?;

        let results: Vec<serde_json::Value> = memories
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "content": m.content,
                    "similarity": m.similarity.unwrap_or(0.0),
                    "created_at": m.created_at
                })
            })
            .collect();

        Ok(serde_json::to_value(results)?)
    }

    /// List recent memories.
    fn list(&self, project_id: &str, limit: usize) -> Result<serde_json::Value, Error> {
        let store = self.0.lock().unwrap();
        let memories = store.list(project_id, limit)?;

        let results: Vec<serde_json::Value> = memories
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "content": m.content,
                    "created_at": m.created_at
                })
            })
            .collect();

        Ok(serde_json::to_value(results)?)
    }
}

/// MCP tool handler.
pub struct ToolHandler {
    tool_router: ToolRouter<Self>,
    store: StoreWrapper,
    project_id: String,
}

impl ToolHandler {
    pub fn new(store: Arc<Mutex<MemoryStore>>, project_id: String) -> Self {
        Self {
            tool_router: Self::tool_router(),
            store: StoreWrapper(store),
            project_id,
        }
    }
}

/// MCP error types mapped to rmcp ErrorData.
#[derive(Debug)]
pub struct McpError;

impl McpError {
    pub fn invalid_input(message: &str) -> rmcp::ErrorData {
        rmcp::ErrorData::new(
            rmcp::model::ErrorCode::INVALID_REQUEST,
            message.to_string(),
            Some(serde_json::json!({"type": "invalid_input"})),
        )
    }

    pub fn internal_error(message: &str) -> rmcp::ErrorData {
        rmcp::ErrorData::new(
            rmcp::model::ErrorCode::INTERNAL_ERROR,
            message.to_string(),
            Some(serde_json::json!({"type": "internal_error"})),
        )
    }

    pub fn conflict(proposed: &str, conflicts: Vec<ConflictMemory>) -> rmcp::ErrorData {
        rmcp::ErrorData::new(
            rmcp::model::ErrorCode::INVALID_REQUEST,
            format!(
                "Proposed memory conflicts with {} existing memory(ies)",
                conflicts.len()
            ),
            Some(serde_json::json!({
                "type": "conflict",
                "proposed": proposed,
                "conflicts": conflicts
            })),
        )
    }

    fn from_serde_error(e: serde_json::Error) -> rmcp::ErrorData {
        rmcp::ErrorData::new(
            rmcp::model::ErrorCode::INTERNAL_ERROR,
            format!("Serialization error: {}", e),
            Some(serde_json::json!({"type": "internal_error"})),
        )
    }
}

/// Conversion from library errors to MCP errors.
impl From<Error> for rmcp::ErrorData {
    fn from(e: Error) -> Self {
        match e {
            Error::EmptyInput => McpError::invalid_input("Text cannot be empty"),
            Error::InputTooLong {
                max_length,
                actual_length,
            } => McpError::invalid_input(&format!(
                "Input too long: {} characters (max: {})",
                actual_length, max_length
            )),
            _ => McpError::internal_error(&e.to_string()),
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

        // Serialize metadata
        let metadata_str = match &params.metadata {
            Some(meta) => serde_json::to_string(meta)
                .map_err(|e| McpError::invalid_input(&format!("Invalid metadata: {}", e)))?,
            None => "null".to_string(),
        };

        let value = self
            .store
            .ingest(&self.project_id, &params.text, &metadata_str, false)?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).unwrap_or_default(),
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

        let value = self
            .store
            .search(&self.project_id, &params.query, limit)
            .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).unwrap_or_default(),
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

        let value = self
            .store
            .list(&self.project_id, limit)
            .map_err(|e: Error| -> rmcp::ErrorData { e.into() })?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).unwrap_or_default(),
        )]))
    }
}

#[tool_handler]
impl rmcp::ServerHandler for ToolHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}
