//! MCP tool implementations.

use crate::errors::Error;
use crate::mcp::params::*;
use crate::memory::MemoryStore;
use crate::memory::lifecycle::{MemoryStatus, MemoryType};
use crate::memory_types::ConflictMemory as InternalConflictMemory;
use crate::memory_types::IngestPolicy;
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
pub(crate) struct StoreWrapper(Arc<Mutex<MemoryStore>>);

impl StoreWrapper {
    /// Create a new StoreWrapper from a shared MemoryStore.
    #[allow(dead_code)] // Used in cfg(test) builds
    pub(crate) fn new(store: Arc<Mutex<MemoryStore>>) -> Self {
        Self(store)
    }

    /// Store a memory with conflict detection.
    #[allow(dead_code)] // Used for backward compatibility
    pub(crate) fn ingest(
        &self,
        project_id: &str,
        text: &str,
        metadata: &str,
        force: bool,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();
        let policy = if force {
            IngestPolicy::Force
        } else {
            IngestPolicy::ConflictAware
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

    /// Store a memory with explicit type and status.
    #[allow(dead_code)] // Used by store_memory with type/status
    pub(crate) fn ingest_with_type_status(
        &self,
        project_id: &str,
        text: &str,
        metadata: &str,
        force: bool,
        memory_type: &str,
        status: &str,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();
        let policy = if force {
            IngestPolicy::Force
        } else {
            IngestPolicy::ConflictAware
        };

        match store
            .ingest_with_type_status(
                project_id,
                text,
                Some(metadata),
                policy,
                memory_type,
                status,
            )
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

    /// Supersede an existing memory with new content.
    #[allow(dead_code)] // Used by supersede_memory tool
    pub(crate) fn supersede(
        &self,
        project_id: &str,
        new_text: &str,
        metadata: &str,
        memory_type: &str,
        old_memory_id: &str,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();

        // Use mock embedding if embedder is not loaded (test mode)
        let embedding = if store.embedder.is_none() {
            crate::memory::crud::mock_embedding_for_content(new_text)
        } else {
            store.embedder()?.embed(new_text)?
        };

        let metadata_str = if metadata == "null" {
            None
        } else {
            Some(metadata)
        };

        let new_id = match store.db.supersede(
            project_id,
            new_text,
            &embedding,
            metadata_str,
            memory_type,
            old_memory_id,
        ) {
            Ok(id) => id,
            Err(err) => {
                let err: Error = err.into();
                return Err(err.into());
            }
        };

        serde_json::to_value(SuccessResponse {
            id: new_id,
            status: "superseded".to_string(),
        })
        .map_err(McpError::from_serde_error)
    }

    /// Search memories by semantic meaning.
    #[allow(dead_code)] // Used in tests but not in actual MCP flow (server uses store.search directly)
    pub(crate) fn search(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
        memory_types: Option<Vec<&str>>,
        statuses: Option<Vec<&str>>,
    ) -> Result<serde_json::Value, Error> {
        let mut store = self.0.lock().unwrap();
        let memories = store.search(
            project_id,
            query,
            limit,
            recency_weight,
            crate::memory::SearchOptions {
                memory_types,
                statuses,
                touch: true,
            },
        )?;

        let results: Vec<serde_json::Value> = memories
            .into_iter()
            .map(|m| {
                // Parse metadata string to JSON value, or use null if None/invalid
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
                    "metadata": metadata_value
                })
            })
            .collect();

        Ok(serde_json::to_value(results)?)
    }

    /// Search memories by hybrid (semantic + BM25 with RRF fusion).
    #[allow(dead_code)] // MCP tool, available for future use
    pub(crate) fn search_hybrid(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
        memory_types: Option<Vec<&str>>,
        statuses: Option<Vec<&str>>,
    ) -> Result<serde_json::Value, Error> {
        let mut store = self.0.lock().unwrap();
        let options = crate::memory::SearchOptions {
            memory_types,
            statuses,
            touch: true,
        };
        let memories = store.search_hybrid(
            project_id,
            query,
            limit,
            recency_weight,
            options,
        )?;

        let results: Vec<serde_json::Value> = memories
            .into_iter()
            .map(|m| {
                // Parse metadata string to JSON value, or use null if None/invalid
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
                    "metadata": metadata_value
                })
            })
            .collect();

        Ok(serde_json::to_value(results)?)
    }

    /// List recent memories.
    pub(crate) fn list(
        &self,
        project_id: &str,
        limit: usize,
        memory_types: Option<&[&str]>,
        statuses: Option<&[&str]>,
    ) -> Result<serde_json::Value, Error> {
        let store = self.0.lock().unwrap();
        let memories = store.list(project_id, limit, memory_types, statuses)?;

        let results: Vec<serde_json::Value> = memories
            .into_iter()
            .map(|m| {
                // Parse metadata string to JSON value, or use null if None/invalid
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
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                    "project_id": m.project_id,
                    "metadata": metadata_value
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
            // User input errors → INVALID_REQUEST with invalid_input type
            Error::EmptyInput => McpError::invalid_input("Text cannot be empty"),
            Error::InputTooLong {
                max_length,
                actual_length,
            } => McpError::invalid_input(&format!(
                "Input too long: {} characters (max: {})",
                actual_length, max_length
            )),
            Error::InvalidInput(msg) => McpError::invalid_input(&msg),
            Error::Validation(msg) => McpError::invalid_input(&msg),
            Error::InvalidTimestamp { timestamp, error } => {
                McpError::invalid_input(&format!("Invalid timestamp '{}': {}", timestamp, error))
            }
            Error::ContentTooLong {
                token_count,
                max_tokens,
            } => McpError::invalid_input(&format!(
                "Content exceeds {}-token embedding limit (measured: {} tokens)",
                max_tokens, token_count
            )),
            // Not found → INVALID_REQUEST with not_found type
            Error::NotFound(msg) => rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_REQUEST,
                msg,
                Some(serde_json::json!({"type": "not_found"})),
            ),
            // Internal/unrecoverable errors → INTERNAL_ERROR
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

        // Parse and validate memory_type (default: "fact")
        let memory_type = params.memory_type.as_deref().unwrap_or("fact");
        let _ = MemoryType::from_str(memory_type)
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
                memory_type,
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
            memory_type,
            status_str,
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
        let no_touch = params.no_touch.unwrap_or(false);

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
                touch: !no_touch,
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
        let _ = MemoryType::from_str(memory_type)
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
            memory_type,
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
