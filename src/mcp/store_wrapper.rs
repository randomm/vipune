//! StoreWrapper for MCP tool implementations.

use crate::errors::Error;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory_types::ConflictMemory as InternalConflictMemory;
use crate::memory_types::IngestPolicy;
use std::sync::{Arc, Mutex};

use crate::memory::{MemoryStore, NewMemory, NewSupersedingMemory, UpdateParams};

/// Wrapper for MemoryStore to allow async-sound use.
///
/// Since MCP stdio handles requests sequentially, the Mutex will never contend.
pub(crate) struct StoreWrapper(pub(crate) Arc<Mutex<MemoryStore>>);

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

    /// Store a memory with the full lifecycle the `store_memory` tool carries.
    ///
    /// The row and its level travel together in the store's own `NewMemory` +
    /// `MemoryImportance` params — no MCP-side row struct that could drift from
    /// them — so the frame that parses the strings is not the frame that drops
    /// one (#256, the MCP twin of the CLI defect #238). `force` decides the
    /// ingest policy; the other three are the parsed lifecycle columns.
    pub(crate) fn ingest_with_lifecycle(
        &self,
        row: &NewMemory<'_>,
        force: bool,
        importance: MemoryImportance,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();
        let policy = if force {
            IngestPolicy::Force
        } else {
            IngestPolicy::ConflictAware
        };

        match store
            .ingest_with_importance(row, policy, importance)
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

    /// Supersede an existing memory with new content, at the `active` status and
    /// the default importance — what the `supersede_memory` tool drives (it has
    /// neither a `status` nor an `importance` parameter).
    #[allow(dead_code)] // Used by supersede_memory tool
    pub(crate) fn supersede(
        &self,
        project_id: &str,
        new_text: &str,
        metadata: &str,
        memory_type: MemoryType,
        old_memory_id: &str,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let row = NewMemory {
            project_id,
            content: new_text,
            metadata: Some(metadata),
            memory_type,
            status: MemoryStatus::Active,
        };
        self.supersede_with_row(&row, old_memory_id, MemoryImportance::default())
    }

    /// Supersede an existing memory with a fully specified replacement row — what
    /// `store_memory`'s `supersedes` path drives, since that tool takes `status`
    /// and `importance` (#258, #256). `supersede_memory` reaches the same store
    /// call through [`Self::supersede`], which composes the row with `active` and
    /// the default level; the two tools differ only in their schemas, not in a
    /// second write path.
    ///
    /// The replacement row is the store's own [`NewMemory`] — the same type the
    /// add path carries — composed with the replaced id and the level into
    /// [`NewSupersedingMemory`] right here, so the two INSERT paths cannot drift
    /// column-for-column and no MCP-side row struct duplicates `NewMemory`.
    ///
    /// This frame adds nothing to the insert-status rule: it takes the row whose
    /// `status` is a `MemoryStatus`, the same type the store takes, so no string
    /// can become an unvalidated status one frame lower, and
    /// [`MemoryStore::supersede_with`] is the one place that rejects
    /// `superseded`/`deprecated` — the error arrives back as `invalid_input`
    /// through `From<Error> for rmcp::ErrorData`.
    pub(crate) fn supersede_with_row(
        &self,
        row: &NewMemory<'_>,
        old_memory_id: &str,
        importance: MemoryImportance,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();

        let new_id = match store.supersede_with(&NewSupersedingMemory {
            row: *row,
            old_id: old_memory_id,
            // The level the caller named, defaulted by name to `medium` in the
            // tool handler — never the column DEFAULT (#238, #256). On the
            // supersedes path it applies to the NEW row, matching the CLI.
            importance,
        }) {
            Ok(id) => id,
            Err(err) => {
                return Err(err.into());
            }
        };

        serde_json::to_value(SuccessResponse {
            id: new_id,
            status: "superseded".to_string(),
        })
        .map_err(McpError::from_serde_error)
    }

    /// Search memories by semantic meaning, returning raw Memory structs.
    pub(crate) fn search_raw(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
        options: crate::memory::SearchOptions,
    ) -> Result<Vec<crate::Memory>, Error> {
        let mut store = self.0.lock().unwrap();
        store.search(project_id, query, limit, recency_weight, options)
    }

    /// Search memories by hybrid (semantic + BM25), returning raw Memory structs.
    pub(crate) fn search_hybrid_raw(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
        recency_weight: f64,
        options: crate::memory::SearchOptions,
    ) -> Result<Vec<crate::Memory>, Error> {
        let mut store = self.0.lock().unwrap();
        store.search_hybrid(project_id, query, limit, recency_weight, options)
    }

    /// Search memories by semantic meaning.
    #[allow(dead_code)] // Used in tests but not in actual MCP flow
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
            },
        )?;

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
        };
        let memories = store.search_hybrid(project_id, query, limit, recency_weight, options)?;

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

    /// Access the inner Arc<Mutex<MemoryStore>> for ToolHandler.
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &Arc<Mutex<MemoryStore>> {
        &self.0
    }

    /// Get a specific memory by ID scoped to a project.
    #[allow(dead_code)] // Used by get_memory tool
    pub(crate) fn get(
        &self,
        id: &str,
        project_id: &str,
    ) -> Result<Option<crate::Memory>, rmcp::ErrorData> {
        let store = self.0.lock().map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                format!("store lock poisoned: {e}"),
                None,
            )
        })?;
        store
            .get(id, project_id)
            .map_err(|e| -> rmcp::ErrorData { e.into() })
    }

    /// Delete a memory by ID scoped to a project.
    #[allow(dead_code)] // Used by delete_memory tool
    pub(crate) fn delete(&self, id: &str, project_id: &str) -> Result<bool, rmcp::ErrorData> {
        let store = self.0.lock().map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                format!("store lock poisoned: {e}"),
                None,
            )
        })?;
        store
            .delete(id, project_id)
            .map_err(|e| -> rmcp::ErrorData { e.into() })
    }

    /// Update a memory's content, metadata, type, or status.
    #[allow(dead_code)] // Used by update_memory tool
    pub(crate) fn update(
        &self,
        id: &str,
        project_id: &str,
        params: UpdateParams<'_>,
    ) -> Result<(), rmcp::ErrorData> {
        let mut store = self.0.lock().map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                format!("store lock poisoned: {e}"),
                None,
            )
        })?;
        store
            .update(id, project_id, params)
            .map_err(|e| -> rmcp::ErrorData { e.into() })
    }

    /// Update retrieval telemetry for a list of memory IDs.
    #[allow(dead_code)] // Used by search_memories and get_memory tools
    pub(crate) fn touch_memories(&self, ids: &[&str]) -> Result<(), rmcp::ErrorData> {
        let store = self.0.lock().map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                format!("store lock poisoned: {e}"),
                None,
            )
        })?;
        store
            .touch_memories(ids)
            .map_err(|e| -> rmcp::ErrorData { e.into() })
    }
}

/// Response type for successful operations.
#[derive(serde::Serialize)]
struct SuccessResponse {
    id: String,
    status: String,
}

/// Conflict memory type for MCP errors.
#[derive(serde::Serialize)]
pub struct ConflictMemory {
    pub id: String,
    pub content: String,
    pub similarity: f64,
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

    pub fn from_serde_error(e: serde_json::Error) -> rmcp::ErrorData {
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
            Error::NotFound(msg) => rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INVALID_REQUEST,
                msg,
                Some(serde_json::json!({"type": "not_found"})),
            ),
            // Database errors keep the INTERNAL_ERROR code and the rendered
            // message (issue #255 changed only the message body: the underlying
            // SQLite text is now included instead of the bare "Database
            // error"). A locked database is reported as such so an MCP client
            // can retry rather than treat it as a server defect.
            Error::DatabaseLocked => rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                e.to_string(),
                Some(serde_json::json!({"type": "database_locked"})),
            ),
            Error::Database { .. }
            | Error::CorruptEmbedding { .. }
            | Error::InvalidBlobSize { .. }
            | Error::MismatchedDimensions { .. } => McpError::internal_error(&e.to_string()),
            Error::EmbedderUnavailable { reason } => rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                reason,
                Some(serde_json::json!({"type": "embedder_unavailable"})),
            ),
            _ => McpError::internal_error(&e.to_string()),
        }
    }
}
