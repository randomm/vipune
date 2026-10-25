//! StoreWrapper for MCP tool implementations.

use crate::errors::Error;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory_types::ConflictMemory as InternalConflictMemory;
use crate::memory_types::IngestPolicy;
use std::sync::{Arc, Mutex};

use crate::memory::{MemoryStore, NewMemory, NewSupersedingMemory, UpdateParams};

/// The caller-supplied columns of a row an MCP write tool is about to create:
/// the row itself plus, for the supersede arm, the id it replaces.
///
/// Grouping them keeps the two write wrappers under clippy's argument ceiling
/// without an `allow` — the same reason `NewMemory` exists one layer down
/// (`src/memory/add.rs`) — and keeps every level, type and status a named field
/// at the call site rather than a positional tail an extra argument can slip
/// past (#238, #256, #258).
pub(crate) struct McpWrite<'a> {
    /// Project the row belongs to.
    pub project_id: &'a str,
    /// Text content to store.
    pub text: &'a str,
    /// Serialised metadata, or the literal `"null"` for none.
    pub metadata: &'a str,
    /// Memory type of the new row.
    pub memory_type: MemoryType,
    /// Lifecycle status of the new row.
    pub status: MemoryStatus,
    /// Importance of the new row — what decides the `high`/`critical` prune
    /// exemption.
    pub importance: MemoryImportance,
    /// Set on the supersede arm: the id the new row replaces.
    pub supersedes: Option<&'a str>,
    /// Set on the ingest arm: bypass the conflict gate.
    pub force: bool,
}

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

    /// Store a memory with explicit type, status and importance — the one
    /// ingest path `store_memory` drives. The level is a parsed
    /// [`MemoryImportance`], so the column gets the canonical `as_str` spelling
    /// rather than the argument string, and `medium` is bound by name (the
    /// store's own default) rather than by the INSERT omitting the column.
    pub(crate) fn ingest_with_type_status(
        &self,
        row: &McpWrite<'_>,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();
        let policy = if row.force {
            IngestPolicy::Force
        } else {
            IngestPolicy::ConflictAware
        };

        match store
            .ingest_with_importance(
                &NewMemory {
                    project_id: row.project_id,
                    content: row.text,
                    metadata: Some(row.metadata),
                    memory_type: row.memory_type,
                    status: row.status,
                },
                policy,
                row.importance,
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

    /// Supersede an existing memory with new content, at the `active` status and
    /// default `medium` importance — what the `supersede_memory` tool drives (it
    /// has neither a `status` nor an `importance` parameter). The level is bound
    /// by name here rather than left to the INSERT's column DEFAULT (#238).
    pub(crate) fn supersede(
        &self,
        project_id: &str,
        new_text: &str,
        metadata: &str,
        memory_type: MemoryType,
        old_memory_id: &str,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        self.supersede_with_status(&McpWrite {
            project_id,
            text: new_text,
            metadata,
            memory_type,
            status: MemoryStatus::Active,
            importance: MemoryImportance::default(),
            supersedes: Some(old_memory_id),
            force: false,
        })
    }

    /// Supersede an existing memory with new content and an explicit lifecycle
    /// status and importance on the replacement row — what `store_memory`'s
    /// `supersedes` path drives. `supersede_memory` reaches the same store call
    /// through [`Self::supersede`], which passes `active`/`medium`; the two tools
    /// differ only in their schemas, not in a second write path.
    ///
    /// The `supersedes` arm of `store_memory` used to call [`Self::supersede`]
    /// with the status it had just parsed and validated still in a local, so a
    /// caller passing `status: "candidate"` together with `supersedes` got an
    /// `active` row and no warning (issue #258). The status is threaded to the
    /// store here so the MCP surface matches the CLI, and from #256 the level
    /// travels the same frame for the same reason: `vipune add --importance`
    /// sets it on the row `--supersedes` creates.
    ///
    /// This frame adds nothing to the insert-status rule: it takes
    /// `MemoryStatus`, the same type the store takes, so no string can become an
    /// unvalidated status one frame lower, and
    /// [`MemoryStore::supersede_with`] is the one place that rejects
    /// `superseded`/`deprecated` — the error arrives back as `invalid_input`
    /// through `From<Error> for rmcp::ErrorData`.
    pub(crate) fn supersede_with_status(
        &self,
        row: &McpWrite<'_>,
    ) -> Result<serde_json::Value, rmcp::ErrorData> {
        let mut store = self.0.lock().unwrap();

        let metadata_str = if row.metadata == "null" {
            None
        } else {
            Some(row.metadata)
        };

        // The supersede arm is the only caller, so the replaced id is always
        // there; a missing one is a wiring bug, not user input, and is answered
        // as an internal error rather than by replacing nothing.
        let old_memory_id = match row.supersedes {
            Some(id) => id,
            None => return Err(McpError::internal_error("supersede requires an id")),
        };

        let new_id = match store.supersede_with(&NewSupersedingMemory {
            row: NewMemory {
                project_id: row.project_id,
                content: row.text,
                metadata: metadata_str,
                memory_type: row.memory_type,
                status: row.status,
            },
            old_id: old_memory_id,
            // The level is the caller's, never the column DEFAULT (#238, #256).
            importance: row.importance,
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
            Error::EmbedderUnavailable { reason } => rmcp::ErrorData::new(
                rmcp::model::ErrorCode::INTERNAL_ERROR,
                reason,
                Some(serde_json::json!({"type": "embedder_unavailable"})),
            ),
            _ => McpError::internal_error(&e.to_string()),
        }
    }
}
