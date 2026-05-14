//! Parameter structs for MCP tools.

use rmcp::schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the store_memory tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct StoreMemoryParams {
    /// The information to remember. Write as a clear, self-contained fact.
    #[schemars(
        description = "The information to remember. Write as a clear, self-contained fact."
    )]
    pub text: String,

    /// Optional structured labels like {"topic": "auth", "type": "decision"}.
    #[schemars(
        description = "Optional structured labels like {\"topic\": \"auth\", \"type\": \"decision\"}."
    )]
    pub metadata: Option<serde_json::Value>,

    /// Memory type: fact, preference, procedure, guard, observation (default: fact).
    #[serde(default)]
    #[schemars(
        description = "Memory type: fact, preference, procedure, guard, observation (default: fact)."
    )]
    pub memory_type: Option<String>,

    /// Memory status: active, candidate (default: active).
    #[serde(default)]
    #[schemars(description = "Memory status: active, candidate (default: active).")]
    pub status: Option<String>,

    /// ID of memory to supersede (creates new memory, marks old as superseded).
    #[serde(default)]
    #[schemars(
        description = "ID of memory to supersede (creates new memory, marks old as superseded)."
    )]
    pub supersedes: Option<String>,

    /// Force store even if conflicts detected (default: false).
    #[serde(default)]
    #[schemars(description = "Force store even if conflicts detected (default: false).")]
    pub force: Option<bool>,
}

/// Parameters for the supersede_memory tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SupersedeMemoryParams {
    /// The new text content that replaces the old memory.
    #[schemars(description = "The new text content that replaces the old memory.")]
    pub new_text: String,

    /// ID of the memory to supersede.
    #[schemars(description = "ID of the memory to supersede.")]
    pub old_memory_id: String,

    /// Memory type for the new memory (default: same as old, or fact).
    #[serde(default)]
    #[schemars(description = "Memory type for the new memory (default: same as old, or fact).")]
    pub memory_type: Option<String>,

    /// Optional metadata for the new memory.
    #[serde(default)]
    #[schemars(description = "Optional metadata for the new memory.")]
    pub metadata: Option<serde_json::Value>,
}

/// Parameters for the search_memories tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchMemoriesParams {
    /// What to look for in natural language.
    #[schemars(description = "What to look for in natural language.")]
    pub query: String,

    /// Maximum number of results to return.
    #[schemars(description = "Maximum number of results to return (default: 5).")]
    pub limit: Option<usize>,

    /// Filter by memory types.
    #[schemars(description = "Filter by memory types (e.g., fact, preference, procedure).")]
    pub memory_types: Option<Vec<String>>,

    /// Filter by statuses.
    #[schemars(description = "Filter by statuses (e.g., active, candidate).")]
    pub statuses: Option<Vec<String>>,

    /// Recency weight for search results (0.0 = pure semantic, 1.0 = pure recency).
    /// Defaults to config value (typically 0.3).
    #[serde(default)]
    #[schemars(description = "Recency weight (0.0-1.0). Default: config value (0.3).")]
    pub recency_weight: Option<f64>,

    /// Whether to use hybrid search (semantic + BM25). Default: config value.
    #[serde(default)]
    #[schemars(description = "Use hybrid search (semantic + BM25). Default: config value.")]
    pub hybrid: Option<bool>,

    /// Whether to skip updating retrieval telemetry (retrieval_count, last_retrieved_at).
    /// Default: false (telemetry is updated).
    #[serde(default)]
    #[schemars(description = "Skip updating retrieval telemetry (default: false).")]
    pub no_touch: Option<bool>,
}

/// Parameters for the list_memories tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListMemoriesParams {
    /// Maximum number of memories to list.
    #[schemars(description = "Maximum number of memories to list (default: 10).")]
    pub limit: Option<usize>,

    /// Filter by memory types.
    #[schemars(description = "Filter by memory types (e.g., fact, preference, procedure).")]
    pub memory_types: Option<Vec<String>>,

    /// Filter by statuses.
    #[schemars(description = "Filter by statuses (e.g., active, candidate).")]
    pub statuses: Option<Vec<String>>,

    /// Whether to skip updating retrieval telemetry. Default: false.
    #[serde(default)]
    #[schemars(description = "Skip updating retrieval telemetry (default: false).")]
    pub no_touch: Option<bool>,
}

/// Response when a memory is successfully added.
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct SuccessResponse {
    /// Memory ID
    pub id: String,
    /// Status
    pub status: String,
}

/// Parameters for the get_memory tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetMemoryParams {
    /// ID of the memory to retrieve.
    #[schemars(description = "ID of the memory to retrieve.")]
    pub id: String,
    /// Skip updating retrieval telemetry (default: false).
    #[serde(default)]
    #[schemars(description = "Skip updating retrieval telemetry (default: false).")]
    pub no_touch: Option<bool>,
}

/// Parameters for the delete_memory tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DeleteMemoryParams {
    /// ID of the memory to delete.
    #[schemars(description = "ID of the memory to delete.")]
    pub id: String,
}

/// Parameters for the update_memory tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UpdateMemoryParams {
    /// ID of the memory to update.
    #[schemars(description = "ID of the memory to update.")]
    pub id: String,
    /// New text content for the memory.
    #[serde(default)]
    #[schemars(description = "New text content for the memory.")]
    pub text: Option<String>,
    /// New metadata as JSON object.
    #[serde(default)]
    #[schemars(description = "New metadata as JSON object.")]
    pub metadata: Option<serde_json::Value>,
    /// New memory type: fact, preference, procedure, guard, observation.
    #[serde(default)]
    #[schemars(description = "New memory type: fact, preference, procedure, guard, observation.")]
    pub memory_type: Option<String>,
    /// New memory status: active, candidate.
    #[serde(default)]
    #[schemars(description = "New memory status: active, candidate.")]
    pub status: Option<String>,
}

/// A conflicting memory.
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ConflictMemory {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Similarity score
    pub similarity: f64,
}
