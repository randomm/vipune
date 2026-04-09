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
}

/// Parameters for the list_memories tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListMemoriesParams {
    /// Maximum number of memories to list.
    #[schemars(description = "Maximum number of memories to list (default: 10).")]
    pub limit: Option<usize>,
}

/// Response when a memory is successfully added.
#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    /// Memory ID
    pub id: String,
    /// Status
    pub status: String,
}

/// A conflicting memory.
#[derive(Debug, Serialize)]
pub struct ConflictMemory {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Similarity score
    pub similarity: f64,
}
