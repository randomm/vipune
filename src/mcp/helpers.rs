//! Helper functions for MCP tool implementations.

use crate::sqlite::Memory;

/// Formats a memory (with similarity score) as JSON for MCP responses.
pub fn format_memory_json(memory: &Memory) -> serde_json::Value {
    let metadata_value = parse_metadata(&memory.metadata);
    let similarity = memory.similarity.unwrap_or(0.0);

    serde_json::json!({
        "id": memory.id,
        "content": memory.content,
        "similarity": similarity,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "project_id": memory.project_id,
        "memory_type": memory.memory_type,
        "status": memory.status,
        "metadata": metadata_value,
        "retrieval_count": memory.retrieval_count,
        "last_retrieved_at": memory.last_retrieved_at
    })
}

/// Parses metadata string into JSON value.
fn parse_metadata(metadata: &Option<String>) -> serde_json::Value {
    match metadata {
        Some(meta_str) if meta_str.trim() != "null" => {
            serde_json::from_str::<serde_json::Value>(meta_str)
                .unwrap_or_else(|_| serde_json::Value::String(meta_str.clone()))
        }
        _ => serde_json::Value::Null,
    }
}
