//! Helper functions for MCP tool implementations.

use crate::mcp::store_wrapper::McpError;
use crate::sqlite::Memory;

/// Serialises a tool's `metadata` argument for the storage column.
///
/// One rule for all three write tools: an absent argument and a JSON `null`
/// both become `None` — "no metadata given". What the callers do with `None`
/// differs: `store_memory`/`supersede_memory` store SQL NULL, while
/// `update_memory` leaves the stored column unchanged. Every other value is
/// passed on as its own JSON text, still subject to the store's validation —
/// so the *string* `"null"` keeps its quotes and never collapses into the NULL
/// spelling (`serde_json::to_string` is what adds them).
///
/// rmcp/serde already deserialises an explicit JSON `null` into `None` for the
/// `Option<Value>` tool field, so the `Value::Null` arm is belt-and-braces.
pub(super) fn metadata_column(
    metadata: &Option<serde_json::Value>,
) -> Result<Option<String>, rmcp::ErrorData> {
    match metadata {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => serde_json::to_string(value)
            .map(Some)
            .map_err(|e| McpError::invalid_input(&format!("Invalid metadata: {e}"))),
    }
}

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

#[cfg(test)]
mod tests {
    use super::metadata_column;
    use serde_json::json;

    /// The single write-side rule (#309): absent and JSON `null` are the only
    /// spellings of "no metadata". Everything else is stored as its own JSON
    /// text, which is why a match on the *serialised* string would be wrong —
    /// the string `"null"` must keep its quotes.
    #[test]
    fn metadata_column_maps_only_absent_and_json_null_to_none() {
        let cases = [
            ("absent", None, None),
            ("json null", Some(json!(null)), None),
            ("the string null", Some(json!("null")), Some("\"null\"")),
            ("a number", Some(json!(0)), Some("0")),
            ("a boolean", Some(json!(false)), Some("false")),
            ("an empty string", Some(json!("")), Some(r#""""#)),
            (
                "a document",
                Some(json!({ "topic": "auth" })),
                Some(r#"{"topic":"auth"}"#),
            ),
        ];

        for (label, sent, expected) in cases {
            let got = metadata_column(&sent)
                .unwrap_or_else(|err| panic!("{label}: must serialise, got {err:?}"));
            assert_eq!(got.as_deref(), expected, "{label}");
        }
    }
}
