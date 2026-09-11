//! JSON response types and formatting for CLI output.

use serde::Serialize;

/// Response for successful memory addition.
#[derive(Serialize)]
pub struct AddResponse {
    /// Operation status ("added").
    pub status: String,
    /// Unique identifier of the added memory.
    pub id: String,
}

/// Response for search results.
#[derive(Serialize)]
pub struct SearchResponse {
    /// List of search results ranked by relevance.
    pub results: Vec<SearchResultItem>,
}

/// Individual search result item.
#[derive(Serialize)]
pub struct SearchResultItem {
    /// Unique identifier of the memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Relevance score (0.0 to 1.0, higher is better).
    pub similarity: f64,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Number of times this memory was retrieved via search or get.
    pub retrieval_count: i64,
    /// RFC3339 timestamp of last retrieval (null if never retrieved).
    pub last_retrieved_at: Option<String>,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    pub status: String,
}

/// Response for retrieving a specific memory.
#[derive(Serialize)]
pub struct GetResponse {
    /// Unique identifier of the memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Project identifier for this memory.
    pub project_id: String,
    /// Optional user-provided metadata (JSON string).
    pub metadata: Option<String>,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Last update timestamp in RFC3339 format.
    pub updated_at: String,
    /// Number of times this memory was retrieved via search or get.
    pub retrieval_count: i64,
    /// RFC3339 timestamp of last retrieval (null if never retrieved).
    pub last_retrieved_at: Option<String>,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    pub status: String,
}

/// Response for listing memories.
#[derive(Serialize)]
pub struct ListResponse {
    /// List of memories ordered by creation time (newest first).
    pub memories: Vec<ListItem>,
}

/// Individual list item.
#[derive(Serialize)]
pub struct ListItem {
    /// Unique identifier of the memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Number of times this memory was retrieved via search or get.
    pub retrieval_count: i64,
    /// RFC3339 timestamp of last retrieval (null if never retrieved).
    pub last_retrieved_at: Option<String>,
    /// Memory type (fact, preference, procedure, guard, observation).
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    pub status: String,
}

/// Response for successful memory deletion.
#[derive(Serialize)]
pub struct DeleteResponse {
    /// Operation status ("deleted").
    pub status: String,
    /// Unique identifier of the deleted memory.
    pub id: String,
}

/// Response for successful memory update.
#[derive(Serialize)]
pub struct UpdateResponse {
    /// Operation status ("updated").
    pub status: String,
    /// Unique identifier of the updated memory.
    pub id: String,
}

/// Response for error cases.
#[derive(Serialize)]
pub struct ErrorResponse {
    /// Error message describing what went wrong.
    pub error: String,
}

/// Response for conflict detection.
#[derive(Serialize)]
pub struct ConflictsResponse {
    /// Operation status ("conflicts").
    pub status: String,
    /// The proposed memory content.
    pub proposed: String,
    /// List of conflicting memories.
    pub conflicts: Vec<ConflictMemoryResponse>,
}

/// Individual conflicting memory in response.
#[derive(Serialize)]
pub struct ConflictMemoryResponse {
    /// Unique identifier of the conflicting memory.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Similarity score indicating the degree of conflict (0.0 to 1.0).
    pub similarity: f64,
}

/// Response for token validation.
#[derive(Serialize)]
pub struct ValidateResponse {
    /// Token count of the validated text.
    pub token_count: usize,
    /// Maximum allowed tokens for embedding.
    pub max_tokens: usize,
    /// Whether the text is within the embedding limit.
    pub within_limit: bool,
}

/// Response for `doctor --embeddings` per-project report.
#[derive(Serialize)]
pub struct DoctorResponse {
    /// Project identifier.
    pub project_id: String,
    /// Total number of rows in the project.
    pub total_rows: usize,
    /// Number of rows with real (L2-normalised) embeddings.
    pub real_rows: usize,
    /// Number of rows with mock embeddings.
    pub mock_rows: usize,
    /// Number of rows with unknown/corrupted embeddings.
    pub unknown_rows: usize,
}

/// Response for `reindex` per-project report.
#[derive(Serialize)]
pub struct ReindexResponse {
    /// Project identifier.
    pub project_id: String,
    /// Number of rows successfully re-embedded.
    pub reindexed: usize,
    /// Number of rows skipped (unknown/corrupted embeddings).
    pub skipped: usize,
    /// Rows that failed during re-embedding.
    pub failed: Vec<ReindexFailure>,
}

/// A single failed row during reindex.
#[derive(Serialize, Clone)]
pub struct ReindexFailure {
    /// Memory ID that failed.
    pub id: String,
    /// Error message describing the failure.
    pub error: String,
}

/// A suspected split pair detected by `doctor --projects`.
///
/// `pair[0]` is the segment after the first `/` in `pair[1]`. For the common case
/// (owned id is `owner/repo`), `pair[0]` is the bare id `repo`. For multi-slash ids
/// like `group/subgroup/project`, `pair[0]` is `subgroup/project`.
/// `row_counts[0]` corresponds to `pair[0]`, `row_counts[1]` to `pair[1]`.
/// Ordering is stable so `--json` consumers can attribute counts unambiguously.
#[derive(Serialize)]
pub struct DoctorProjectsSuspectedSplit {
    /// Pair of project ids: `[segment, owned_id]`.
    pub pair: [String; 2],
    /// Row counts for each side: `[count_segment, count_owned]`.
    pub row_counts: [usize; 2],
}

/// Response for `doctor --projects` scan.
#[derive(Serialize)]
pub struct DoctorProjectsResponse {
    /// List of suspected split pairs. Empty when no splits are found.
    pub suspected_splits: Vec<DoctorProjectsSuspectedSplit>,
}

/// A single under-populated project reported by `doctor --fts`.
///
/// `memory_rows` is the total number of rows in that project; `missing_from_fts` is
/// the number of that project's rowids absent from `memories_fts`.
#[derive(Serialize)]
pub struct DoctorFtsProject {
    /// Project identifier.
    pub project_id: String,
    /// Total number of rows in this project.
    pub memory_rows: usize,
    /// Rows in this project whose rowid is missing from `memories_fts`.
    pub missing_from_fts: usize,
}

/// Response for `doctor --fts` FTS desync check.
///
/// `in_sync` is true when both the under-population and orphan directions report
/// zero. `underpopulated_by_project` lists each project with missing FTS rows (empty
/// when in sync). `orphan_rows` is a GLOBAL count of `memories_fts` rowids with no
/// matching `memories` row — never attributed to a project, because an orphan row's
/// content in an external-content table is undefined. When `--repair` is passed,
/// `repaired` is true if a rebuild ran (false if the pre-check found zero desync and
/// the rebuild was skipped); `actions` counts the rows a rebuild re-indexed.
#[derive(Serialize)]
pub struct DoctorFtsResponse {
    /// True when the FTS index is in sync with the memories table.
    pub in_sync: bool,
    /// Per-project under-population reports (empty when in sync).
    pub underpopulated_by_project: Vec<DoctorFtsProject>,
    /// Global count of orphan FTS rows (no matching memories row).
    pub orphan_rows: usize,
    /// Total number of desynced rows (under-population + orphans).
    pub total_desynced: usize,
    /// Whether `--repair` ran a rebuild (false when the pre-check found zero desync).
    pub repaired: bool,
    /// Number of FTS rows the rebuild re-indexed (0 when skipped).
    pub actions: usize,
}

/// Response for `backup` operation.
#[derive(Serialize)]
pub struct BackupResponse {
    /// Path of the source database that was backed up.
    pub source: String,
    /// Path of the produced backup file.
    pub destination: String,
    /// Number of rows in the produced backup (matches source by construction).
    pub rows: usize,
    /// Size of the produced backup file in bytes.
    pub bytes: u64,
}

/// Response for `project merge` operation.
#[derive(Serialize)]
pub struct MergeResponse {
    /// Source project id (rows moved from this).
    pub from: String,
    /// Target project id (rows moved to this).
    pub to: String,
    /// Number of rows that were moved.
    pub rows_moved: usize,
}

/// Response for `export` operation (see `ExportRowLine` for the 12-field JSONL row format).
#[derive(Serialize)]
#[allow(dead_code)] // Consumed by task-a (export handler) in the consolidated tree
pub struct ExportResponse {
    /// Number of data rows exported (excluding the header line).
    pub rows: usize,
    /// Path the JSONL file was written to.
    pub path: String,
}

/// One 12-field JSONL export row. DB `type` is renamed to `memory_type`; `embedding` is
/// base64 of the stored BLOB (`""` for empty); `superseded_by`/`last_retrieved_at` nullable.
#[derive(Serialize, Clone, Debug)]
#[allow(dead_code)] // Consumed by task-a (export handler) in the consolidated tree
pub struct ExportRowLine {
    /// Memory id.
    pub id: String,
    /// Project identifier that owns this memory.
    pub project_id: String,
    /// Memory content.
    pub content: String,
    /// Optional user-provided metadata (JSON string), null if absent.
    pub metadata: Option<String>,
    /// Base64 of the exact 384xf32-LE embedding BLOB, or `""` for NULL/empty.
    pub embedding: String,
    /// Creation timestamp in RFC3339 format.
    pub created_at: String,
    /// Last update timestamp in RFC3339 format.
    pub updated_at: String,
    /// Memory type (fact, preference, procedure, guard, observation). Renamed from DB `type`.
    pub memory_type: String,
    /// Lifecycle status (active, candidate, superseded, deprecated).
    pub status: String,
    /// ID of the memory that superseded this one (null if none).
    pub superseded_by: Option<String>,
    /// Number of times this memory was retrieved.
    pub retrieval_count: i64,
    /// RFC3339 timestamp of last retrieval (null if never retrieved).
    pub last_retrieved_at: Option<String>,
}

/// Response for `import` operation (all-or-nothing; `skipped` = existing-id rows, not errors).
#[derive(Serialize)]
#[allow(dead_code)] // Consumed by task-b (import handler) in the consolidated tree
pub struct ImportResponse {
    /// Number of rows newly inserted into the destination DB.
    pub inserted: usize,
    /// Number of rows skipped because their id already existed in the destination.
    pub skipped: usize,
    /// Total data rows read from the JSONL (inserted + skipped). Excludes the header line.
    pub rows_total: usize,
}

/// Serialize a value as formatted JSON and print to stdout.
///
/// Exits with status 1 if serialization fails.
pub fn print_json<T: Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("Failed to serialize JSON: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_add_response() {
        let response = AddResponse {
            status: "added".to_string(),
            id: "test-id".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"added\""));
        assert!(json.contains("\"id\":\"test-id\""));
    }

    #[test]
    fn test_serialize_search_response() {
        let response = SearchResponse {
            results: vec![SearchResultItem {
                id: "test-id".to_string(),
                content: "test content".to_string(),
                similarity: 0.95,
                created_at: "2024-01-01T00:00:00Z".to_string(),
                retrieval_count: 3,
                last_retrieved_at: Some("2024-01-02T00:00:00Z".to_string()),
                memory_type: "guard".to_string(),
                status: "active".to_string(),
            }],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"results\""));
        assert!(json.contains("\"similarity\":0.95"));
        assert!(json.contains("\"retrieval_count\":3"));
        assert!(json.contains("\"last_retrieved_at\":\"2024-01-02T00:00:00Z\""));
        // #178: type/status must be observable in JSON, not just filterable.
        assert!(json.contains("\"memory_type\":\"guard\""));
        assert!(json.contains("\"status\":\"active\""));
    }

    #[test]
    fn test_serialize_search_result_null_last_retrieved() {
        let response = SearchResponse {
            results: vec![SearchResultItem {
                id: "test-id".to_string(),
                content: "test content".to_string(),
                similarity: 0.95,
                created_at: "2024-01-01T00:00:00Z".to_string(),
                retrieval_count: 0,
                last_retrieved_at: None,
                memory_type: "fact".to_string(),
                status: "active".to_string(),
            }],
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"retrieval_count\":0"));
        assert!(json.contains("\"last_retrieved_at\":null"));
    }

    #[test]
    fn test_serialize_list_item_with_retrieval_fields() {
        let item = ListItem {
            id: "test-id".to_string(),
            content: "test content".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            retrieval_count: 5,
            last_retrieved_at: Some("2024-01-02T12:00:00Z".to_string()),
            memory_type: "fact".to_string(),
            status: "active".to_string(),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"retrieval_count\":5"));
        assert!(json.contains("\"last_retrieved_at\":\"2024-01-02T12:00:00Z\""));

        let item_none = ListItem {
            id: "test-id".to_string(),
            content: "test content".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            retrieval_count: 0,
            last_retrieved_at: None,
            memory_type: "fact".to_string(),
            status: "active".to_string(),
        };
        let json_none = serde_json::to_string(&item_none).unwrap();
        assert!(json_none.contains("\"retrieval_count\":0"));
        assert!(json_none.contains("\"last_retrieved_at\":null"));
    }

    #[test]
    fn test_serialize_backup_response() {
        let response = BackupResponse {
            source: "/tmp/memories.db".to_string(),
            destination: "/tmp/memories-backup.db".to_string(),
            rows: 42,
            bytes: 2048,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"source\":\"/tmp/memories.db\""));
        assert!(json.contains("\"destination\":\"/tmp/memories-backup.db\""));
        assert!(json.contains("\"rows\":42"));
        assert!(json.contains("\"bytes\":2048"));
    }

    #[test]
    fn test_serialize_export_row_line_pins_memory_type_not_type() {
        // Gate resolution: DB `type` column is renamed to `memory_type` in the JSONL
        // row; `type` must NOT appear as a key. Nullable fields serialize as null.
        let row = ExportRowLine {
            id: "id-1".to_string(),
            project_id: "proj".to_string(),
            content: "c".to_string(),
            metadata: None,
            embedding: "AA==".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            memory_type: "guard".to_string(),
            status: "active".to_string(),
            superseded_by: None,
            retrieval_count: 3,
            last_retrieved_at: Some("2024-01-02T00:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&row).unwrap();
        assert!(json.contains("\"memory_type\":\"guard\""));
        assert!(json.contains("\"id\":\"id-1\""));
        assert!(json.contains("\"project_id\":\"proj\""));
        assert!(json.contains("\"embedding\":\"AA==\""));
        assert!(json.contains("\"retrieval_count\":3"));
        assert!(json.contains("\"superseded_by\":null"));
        assert!(json.contains("\"last_retrieved_at\":\"2024-01-02T00:00:00Z\""));
        assert!(
            !json.contains("\"type\":"),
            "must not have bare \"type\" key, got: {}",
            json
        );
        // With superseded_by set, it serializes to the string value.
        let row2 = ExportRowLine {
            superseded_by: Some("x".to_string()),
            ..row
        };
        let j2 = serde_json::to_string(&row2).unwrap();
        assert!(j2.contains("\"superseded_by\":\"x\""));
    }

    #[test]
    fn test_serialize_export_import_backup_responses() {
        let e = ExportResponse {
            rows: 42,
            path: "p.jsonl".to_string(),
        };
        let ej = serde_json::to_string(&e).unwrap();
        assert!(ej.contains("\"rows\":42"));
        assert!(ej.contains("\"path\":\"p.jsonl\""));

        let i = ImportResponse {
            inserted: 10,
            skipped: 5,
            rows_total: 15,
        };
        let ij = serde_json::to_string(&i).unwrap();
        assert!(ij.contains("\"inserted\":10"));
        assert!(ij.contains("\"skipped\":5"));
        assert!(ij.contains("\"rows_total\":15"));

        let b = BackupResponse {
            source: "b.db".to_string(),
            destination: "b-backup.db".to_string(),
            rows: 0,
            bytes: 2048,
        };
        let bj = serde_json::to_string(&b).unwrap();
        assert!(bj.contains("\"bytes\":2048"));
        assert!(bj.contains("\"source\":\"b.db\""));
    }

    #[test]
    fn test_serialize_get_and_list_items_carry_type_and_status() {
        let get = GetResponse {
            id: "test-id".to_string(),
            content: "test content".to_string(),
            project_id: "proj".to_string(),
            metadata: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            retrieval_count: 2,
            last_retrieved_at: Some("2024-01-02T00:00:00Z".to_string()),
            memory_type: "procedure".to_string(),
            status: "candidate".to_string(),
        };
        let get_json = serde_json::to_string(&get).unwrap();
        assert!(get_json.contains("\"retrieval_count\":2"));
        assert!(get_json.contains("\"memory_type\":\"procedure\""));
        assert!(get_json.contains("\"status\":\"candidate\""));

        let list = ListResponse {
            memories: vec![ListItem {
                id: "test-id".to_string(),
                content: "test content".to_string(),
                created_at: "2024-01-01T00:00:00Z".to_string(),
                retrieval_count: 0,
                last_retrieved_at: None,
                memory_type: "observation".to_string(),
                status: "deprecated".to_string(),
            }],
        };
        let list_json = serde_json::to_string(&list).unwrap();
        assert!(list_json.contains("\"memory_type\":\"observation\""));
        assert!(list_json.contains("\"status\":\"deprecated\""));
    }
}
