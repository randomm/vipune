//! Data migration from remory SQLite databases or JSON exports.

use crate::config::Config;
use crate::errors::Error;
use crate::memory::MemoryStore;
use crate::sqlite::{blob_to_vec, vec_to_blob, Database};
use chrono::Utc;
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;
use uuid::Uuid;

/// Import format for JSON-based imports.
#[derive(Debug, Deserialize)]
pub struct JsonMemory {
    pub content: String,
    pub project_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

/// Import statistics for reporting.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ImportStats {
    pub total_memories: usize,
    pub imported_memories: usize,
    pub skipped_duplicates: usize,
    pub skipped_corrupted: usize,
    pub projects: HashSet<String>,
}

impl ImportStats {
    fn new() -> Self {
        Self::default()
    }
}

/// Import memories from a remory SQLite database.
///
/// Opens the source database read-only, migrates memories to vipune schema.
///
/// # Arguments
///
/// * `db_path` - Path to remory SQLite database
/// * `dry_run` - If true, only report what would be imported
/// * `vipune_db` - Path to vipune database
/// * `model_id` - HuggingFace model ID for re-embedding if needed
/// * `config` - Configuration for similarity threshold
///
/// # Returns
///
/// Import statistics showing how many memories were imported, skipped, or failed.
pub fn import_from_sqlite(
    db_path: &Path,
    dry_run: bool,
    vipune_db: &Path,
    model_id: &str,
    config: Config,
) -> Result<ImportStats, Error> {
    if !db_path.exists() {
        return Err(Error::FileNotFound(db_path.to_path_buf()));
    }

    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY;
    let src_conn = Connection::open_with_flags(db_path, flags)?;

    let mut stats = ImportStats::new();

    let mut stmt = src_conn.prepare(
        r#"
        SELECT id, content, project_id, user_id, metadata, embedding, created_at, updated_at
        FROM memories
        "#,
    )?;

    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<Vec<u8>>>(5)?,
            row.get::<_, Option<String>>(6)?,
            row.get::<_, Option<String>>(7)?,
        ))
    })?;

    // Create a single store instance for duplicate checking and import
    // (used in both dry-run and normal modes)
    let mut store = MemoryStore::new(vipune_db, model_id, config.clone())?;

    for row_result in rows {
        let (id, content, project_id, user_id, metadata, embedding, created_at, updated_at) =
            row_result.map_err(Error::SQLite)?;

        stats.total_memories += 1;

        let project_id: String = project_id
            .or(user_id)
            .unwrap_or_else(|| "default".to_string());
        stats.projects.insert(project_id.clone());

        // Check for duplicates in all modes (before actual import)
        if is_duplicate(&mut store, &project_id, &content)? {
            stats.skipped_duplicates += 1;
            continue;
        }

        if !dry_run {
            // Provide fallback timestamps for nullable columns
            let now = Utc::now().to_rfc3339();
            let created = created_at.as_ref().unwrap_or(&now).clone();
            let updated = updated_at.as_ref().unwrap_or(&now).clone();

            let embedding_vec = match embedding {
                Some(blob) => match blob_to_vec(&blob) {
                    Ok(vec) => vec,
                    Err(e) => {
                        eprintln!(
                            "Warning: corrupted embedding for memory '{}': {} - re-embedding",
                            id.as_ref().unwrap_or(&"unknown".to_string()),
                            e
                        );
                        store.embedder.embed(&content)?
                    }
                },
                None => {
                    eprintln!(
                        "Warning: missing embedding for memory '{}', re-embedding",
                        id.as_ref().unwrap_or(&"unknown".to_string())
                    );
                    store.embedder.embed(&content)?
                }
            };

            if let Err(e) = insert_with_params(
                &store.db,
                &project_id,
                &content,
                &embedding_vec,
                metadata.as_deref(),
                &created,
                &updated,
            ) {
                eprintln!(
                    "Warning: failed to import memory '{}': {}",
                    id.as_ref().unwrap_or(&"unknown".to_string()),
                    e
                );
                stats.skipped_corrupted += 1;
                continue;
            }

            stats.imported_memories += 1;
        } else {
            // In dry-run mode, count what would have been imported (passed duplicate check)
            stats.imported_memories += 1;
        }
    }

    Ok(stats)
}

/// Import memories from a JSON export file.
///
/// Parses JSON array and generates embeddings for each memory.
///
/// # Arguments
///
/// * `json_path` - Path to JSON file
/// * `vipune_db` - Path to vipune database
/// * `model_id` - HuggingFace model ID for embedding generation
/// * `config` - Configuration for similarity threshold
///
/// # Returns
///
/// Import statistics.
pub fn import_from_json(
    json_path: &Path,
    vipune_db: &Path,
    model_id: &str,
    config: Config,
) -> Result<ImportStats, Error> {
    if !json_path.exists() {
        return Err(Error::FileNotFound(json_path.to_path_buf()));
    }

    let content = std::fs::read_to_string(json_path)?;
    let memories: Vec<JsonMemory> = serde_json::from_str(&content)?;

    let mut stats = ImportStats::new();
    let mut store = MemoryStore::new(vipune_db, model_id, config)?;

    stats.total_memories = memories.len();

    for memory in memories {
        let project_id = memory.project_id.unwrap_or_else(|| "default".to_string());
        stats.projects.insert(project_id.clone());

        if is_duplicate(&mut store, &project_id, &memory.content)? {
            stats.skipped_duplicates += 1;
            continue;
        }

        let embedding = store.embedder.embed(&memory.content)?;

        let now = Utc::now().to_rfc3339();
        let created = memory.created_at.unwrap_or_else(|| now.clone());
        let updated = memory.updated_at.unwrap_or(now);

        let metadata_json = memory
            .metadata
            .map(|v| serde_json::to_string(&v))
            .transpose()?;

        if let Err(e) = insert_with_params(
            &store.db,
            &project_id,
            &memory.content,
            &embedding,
            metadata_json.as_deref(),
            &created,
            &updated,
        ) {
            eprintln!(
                "Warning: failed to import memory '{}': {}",
                memory.content, e
            );
            stats.skipped_corrupted += 1;
            continue;
        }

        stats.imported_memories += 1;
    }

    Ok(stats)
}

/// Check if a memory would be a duplicate (similar content exists).
///
/// Requires `&mut MemoryStore` because embedding generation (`embed()`) mutates
/// internal ONNX tensor allocations. This is a design constraint of the embedding engine.
fn is_duplicate(store: &mut MemoryStore, project_id: &str, content: &str) -> Result<bool, Error> {
    let embedding = store.embedder.embed(content)?;
    let similars =
        store
            .db
            .find_similar(project_id, &embedding, store.config.similarity_threshold)?;
    Ok(!similars.is_empty())
}

fn insert_with_params(
    db: &Database,
    project_id: &str,
    content: &str,
    embedding: &[f32],
    metadata: Option<&str>,
    created_at: &str,
    updated_at: &str,
) -> Result<String, Error> {
    let id = Uuid::new_v4().to_string();
    let blob = vec_to_blob(embedding)?;

    db.conn().execute(
        r#"
        INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#,
        rusqlite::params![&id, project_id, content, &blob, metadata, created_at, updated_at],
    )?;

    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_import_stats_default() {
        let stats = ImportStats::new();
        assert_eq!(stats.total_memories, 0);
        assert_eq!(stats.imported_memories, 0);
        assert_eq!(stats.skipped_duplicates, 0);
        assert_eq!(stats.skipped_corrupted, 0);
        assert!(stats.projects.is_empty());
    }

    #[test]
    fn test_json_memory_deserialize() {
        let json = r#"{
            "content": "test memory",
            "project_id": "test-project",
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }"#;
        let memory: JsonMemory = serde_json::from_str(json).unwrap();
        assert_eq!(memory.content, "test memory");
        assert_eq!(memory.project_id, Some("test-project".to_string()));
    }

    #[test]
    fn test_import_from_nonexistent_file() {
        let result = import_from_sqlite(
            Path::new("/nonexistent/db.sqlite"),
            false,
            Path::new("/tmp/test.db"),
            "BAAI/bge-small-en-v1.5",
            Config::default(),
        );
        assert!(matches!(result, Err(Error::FileNotFound(_))));
    }

    #[test]
    fn test_create_mock_remory_db_and_import() {
        let dir = TempDir::new().unwrap();
        let remory_db = dir.path().join("remory.db");
        let vipune_db = dir.path().join("vipune.db");

        let conn = Connection::open(&remory_db).unwrap();
        conn.execute(
            r#"
              CREATE TABLE memories (
                  id TEXT PRIMARY KEY,
                  content TEXT NOT NULL,
                  project_id TEXT,
                  user_id TEXT,
                  metadata TEXT,
                  embedding BLOB,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
              )
              "#,
            [],
        )
        .unwrap();

        let embedding_blob = vec_to_blob(&vec![0.5f32; 384]).unwrap();
        conn.execute(
              r#"
              INSERT INTO memories (id, content, project_id, user_id, metadata, embedding, created_at, updated_at)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
              "#,
              rusqlite::params![
                  "test-id-1",
                  "test content",
                  "test-project",
                  None::<String>,
                  r#"{"key":"value"}"#,
                  embedding_blob,
                  "2024-01-01T00:00:00Z",
                  "2024-01-01T00:00:00Z"
              ],
          )
          .unwrap();

        let stats = import_from_sqlite(
            &remory_db,
            true,
            &vipune_db,
            "BAAI/bge-small-en-v1.5",
            Config::default(),
        )
        .unwrap();

        assert_eq!(stats.total_memories, 1);
        assert_eq!(stats.imported_memories, 1);
        assert_eq!(stats.skipped_duplicates, 0);
        assert_eq!(stats.skipped_corrupted, 0);
    }
}
