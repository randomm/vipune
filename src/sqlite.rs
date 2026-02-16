// Dead code suppression justified: All items are used in tests
// but dead_code linter cannot cross cfg(test) boundaries
#![allow(dead_code)]

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Result as SqliteResult};
use std::path::Path;
use uuid::Uuid;

pub struct Database {
    pub conn: Connection,
}

pub struct Memory {
    pub id: String,
    pub project_id: String,
    pub content: String,
    pub metadata: Option<String>,
    pub similarity: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug)]
pub enum Error {
    Sqlite(String),
    InvalidBlobSize { expected: usize, actual: usize },
    MismatchedDimensions { expected: usize, actual: usize },
    EmptyVector,
    InvalidEmbedding(String),
    InvalidLimit(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Sqlite(msg) => write!(f, "Database error: {}", msg),
            Error::InvalidBlobSize { expected, actual } => {
                write!(
                    f,
                    "Invalid BLOB size: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            Error::MismatchedDimensions { expected, actual } => {
                write!(
                    f,
                    "Mismatched dimensions: expected {} dimensions, got {} dimensions",
                    expected, actual
                )
            }
            Error::EmptyVector => write!(f, "Cannot compute similarity with empty vector"),
            Error::InvalidEmbedding(msg) => write!(f, "Invalid embedding: {}", msg),
            Error::InvalidLimit(msg) => write!(f, "Invalid limit: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<rusqlite::Error> for Error {
    fn from(err: rusqlite::Error) -> Self {
        Error::Sqlite(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

const EMBEDDING_DIMS: usize = 384;
const EMBEDDING_BLOB_SIZE: usize = EMBEDDING_DIMS * 4; // 384 f32 values × 4 bytes each

pub fn vec_to_blob(vec: &[f32]) -> Result<Vec<u8>> {
    if vec.len() != EMBEDDING_DIMS {
        return Err(Error::MismatchedDimensions {
            expected: EMBEDDING_DIMS,
            actual: vec.len(),
        });
    }
    Ok(vec.iter().flat_map(|&x| x.to_le_bytes()).collect())
}

pub fn blob_to_vec(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() != EMBEDDING_BLOB_SIZE {
        return Err(Error::InvalidBlobSize {
            expected: EMBEDDING_BLOB_SIZE,
            actual: blob.len(),
        });
    }
    let mut vec = Vec::with_capacity(EMBEDDING_DIMS);
    for chunk in blob.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        vec.push(val);
    }
    Ok(vec)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::EmptyVector);
    }

    if a.len() != b.len() {
        return Err(Error::MismatchedDimensions {
            expected: a.len(),
            actual: b.len(),
        });
    }

    if a.iter().any(|x| x.is_nan() || x.is_infinite())
        || b.iter().any(|x| x.is_nan() || x.is_infinite())
    {
        return Err(Error::InvalidEmbedding(
            "Vector contains NaN or infinite values".to_string(),
        ));
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

fn create_schema(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            content='memories',
            content_rowid='rowid'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
            INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
        "#,
    )?;
    Ok(())
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        let mut conn = Connection::open(path)?;
        create_schema(&mut conn)?;
        Ok(Self { conn })
    }

    pub fn insert(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![&id, project_id, content, &blob, metadata, &now, &now],
        )?;

        Ok(id)
    }

    pub fn search(
        &self,
        project_id: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Memory>> {
        if limit == 0 {
            return Err(Error::InvalidLimit(
                "Limit must be greater than 0".to_string(),
            ));
        }
        if limit > i64::MAX as usize || limit > 10_000 {
            return Err(Error::InvalidLimit(format!(
                "Limit {} exceeds maximum allowed (10,000)",
                limit
            )));
        }

        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, created_at, updated_at, embedding
            FROM memories
            WHERE project_id = ?1
            "#,
        )?;

        let mut memories: Vec<Memory> = Vec::new();

        let rows = stmt.query_map([project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Vec<u8>>(6)?,
            ))
        })?;

        for row_result in rows {
            let (id, pid, content, metadata, created_at, updated_at, blob) = row_result?;
            let stored_embedding = blob_to_vec(&blob)?;
            let similarity = Some(cosine_similarity(query_embedding, &stored_embedding)?);

            memories.push(Memory {
                id,
                project_id: pid,
                content,
                metadata,
                similarity,
                created_at,
                updated_at,
            });
        }

        memories.sort_by(|a, b| {
            b.similarity
                .unwrap_or(0.0)
                .partial_cmp(&a.similarity.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        memories.truncate(limit);
        Ok(memories)
    }

    pub fn find_similar(
        &self,
        project_id: &str,
        embedding: &[f32],
        threshold: f64,
    ) -> Result<Vec<Memory>> {
        let all_results = self.search(project_id, embedding, 10_000)?;
        Ok(all_results
            .into_iter()
            .filter(|m| m.similarity.unwrap_or(0.0) >= threshold)
            .collect())
    }

    pub fn get(&self, id: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, created_at, updated_at
            FROM memories
            WHERE id = ?1
            "#,
        )?;

        let result = stmt
            .query_row([id], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    metadata: row.get(3)?,
                    similarity: None,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                })
            })
            .optional()?;

        Ok(result)
    }

    pub fn list(&self, project_id: &str, limit: usize) -> Result<Vec<Memory>> {
        if limit > i64::MAX as usize || limit > 10_000 {
            return Err(Error::InvalidLimit(format!(
                "Limit {} exceeds maximum allowed (10,000)",
                limit
            )));
        }

        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, project_id, content, metadata, created_at, updated_at
            FROM memories
            WHERE project_id = ?1
            ORDER BY created_at DESC
            LIMIT ?2
            "#,
        )?;

        let memories: SqliteResult<Vec<Memory>> = stmt
            .query_map(params![project_id, limit as i64], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    metadata: row.get(3)?,
                    similarity: None,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                })
            })?
            .collect();

        Ok(memories?)
    }

    pub fn update(&self, id: &str, content: &str, embedding: &[f32]) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        let rows = self.conn.execute(
            r#"
            UPDATE memories
            SET content = ?1, embedding = ?2, updated_at = ?3
            WHERE id = ?4
            "#,
            params![content, &blob, &now, id],
        )?;

        if rows == 0 {
            return Err(Error::Sqlite(format!("No memory found with id: {}", id)));
        }

        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", [id])?;
        Ok(rows > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0f32; 100];
        let b = vec![0.5f32; 384];
        let result = cosine_similarity(&a, &b);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::MismatchedDimensions { .. })));
    }

    #[test]
    fn test_cosine_similarity_same_length() {
        let a = vec![1.0f32; 384];
        let b = vec![0.5f32; 384];
        let result = cosine_similarity(&a, &b);
        assert!(result.is_ok());
        let similarity = result.unwrap();
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_to_blob_valid() {
        let vec = vec![0.0f32; 384];
        let result = vec_to_blob(&vec);
        assert!(result.is_ok());
        let blob = result.unwrap();
        assert_eq!(blob.len(), 1536);
    }

    #[test]
    fn test_vec_to_blob_invalid_dimensions() {
        let vec = vec![0.0f32; 100];
        let result = vec_to_blob(&vec);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::MismatchedDimensions { .. })));
        if let Err(Error::MismatchedDimensions { expected, actual }) = result {
            assert_eq!(expected, 384);
            assert_eq!(actual, 100);
        }
    }

    #[test]
    fn test_blob_to_vec_valid() {
        let mut blob = vec![0u8; 1536];
        blob[0] = 0x00;
        blob[1] = 0x00;
        blob[2] = 0x80;
        blob[3] = 0x3F; // 1.0f32 in little-endian
        let result = blob_to_vec(&blob);
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec.len(), 384);
        assert_eq!(vec[0], 1.0);
    }
}
