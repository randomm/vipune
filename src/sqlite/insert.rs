//! Database insert operations.
//!
//! Owns the column set of a new row: both new-row INSERTs below take a
//! [`NewRow`], so the add path and the supersede path cannot learn to name
//! different columns — the shape that made `--importance` a silent no-op in
//! issue #238.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use super::{Result, vec_to_blob};

/// The caller-supplied columns of one new row, with the lifecycle fields
/// already reduced to their canonical column spellings.
///
/// Build the values from the parsed lifecycle enums (`MemoryType::as_str`,
/// `MemoryStatus::as_str`, `MemoryImportance::as_str`) — the column has no
/// CHECK constraint, so the canonical spelling is what read-side parsing and
/// the `high`/`critical` prune exemption depend on.
pub(crate) struct NewRow<'a> {
    /// Project that owns the row.
    pub project_id: &'a str,
    /// Text content to store.
    pub content: &'a str,
    /// Optional JSON metadata string.
    pub metadata: Option<&'a str>,
    /// Canonical memory type (`fact`, `preference`, `procedure`, `guard`,
    /// `observation`).
    pub memory_type: &'a str,
    /// Canonical lifecycle status (`active`, `candidate`).
    pub status: &'a str,
    /// Canonical importance level (`low`, `medium`, `high`, `critical`).
    pub importance: &'a str,
}

impl super::Database {
    /// Insert a new memory with embedding at the `medium` importance default.
    ///
    /// The entry point for a caller that names no level: it delegates to
    /// [`Self::insert_with_importance`], which binds `medium` by name rather
    /// than leaving the column to its `NOT NULL DEFAULT`.
    ///
    /// # Errors
    ///
    /// Returns error if the embedding has invalid dimensions or database write fails.
    pub fn insert(
        &self,
        project_id: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<&str>,
        memory_type: &str,
        status: &str,
    ) -> Result<String> {
        self.insert_with_importance(
            &NewRow {
                project_id,
                content,
                metadata,
                memory_type,
                status,
                importance: "medium",
            },
            embedding,
        )
    }

    /// Insert a new memory binding the `importance` column explicitly.
    ///
    /// `vipune add --importance <level>` writes through here. The column is
    /// named in the INSERT and bound to the caller's canonical level because
    /// `prune` reads it for the `high`/`critical` exemption: an omitted column
    /// silently reset every added row to the DEFAULT (issue #238).
    ///
    /// # Errors
    ///
    /// Returns error if the embedding has invalid dimensions or database write fails.
    pub(crate) fn insert_with_importance(
        &self,
        row: &NewRow<'_>,
        embedding: &[f32],
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let blob = vec_to_blob(embedding)?;

        self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, importance)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                &id,
                row.project_id,
                row.content,
                &blob,
                row.metadata,
                &now,
                &now,
                row.memory_type,
                row.status,
                row.importance,
            ],
        )?;

        Ok(id)
    }
}
