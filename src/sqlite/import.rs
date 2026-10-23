//! Restore support for the import command (issue #195).
//!
//! `Database::insert_with_id` is the all-columns, caller-supplied-id insert
//! used by the import path. Unlike `insert()` (which mints a fresh UUID and
//! leaves `superseded_by`/`retrieval_count`/`last_retrieved_at` at schema
//! defaults), it restores all 12 columns verbatim from an export so an
//! export → import round-trip is byte-identical. Embedding blobs arrive
//! as raw bytes; they are written as-is with no dimension validation at
//! this layer (dimension checks happen in the import handler so a whole
//! import can abort all-or-nothing).

use super::Database;
use crate::sqlite::Result;
use rusqlite::params;

impl Database {
    /// Insert a memory row under an explicit id, restoring all 12 columns.
    ///
    /// `embedding_blob` is the raw BLOB (already base64-decoded by the
    /// caller) and is written byte-for-byte. Returns `true` if the row was
    /// inserted; a duplicate id is NOT a silent `false` — the caller is
    /// expected to pre-filter against `existing_ids()` (the skip set) before
    /// calling, and a duplicate that slips through surfaces as a PK-constraint
    /// error, never a silent upsert or overwrite.
    ///
    /// # Errors
    ///
    /// Returns error if the database write fails, including a primary-key
    /// constraint violation when the id already exists.
    #[allow(dead_code)] // used by bin target (import); exercised by lib tests
    #[allow(clippy::too_many_arguments)] // signature mirrors all 12 table columns
    pub fn insert_with_id(
        &self,
        id: &str,
        project_id: &str,
        content: &str,
        embedding_blob: &[u8],
        metadata: Option<&str>,
        created_at: &str,
        updated_at: &str,
        memory_type: &str,
        status: &str,
        superseded_by: Option<&str>,
        retrieval_count: i64,
        last_retrieved_at: Option<&str>,
    ) -> Result<bool> {
        let rows = self.conn.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            params![
                id,
                project_id,
                content,
                embedding_blob,
                metadata,
                created_at,
                updated_at,
                memory_type,
                status,
                superseded_by,
                retrieval_count,
                last_retrieved_at,
            ],
        )?;
        Ok(rows > 0)
    }

    /// Collect the set of ids already present in the database (all projects).
    ///
    /// Called by the import path BEFORE the import transaction begins, so
    /// already-present rows are skipped and counted rather than erroring or
    /// upserting.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn existing_ids(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT id FROM memories")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut ids = Vec::new();
        for row_result in rows {
            ids.push(row_result?);
        }
        Ok(ids)
    }

    /// Borrow the underlying SQLite connection mutably (used by the import
    /// path to open a single all-or-nothing transaction on the same
    /// connection that built the skip set and applied the nonzero
    /// busy_timeout).
    #[allow(dead_code)] // used by bin target (import), not by lib
    pub(crate) fn connection(&mut self) -> &mut rusqlite::Connection {
        &mut self.conn
    }
}
