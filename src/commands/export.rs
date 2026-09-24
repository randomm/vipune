//! `vipune export` handler and JSONL writer (issue #195).
//!
//! The export produces a JSONL file: one header line followed by one 12-field
//! line per row, across ALL projects, with no row cap and no status/type
//! filter. The embedding is serialised as base64 of the exact stored BLOB
//! (1536 bytes for a 384xf32-LE vector), so the round-trip contract is
//! byte-identity, not cosine similarity. NULL or empty BLOBs export as `""`
//! and are only rejected later, at import time.
//!
//! # JSONL schema
//!
//! Header line (not counted as a data row):
//! `{"type":"export","format_version":1,"embedding_dims":384,"model_id":...,
//! "model_revision":...,"exported_at":<RFC3339>,"rows":<N>}`
//!
//! The model id + revision record the identity the store's embeddings were
//! produced with (the recorded identity, or the default bge identity when no
//! row exists — see `crate::sqlite::identity`). Importers compare these
//! against the destination database's identity and refuse on mismatch.
//!
//! The header identity is ADVISORY, not proof: it is whatever the source
//! store recorded at export time (or its default), and a hand-edited or
//! mis-labelled header is not verified against the actual vectors. The
//! import-side refusal is a guard against the common mistake, not a
//! guarantee of vector provenance.
//!
//! Row line (12 fields; the DB column `type` is renamed `memory_type` so the
//! JSON key never collides with the header's `type` discriminator):
//! `{"id", "project_id", "content", "metadata", "embedding" (base64),
//! "created_at", "updated_at", "memory_type", "status", "superseded_by",
//! "retrieval_count", "last_retrieved_at"}`
//!
//! The global `--project` flag is ignored: export is cross-project by
//! contract, and a stderr warning is emitted when it is passed.

use crate::errors::Error;
use crate::sqlite::Database;
use crate::sqlite::export_scan::ExportRow;
use crate::sqlite::identity;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

/// Format version of the JSONL export, written in the header line.
pub const EXPORT_FORMAT_VERSION: u32 = 1;

use crate::embedding::EMBEDDING_DIMS;

/// Base64 standard-alphabet encoder — vendored so the export does not need a
/// new crate dependency. 1536-byte blobs encode to a 2048-char string, so the
/// default stack allocation is far smaller than any real input.
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Decode a base64 string (standard alphabet) back to bytes.
///
/// Inverse of [`base64_encode`] (used by the round-trip tests). Rejects
/// padding in the wrong position and any character outside the alphabet.
///
/// Test-only helper: the `export` command never decodes (import uses the
/// `base64` crate), so the `#[cfg(test)]` visibility keeps it out of the
/// production build instead of carrying a dead-code suppression.
#[cfg(test)]
pub(crate) fn base64_decode(input: &str) -> Result<Vec<u8>, Error> {
    fn value(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    fn decode_value(c: u8) -> Result<u32, Error> {
        value(c).ok_or_else(|| {
            Error::InvalidInput(format!("invalid base64 character: {:?}", c as char))
        })
    }

    let bytes: Vec<u8> = input.bytes().collect();
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidInput(format!(
            "base64 input length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        // Decode the two always-present high bits first, then peel off any
        // trailing padding before decoding the remaining two sextets. This
        // keeps the padding checks local to each of the four legal layouts:
        // `XXXX`, `XXXX=`, `XXXX==` (only `==`), never `X==X`/`===X`.
        let v0 = decode_value(chunk[0])?;
        let v1 = decode_value(chunk[1])?;
        if chunk[1] == b'=' {
            return Err(Error::InvalidInput("misplaced base64 padding".to_string()));
        }
        out.push(((v0 << 2) | (v1 >> 4)) as u8);
        if chunk[2] == b'=' {
            // `==` padding: only the high 4 bits were carried above.
            if chunk[3] != b'=' {
                return Err(Error::InvalidInput("misplaced base64 padding".to_string()));
            }
            continue;
        }
        let v2 = decode_value(chunk[2])?;
        out.push(((v1 << 4) | (v2 >> 2)) as u8);
        if chunk[3] == b'=' {
            // `=` padding on the final byte: no third octet.
            continue;
        }
        let v3 = decode_value(chunk[3])?;
        out.push(((v2 << 6) | v3) as u8);
    }
    Ok(out)
}

struct ExportJsonlRow {
    /// Global primary key.
    id: String,
    /// Project id the row carries (no validation at import time).
    project_id: String,
    /// Memory content.
    content: String,
    /// User-provided JSON metadata, or JSON null.
    metadata: Option<String>,
    /// Base64 of the exact stored embedding BLOB (`""` for NULL/empty).
    embedding: String,
    /// RFC3339 creation timestamp, restored verbatim.
    created_at: String,
    /// RFC3339 last-update timestamp, restored verbatim.
    updated_at: String,
    /// Memory type — renamed from the DB column `type`.
    memory_type: String,
    /// Lifecycle status.
    status: String,
    /// Id of the superseding memory, or JSON null.
    superseded_by: Option<String>,
    /// Retrieval counter, restored verbatim.
    retrieval_count: i64,
    /// Last retrieval timestamp, or JSON null.
    last_retrieved_at: Option<String>,
}

impl ExportJsonlRow {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "project_id": self.project_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "memory_type": self.memory_type,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "retrieval_count": self.retrieval_count,
            "last_retrieved_at": self.last_retrieved_at,
        })
    }
}

impl From<&ExportRow> for ExportJsonlRow {
    fn from(row: &ExportRow) -> Self {
        Self {
            id: row.id.clone(),
            project_id: row.project_id.clone(),
            content: row.content.clone(),
            metadata: row.metadata.clone(),
            embedding: base64_encode(&row.embedding_blob),
            created_at: row.created_at.clone(),
            updated_at: row.updated_at.clone(),
            memory_type: row.memory_type.clone(),
            status: row.status.clone(),
            superseded_by: row.superseded_by.clone(),
            retrieval_count: row.retrieval_count,
            last_retrieved_at: row.last_retrieved_at.clone(),
        }
    }
}

/// Response struct for `vipune export --json`.
///
/// Note: task-d of issue #195 owns `src/output.rs` and will add the shared
/// response struct there. Until it lands, this command-local struct carries
/// the spec-final field set (`rows`, `format_version`, `path`) so the export
/// command builds and its tests run independently. Replace the call site with
/// `crate::output::ExportResponse` and delete this once task-d is merged.
#[derive(serde::Serialize)]
pub struct ExportResponse {
    /// Number of data rows written (header line not counted).
    pub rows: usize,
    /// Format version of the JSONL file that was written.
    pub format_version: u32,
    /// Destination file path as given on the command line.
    pub path: String,
}

/// Write the JSONL export to `out`: one header line, then one line per row,
/// all with LF line endings. Every field goes through `serde_json`, so
/// content containing quotes, newlines, or unicode serialises safely.
///
/// The header is NOT counted in `rows`.
///
/// # Errors
///
/// Returns an error if any row fails to serialize or the write fails.
pub fn write_jsonl<W: Write>(
    out: &mut W,
    rows: &[ExportRow],
    exported_at: &str,
    model_id: &str,
    model_revision: &str,
) -> std::result::Result<usize, Error> {
    let header = serde_json::json!({
        "type": "export",
        "format_version": EXPORT_FORMAT_VERSION,
        "embedding_dims": EMBEDDING_DIMS,
        "model_id": model_id,
        "model_revision": model_revision,
        "exported_at": exported_at,
        "rows": rows.len(),
    });
    writeln!(
        out,
        "{}",
        serde_json::to_string(&header).map_err(Error::from)?
    )?;
    for row in rows {
        let line =
            serde_json::to_string(&ExportJsonlRow::from(row).to_json()).map_err(Error::from)?;
        writeln!(out, "{line}")?;
    }
    Ok(rows.len())
}

/// Run the export command.
///
/// Opens the database at `db_path` (the path resolved AFTER the `--db-path`
/// CLI override is applied), scans ALL rows across ALL projects (uncapped,
/// raw blobs), and writes the JSONL file to `output_path`.
///
/// # Arguments
///
/// * `db_path` - Resolved database path (honours `--db-path`)
/// * `output_path` - Destination JSONL file
/// * `project` - The global `--project` flag, if passed (ignored, warned)
/// * `json` - If true, emit the JSON response; else human-readable
///
/// # Errors
///
/// Returns an error if the database cannot be opened, the scan fails, or the
/// file cannot be written.
pub fn handle_export(
    db_path: &Path,
    output_path: &Path,
    project: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    if let Some(project) = project {
        eprintln!(
            "Note: --project '{}' is ignored for export — export covers ALL projects.",
            project
        );
    }

    let db = Database::open(db_path).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string(),
            );
        }
        Error::Config(msg)
    })?;

    // The header records the identity the store's embeddings were produced
    // with (the recorded identity, or the default bge identity when no row
    // exists — the zero-change contract for pre-v6 stores).
    let (recorded, _) = identity::read_identity_and_marker(db.conn())
        .map_err(|e| Error::Config(format!("identity read failed: {e}")))?;
    let identity = recorded.unwrap_or_else(identity::ModelIdentity::default_identity);

    let rows = db
        .scan_all_rows()
        .map_err(|e| Error::InvalidInput(format!("export scan failed: {e}")))?;

    let exported_at = chrono::Utc::now().to_rfc3339();
    let mut file = std::fs::File::create(output_path)
        .map_err(|e| Error::Config(format!("cannot create {}: {e}", output_path.display())))?;
    let count = write_jsonl(
        &mut file,
        &rows,
        &exported_at,
        &identity.model_id,
        &identity.revision,
    )?;

    let response = ExportResponse {
        rows: count,
        format_version: EXPORT_FORMAT_VERSION,
        path: output_path.display().to_string(),
    };

    if json {
        crate::output::print_json(&response);
    } else {
        println!("Exported {} row(s) to {}", count, output_path.display());
    }

    Ok(ExitCode::SUCCESS)
}
