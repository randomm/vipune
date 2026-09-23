//! `vipune import` handler (issue #195).
//!
//! Restores memories from a JSONL export file (or `-` for stdin) into the
//! database in a SINGLE TRANSACTION, ALL-OR-NOTHING:
//!
//! - The upfront skip-set (`SELECT id FROM memories`) runs on the same
//!   connection BEFORE `BEGIN`; rows whose id already exists are skipped and
//!   counted, never upserted, so re-running an interrupted import is safe.
//! - Any malformed line, wrong-dimension embedding, or empty/corrupt base64
//!   blob aborts the whole import — the transaction is dropped (never
//!   committed), zero rows are written, and the handler exits non-zero
//!   naming the offending line number (the header is line 1; data rows
//!   start at line 2).
//! - The busy_timeout is NONZERO (unlike reindex/doctor, which fail fast
//!   with ZERO) so a restore waits for a running MCP server rather than
//!   failing instantly on `database is locked`.
//! - `--project` is ignored (cross-project by contract) with a stderr warning.
//! - The header's model identity (`model_id` + `model_revision`) is compared
//!   against the destination database's recorded identity: a mismatch is
//!   refused, and headers without identity (pre-#217 exports) are treated as
//!   the default bge identity, so legacy exports still import into default
//!   stores.
//!
//! JSONL schema — header line:
//!   `{"type":"export","version":1,"embedding_dims":384,"model_id":...,"model_revision":...,"exported_at":...,"rows":N}`
//! Row line (12 fields; DB column `type` is named `memory_type`):
//!   `id`, `project_id`, `content`, `metadata` (nullable), `embedding`
//!   (base64 of the exact stored BLOB), `created_at`, `updated_at`,
//!   `memory_type`, `status`, `superseded_by` (nullable),
//!   `retrieval_count`, `last_retrieved_at` (nullable).

use crate::errors::Error;
use crate::output::print_json;
use crate::sqlite::Database;
use crate::sqlite::model_identity;
use base64::Engine;
use serde::Deserialize;
use std::io::BufRead;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use super::ImportResponse;

use crate::embedding::EMBEDDING_DIMS;

/// Busy timeout for the import connection: a restore should WAIT for a
/// running MCP server to release its lock, unlike reindex/doctor which use
/// `Duration::ZERO` for fast-fail.
const IMPORT_BUSY_TIMEOUT: Duration = Duration::from_secs(30);

/// One JSONL row (12 columns; `memory_type` is the JSON name for the DB
/// `type` column).
#[derive(Deserialize)]
struct ImportRow {
    id: String,
    project_id: String,
    content: String,
    #[serde(default)]
    metadata: Option<String>,
    embedding: String,
    created_at: String,
    updated_at: String,
    memory_type: String,
    status: String,
    #[serde(default)]
    superseded_by: Option<String>,
    retrieval_count: i64,
    #[serde(default)]
    last_retrieved_at: Option<String>,
}

/// Read the JSONL source (path, or `-` for stdin).
fn read_source(source: &str) -> Result<String, Error> {
    if source == "-" {
        let stdin = std::io::stdin();
        let mut reader = std::io::BufReader::new(stdin.lock());
        let mut content = String::new();
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(Error::from)? != 0 {
            content.push_str(&line);
            line.clear();
        }
        Ok(content)
    } else {
        std::fs::read_to_string(source).map_err(|e| {
            Error::InvalidInput(format!("cannot read import source '{}': {}", source, e))
        })
    }
}

/// Parse one data line into a row and its decoded, dimension-checked blob.
///
/// Line numbers in errors are 1-based with the header as line 1.
fn parse_row(line: &str, line_no: usize) -> Result<(ImportRow, Vec<u8>), Error> {
    // Tolerate CRLF-terminated input (Windows-edited files): a trailing
    // `\r` would otherwise corrupt the last JSON field.
    let line = line.trim_end_matches('\r');
    if line.trim().is_empty() {
        return Err(Error::InvalidInput(format!("line {}: empty row", line_no)));
    }

    let row: ImportRow = serde_json::from_str(line)
        .map_err(|e| Error::InvalidInput(format!("line {}: malformed JSON: {}", line_no, e)))?;

    if row.embedding.is_empty() {
        return Err(Error::InvalidInput(format!(
            "line {}: empty embedding (NULL/empty blobs are not restorable)",
            line_no
        )));
    }

    let blob = base64::engine::general_purpose::STANDARD
        .decode(&row.embedding)
        .map_err(|e| {
            Error::InvalidInput(format!("line {}: invalid base64 embedding: {}", line_no, e))
        })?;

    if blob.len() != EMBEDDING_DIMS * 4 {
        return Err(Error::InvalidInput(format!(
            "line {}: wrong-dimension embedding: expected {} bytes ({}xf32), got {} bytes",
            line_no,
            EMBEDDING_DIMS * 4,
            EMBEDDING_DIMS,
            blob.len()
        )));
    }

    Ok((row, blob))
}

/// The model identity carried by an export header. `None` fields mean the
/// header predates issue #217 (no identity keys) — such exports are treated
/// as the default bge identity, so legacy files still import into default
/// stores (zero-change contract).
#[derive(Debug, Clone, PartialEq, Eq)]
struct HeaderIdentity {
    model_id: Option<String>,
    model_revision: Option<String>,
}

impl HeaderIdentity {
    /// Resolve to a concrete identity: absent fields fall back to the default
    /// bge identity's value.
    fn resolved(&self) -> model_identity::ModelIdentity {
        let default = model_identity::default_identity();
        model_identity::ModelIdentity {
            model_id: self.model_id.clone().unwrap_or(default.model_id),
            revision: self.model_revision.clone().unwrap_or(default.revision),
        }
    }
}

/// Validate the header line: must be JSON with `type: "export"`, a version,
/// and a `rows` count. Returns the declared row count and the header's model
/// identity (None fields = identity absent, i.e. a legacy export).
fn validate_header(line: &str) -> Result<(usize, HeaderIdentity), Error> {
    let line = line.trim_end_matches('\r');
    let value: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| Error::InvalidInput(format!("line 1: malformed header JSON: {}", e)))?;
    let obj = value
        .as_object()
        .ok_or_else(|| Error::InvalidInput("line 1: header must be a JSON object".to_string()))?;

    if obj.get("type").and_then(|v| v.as_str()) != Some("export") {
        return Err(Error::InvalidInput(
            "line 1: not an export file (missing type=export)".to_string(),
        ));
    }
    if obj.get("version").is_none() {
        return Err(Error::InvalidInput(
            "line 1: header missing version".to_string(),
        ));
    }
    let rows = obj
        .get("rows")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::InvalidInput("line 1: header missing rows".to_string()))?;
    let identity = HeaderIdentity {
        model_id: obj
            .get("model_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        model_revision: obj
            .get("model_revision")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    };
    Ok((rows as usize, identity))
}

/// Run the import. Returns the response on success, or an `Err` whose
/// message names the offending line number on all-or-nothing failure.
///
/// Every data line is parsed and dimension-checked BEFORE the transaction
/// opens, so a late-line error can never leave a partial write behind.
pub(crate) fn run_import(db_path: &Path, source: &str) -> Result<ImportResponse, Error> {
    // Nonzero busy_timeout: a restore waits for the MCP server rather than
    // failing instantly (deliberate divergence from reindex/doctor).
    let mut db = Database::open(db_path).map_err(|e| Error::Config(e.to_string()))?;
    db.set_busy_timeout(IMPORT_BUSY_TIMEOUT)
        .map_err(Error::from)?;

    // Upfront skip set on the SAME connection, BEFORE the transaction:
    // ids already present in the destination are skipped and counted.
    let skip_set: std::collections::HashSet<String> = db.existing_ids()?.into_iter().collect();

    let content = read_source(source)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err(Error::InvalidInput(
            "import source is empty (no header line)".to_string(),
        ));
    }

    // Validate header (line 1) and check its model identity against the
    // destination database: a mismatched identity means the exported vectors
    // were produced by a different model and would silently degrade search
    // quality, so the import is refused (use `reindex --force` after export
    // into a matching store, or re-export from a matching store).
    let (_rows, header_identity) = validate_header(lines[0])?;
    let (recorded, marker) = model_identity::read_identity(db.conn())
        .map_err(|e| Error::SqliteModule(format!("identity read failed: {e}")))?;
    if marker.is_some() {
        return Err(Error::InvalidInput(format!(
            "import refused: destination database is in the middle of a model migration ({}); complete it with `vipune reindex --force` and retry",
            marker.unwrap_or_default()
        )));
    }
    let destination = recorded.unwrap_or_else(model_identity::default_identity);
    let source = header_identity.resolved();
    if source != destination {
        return Err(Error::InvalidInput(format!(
            "import refused: export was produced with model {} but the destination database uses {}. Re-import into a store embedded with the same model, or complete the migration with `vipune reindex --force`.",
            source.display(),
            destination.display()
        )));
    }

    // Parse and validate every data line up front (all-or-nothing).
    let mut parsed: Vec<(usize, ImportRow, Vec<u8>)> = Vec::new();
    for (idx, line) in lines.iter().enumerate().skip(1) {
        let line_no = idx + 1;
        if line.trim().is_empty() {
            continue;
        }
        let (row, blob) = parse_row(line, line_no)?;
        parsed.push((line_no, row, blob));
    }

    // Single transaction: insert every non-skipped row, commit once.
    let (mut inserted, mut skipped) = (0usize, 0usize);
    let tx = db.connection().transaction()?;
    for (line_no, row, blob) in parsed {
        if skip_set.contains(&row.id) {
            skipped += 1;
            continue;
        }
        let inserted_here = tx.execute(
            r#"
            INSERT INTO memories (id, project_id, content, embedding, metadata, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            rusqlite::params![
                row.id,
                row.project_id,
                row.content,
                blob,
                row.metadata,
                row.created_at,
                row.updated_at,
                row.memory_type,
                row.status,
                row.superseded_by,
                row.retrieval_count,
                row.last_retrieved_at,
            ],
        )?;
        if inserted_here == 0 {
            // PK conflict the skip set did not see (id inserted between the
            // SELECT and now). All-or-nothing: abort the whole import.
            return Err(Error::InvalidInput(format!(
                "line {}: id {} already exists",
                line_no, row.id
            )));
        }
        inserted += 1;
    }
    tx.commit().map_err(Error::from)?;

    Ok(ImportResponse { inserted, skipped })
}

/// Public entry point invoked by `Commands::Import`.
pub fn handle_import(
    db_path: &Path,
    source: Option<&str>,
    project_flag: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    if let Some(project) = project_flag {
        eprintln!(
            "warning: --project ({project}) is ignored for import; import restores all projects in the file"
        );
    }
    let source_str = source.unwrap_or("-");
    match run_import(db_path, source_str) {
        Ok(response) => {
            if json {
                print_json(&response);
            } else {
                println!(
                    "Import complete: {} inserted, {} skipped (already present)",
                    response.inserted, response.skipped
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            // All-or-nothing: nothing was written; report and exit non-zero.
            eprintln!("Import failed: {}", e);
            Err(Error::InvalidInput(e.to_string()))
        }
    }
}
