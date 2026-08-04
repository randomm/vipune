//! `vipune doctor --embeddings` and `vipune doctor --projects` handlers.
//!
//! `--embeddings`: reports per-project total / real / mock / unknown rows.
//! `--projects`: scans all projects for suspected split pairs (bare id vs owner/repo).

use crate::errors::Error;
use crate::output::{DoctorProjectsResponse, DoctorResponse, print_json};
use crate::sqlite::Database;
use crate::sqlite::embedding::classify_embedding;
use rusqlite::{Connection, OpenFlags};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::ExitCode;

/// Wrap a database error, converting SQLITE_BUSY into the actionable MCP-server message.
fn wrap_busy<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(Error::SqliteModule(msg)) if msg.contains("database is locked") => {
            Err(Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            ))
        }
        Err(e) => Err(e),
    }
}

/// Run the embedding doctor check on the database.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database
/// * `project_filter` - If Some, only check this project; if None, check all projects
/// * `json` - If true, output JSON; otherwise human-readable
///
/// # Errors
///
/// Returns error if the database cannot be opened or queried.
pub fn handle_doctor(
    db_path: &Path,
    project_filter: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open database
    let db = Database::open(db_path).map_err(|e| {
        let err_msg = e.to_string();
        if err_msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            );
        }
        Error::Config(err_msg)
    })?;

    // Determine which projects to audit
    let projects: Vec<String> = if let Some(filter) = project_filter {
        vec![filter.to_string()]
    } else {
        wrap_busy(db.list_all_project_ids().map_err(Error::from))?
    };

    if projects.is_empty() {
        if json {
            print_json(&[DoctorResponse {
                project_id: "(none)".to_string(),
                total_rows: 0,
                real_rows: 0,
                mock_rows: 0,
                unknown_rows: 0,
            }]);
        } else {
            println!("No projects found in database.");
        }
        return Ok(ExitCode::SUCCESS);
    }

    let mut responses: Vec<DoctorResponse> = vec![];

    for project_id in &projects {
        let result = wrap_busy(audit_project(&db, project_id))?;
        responses.push(DoctorResponse {
            project_id: project_id.clone(),
            total_rows: result.total,
            real_rows: result.real_count,
            mock_rows: result.mock_count,
            unknown_rows: result.unknown_count,
        });

        if !json {
            println!("Project: {}", project_id);
            println!("  Total rows: {}", result.total);
            println!("  Real:     {}", result.real_count);
            println!("  Mock:     {}", result.mock_count);
            println!("  Unknown:  {}", result.unknown_count);
            if result.mock_count > 0 || result.unknown_count > 0 {
                println!(
                    "  → Run 'vipune reindex' to repair mock rows ({} candidates)",
                    result.mock_count
                );
            }
        }
    }

    // Print JSON: single array of all project responses
    if json {
        print_json(&responses);
    }

    Ok(ExitCode::SUCCESS)
}

struct AuditResult {
    total: usize,
    real_count: usize,
    mock_count: usize,
    unknown_count: usize,
}

fn audit_project(db: &Database, project_id: &str) -> Result<AuditResult, Error> {
    let rows = db.list_all_rows_for_project(project_id)?;

    let mut result = AuditResult {
        total: 0,
        real_count: 0,
        mock_count: 0,
        unknown_count: 0,
    };

    for (_id, _content, embedding) in rows {
        result.total += 1;
        match classify_embedding(&embedding) {
            crate::sqlite::embedding::EmbeddingClass::Real => result.real_count += 1,
            crate::sqlite::embedding::EmbeddingClass::Mock => result.mock_count += 1,
            crate::sqlite::embedding::EmbeddingClass::Unknown => result.unknown_count += 1,
        }
    }

    Ok(result)
}

/// Run the project split detection scan.
///
/// Scans ALL projects in the database regardless of `-p/--project`. A split spans
/// two project_ids by definition, so a scoped scan structurally cannot see one.
/// Performs no database writes.
///
/// Heuristic: id `A` is a suspected split of id `B` when `A` equals the segment
/// after `/` in `B`. Example: `pi-an` pairs with `randomm/pi-an`.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database
/// * `project_filter` - Intended project filter (always ignored; warn if Some)
/// * `json` - If true, output JSON; otherwise human-readable
///
/// # Errors
///
/// Returns error if the database cannot be opened or queried.
pub fn handle_doctor_projects(
    db_path: &Path,
    project_filter: Option<&str>,
    json: bool,
) -> Result<ExitCode, Error> {
    // Warn if -p was passed alongside --projects (silently ignored but user likely expects it to apply)
    if let Some(filter) = project_filter {
        eprintln!(
            "Warning: -p/--project is ignored for doctor --projects (scan must cover all projects to detect splits). Filter '{}' was not applied.",
            filter
        );
    }

    // Open database in READ-ONLY mode — this is a diagnostic that must not modify the DB.
    // Any accidental write will fail with SQLITE_READONLY instead of silently corrupting data.
    let db = Database::from_conn(
        Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
            let err_msg = e.to_string();
            if err_msg.contains("database is locked") {
                return Error::Config(
                    "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
                );
            }
            Error::Config(err_msg)
        })?,
    );

    // Gather all project ids with row counts
    let project_ids = db.list_all_project_ids().map_err(Error::from)?;

    let mut counts: HashMap<String, usize> = HashMap::new();
    for pid in &project_ids {
        counts.insert(
            pid.clone(),
            db.count_rows_for_project(pid).map_err(Error::from)?,
        );
    }

    // Detect suspected split pairs.
    // Heuristic: bare_id == repo segment after "/" in "owner/repo".
    // pair[0] = bare_id (no "/"), pair[1] = owner/repo_id.
    let mut suspected_splits: Vec<(&str, &str)> = Vec::new();
    let mut reported: HashSet<&str> = HashSet::new();

    // Collect owner/repo ids (those containing "/").
    let owned_ids: Vec<&str> = project_ids
        .iter()
        .filter(|s| s.contains('/'))
        .map(|s| s.as_str())
        .collect();

    for owned in &owned_ids {
        // Extract the repo segment (after the first "/")
        let repo_segment = match owned.split_once('/') {
            Some((_, repo)) => repo,
            None => continue,
        };
        // Check if the bare id exists AND hasn't already been reported
        if counts.contains_key(repo_segment) && !reported.contains(repo_segment) {
            reported.insert(repo_segment);
            // Avoid self-pair: bare id must not equal the full owned id
            // (can't happen since bare has no "/", but be explicit)
            if repo_segment != *owned {
                suspected_splits.push((repo_segment, owned));
            }
        }
    }

    // Sort by bare_id for deterministic output.
    suspected_splits.sort_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(b.1)));

    // Build response.
    let response = DoctorProjectsResponse {
        suspected_splits: suspected_splits
            .into_iter()
            .map(
                |(bare, owned)| crate::output::DoctorProjectsSuspectedSplit {
                    pair: [bare.to_string(), owned.to_string()],
                    row_counts: [
                        *counts.get(bare).unwrap_or(&0),
                        *counts.get(owned).unwrap_or(&0),
                    ],
                },
            )
            .collect(),
    };

    if json {
        print_json(&response);
    } else {
        print_human_projects(&response);
    }

    Ok(ExitCode::SUCCESS)
}

/// Print human-readable output for the projects doctor check.
fn print_human_projects(response: &DoctorProjectsResponse) {
    if response.suspected_splits.is_empty() {
        println!("No suspected project splits found.");
        return;
    }

    println!("Suspected project splits:");
    println!();

    for split in &response.suspected_splits {
        println!(
            "  '{}' ({} rows)  +  '{}' ({} rows)",
            split.pair[0], split.row_counts[0], split.pair[1], split.row_counts[1]
        );
    }

    println!();
    println!(
        "These are suspected pairs — confirm they represent the same repository before merging."
    );
    println!("Known false positive: a genuinely separate project whose directory name matches");
    println!("another project's repo name (e.g. 'ci-runner' vs 'team/ci-runner').");
    println!();
    println!("To merge confirmed pairs, run:");
    println!("  vipune project merge <from> <to>");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::Database;
    use tempfile::TempDir;

    /// Helper: create a test database with known embeddings.
    fn setup_test_db() -> (TempDir, Database) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        (dir, db)
    }

    #[test]
    fn test_doctor_empty_database() {
        let (_dir, db) = setup_test_db();
        let result = audit_project(&db, "nonexistent-project").unwrap();
        assert_eq!(result.total, 0);
        assert_eq!(result.real_count, 0);
        assert_eq!(result.mock_count, 0);
        assert_eq!(result.unknown_count, 0);
    }

    #[test]
    fn test_doctor_classifies_real_embeddings() {
        let (_dir, db) = setup_test_db();
        // Insert a real (L2-normalised) embedding
        let mut vec = vec![0.0f32; 384];
        vec[0] = 1.0; // norm = 1.0 → Real
        db.insert("test-proj", "real memory", &vec, None, "fact", "active")
            .unwrap();

        let result = audit_project(&db, "test-proj").unwrap();
        assert_eq!(result.total, 1);
        assert_eq!(result.real_count, 1);
        assert_eq!(result.mock_count, 0);
        assert_eq!(result.unknown_count, 0);
    }

    #[test]
    fn test_doctor_classifies_mock_embeddings() {
        let (_dir, db) = setup_test_db();
        // Insert a mock-like embedding (uniform ones, norm ≈ 19.6)
        let vec = vec![1.0f32; 384];
        db.insert("test-proj", "mock memory", &vec, None, "fact", "active")
            .unwrap();

        let result = audit_project(&db, "test-proj").unwrap();
        assert_eq!(result.total, 1);
        assert_eq!(result.real_count, 0);
        assert_eq!(result.mock_count, 1);
        assert_eq!(result.unknown_count, 0);
    }

    #[test]
    fn test_doctor_mixed_classifications() {
        let (_dir, db) = setup_test_db();
        // Real: norm = 1.0
        let mut real_vec = vec![0.0f32; 384];
        real_vec[0] = 1.0;
        db.insert("proj", "real", &real_vec, None, "fact", "active")
            .unwrap();
        // Mock: norm ≈ 19.6
        let mock_vec = vec![1.0f32; 384];
        db.insert("proj", "mock", &mock_vec, None, "fact", "active")
            .unwrap();
        // Unknown: norm = 0
        let unknown_vec = vec![0.0f32; 384];
        db.insert("proj", "unknown", &unknown_vec, None, "fact", "active")
            .unwrap();

        let result = audit_project(&db, "proj").unwrap();
        assert_eq!(result.total, 3);
        assert_eq!(result.real_count, 1);
        assert_eq!(result.mock_count, 1);
        assert_eq!(result.unknown_count, 1);
    }
}
