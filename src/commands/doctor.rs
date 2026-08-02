//! `vipune doctor --embeddings` handler.
//!
//! Reports per-project: total / real / mock / unknown rows using the L2-norm classifier.

use crate::errors::Error;
use crate::output::{DoctorResponse, print_json};
use crate::sqlite::Database;
use crate::sqlite::embedding::classify_embedding;
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
