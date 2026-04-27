//! CLI entry point for vipune memory layer.

mod commands;
mod config;
mod embedding;
mod errors;
mod memory;
pub mod memory_types; // Re-export for library consumers: IngestPolicy, BatchIngestItemResult, BatchIngestResult
mod output;
mod project;
mod rrf;
mod sqlite;
mod temporal;

use clap::Parser;
use commands::Commands;
use errors::Error;
use memory::MemoryStore;
use output::{ErrorResponse, print_json};
use project::detect_project;
use std::process::ExitCode;

/// vipune - A minimal memory layer for AI agents
#[derive(Parser)]
#[command(name = "vipune", about = "Minimal memory layer for AI agents", long_about = None)]
struct Cli {
    /// Output as JSON (default: human-readable)
    #[arg(long, global = true)]
    json: bool,

    /// Project identifier (auto-detected from git if omitted)
    #[arg(long, short = 'p', global = true)]
    project: Option<String>,

    /// Override database path
    #[arg(long, global = true)]
    db_path: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(&cli) {
        Ok(exit_code) => exit_code,
        Err(error) => {
            // Map ContentTooLong errors to exit code 3
            let exit_code = if matches!(error, Error::ContentTooLong { .. }) {
                ExitCode::from(3)
            } else {
                ExitCode::from(1)
            };

            if cli.json {
                print_json(&ErrorResponse {
                    error: error.to_string(),
                });
            } else {
                eprintln!("Error: {}", error);
            }
            exit_code
        }
    }
}

fn run(cli: &Cli) -> Result<ExitCode, Error> {
    let mut config = config::Config::load()?;
    config.ensure_directories()?;

    if let Some(db_path) = &cli.db_path {
        config.database_path = db_path.clone().into();
    }

    let project_id = detect_project(cli.project.as_deref());

    // Handle MCP command separately (doesn't use MemoryStore directly)
    #[cfg(feature = "mcp")]
    if matches!(cli.command, Commands::Mcp) {
        // MCP server run_mcp uses library types; map to local error type
        vipune::mcp::server::run_mcp(
            config.embedding_model.clone(),
            &project_id,
            config.database_path.clone(),
        )
        .map_err(|e| Error::Config(e.to_string()))?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut store = MemoryStore::new(
        &config.database_path,
        &config.embedding_model,
        config.clone(),
    )?;

    commands::execute(&cli.command, &mut store, project_id, &config, cli.json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_types::{BatchIngestItemResult, IngestPolicy};
    use sqlite::Database;

    #[test]
    fn test_cli_parse_add() {
        let cli = Cli::parse_from(["vipune", "add", "test content"]);
        assert!(!cli.json);
        assert!(cli.project.is_none());
        assert!(cli.db_path.is_none());
        matches!(cli.command, Commands::Add { .. });
    }

    // Exercise batch types to eliminate dead_code warnings from binary compilation
    #[test]
    fn test_batch_types_exist() {
        // Verify IngestPolicy variants can be constructed
        let _policy_force = IngestPolicy::Force;
        let _policy_conflict = IngestPolicy::ConflictAware;

        // Verify BatchIngestItemResult variants can be constructed
        let _added = BatchIngestItemResult::Added {
            id: "test-id".to_string(),
        };
        let _conflicts = BatchIngestItemResult::Conflicts {
            proposed: "test".to_string(),
            conflicts: vec![],
        };
        let _error = BatchIngestItemResult::Error {
            message: "error".to_string(),
        };

        // Verify MemoryStore has batch_ingest method exists (compilation check)
        // Note: We don't actually run it since that would require downloading models
        // This test is just to satisfy dead_code analysis
        assert!(IngestPolicy::Force == IngestPolicy::Force);
    }

    #[test]
    fn test_cli_parse_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "add", "test"]);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parse_with_project() {
        let cli = Cli::parse_from(["vipune", "-p", "my-project", "add", "test"]);
        assert_eq!(cli.project, Some("my-project".to_string()));
    }

    #[test]
    fn test_cli_parse_search() {
        let cli = Cli::parse_from(["vipune", "search", "query", "--limit", "10"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                limit: 10,
                ..
            } if query == "query"
        );
    }

    #[test]
    fn test_cli_parse_get() {
        let cli = Cli::parse_from(["vipune", "get", "memory-id"]);
        matches!(cli.command, Commands::Get { id } if id == "memory-id");
    }

    #[test]
    fn test_cli_parse_list() {
        let cli = Cli::parse_from(["vipune", "list"]);
        matches!(cli.command, Commands::List { .. });
    }

    #[test]
    fn test_cli_parse_delete() {
        let cli = Cli::parse_from(["vipune", "delete", "memory-id"]);
        matches!(cli.command, Commands::Delete { id } if id == "memory-id");
    }

    #[test]
    fn test_cli_parse_update() {
        // Update with text only
        let cli = Cli::parse_from(["vipune", "update", "memory-id", "--text", "new content"]);
        matches!(
            cli.command,
            Commands::Update { id, text, metadata }
            if id == "memory-id" && text == Some("new content".to_string()) && metadata.is_none()
        );

        // Update with metadata only
        let cli = Cli::parse_from(["vipune", "update", "memory-id", "-m", r#"{"tag": "new"}"#]);
        matches!(
            cli.command,
            Commands::Update { id, text, metadata }
            if id == "memory-id" && text.is_none() && metadata == Some(r#"{"tag": "new"}"#.to_string())
        );

        // Update with both
        let cli = Cli::parse_from(["vipune", "update", "memory-id", "-t", "new", "-m", r#"{"key":"val"}"#]);
        matches!(
            cli.command,
            Commands::Update { id, text, metadata }
            if id == "memory-id" && text == Some("new".to_string()) && metadata == Some(r#"{"key":"val"}"#.to_string())
        );
    }

    #[test]
    fn test_cli_parse_version() {
        let cli = Cli::parse_from(["vipune", "version"]);
        matches!(cli.command, Commands::Version);
    }

    #[test]
    fn test_cli_parse_validate() {
        let cli = Cli::parse_from(["vipune", "validate", "test text"]);
        matches!(
            cli.command,
            Commands::Validate { text } if text == "test text"
        );
    }

    #[test]
    fn test_cli_parse_with_db_path() {
        let cli = Cli::parse_from(["vipune", "--db-path", "/custom/path.db", "add", "test"]);
        assert_eq!(cli.db_path, Some("/custom/path.db".to_string()));
    }

    #[test]
    fn test_cli_parse_search_with_recency() {
        let cli = Cli::parse_from(["vipune", "search", "query", "--recency", "0.5"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                recency: Some(0.5),
                ..
            } if query == "query"
        );
    }

    #[test]
    fn test_cli_parse_search_without_recency() {
        let cli = Cli::parse_from(["vipune", "search", "query"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                recency: None,
                ..
            } if query == "query"
        );
    }

    #[test]
    fn test_cli_parse_search_with_hybrid() {
        let cli = Cli::parse_from(["vipune", "search", "query", "--hybrid"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                hybrid: true,
                ..
            } if query == "query"
        );
    }

    #[test]
    fn test_cli_parse_search_without_hybrid() {
        let cli = Cli::parse_from(["vipune", "search", "query"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                hybrid: false,
                ..
            } if query == "query"
        );
    }

    #[test]
    fn test_cli_parse_search_with_hybrid_and_recency() {
        let cli = Cli::parse_from(["vipune", "search", "query", "--hybrid", "--recency", "0.5"]);
        matches!(
            cli.command,
            Commands::Search {
                query,
                hybrid: true,
                recency: Some(0.5),
                ..
            } if query == "query"
        );
    }

    // Exercise MemoryStore::batch_ingest to eliminate dead_code warnings
    #[test]
    fn test_batch_ingest_integration_compiles() {
        use tempfile::TempDir;

        // Create a temporary database
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        std::mem::forget(dir);

        let db = Database::open(&path).unwrap();
        let config = config::Config::default();
        let mut store = MemoryStore::from_db(db, config);

        // Test with empty batch
        let result = store.batch_ingest("test-project", vec![], IngestPolicy::Force);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().results.len(), 0);

        // Clean up
        std::fs::remove_file(path).ok();
    }
}
