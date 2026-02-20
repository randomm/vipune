//! CLI entry point for vipune memory layer.

mod commands;
mod config;
mod embedding;
mod errors;
mod memory;
mod memory_types;
mod output;
mod project;
mod rrf;
mod sqlite;
mod temporal;

use clap::Parser;
use commands::Commands;
use errors::Error;
use memory::MemoryStore;
use output::{print_json, ErrorResponse};
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
            if cli.json {
                print_json(&ErrorResponse {
                    error: error.to_string(),
                });
            } else {
                eprintln!("Error: {}", error);
            }
            ExitCode::from(1)
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

    #[test]
    fn test_cli_parse_add() {
        let cli = Cli::parse_from(&["vipune", "add", "test content"]);
        assert_eq!(cli.json, false);
        assert!(cli.project.is_none());
        assert!(cli.db_path.is_none());
        matches!(cli.command, Commands::Add { .. });
    }

    #[test]
    fn test_cli_parse_with_json() {
        let cli = Cli::parse_from(&["vipune", "--json", "add", "test"]);
        assert_eq!(cli.json, true);
    }

    #[test]
    fn test_cli_parse_with_project() {
        let cli = Cli::parse_from(&["vipune", "-p", "my-project", "add", "test"]);
        assert_eq!(cli.project, Some("my-project".to_string()));
    }

    #[test]
    fn test_cli_parse_search() {
        let cli = Cli::parse_from(&["vipune", "search", "query", "--limit", "10"]);
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
        let cli = Cli::parse_from(&["vipune", "get", "memory-id"]);
        matches!(cli.command, Commands::Get { id } if id == "memory-id");
    }

    #[test]
    fn test_cli_parse_list() {
        let cli = Cli::parse_from(&["vipune", "list"]);
        matches!(cli.command, Commands::List { .. });
    }

    #[test]
    fn test_cli_parse_delete() {
        let cli = Cli::parse_from(&["vipune", "delete", "memory-id"]);
        matches!(cli.command, Commands::Delete { id } if id == "memory-id");
    }

    #[test]
    fn test_cli_parse_update() {
        let cli = Cli::parse_from(&["vipune", "update", "memory-id", "new content"]);
        matches!(
            cli.command,
            Commands::Update { id, text } if id == "memory-id" && text == "new content"
        );
    }

    #[test]
    fn test_cli_parse_version() {
        let cli = Cli::parse_from(&["vipune", "version"]);
        matches!(cli.command, Commands::Version);
    }

    #[test]
    fn test_cli_parse_with_db_path() {
        let cli = Cli::parse_from(&["vipune", "--db-path", "/custom/path.db", "add", "test"]);
        assert_eq!(cli.db_path, Some("/custom/path.db".to_string()));
    }

    #[test]
    fn test_cli_parse_search_with_recency() {
        let cli = Cli::parse_from(&["vipune", "search", "query", "--recency", "0.5"]);
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
        let cli = Cli::parse_from(&["vipune", "search", "query"]);
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
        let cli = Cli::parse_from(&["vipune", "search", "query", "--hybrid"]);
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
        let cli = Cli::parse_from(&["vipune", "search", "query"]);
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
        let cli = Cli::parse_from(&["vipune", "search", "query", "--hybrid", "--recency", "0.5"]);
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
}
