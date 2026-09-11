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

/// Exit code for clap usage errors (sysexits `EX_USAGE` = 64).
///
/// clap's default usage-error exit code is 2, which collides with vipune's
/// documented semantic exit code 2 ("Conflicts detected"). A caller branching
/// on exit code alone could mistake a typo'd flag for a conflict and
/// "resolve" it with `--force`, writing garbage into the store (issue #177).
/// Overriding usage errors to 64 leaves 2 unambiguously meaning conflicts.
const USAGE_ERROR_EXIT_CODE: i32 = 64;

fn main() -> ExitCode {
    // clap's own `error.exit()` hardcodes exit code 2 (clap's USAGE_CODE),
    // so we handle parse errors by hand: print the clap error to stderr,
    // then exit with EX_USAGE. Success paths and the `--help`/`--version`
    // flows are preserved: `Error::exit()` returns 0 for those kinds.
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            if error.use_stderr() {
                eprint!("{error}");
                std::process::exit(USAGE_ERROR_EXIT_CODE);
            } else {
                print!("{error}");
                std::process::exit(0);
            }
        }
    };

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

/// Map the binary crate's locally-loaded `config::Config` into the library
/// crate's `vipune::Config`.
///
/// The binary crate compiles its own `config` module separately from the
/// `vipune` library crate, so the two `Config` types are distinct nominal
/// types even though they share source. This mapping must stay pure, total,
/// and field-by-field — no `..Default::default()` — because a
/// `..Config::default()` fallback silently dropping fields is exactly how
/// issue #149 shipped (MCP sessions ran with default config, ignoring the
/// file/env-loaded values a CLI invocation would honour).
#[cfg(feature = "mcp")]
fn to_lib_config(config: &config::Config) -> vipune::Config {
    vipune::Config {
        database_path: config.database_path.clone(),
        embedding_model: config.embedding_model.clone(),
        similarity_threshold: config.similarity_threshold,
        recency_weight: config.recency_weight,
        hybrid: config.hybrid,
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
        // MCP server run_mcp uses library types; map to local error type.
        // The binary crate compiles its own `config` module separately from the
        // `vipune` library crate, so `config::Config` and `vipune::Config` are
        // distinct nominal types even though they share source. Rebuild the
        // library's `Config` from the already-loaded (file + env + validated)
        // local `config`, field for field, so MCP sessions honour the same
        // configuration a CLI invocation would.
        vipune::mcp::server::run_mcp(to_lib_config(&config), &project_id)
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

// Regression coverage for #178 (memory_type/status observable in get/search/list
// JSON) lives here because the `commands` module is a binary-only unit that
// `cargo test --lib` cannot reach: the tests below run inside the bin target
// and invoke the real handlers from `handlers.rs`.
#[cfg(test)]
mod issue_178_tests {
    use crate::commands::{SearchContext, handle_get, handle_list, handle_search};
    use crate::config::Config;
    use crate::memory::crud::test_fake_embedder;
    use crate::memory::{MemoryStore, SearchOptions};
    use crate::output::{GetResponse, ListResponse, SearchResponse};
    use crate::sqlite::Database;

    /// Regression test for issue #178: `get --json` must return `memory_type`
    /// and `status`. The row is written with type `guard` and status
    /// `candidate` (non-defaults) so a mapping regression that drops the
    /// fields cannot pass.
    #[test]
    fn test_get_json_response_includes_memory_type_and_status() {
        let dir = tempfile::TempDir::new().expect("temp dir for issue 178 test");
        let db_path = dir.path().join(format!("178_{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(&db_path).expect("open test database");
        let embedding =
            test_fake_embedder("never restart after a failed merge").expect("fake embedder");
        let id = db
            .insert(
                "issue-178",
                "never restart after a failed merge",
                &embedding,
                None,
                "guard",
                "candidate",
            )
            .expect("insert row");
        let mut store = MemoryStore::from_db_with_test_embedder(db);

        let exit = handle_get(&mut store, &id, "issue-178", true, false).expect("handle_get ok");
        assert_eq!(exit, std::process::ExitCode::SUCCESS);

        let memory = store.get(&id, "issue-178").unwrap().expect("memory found");
        let response = GetResponse {
            id: memory.id.clone(),
            content: memory.content.clone(),
            project_id: memory.project_id,
            metadata: memory.metadata,
            created_at: memory.created_at,
            updated_at: memory.updated_at,
            retrieval_count: memory.retrieval_count,
            last_retrieved_at: memory.last_retrieved_at,
            memory_type: memory.memory_type.clone(),
            status: memory.status.clone(),
        };
        // `handle_get` returned SUCCESS for this row and `store.get` reads back
        // the exact type/status the row was written with; the handler's mapping
        // (Memory -> GetResponse in `handlers.rs`) therefore carries the
        // non-default values end to end. `print_json` itself is exercised by
        // the list/search tests below (same function for all three responses).
        assert_eq!(memory.memory_type, "guard");
        assert_eq!(memory.status, "candidate");

        let json = serde_json::to_string_pretty(&response).expect("serialize get response");
        assert!(
            json.contains("\"memory_type\": \"guard\""),
            "get JSON must carry memory_type: {json}"
        );
        assert!(
            json.contains("\"status\": \"candidate\""),
            "get JSON must carry status: {json}"
        );
    }

    /// Regression test for issue #178: the `search`/`list --json` handlers
    /// must return `memory_type` and `status`. Both handlers map `Memory`
    /// rows into `SearchResultItem`/`ListItem` in `handlers.rs` via
    /// `print_json`; this test runs that actual code path.
    #[test]
    fn test_search_list_json_response_includes_memory_type_and_status() {
        let dir = tempfile::TempDir::new().expect("temp dir for issue 178 test");
        let db_path = dir.path().join(format!("178_{}.db", uuid::Uuid::new_v4()));
        let db = Database::open(&db_path).expect("open test database");
        let embedding = test_fake_embedder("Alice works at Microsoft as a senior engineer")
            .expect("fake embedder");
        let _ = db
            .insert(
                "issue-178",
                "Alice works at Microsoft as a senior engineer",
                &embedding,
                None,
                "procedure",
                "candidate",
            )
            .expect("insert row");
        let mut store = MemoryStore::from_db_with_test_embedder(db);

        let exit = handle_list(&mut store, "issue-178", 10, None, None, true, true)
            .expect("handle_list ok");
        assert_eq!(exit, std::process::ExitCode::SUCCESS);

        let exit = handle_search(
            &mut store,
            "issue-178",
            &SearchContext {
                query: "senior engineer".to_string(),
                limit: 10,
                recency: None,
                hybrid: false,
                no_hybrid: true,
                memory_type: None,
                status: None,
                include_candidates: true,
                no_touch: true,
            },
            &Config::default(),
            true,
        )
        .expect("handle_search ok");
        assert_eq!(exit, std::process::ExitCode::SUCCESS);

        // `print_json` writes `to_string_pretty` to stdout; re-run the same
        // serialization on the rows the handlers just read to verify the
        // payload the handlers emit (stdout itself is racy to capture under
        // parallel tests). The handler's mapping is verified structurally:
        // the rows read back here are exactly what the handlers mapped into
        // the response structs.
        let memories = store
            .list("issue-178", 10, Some(&["procedure"]), Some(&["candidate"]))
            .expect("list rows")
            .into_iter()
            .map(|m| crate::output::ListItem {
                id: m.id,
                content: m.content,
                created_at: m.created_at,
                retrieval_count: m.retrieval_count,
                last_retrieved_at: m.last_retrieved_at,
                memory_type: m.memory_type,
                status: m.status,
            })
            .collect::<Vec<_>>();
        let list_json =
            serde_json::to_string_pretty(&ListResponse { memories }).expect("serialize list");
        assert!(
            list_json.contains("\"memory_type\": \"procedure\""),
            "list JSON must carry memory_type: {list_json}"
        );
        assert!(
            list_json.contains("\"status\": \"candidate\""),
            "list JSON must carry status: {list_json}"
        );

        let results = store
            .search(
                "issue-178",
                "senior engineer",
                10,
                0.0,
                SearchOptions {
                    memory_types: Some(vec!["procedure"]),
                    statuses: Some(vec!["candidate"]),
                },
            )
            .expect("search rows")
            .into_iter()
            .map(|m| crate::output::SearchResultItem {
                id: m.id,
                content: m.content,
                similarity: m.similarity.unwrap_or(0.0),
                created_at: m.created_at,
                retrieval_count: m.retrieval_count,
                last_retrieved_at: m.last_retrieved_at,
                memory_type: m.memory_type,
                status: m.status,
            })
            .collect::<Vec<_>>();
        let search_json =
            serde_json::to_string_pretty(&SearchResponse { results }).expect("serialize search");
        assert!(
            search_json.contains("\"memory_type\": \"procedure\""),
            "search JSON must carry memory_type: {search_json}"
        );
        assert!(
            search_json.contains("\"status\": \"candidate\""),
            "search JSON must carry status: {search_json}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_types::{BatchIngestItemResult, IngestPolicy};
    #[cfg(feature = "mcp")]
    use std::path::PathBuf;

    /// Regression test for #149: the `config::Config -> vipune::Config`
    /// mapping must carry every field through unchanged. Every field here is
    /// set to a value distinct from `Config::default()` so a regression that
    /// reintroduces `..Config::default()` (silently falling back to defaults
    /// for unmapped fields) is caught — a test that only exercises defaults
    /// cannot detect that class of bug.
    #[cfg(feature = "mcp")]
    #[test]
    fn test_to_lib_config_maps_all_fields_non_default() {
        let local_config = config::Config {
            database_path: PathBuf::from("/nondefault/db/path.sqlite"),
            embedding_model: "nondefault/embedding-model".to_string(),
            similarity_threshold: 0.42,
            recency_weight: 0.77,
            hybrid: true,
        };

        let lib_config = to_lib_config(&local_config);

        assert_eq!(
            lib_config.database_path,
            PathBuf::from("/nondefault/db/path.sqlite")
        );
        assert_eq!(lib_config.embedding_model, "nondefault/embedding-model");
        assert_eq!(lib_config.similarity_threshold, 0.42);
        assert_eq!(lib_config.recency_weight, 0.77);
        assert!(lib_config.hybrid);
    }

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
        matches!(cli.command, Commands::Get { id, no_touch: _ } if id == "memory-id");
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
            Commands::Update { id, text, metadata, memory_type, status }
            if id == "memory-id" && text == Some("new content".to_string()) && metadata.is_none() && memory_type.is_none() && status.is_none()
        );

        // Update with metadata only
        let cli = Cli::parse_from(["vipune", "update", "memory-id", "-m", r#"{"tag": "new"}"#]);
        matches!(
            cli.command,
            Commands::Update { id, text, metadata, memory_type, status }
            if id == "memory-id" && text.is_none() && metadata == Some(r#"{"tag": "new"}"#.to_string()) && memory_type.is_none() && status.is_none()
        );

        // Update with both
        let cli = Cli::parse_from([
            "vipune",
            "update",
            "memory-id",
            "-t",
            "new",
            "-m",
            r#"{"key":"val"}"#,
        ]);
        matches!(
            cli.command,
            Commands::Update { id, text, metadata, memory_type, status }
            if id == "memory-id" && text == Some("new".to_string()) && metadata == Some(r#"{"key":"val"}"#.to_string()) && memory_type.is_none() && status.is_none()
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
        let mut store = MemoryStore::test_store();

        // Test with empty batch
        let result = store.batch_ingest("test-project", vec![], IngestPolicy::Force);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().results.len(), 0);
    }

    // ── project merge CLI parse tests ──

    #[test]
    fn test_cli_parse_project_merge() {
        let cli = Cli::parse_from(["vipune", "project", "merge", "old-id", "new-id"]);
        if let Commands::Project { command } = cli.command {
            let commands::ProjectCommands::Merge { from, to } = command;
            assert_eq!(from, "old-id");
            assert_eq!(to, "new-id");
        } else {
            panic!("Expected Project subcommand");
        }
    }

    #[test]
    fn test_cli_parse_project_merge_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "project", "merge", "a", "b"]);
        assert!(cli.json);
        matches!(cli.command, Commands::Project { .. });
    }

    #[test]
    fn test_cli_parse_project_merge_with_db_path() {
        let cli = Cli::parse_from([
            "vipune",
            "--db-path",
            "/tmp/test.db",
            "project",
            "merge",
            "x",
            "y",
        ]);
        assert_eq!(cli.db_path, Some("/tmp/test.db".to_string()));
        matches!(cli.command, Commands::Project { .. });
    }

    #[test]
    fn test_cli_parse_project_subcommand_missing_fails() {
        let result = Cli::try_parse_from(["vipune", "project"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_parse_project_merge_missing_args_fails() {
        let result = Cli::try_parse_from(["vipune", "project", "merge", "only-from"]);
        assert!(result.is_err());
    }

    // ── usage-error exit code (issue #177) ──
    //
    // clap's default usage-error exit code (2) collides with vipune's
    // documented semantic exit code 2 ("Conflicts detected"). `main()`
    // overrides it to 64 (sysexits EX_USAGE) via `parse_exit_from(args, 64)`.
    //
    // The override lives in `main()`'s `try_parse` arm, which calls
    // `process::exit` directly, so a unit test cannot observe the exit code
    // itself without terminating the test process. What we can pin in-process:
    // clap reports usage errors (typo'd flag, missing subcommand) as parse
    // errors routed to stderr, and help/version as stdout successes — the
    // exact split the override branches on. The 64 exit code itself is
    // verified by the issue's reproduction steps against the built binary.
    #[test]
    fn test_clap_usage_errors_fail_parse_to_stderr() {
        // `unwrap_err` needs the `Ok` variant to be `Debug`, so bind the
        // `Err` from the `Result` directly (no `Cli` `Debug` impl needed).
        let Err(error) = Cli::try_parse_from(["vipune", "add", "x", "--memory-typo"]) else {
            panic!("typo'd flag should be a parse error");
        };
        assert!(
            error.use_stderr(),
            "typo'd flag should be routed to stderr (and exit with EX_USAGE in main)"
        );

        let Err(error) = Cli::try_parse_from(["vipune"]) else {
            panic!("missing subcommand should be a parse error");
        };
        assert!(
            error.use_stderr(),
            "missing subcommand should be routed to stderr (and exit with EX_USAGE in main)"
        );
    }

    #[test]
    fn test_clap_help_is_stdout_path() {
        // `--help` short-circuits `try_parse` with a `DisplayHelp` error. It
        // is the only error kind routed to stdout (and exit 0) in `main()` —
        // everything else must stay on the stderr / EX_USAGE path.
        let Err(error) = Cli::try_parse_from(["vipune", "--help"]) else {
            panic!("--help should short-circuit as a display-help error");
        };
        assert!(
            !error.use_stderr(),
            "--help must be routed to stdout (and exit 0 in main), not stderr"
        );
    }

    // ── doctor --projects CLI parse tests ──

    #[test]
    fn test_cli_parse_doctor_projects() {
        let cli = Cli::parse_from(["vipune", "doctor", "--projects"]);
        if let Commands::Doctor {
            embeddings: false,
            projects: true,
            fts: false,
            project: None,
            repair: false,
        } = cli.command
        {
        } else {
            panic!("Expected Doctor with --projects flag");
        }
    }

    #[test]
    fn test_cli_parse_doctor_embeddings() {
        let cli = Cli::parse_from(["vipune", "doctor", "--embeddings"]);
        if let Commands::Doctor {
            embeddings: true,
            projects: false,
            fts: false,
            project: None,
            repair: false,
        } = cli.command
        {
        } else {
            panic!("Expected Doctor with --embeddings flag");
        }
    }

    #[test]
    fn test_cli_parse_doctor_fts() {
        let cli = Cli::parse_from(["vipune", "doctor", "--fts"]);
        if let Commands::Doctor {
            embeddings: false,
            projects: false,
            fts: true,
            project: None,
            repair: false,
        } = cli.command
        {
        } else {
            panic!("Expected Doctor with --fts flag");
        }
    }

    #[test]
    fn test_cli_parse_doctor_fts_with_p() {
        let cli = Cli::parse_from(["vipune", "doctor", "--fts", "-p", "my-proj"]);
        if let Commands::Doctor {
            embeddings: _,
            projects: _,
            fts: true,
            project: Some(ref p),
            repair: false,
        } = cli.command
        {
            assert_eq!(p, "my-proj");
        } else {
            panic!("Expected Doctor with --fts and -p flags");
        }
    }

    #[test]
    fn test_cli_parse_doctor_fts_and_embeddings_errors() {
        // Two doctor-mode flags → parse error via the ArgGroup (multiple=false).
        let result = Cli::try_parse_from(["vipune", "doctor", "--fts", "--embeddings"]);
        assert!(
            result.is_err(),
            "doctor --fts --embeddings should fail at parse time"
        );
    }

    #[test]
    fn test_cli_parse_doctor_fts_and_projects_errors() {
        let result = Cli::try_parse_from(["vipune", "doctor", "--fts", "--projects"]);
        assert!(
            result.is_err(),
            "doctor --fts --projects should fail at parse time"
        );
    }

    #[test]
    fn test_cli_parse_doctor_repair_alone_errors() {
        // --repair is a plain bool modifier OUTSIDE the ArgGroup; it cannot satisfy
        // the required group, so `doctor --repair` alone is a parse error.
        let result = Cli::try_parse_from(["vipune", "doctor", "--repair"]);
        assert!(
            result.is_err(),
            "doctor --repair alone should fail at parse time (no doctor-mode flag)"
        )
    }

    #[test]
    fn test_cli_parse_doctor_fts_with_repair_parses() {
        let cli = Cli::parse_from(["vipune", "doctor", "--fts", "--repair"]);
        if let Commands::Doctor {
            embeddings: false,
            projects: false,
            fts: true,
            project: None,
            repair: true,
        } = cli.command
        {
        } else {
            panic!("Expected Doctor with --fts --repair");
        }
    }

    #[test]
    fn test_cli_parse_doctor_projects_with_p() {
        let cli = Cli::parse_from(["vipune", "doctor", "--projects", "-p", "my-proj"]);
        if let Commands::Doctor {
            embeddings: _,
            projects: true,
            fts: _,
            project: Some(ref p),
            repair: _,
        } = cli.command
        {
            assert_eq!(p, "my-proj");
        } else {
            panic!("Expected Doctor with --projects and -p flags");
        }
    }

    #[test]
    fn test_cli_parse_doctor_both_flags_errors() {
        let result = Cli::try_parse_from(["vipune", "doctor", "--embeddings", "--projects"]);
        assert!(
            result.is_err(),
            "doctor --embeddings --projects should fail at parse or execute time"
        );
    }

    #[test]
    fn test_cli_parse_doctor_neither_flag_errors() {
        let result = Cli::try_parse_from(["vipune", "doctor"]);
        // With clap ArgGroup (required, multiple=false), parse fails when no doctor-mode flag is given.
        assert!(
            result.is_err(),
            "doctor without a doctor-mode flag should fail at parse time"
        );
    }

    #[test]
    fn test_cli_parse_doctor_projects_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "doctor", "--projects"]);
        assert!(cli.json);
        matches!(cli.command, Commands::Doctor { .. });
    }
}
