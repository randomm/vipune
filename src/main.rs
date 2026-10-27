//! CLI entry point for vipune memory layer.
//!
//! The binary declares **no** modules: the command surface and its response
//! DTOs live in the library (`vipune::commands`, and `vipune::commands::output`
//! for the JSON DTOs) and everything else is imported from the library crate,
//! so no library module is compiled a second time into the bin target (issue
//! #267). The `vipune::` items below are the very same nominal types the library
//! exposes, which is also what lets `run` hand the loaded config straight to
//! `run_mcp` without a field-by-field mapping.

use clap::Parser;
use std::process::ExitCode;
use vipune::Config;
use vipune::commands;
use vipune::commands::Commands;
use vipune::commands::HookEvent;
use vipune::commands::handle_hook_event;
use vipune::commands::output::{ErrorResponse, print_json};
use vipune::errors::Error;
use vipune::memory::MemoryStore;
use vipune::project::detect_project;

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

/// Map a `Commands::Hook` subcommand to its `HookEvent` counterpart, if it is
/// one of the five event subcommands. Returns `None` for `Install` and
/// `Uninstall` (those go through the normal command path) and for non-hook
/// commands.
fn hook_event_from_command(command: &Commands) -> Option<commands::HookEvent> {
    match command {
        Commands::Hook {
            command: commands::HookCommands::SessionStart,
        } => Some(HookEvent::SessionStart),
        Commands::Hook {
            command: commands::HookCommands::UserPromptSubmit,
        } => Some(HookEvent::UserPromptSubmit),
        Commands::Hook {
            command: commands::HookCommands::PreToolUse,
        } => Some(HookEvent::PreToolUse),
        Commands::Hook {
            command: commands::HookCommands::PostToolUse,
        } => Some(HookEvent::PostToolUse),
        Commands::Hook {
            command: commands::HookCommands::PreCompact,
        } => Some(HookEvent::PreCompact),
        _ => None,
    }
}

fn run(cli: &Cli) -> Result<ExitCode, Error> {
    let mut config = Config::load()?;
    config.ensure_directories()?;

    if let Some(db_path) = &cli.db_path {
        config.database_path = db_path.clone().into();
    }

    let project_id = detect_project(cli.project.as_deref());

    // Handle hook event subcommands separately — the hook path must NEVER
    // load the ONNX model. The MCP early-return above is the existing
    // precedent for this pattern. We intercept all five event subcommands
    // (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PreCompact)
    // but NOT Install/Uninstall (those run via the normal path and don't
    // touch the DB or the embedder).
    if let Some(event) = hook_event_from_command(&cli.command) {
        return handle_hook_event(&config, cli.json, event);
    }

    // Handle MCP command separately (doesn't use MemoryStore directly)
    #[cfg(feature = "mcp")]
    if matches!(cli.command, Commands::Mcp) {
        // MCP server run_mcp uses library types; map to local error type.
        // The bin and the library share one `Config` type now (issue #267), so
        // the already-loaded (file + env + validated) config is what MCP
        // sessions get — the same values a CLI invocation honours. The field
        // completeness that `to_lib_config` used to guarantee by hand is
        // structural: there is no second type to map onto.
        vipune::mcp::server::run_mcp(config, &project_id)
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

    /// Regression coverage for #149 — MCP sessions must honour the file/env
    /// config a CLI invocation honours, not `Config::default()`.
    ///
    /// That used to be a field-by-field mapping (`to_lib_config`) whose every
    /// field this test had to set non-default to catch a `..Config::default()`
    /// fallback. The bin and the library share one `Config` type now (issue
    /// #267), so `run` hands the loaded struct to `run_mcp` directly and there
    /// is no mapping left to drop a field. What still needs pinning — and what
    /// this test now pins — is that the struct keeps every knob the CLI and MCP
    /// both read: a field removed from `Config` compiles here but fails the
    /// exhaustive destructuring below.
    #[cfg(feature = "mcp")]
    #[test]
    fn test_config_is_the_single_type_shared_with_mcp() {
        let config = Config {
            database_path: std::path::PathBuf::from("/nondefault/db/path.sqlite"),
            embedding_model: "nondefault/embedding-model".to_string(),
            similarity_threshold: 0.42,
            recency_weight: 0.77,
            hybrid: true,
            decay_refresh_days: 14.0,
            promotion_threshold: 7,
            prune_retrieval_limit: 3,
            prune_min_age_days: 9.5,
        };

        // Exhaustive destructuring: adding a field to `Config` without wiring it
        // through makes this fail to compile, which is the guard #149 needed.
        let Config {
            database_path,
            embedding_model,
            similarity_threshold,
            recency_weight,
            hybrid,
            decay_refresh_days,
            promotion_threshold,
            prune_retrieval_limit,
            prune_min_age_days,
        } = config;

        assert_eq!(
            database_path,
            std::path::PathBuf::from("/nondefault/db/path.sqlite")
        );
        assert_eq!(embedding_model, "nondefault/embedding-model");
        assert_eq!(similarity_threshold, 0.42);
        assert_eq!(recency_weight, 0.77);
        assert!(hybrid);
        assert_eq!(decay_refresh_days, 14.0);
        assert_eq!(promotion_threshold, 7);
        assert_eq!(prune_retrieval_limit, 3);
        assert_eq!(prune_min_age_days, 9.5);
    }

    #[test]
    fn test_cli_parse_add() {
        let cli = Cli::parse_from(["vipune", "add", "test content"]);
        assert!(!cli.json);
        assert!(cli.project.is_none());
        assert!(cli.db_path.is_none());
        matches!(cli.command, Commands::Add { .. });
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
            Commands::Update { id, text, metadata, memory_type, status, importance }
            if id == "memory-id" && text == Some("new content".to_string()) && metadata.is_none() && memory_type.is_none() && status.is_none() && importance.is_none()
        );

        // Update with metadata only
        let cli = Cli::parse_from(["vipune", "update", "memory-id", "-m", r#"{"tag": "new"}"#]);
        matches!(
            cli.command,
            Commands::Update { id, text, metadata, memory_type, status, importance }
            if id == "memory-id" && text.is_none() && metadata == Some(r#"{"tag": "new"}"#.to_string()) && memory_type.is_none() && status.is_none() && importance.is_none()
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
            Commands::Update { id, text, metadata, memory_type, status, importance }
            if id == "memory-id" && text == Some("new".to_string()) && metadata == Some(r#"{"key":"val"}"#.to_string()) && memory_type.is_none() && status.is_none() && importance.is_none()
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

    // ── backup CLI parse tests ──

    #[test]
    fn test_cli_parse_backup() {
        let cli = Cli::parse_from(["vipune", "backup"]);
        matches!(cli.command, Commands::Backup { output: None });
    }

    #[test]
    fn test_cli_parse_backup_with_output() {
        let cli = Cli::parse_from(["vipune", "backup", "--output", "/tmp/backup.db"]);
        matches!(
            cli.command,
            Commands::Backup {
                output: Some(p)
            } if p.to_string_lossy() == "/tmp/backup.db"
        );
    }

    #[test]
    fn test_cli_parse_backup_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "backup"]);
        assert!(cli.json);
        matches!(cli.command, Commands::Backup { .. });
    }

    #[test]
    fn test_cli_parse_backup_with_db_path() {
        let cli = Cli::parse_from(["vipune", "--db-path", "/tmp/seeded.db", "backup"]);
        assert_eq!(cli.db_path, Some("/tmp/seeded.db".to_string()));
        matches!(cli.command, Commands::Backup { .. });
    }

    #[test]
    fn test_cli_parse_doctor_projects_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "doctor", "--projects"]);
        assert!(cli.json);
        matches!(cli.command, Commands::Doctor { .. });
    }

    // ── export CLI parse tests ──

    #[test]
    fn test_cli_parse_export() {
        let cli = Cli::parse_from(["vipune", "export", "/tmp/out.jsonl"]);
        matches!(
            cli.command,
            Commands::Export {
                ref output_path
            } if output_path == "/tmp/out.jsonl"
        );
    }

    #[test]
    fn test_cli_parse_export_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "export", "out.jsonl"]);
        assert!(cli.json);
        matches!(cli.command, Commands::Export { .. });
    }

    #[test]
    fn test_cli_parse_export_with_db_path() {
        let cli = Cli::parse_from([
            "vipune",
            "--db-path",
            "/tmp/seeded.db",
            "export",
            "out.jsonl",
        ]);
        assert_eq!(cli.db_path, Some("/tmp/seeded.db".to_string()));
        matches!(cli.command, Commands::Export { .. });
    }

    #[test]
    fn test_cli_parse_export_with_stray_project_parses_but_is_ignored() {
        // The global --project flag parses; the handler ignores it (with a
        // stderr warning) because export is cross-project by contract.
        let cli = Cli::parse_from(["vipune", "-p", "my-proj", "export", "out.jsonl"]);
        assert_eq!(cli.project, Some("my-proj".to_string()));
        matches!(cli.command, Commands::Export { .. });
    }

    #[test]
    fn test_cli_parse_export_missing_output_path_fails() {
        let result = Cli::try_parse_from(["vipune", "export"]);
        assert!(result.is_err());
    }

    // ── import CLI parse tests (issue #195) ──

    #[test]
    fn test_cli_parse_import_with_source() {
        let cli = Cli::parse_from(["vipune", "import", "/tmp/export.jsonl"]);
        if let Commands::Import { source } = cli.command {
            assert_eq!(source, Some("/tmp/export.jsonl".to_string()));
        } else {
            panic!("Expected Import subcommand");
        }
    }

    #[test]
    fn test_cli_parse_import_stdin() {
        let cli = Cli::parse_from(["vipune", "import", "-"]);
        if let Commands::Import { source } = cli.command {
            assert_eq!(source, Some("-".to_string()));
        } else {
            panic!("Expected Import subcommand");
        }
    }

    #[test]
    fn test_cli_parse_import_no_source_defaults_to_stdin() {
        let cli = Cli::parse_from(["vipune", "import"]);
        if let Commands::Import { source } = cli.command {
            assert!(source.is_none(), "no source should default to stdin");
        } else {
            panic!("Expected Import subcommand");
        }
    }

    #[test]
    fn test_cli_parse_import_with_json() {
        let cli = Cli::parse_from(["vipune", "--json", "import", "-"]);
        assert!(cli.json);
        matches!(cli.command, Commands::Import { .. });
    }

    #[test]
    fn test_cli_parse_import_with_db_path() {
        let cli = Cli::parse_from(["vipune", "--db-path", "/tmp/test.db", "import", "-"]);
        assert_eq!(cli.db_path, Some("/tmp/test.db".to_string()));
        matches!(cli.command, Commands::Import { .. });
    }

    #[test]
    fn test_cli_parse_import_with_project_flag_parses_but_is_ignored() {
        // --project is a global flag that parses, but import ignores it (with a
        // stderr warning at execute time). Parse must succeed.
        let cli = Cli::parse_from(["vipune", "--project", "my-proj", "import", "-"]);
        assert_eq!(cli.project, Some("my-proj".to_string()));
        matches!(cli.command, Commands::Import { .. });
    }

    // ── hook CLI parse tests (issue #191) ──

    #[test]
    fn test_cli_parse_hook_install() {
        let cli = Cli::parse_from(["vipune", "hook", "install"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::Install);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_uninstall() {
        let cli = Cli::parse_from(["vipune", "hook", "uninstall"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::Uninstall);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_session_start() {
        let cli = Cli::parse_from(["vipune", "hook", "session-start"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::SessionStart);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_user_prompt_submit() {
        let cli = Cli::parse_from(["vipune", "hook", "user-prompt-submit"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::UserPromptSubmit);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_pre_tool_use() {
        let cli = Cli::parse_from(["vipune", "hook", "pre-tool-use"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::PreToolUse);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_post_tool_use() {
        let cli = Cli::parse_from(["vipune", "hook", "post-tool-use"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::PostToolUse);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_pre_compact() {
        let cli = Cli::parse_from(["vipune", "hook", "pre-compact"]);
        if let Commands::Hook { command } = cli.command {
            matches!(command, commands::HookCommands::PreCompact);
        } else {
            panic!("Expected Hook subcommand");
        }
    }

    #[test]
    fn test_cli_parse_hook_missing_subcommand_fails() {
        let result = Cli::try_parse_from(["vipune", "hook"]);
        assert!(
            result.is_err(),
            "hook without subcommand should fail at parse time"
        );
    }

    #[test]
    fn test_cli_parse_hook_with_db_path() {
        let cli = Cli::parse_from(["vipune", "--db-path", "/tmp/test.db", "hook", "install"]);
        assert_eq!(cli.db_path, Some("/tmp/test.db".to_string()));
        matches!(cli.command, Commands::Hook { .. });
    }
}
