//! Hook pipeline orchestration (issue #213).
//!
//! Wires together the payload parser, the zero-LLM extractor, project
//! detection via `detect_project_at`, and the dedup-aware `insert_with_hash`
//! path. All non-fatal failures are silently swallowed (exit 0, empty
//! stdout) — the hook must never surface an error to the agent mid-session.
//!
//! The hook path never touches the ONNX embedder. It inserts placeholder
//! embeddings (via `embedding::placeholder_embedding`) that classify as
//! `Mock` so a later `reindex` run backfills them with real vectors.

use std::io::Read;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use crate::config::Config;
use crate::errors::Error;
use crate::hook::embedding::placeholder_embedding;
use crate::hook::extract::extract_candidate;
use crate::hook::payload::{HookEvent, HookPayload, parse_hook_payload};
use crate::project::detect_project_at;
use crate::sqlite::Database;
use crate::sqlite::hash::content_hash;

/// Short busy timeout applied to the hook's database connection.
///
/// If the database is locked (e.g. another vipune command holds a write
/// lock), we wait at most this long and then silently give up — the hook
/// must not block the agent mid-session. A 2-second wait covers the common
/// case where a `prune` or `reindex` is briefly holding the lock.
const HOOK_DB_BUSY_TIMEOUT: Duration = Duration::from_secs(2);

/// Entry point for the hook pipeline. Called by the command surface
/// (`src/commands/hook_run.rs`) with the event type and the full stdin
/// payload.
///
/// Always returns `Ok(ExitCode::SUCCESS)` — any non-fatal failure (malformed
/// payload, DB lock, credential block, empty extraction, unknown event) maps
/// to silent success. The hook must never surface an error to the agent.
pub fn run_hook_event(
    config: &Config,
    event: HookEvent,
    stdin_input: &str,
) -> Result<ExitCode, Error> {
    // 1. Parse payload. SessionStart returns None (no-op).
    let payload: HookPayload = match parse_hook_payload(event, stdin_input) {
        Some(p) => p,
        None => return Ok(ExitCode::SUCCESS),
    };

    // 2. Project scoping: use payload.cwd, not process cwd. If cwd is absent
    //    the candidate is silently dropped (no fallback to process cwd —
    //    that would leak the wrong project's DB into the agent's session).
    let Some(cwd) = &payload.cwd else {
        return Ok(ExitCode::SUCCESS);
    };
    let project_id = detect_project_at(cwd, None);

    // 3. Zero-LLM extraction with credential hard-block.
    let Some(candidate) = extract_candidate(&payload.candidate_source) else {
        return Ok(ExitCode::SUCCESS);
    };

    // 4. Open database with a short busy timeout (fast-fail on lock).
    let db = match open_hook_db(&config.database_path) {
        Some(db) => db,
        None => return Ok(ExitCode::SUCCESS),
    };

    // 5. Insert with placeholder embedding. A UNIQUE constraint violation
    //    (either from a pre-existing row or a concurrent hook event that
    //    raced past any pre-check) is treated as a silent skip — the row
    //    was already inserted by the other invocation, so the outcome is
    //    correct.
    let hash = content_hash(&candidate);
    let embedding = placeholder_embedding(&candidate);
    let result = db.insert_with_hash(
        &project_id,
        &candidate,
        &embedding,
        None,
        "observation",
        "candidate",
        &hash,
    );

    match result {
        Ok(_) => Ok(ExitCode::SUCCESS),
        Err(e) if is_unique_constraint_violation(&e) => {
            // Concurrent race or pre-existing row: another hook invocation
            // inserted the same (project_id, content_hash) pair first.
            // Silent skip — the outcome is correct.
            Ok(ExitCode::SUCCESS)
        }
        Err(_) => {
            // Any other insert failure (I/O, I/O on a locked DB, etc.) is
            // also silent — the hook never surfaces errors to the agent.
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Open the hook database with a short busy timeout.
///
/// Returns `None` on any open failure (path not writable, permissions, etc.)
/// so the caller can silently give up.
fn open_hook_db(db_path: &Path) -> Option<Database> {
    let db = Database::open(db_path).ok()?;
    db.set_busy_timeout(HOOK_DB_BUSY_TIMEOUT).ok()?;
    Some(db)
}

/// Detect a SQLite UNIQUE constraint violation in a `crate::sqlite::Error`.
///
/// `rusqlite` returns `rusqlite::Error::SqliteFailure(ErrorCode::ConstraintViolation, ..)`
/// when an insert violates the `idx_memories_dedup` unique index. The sqlite
/// module wraps that into `Error::Sqlite(msg)` — we match on the canonical
/// "UNIQUE constraint failed" text, which is stable across rusqlite versions.
fn is_unique_constraint_violation(e: &crate::sqlite::Error) -> bool {
    if let crate::sqlite::Error::Sqlite(msg) = e {
        msg.contains("UNIQUE constraint failed")
    } else {
        false
    }
}

/// Read the full stdin into a string, returning empty string on failure.
///
/// The command surface calls this once and passes the result to
/// `run_hook_event`. Reading stdin separately (rather than inside
/// `run_hook_event`) keeps the function pure with respect to I/O — it is
/// easier to test and to reason about.
pub fn read_stdin() -> String {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        String::new()
    } else {
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hook::payload::HookEvent;
    use crate::sqlite::Database;
    use std::sync::Mutex;

    // Mutex to serialise tests that use a shared temp dir pattern.
    static DB_MUTEX: Mutex<()> = Mutex::new(());

    fn make_config_with_tmp_db() -> (Config, tempfile::TempDir, std::path::PathBuf) {
        let _guard = DB_MUTEX.lock().unwrap();
        let dir = tempfile::TempDir::new().expect("temp dir");
        let db_path = dir.path().join(format!("hook_{}.db", uuid::Uuid::new_v4()));
        let config = Config {
            database_path: db_path.clone(),
            ..Default::default()
        };
        (config, dir, db_path)
    }

    fn make_git_repo(tmp: &std::path::Path, name: &str, remote_url: &str) -> std::path::PathBuf {
        let repo_path = tmp.join(name);
        std::fs::create_dir_all(&repo_path).unwrap();
        let git = |args: &[&str]| -> std::process::Output {
            std::process::Command::new("git")
                .args(args)
                .current_dir(&repo_path)
                .env("GIT_AUTHOR_NAME", "test")
                .env("GIT_AUTHOR_EMAIL", "test@test.com")
                .env("GIT_COMMITTER_NAME", "test")
                .env("GIT_COMMITTER_EMAIL", "test@test.com")
                .env("GIT_TERMINAL_PROMPT", "0")
                .output()
                .expect("git ran")
        };
        let init = git(&["init", "-q"]);
        assert!(init.status.success(), "git init failed: {:?}", init);
        let _ = git(&["remote", "add", "origin", remote_url]);
        let _ = git(&["commit", "-q", "--allow-empty", "-m", "init"]);
        repo_path
    }

    // ---- end-to-end pipeline tests ----

    #[test]
    fn user_prompt_submit_extracts_prompt() {
        let (config, tmp, db_path) = make_config_with_tmp_db();
        let repo = make_git_repo(tmp.path(), "repo-a", "git@github.com:owner/repo-a.git");

        let payload = format!(
            r#"{{"prompt": "use cargo fmt before committing", "cwd": "{}", "session_id": "s1"}}"#,
            repo.display()
        );

        let exit = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload)
            .expect("hook should succeed");
        assert_eq!(exit, ExitCode::SUCCESS);

        let db = Database::open(&db_path).expect("open db");
        let count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project_id = 'owner/repo-a'",
                [],
                |r| r.get(0),
            )
            .expect("count query");
        assert_eq!(count, 1, "one row should be inserted for one candidate");
    }

    #[test]
    fn session_start_is_no_op() {
        let (config, _tmp, db_path) = make_config_with_tmp_db();

        let payload = r#"{"session_id": "s1", "cwd": "/tmp"}"#;
        let exit = run_hook_event(&config, HookEvent::SessionStart, payload).expect("ok");
        assert_eq!(exit, ExitCode::SUCCESS);

        let db = Database::open(&db_path).expect("open db");
        let count: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 0, "SessionStart must not insert any row");
    }

    #[test]
    fn jwt_payload_produces_zero_inserts() {
        let (config, tmp, db_path) = make_config_with_tmp_db();
        let repo = make_git_repo(tmp.path(), "repo-jwt", "git@github.com:owner/repo-jwt.git");

        let jwt =
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozj9NabvVH5hU0R3qTc";
        let payload = format!(
            r#"{{"prompt": "here is my token {jwt} please save it", "cwd": "{}"}}"#,
            repo.display()
        );

        let exit = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload).expect("ok");
        assert_eq!(exit, ExitCode::SUCCESS);

        let db = Database::open(&db_path).expect("open db");
        let count: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 0, "JWT-bearing payload must produce zero inserts");
    }

    #[test]
    fn malformed_stdin_exits_zero() {
        let (config, _tmp, _db_path) = make_config_with_tmp_db();

        let exit = run_hook_event(
            &config,
            HookEvent::UserPromptSubmit,
            "not valid json at all",
        )
        .expect("malformed stdin must exit 0");
        assert_eq!(exit, ExitCode::SUCCESS);
    }

    #[test]
    fn empty_prompt_exits_zero() {
        let (config, tmp, db_path) = make_config_with_tmp_db();
        let repo = make_git_repo(
            tmp.path(),
            "repo-empty",
            "git@github.com:owner/repo-empty.git",
        );

        let payload = format!(r#"{{"prompt": "   ", "cwd": "{}"}}"#, repo.display());
        let exit = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload).expect("ok");
        assert_eq!(exit, ExitCode::SUCCESS);

        let db = Database::open(&db_path).expect("open db");
        let count: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 0, "empty/whitespace-only prompt must not insert");
    }

    #[test]
    fn dedup_skips_second_identical_insert() {
        let (config, tmp, db_path) = make_config_with_tmp_db();
        let repo = make_git_repo(
            tmp.path(),
            "repo-dedup",
            "git@github.com:owner/repo-dedup.git",
        );

        let payload = format!(
            r#"{{"prompt": "always use rustfmt before committing", "cwd": "{}"}}"#,
            repo.display()
        );

        let e1 = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload).expect("ok");
        let e2 = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload).expect("ok");
        assert_eq!(e1, ExitCode::SUCCESS);
        assert_eq!(e2, ExitCode::SUCCESS);

        let db = Database::open(&db_path).expect("open db");
        let count: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 1, "identical content twice must yield one row");
    }

    #[test]
    fn project_scoping_uses_payload_cwd_not_process_cwd() {
        let (config, tmp, db_path) = make_config_with_tmp_db();
        let repo_a = make_git_repo(tmp.path(), "repo-a", "git@github.com:owner/repo-a.git");
        let repo_b = make_git_repo(tmp.path(), "repo-b", "git@github.com:owner/repo-b.git");

        let payload_a = format!(
            r#"{{"prompt": "candidate from repo A", "cwd": "{}"}}"#,
            repo_a.display()
        );
        let payload_b = format!(
            r#"{{"prompt": "candidate from repo B", "cwd": "{}"}}"#,
            repo_b.display()
        );

        let _ = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload_a).expect("ok");
        let _ = run_hook_event(&config, HookEvent::UserPromptSubmit, &payload_b).expect("ok");

        let db = Database::open(&db_path).expect("open db");
        let count_a: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project_id = 'owner/repo-a'",
                [],
                |r| r.get(0),
            )
            .expect("count A");
        let count_b: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project_id = 'owner/repo-b'",
                [],
                |r| r.get(0),
            )
            .expect("count B");
        assert_eq!(count_a, 1, "payload A should land in repo-a project");
        assert_eq!(count_b, 1, "payload B should land in repo-b project");
    }

    #[test]
    fn pre_tool_use_extracts_tool_input_object() {
        let (config, tmp, db_path) = make_config_with_tmp_db();
        let repo = make_git_repo(tmp.path(), "repo-pre", "git@github.com:owner/repo-pre.git");

        let payload = format!(
            r#"{{"tool_name": "Bash", "tool_input": {{"command": "git status"}}, "cwd": "{}"}}"#,
            repo.display()
        );

        let exit = run_hook_event(&config, HookEvent::PreToolUse, &payload).expect("ok");
        assert_eq!(exit, ExitCode::SUCCESS);

        let db = Database::open(&db_path).expect("open db");
        let count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE project_id = 'owner/repo-pre'",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(
            count, 1,
            "PreToolUse with tool_input object should insert one row"
        );
    }
}
