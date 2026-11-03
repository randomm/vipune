//! The subprocess + fixture half of the `vipune version` CLI contract tests
//! (see the parent `version_cli_tests` module for the issue #330 story).
//!
//! Split from the assertions file so each stays under the repo's 500-line cap:
//! this one owns the child-process harness (`run_child`, the binary locator,
//! the env neutralisation) and the two fixtures the AC tests need — the
//! unwritable database parent and the future-schema database.

use crate::commands::output::ErrorResponse;
use crate::config::tests_utils::CONFIG_ENV_VARS;
use crate::sqlite::Database;
use crate::sqlite::migrations;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::OnceLock;

/// The knobs besides the config list that must not reach a child: the two
/// `project.rs::detect_project` reads (outside `CONFIG_ENV_VARS`), plus `HOME`
/// and `XDG_CONFIG_HOME`, which `Config::load()` resolves through
/// `dirs::config_dir()` to find `vipune/config.toml`; `HF_HOME`, which the
/// embedding engine's cache-dir resolver reads directly, together with the two
/// HuggingFace credential knobs the embedding pipeline can pick up from the
/// developer's shell; and `VIPUNE_PROMOTION_THRESHOLD`, which
/// `commands/promote.rs` reads (and which is already in `CONFIG_ENV_VARS`).
/// Removing these by default is what keeps a child from silently reading the
/// developer's own config file or credentials; a test that wants a config file
/// sets them explicitly.
const INHERITED_ENV_VARS_EXCLUDED: &[&str] = &[
    "VIPUNE_PROJECT",
    "VIPUNE_DEBUG",
    "HOME",
    "XDG_CONFIG_HOME",
    "HF_HOME",
    "HF_TOKEN",
    "HF_HUB_TOKEN",
];

/// The exit code a child produced, or `None` when a signal killed it — a
/// distinct outcome from `exit(-1)`, which a real child can produce.
pub(crate) type ChildExit = Option<i32>;

/// Path to the compiled `vipune` binary (the test harness builds it).
///
/// Resolved once per test binary and cached: `current_exe()` is fixed for the
/// process, so the walk-up is a property of the harness layout rather than of
/// each child. The first call is still the premise check — if the binary is not
/// next to the harness, every child in the module would fail on it anyway.
fn vipune_bin() -> &'static std::path::PathBuf {
    static BIN: OnceLock<std::path::PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let exe = std::env::current_exe().expect("current_exe");
        let mut path = exe;
        // Walk up from target/debug/deps/<name>-<hash> to target/debug/vipune.
        for _ in 0..4 {
            if let Some(parent) = path.parent() {
                let candidate = parent.join("vipune");
                if candidate.exists() {
                    return candidate;
                }
                path = parent.to_path_buf();
            }
        }
        panic!("could not locate the vipune binary for subprocess capture");
    })
}

/// Spawn `vipune <args…>` as a child with captured stdout/stderr (stdin from
/// `/dev/null`) and return its exit status plus both streams.
///
/// `env` holds the knobs the test wants the child to *see* (the removals below
/// are unconditional); `Command` applies them after `env_remove`, so a set knob
/// wins over the neutralisation.
///
/// The shared env mutex is held only across `spawn()` — the call that snapshots
/// the parent's environ, which races a sibling's `set_var` — and released before
/// `wait_with_output()`, so the child's whole run does not sit inside the
/// critical section every env-writer in this target must wait for (see
/// `docs/testing.md`; the flake behind the rule is issue #233).
///
/// Callers must not hold `ConfigEnvGuard`: `ENV_MUTEX` is not reentrant.
pub(crate) fn run_child(args: &[&str], env: &[(&str, &str)]) -> (ChildExit, String, String) {
    let mut cmd = std::process::Command::new(vipune_bin());
    cmd.args(args);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    for var in CONFIG_ENV_VARS
        .iter()
        .chain(INHERITED_ENV_VARS_EXCLUDED.iter())
    {
        cmd.env_remove(var);
    }
    // Project detection runs in `run()`'s prologue before any command here can
    // fail on its own inputs; pinning it keeps the child from forking `git`
    // (whose absence would surface as a `GitError::Spawn`) for an assertion that
    // says nothing about project detection.
    cmd.env("VIPUNE_PROJECT", "version-cli-tests");
    for (name, value) in env {
        cmd.env(name, value);
    }

    let child = {
        let _lock = crate::config::tests_utils::lock_env_mutex();
        cmd.spawn().expect("spawn vipune")
    };
    let output = child.wait_with_output().expect("wait for vipune");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    // `code()` is None for a signal death; keeping it as `None` rather than
    // collapsing to a sentinel int keeps `exit(0)`, `exit(-1)` and "killed by a
    // signal" distinguishable for the assertions in the sibling module.
    (output.status.code(), stdout, stderr)
}

/// Assert a child exited with `expected` (rather than by signal), with both
/// streams in the failure message.
pub(crate) fn assert_exit(expected: i32, code: ChildExit, stdout: &str, stderr: &str, what: &str) {
    assert_eq!(
        Some(expected),
        code,
        "{what}: expected exit {expected}, got {code:?}; stdout {stdout:?}, stderr {stderr:?}"
    );
}

/// The human-mode half of AC6 for one data command: exit 1, the `Configuration
/// error: Failed to create database directory` framing (which is
/// `ensure_directories`' own boundary — `MemoryStore::new`'s file open renders
/// as a `Database error` instead, so the text also says *which* prologue step
/// rejected the path), and nothing on stdout.
pub(crate) fn assert_fails_human(args: &[&str], expected_word: &str, db_path: &str) {
    let (code, stdout, stderr) = run_child(args, &[("VIPUNE_DATABASE_PATH", db_path)]);

    assert_exit(1, code, &stdout, &stderr, expected_word);
    assert!(
        stderr.contains("Configuration error")
            && stderr.contains("Failed to create database directory"),
        "{expected_word} must keep the existing Configuration error text, got stderr: {stderr:?}"
    );
    assert!(
        stdout.is_empty(),
        "{expected_word} must print the error on stderr, not stdout"
    );
}

/// The JSON half of AC6 for one data command: exit 1 and the same error on
/// stdout, decoded through the CLI's own [`ErrorResponse`] DTO so a field
/// rename is a compile error rather than a silent `None`.
pub(crate) fn assert_fails_json(args: &[&str], expected_word: &str, db_path: &str) {
    let mut json_args = vec!["--json"];
    json_args.extend_from_slice(args);
    let (code, stdout, stderr) = run_child(&json_args, &[("VIPUNE_DATABASE_PATH", db_path)]);

    assert_exit(
        1,
        code,
        &stdout,
        &stderr,
        &format!("{expected_word} --json"),
    );
    let parsed: ErrorResponse = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("{expected_word} --json error must be an ErrorResponse, got {stdout:?}: {e}")
    });
    assert!(
        parsed.error.contains("Configuration error"),
        "{expected_word} --json must report the Configuration error, got: {stdout:?}"
    );
}

/// An unwritable temp parent directory — the "unusable database path" fixture.
/// `Config::ensure_directories()` calls `create_dir_all` on the database's
/// parent, so pointing the child at `<blocked>/nested/x.db` makes the prologue
/// fail with `Configuration error: Failed to create database directory …` and
/// (before the fix) exit 1.
///
/// Restoring the write bits is the `Drop` of this type, not a trailing call in
/// the test: a failing assertion above the fixture's use must not leave a
/// `0o500` tree that `TempDir`'s cleanup cannot remove (leftover scratch is a
/// defect, AGENTS.md "Disk hygiene").
pub(crate) struct BlockedParent {
    dir: tempfile::TempDir,
}

impl BlockedParent {
    pub(crate) fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Build the fixture, or `None` when the write bit turned out not to be
    /// enforced — for the current user (root bypasses it) or by the filesystem
    /// (NFS exports, some containers). The caller then skips its unusable-path
    /// assertion rather than passing vacuously on a premise that does not hold;
    /// the probe is what keeps the `version` assertions from becoming "the
    /// prologue succeeded because the prologue was never actually broken".
    ///
    /// One probe suffices: `ensure_directories`' `create_dir_all` is the step
    /// that trips first, and a `File::create` probe would test the same write
    /// bit on the same directory.
    ///
    /// A chmod that fails for a *different* reason (immutable attribute,
    /// read-only mount) is a hard failure with the path and errno named, not a
    /// silent skip: those are host problems a reviewer needs told apart from an
    /// unenforceable permission bit.
    fn new() -> Option<Self> {
        let dir = tempfile::TempDir::new().expect("temp dir for unwritable parent");
        let path = dir.path().to_path_buf();
        // Best-effort restore before the panic: a `0o500` tree is one `TempDir`
        // cleanup cannot undo, and Drop does not run on a panic through here.
        let chmod =
            |mode: u32| std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode));
        // Hard failure, not a skip: a chmod that errors for any reason other
        // than the fs not enforcing it is a host problem, and a fixture that
        // silently kept its write bits would make the probe below pass for the
        // wrong reason.
        if let Err(e) = chmod(0o500) {
            let _ = chmod(0o700);
            panic!("chmod the fixture parent {}: {e}", path.display());
        }

        // Probe: a directory must NOT be creatable inside it.
        if std::fs::create_dir(path.join("probe")).is_ok() {
            return None;
        }
        Some(Self { dir })
    }
}

impl Drop for BlockedParent {
    fn drop(&mut self) {
        if let Err(error) =
            std::fs::set_permissions(self.dir.path(), std::fs::Permissions::from_mode(0o700))
        {
            eprintln!(
                "could not restore write bits on {}: {error}",
                self.dir.path().display()
            );
        }
    }
}

/// Build the unusable-path fixture for a test, or `None` when this host cannot
/// enforce the write bit — root, or some NFS/container mounts — in which case
/// the caller skips rather than passes vacuously, and the reason is printed so
/// a reviewer can tell the skip from a pass.
pub(crate) fn unwritable_db_path_fixture(ac: &str) -> Option<BlockedParent> {
    let Some(blocked) = BlockedParent::new() else {
        eprintln!(
            "skipping {ac}: the filesystem honours writes for this user, so the unusable-path premise cannot be established"
        );
        return None;
    };
    Some(blocked)
}

/// Build a database whose `PRAGMA user_version` sits one above the ceiling this
/// binary supports, so `Database::open` on the same file refuses it.
///
/// `create_schema` is all `IF NOT EXISTS`, so raising the pragma by hand leaves
/// a fully migrated schema behind — the fixture is a real database that merely
/// claims a newer version, not a corrupt file. The pragma is raised through
/// `Database::set_user_version_for_test`, the storage layer that owns the
/// migration contract, rather than through `Database::conn()`.
pub(crate) fn future_schema_db(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("future.db");
    let mut db = Database::open(&path).expect("open fixture database");
    db.set_user_version_for_test(migrations::ceiling_plus_one())
        .expect("raise user_version");
    drop(db);

    // Premise: the same file must be refused by the binary's own open path, or
    // this fixture would pin nothing.
    assert!(
        Database::open(&path).is_err(),
        "the fixture must be refused by Database::open"
    );
    path
}
