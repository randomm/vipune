//! `vipune version` / `vipune --help` CLI contract (issue #330).
//!
//! `version` is a diagnostic: it reports the binary's name and version and
//! nothing else, so it must keep working when the database path, the config
//! file or the schema is unusable. Until #330 the binary failed here — `run()`
//! called `Config::load()`, `Config::ensure_directories()` and
//! `MemoryStore::new()` before dispatching, so an unusable path, a malformed
//! `VIPUNE_*` value or a future `PRAGMA user_version` turned `vipune version`
//! into exit 1 and *created* the database file it was only meant to report on.
//! The fix short-circuits `Commands::Version` above those three calls in
//! `src/main.rs`; these tests drive the compiled binary as a child process,
//! because every tier of the bug lives in the `run()` prologue and is invisible
//! to an in-process call of `handle_version`.
//!
//! Subprocess capture (rather than in-process fd swapping) is used for the same
//! reason as `reindex_cli_tests`: the test harness's own output reader conflicts
//! with in-process fd swaps.
//!
//! AC1–AC6 map one-to-one onto the `#[test]` fns below, named for the criterion
//! they pin. AC6's other half — hook events and `mcp` are untouched by the
//! reordering — is not re-pinned here: `commands::hook_install_tests` and the
//! `--db-path` wiring tests in `commands::reindex_cli_tests` already drive those
//! paths through the same `run()` prologue, which the reordering leaves in place.

#![cfg(test)]

use crate::config::tests_utils::CONFIG_ENV_VARS;
use crate::sqlite::Database;
use crate::sqlite::migrations;
use std::os::unix::fs::PermissionsExt;

/// The knobs besides the config list that must not reach a child: the two
/// `project.rs::detect_project` reads (outside `CONFIG_ENV_VARS`), plus `HOME`
/// and `XDG_CONFIG_HOME`, which `Config::load()` resolves through
/// `dirs::config_dir()` to find `vipune/config.toml`. Removing the last two by
/// default is what keeps a child from silently reading the developer's own
/// config file; a test that wants a config file sets them explicitly.
const INHERITED_ENV_VARS_EXCLUDED: &[&str] =
    &["VIPUNE_PROJECT", "VIPUNE_DEBUG", "HOME", "XDG_CONFIG_HOME"];

/// The exact human-mode stdout `vipune version` must produce.
fn expected_version_line() -> String {
    format!("{} {}\n", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

/// Path to the compiled `vipune` binary (the test harness builds it).
fn vipune_bin() -> std::path::PathBuf {
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
}

/// How a spawned child's environment is built.
///
/// Every knob the parent might hold is removed, so a sibling test's malformed
/// `VIPUNE_SIMILARITY_THRESHOLD=-1` cannot leak into the child's
/// `Config::load()` and turn a clean pass into a non-zero exit. A test that
/// *wants* a knob to be seen sets it explicitly via [`ChildEnv::set`] (which
/// wins over the removal — `Command` applies `env` after `env_remove`).
struct ChildEnv {
    set: Vec<(String, String)>,
}

impl ChildEnv {
    /// Neutralise every inherited knob — the default for these tests.
    fn clean() -> Self {
        Self { set: Vec::new() }
    }

    /// Set one variable in the child (the knob under test).
    fn set(&mut self, name: &str, value: &str) -> &mut Self {
        self.set.push((name.to_string(), value.to_string()));
        self
    }
}

/// Spawn `vipune <args…>` as a child with captured stdout/stderr (stdin from
/// `/dev/null`) and return its exit code plus both streams.
///
/// The shared env mutex is held only across `spawn()` — the call that snapshots
/// the parent's environ, which races a sibling's `set_var` — and released before
/// `wait_with_output()`, so the child's whole run does not sit inside the
/// critical section every env-writer in this target must wait for (see
/// `docs/testing.md`; the flake behind the rule is issue #233).
///
/// Callers must not hold `ConfigEnvGuard`: `ENV_MUTEX` is not reentrant.
fn run_child(args: &[&str], env: &ChildEnv) -> (i32, String, String) {
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
    for (name, value) in &env.set {
        cmd.env(name, value);
    }

    let child = {
        let _lock = crate::config::tests_utils::lock_env_mutex();
        cmd.spawn().expect("spawn vipune")
    };
    let output = child.wait_with_output().expect("wait for vipune");
    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (code, stdout, stderr)
}

/// A temp parent directory made unwritable — the "unusable database path"
/// fixture. `Config::ensure_directories()` calls `create_dir_all` on the
/// database's parent, so pointing the child at `<blocked>/nested/x.db` makes the
/// prologue fail with `Configuration error: Failed to create database directory
/// …` and (before the fix) exit 1.
///
/// `None` when the write bit turned out not to be enforced — for the current
/// user (root bypasses it) or by the filesystem (NFS exports, some containers).
/// The caller then skips its unusable-path assertion rather than passing
/// vacuously on a premise that does not hold; the probe is what keeps the
/// `version` assertions from becoming "the prologue succeeded because the
/// prologue was never actually broken".
fn unwritable_parent() -> Option<tempfile::TempDir> {
    let dir = tempfile::TempDir::new().expect("temp dir for unwritable parent");
    // Best-effort: drop the write bits so a create inside the parent fails.
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o500))
        .expect("chmod the fixture parent");

    // Probe: a directory must NOT be creatable inside it.
    if std::fs::create_dir(dir.path().join("probe")).is_ok() {
        restore_writable(dir.path());
        return None;
    }
    // Second probe: a *file* must not be creatable either, which is what the
    // CLI's `MemoryStore::new` does once the directory exists — a filesystem
    // that honours mkdir-denial but allows `open(O_CREAT)` would let the
    // data-command assertions pass for the wrong reason, so those tests require
    // both halves. `version` is unaffected either way (it creates nothing).
    if std::fs::File::create(dir.path().join("probe.db")).is_ok() {
        restore_writable(dir.path());
        return None;
    }
    Some(dir)
}

/// Restore the write bits on a fixture parent so `TempDir`'s cleanup can remove
/// the tree (a `0o500` directory blocks its own removal's bookkeeping on some
/// filesystems, and the tests below assert before restoring).
fn restore_writable(path: &std::path::Path) {
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
}

/// Build a database whose `PRAGMA user_version` sits one above the ceiling this
/// binary supports, so `Database::open` on the same file refuses it.
///
/// `create_schema` is all `IF NOT EXISTS`, so raising the pragma by hand leaves
/// a fully migrated schema behind — the fixture is a real database that merely
/// claims a newer version, not a corrupt file.
fn future_schema_db(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("future.db");
    let db = Database::open(&path).expect("open fixture database");
    let ceiling = migrations::ceiling();
    db.conn()
        .pragma_update(None, "user_version", ceiling + 1)
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

/// AC1: `VIPUNE_DATABASE_PATH=<unusable path> vipune version` exits 0 with the
/// exact `"{name} {version}\n"` on stdout, nothing on stderr, and creates
/// neither the parent directory nor the database file.
#[test]
fn test_ac1_version_with_unusable_db_path_is_db_free() {
    let Some(blocked) = unwritable_parent() else {
        eprintln!(
            "skipping AC1: the filesystem honours writes for this user, so the unusable-path premise cannot be established"
        );
        return;
    };
    let nested = blocked.path().join("nested");
    let db_path = nested.join("x.db");

    let mut env = ChildEnv::clean();
    env.set("VIPUNE_DATABASE_PATH", &db_path.to_string_lossy());
    let (code, stdout, stderr) = run_child(&["version"], &env);

    assert_eq!(code, 0, "version must exit 0, stderr was: {stderr:?}");
    assert_eq!(
        stdout,
        expected_version_line(),
        "stdout must be exactly `{{name}} {{version}}` plus a newline"
    );
    assert!(stderr.is_empty(), "stderr must be empty, got: {stderr:?}");
    assert!(
        !nested.exists(),
        "the db parent must not be created by `version`"
    );
    assert!(
        !db_path.exists(),
        "the database file must not be created by `version`"
    );

    restore_writable(blocked.path());
}

/// AC1, second half of the "creates" bug: with a *writable* but absent parent,
/// `version` must not create the directory tree or the database either. (Before
/// the fix this fixture produced both, silently, while still exiting 0 — so the
/// exit code alone did not pin it.)
#[test]
fn test_ac1_version_does_not_create_a_writable_db_path() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let nested = dir.path().join("deep").join("nested");
    let db_path = nested.join("x.db");

    let mut env = ChildEnv::clean();
    env.set("VIPUNE_DATABASE_PATH", &db_path.to_string_lossy());
    let (code, stdout, stderr) = run_child(&["version"], &env);

    assert_eq!(code, 0, "version must exit 0, stderr was: {stderr:?}");
    assert_eq!(stdout, expected_version_line());
    assert!(stderr.is_empty(), "stderr must be empty, got: {stderr:?}");
    assert!(!nested.exists(), "`version` must not create the db parent");
    assert!(
        !db_path.exists(),
        "`version` must not create the database file"
    );
}

/// AC2: `vipune --json version` against an unusable path exits 0 and prints a
/// JSON object with `version` / `name`. Byte layout is not asserted — only the
/// parsed shape, per the criterion.
#[test]
fn test_ac2_json_version_with_unusable_db_path() {
    let Some(blocked) = unwritable_parent() else {
        eprintln!("skipping AC2: unusable-path premise unavailable for this user");
        return;
    };
    // A file inside the blocked parent must not be creatable either: that is
    // `MemoryStore::new`'s side of the prologue (the sqlite file itself), which
    // the `nested/` directory alone only breaks for `ensure_directories`.
    assert!(
        std::fs::File::create(blocked.path().join("probe.db")).is_err(),
        "premise: the blocked parent must refuse file creation as well"
    );
    let db_path = blocked.path().join("nested").join("x.db");

    let mut env = ChildEnv::clean();
    env.set("VIPUNE_DATABASE_PATH", &db_path.to_string_lossy());
    let (code, stdout, stderr) = run_child(&["--json", "version"], &env);

    assert_eq!(
        code, 0,
        "--json version must exit 0, stderr was: {stderr:?}"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json version stdout must parse as JSON");
    let obj = parsed.as_object().expect("JSON object");
    assert_eq!(obj["version"].as_str(), Some(env!("CARGO_PKG_VERSION")));
    assert_eq!(obj["name"].as_str(), Some(env!("CARGO_PKG_NAME")));

    restore_writable(blocked.path());
}

/// AC3: `vipune version` exits 0 against a database file whose `user_version`
/// is one above this binary's ceiling — while `Database::open` on that very
/// file refuses it (asserted in the fixture, so the test proves `version`
/// skipped the open rather than that the file was never unusual).
///
/// Deliberately *no* claim that another command fails on this fixture: the
/// schema is fully migrated, so a data command like `list` succeeds on it.
#[test]
fn test_ac3_version_with_future_schema_exits_zero() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let db_path = future_schema_db(&dir);

    let mut env = ChildEnv::clean();
    env.set("VIPUNE_DATABASE_PATH", &db_path.to_string_lossy());
    let (code, stdout, stderr) = run_child(&["version"], &env);

    assert_eq!(
        code, 0,
        "version must not touch the schema; stderr was: {stderr:?}"
    );
    assert_eq!(stdout, expected_version_line());
    assert!(stderr.is_empty(), "stderr must be empty, got: {stderr:?}");
}

/// AC4: `vipune version` exits 0 with a malformed `VIPUNE_*` value, an
/// unrecognised `VIPUNE_EMBEDDING_MODEL`, and an unparseable `config.toml` —
/// all three at once, since each independently kills `Config::load()`.
///
/// `VIPUNE_EMBEDDING_MODEL` is a *validating* knob (an unknown id is refused by
/// `profile_for`) but not a *fatal* one at load time unless the model is
/// resolved, so it is pinned separately below; the two genuinely fatal knobs —
/// a bad float and a bad TOML file — are pinned together here. Each of the three
/// is asserted to be fatal for a data command in
/// `test_ac6_data_commands_keep_failing_on_bad_config`, which is what makes
/// "version exits 0" a statement about the short-circuit rather than about
/// inputs that nothing would have rejected.
#[test]
fn test_ac4_version_survives_malformed_env_and_config_file() {
    // A config.toml that is not valid TOML, under a redirected HOME *and*
    // XDG_CONFIG_HOME: `loader::load_from_file` reads `dirs::config_dir()`,
    // which on Linux honours `XDG_CONFIG_HOME` and only falls back to
    // `$HOME/.config` when it is unset. Redirecting only HOME would leave the
    // child reading the developer's own config file and the pass would prove
    // nothing.
    let home = tempfile::TempDir::new().expect("temp home");
    let config_dir = home.path().join("config").join("vipune");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "this is not valid toml [[[\n",
    )
    .expect("write malformed config.toml");

    let blocked = unwritable_parent();
    let (db_path, cleanup) = match &blocked {
        Some(dir) => (dir.path().join("nested").join("x.db"), Some(dir)),
        None => (home.path().join("absent").join("x.db"), None),
    };

    let mut env = ChildEnv::clean();
    env.set("VIPUNE_DATABASE_PATH", &db_path.to_string_lossy());
    env.set("VIPUNE_SIMILARITY_THRESHOLD", "-1");
    env.set("VIPUNE_EMBEDDING_MODEL", "nope/not-a-builtin");
    env.set("HOME", &home.path().to_string_lossy());
    env.set(
        "XDG_CONFIG_HOME",
        &home.path().join("config").to_string_lossy(),
    );

    let (code, stdout, stderr) = run_child(&["version"], &env);

    assert_eq!(
        code, 0,
        "version must survive a bad env value, an unknown model and an unparseable config.toml; stderr was: {stderr:?}"
    );
    assert_eq!(stdout, expected_version_line());
    assert!(stderr.is_empty(), "stderr must be empty, got: {stderr:?}");

    if let Some(dir) = cleanup {
        restore_writable(dir.path());
    }
}

/// AC5: `vipune --help` still exits 0 with help text on stdout — the
/// non-regression half of the reordering (clap short-circuits `--help` in
/// `try_parse`, so it never reached the prologue, and no subprocess pin for it
/// existed).
#[test]
fn test_ac5_help_still_exits_zero_with_help_on_stdout() {
    let env = ChildEnv::clean();
    let (code, stdout, stderr) = run_child(&["--help"], &env);

    assert_eq!(code, 0, "--help must exit 0, stderr was: {stderr:?}");
    assert!(
        stdout.contains("vipune"),
        "help text must be printed on stdout, got: {stdout:?}"
    );
    assert!(
        stdout.contains("version"),
        "the subcommand list must be in the help text"
    );
    assert!(stderr.is_empty(), "--help must write nothing to stderr");
}

/// AC6: the data commands keep failing exactly as before with an unusable
/// database path — exit 1, the existing `Configuration error` text on stderr,
/// and the JSON error on stdout under `--json`.
#[test]
fn test_ac6_data_commands_still_fail_on_unusable_db_path() {
    let Some(blocked) = unwritable_parent() else {
        eprintln!("skipping AC6: unusable-path premise unavailable for this user");
        return;
    };
    // Both probes of `unwritable_parent` must hold for these assertions to say
    // anything: `list`/`search`/`add` fail here through `ensure_directories`
    // (the `nested/` create) and, if that were somehow allowed, through
    // `MemoryStore::new` opening the file.
    assert!(
        std::fs::File::create(blocked.path().join("probe.db")).is_err(),
        "premise: the blocked parent must refuse file creation as well"
    );
    let db_path = blocked.path().join("nested").join("x.db");

    for (args, expected_word) in [
        (["list"].as_slice(), "list"),
        (["search", "query"].as_slice(), "search"),
        (["add", "some content"].as_slice(), "add"),
    ] {
        let mut env = ChildEnv::clean();
        env.set("VIPUNE_DATABASE_PATH", &db_path.to_string_lossy());
        let argv: Vec<&str> = args.to_vec();
        let (code, stdout, stderr) = run_child(&argv, &env);

        assert_eq!(
            code, 1,
            "{expected_word} must still exit 1 on an unusable path"
        );
        assert!(
            stderr.contains("Configuration error")
                && stderr.contains("Failed to create database directory"),
            "{expected_word} must keep the existing Configuration error text, got stderr: {stderr:?}"
        );
        assert!(
            stdout.is_empty(),
            "{expected_word} must print the error on stderr, not stdout"
        );

        // JSON mode: the error is reported as JSON on stdout (that is how the
        // CLI renders any error under --json), still exit 1.
        let mut json_args = vec!["--json"];
        json_args.extend_from_slice(&argv);
        let (code, stdout, _stderr) = run_child(&json_args, &env);
        assert_eq!(code, 1, "{expected_word} --json must still exit 1");
        let parsed: serde_json::Value =
            serde_json::from_str(&stdout).expect("--json error output must parse as JSON");
        let message = parsed["error"].as_str().unwrap_or_default();
        assert!(
            message.contains("Configuration error"),
            "{expected_word} --json must report the Configuration error, got: {stdout:?}"
        );
    }

    restore_writable(blocked.path());
}

/// AC6, second premise: the malformed inputs AC4 pins are genuinely fatal for
/// the *other* commands, so AC4's exit 0 comes from the short-circuit and not
/// from inputs nothing would have rejected. Same three knobs, same fixtures,
/// driven through `list` instead of `version`.
#[test]
fn test_ac6_data_commands_keep_failing_on_bad_config() {
    let home = tempfile::TempDir::new().expect("temp home");
    let config_dir = home.path().join("config").join("vipune");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "this is not valid toml [[[\n",
    )
    .expect("write malformed config.toml");

    // An unknown model id alone: fatal through the normal path (validate →
    // profile_for) but not something `version` can see.
    let mut unknown_model = ChildEnv::clean();
    unknown_model.set("VIPUNE_EMBEDDING_MODEL", "nope/not-a-builtin");
    let (code, _stdout, stderr) = run_child(&["list"], &unknown_model);
    assert_eq!(code, 1, "an unknown embedding model must still fail `list`");
    assert!(
        stderr.contains("Unknown embedding model"),
        "expected the unknown-model error, got: {stderr:?}"
    );

    // A malformed float alone.
    let mut bad_threshold = ChildEnv::clean();
    bad_threshold.set("VIPUNE_SIMILARITY_THRESHOLD", "-1");
    let (code, _stdout, stderr) = run_child(&["list"], &bad_threshold);
    assert_eq!(
        code, 1,
        "a malformed similarity threshold must still fail `list`"
    );
    assert!(
        stderr.contains("Invalid similarity threshold"),
        "expected the threshold error, got: {stderr:?}"
    );

    // An unparseable config.toml alone.
    let mut bad_file = ChildEnv::clean();
    bad_file.set("HOME", &home.path().to_string_lossy());
    bad_file.set(
        "XDG_CONFIG_HOME",
        &home.path().join("config").to_string_lossy(),
    );
    let (code, _stdout, stderr) = run_child(&["list"], &bad_file);
    assert_eq!(code, 1, "an unparseable config.toml must still fail `list`");
    assert!(
        stderr.contains("Failed to parse config file"),
        "expected the config-parse error, got: {stderr:?}"
    );

    // And each of the three is survivable for `version` individually, which is
    // the per-knob form of AC4.
    let (code, stdout, stderr) = run_child(&["version"], &unknown_model);
    assert_eq!(
        code, 0,
        "version must survive an unknown model, stderr {stderr:?}"
    );
    assert_eq!(stdout, expected_version_line());
    let (code, stdout, stderr) = run_child(&["version"], &bad_threshold);
    assert_eq!(
        code, 0,
        "version must survive a bad threshold, stderr {stderr:?}"
    );
    assert_eq!(stdout, expected_version_line());
    let (code, stdout, stderr) = run_child(&["version"], &bad_file);
    assert_eq!(
        code, 0,
        "version must survive a bad config.toml, stderr {stderr:?}"
    );
    assert_eq!(stdout, expected_version_line());
}
