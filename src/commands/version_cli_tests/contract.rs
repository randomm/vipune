//! The AC assertions of the `vipune version` CLI contract (issue #330); the
//! child harness and fixtures are in the sibling `child` module.

use super::child::{
    assert_exit, assert_fails_human, assert_fails_json, run_child, unwritable_db_path_fixture,
};
use super::expected_version_line;
use crate::commands::output::version_output;
use crate::sqlite::Database;

/// AC1: `VIPUNE_DATABASE_PATH=<unusable path> vipune version` exits 0 with the
/// exact `"{name} {version}\n"` on stdout, nothing on stderr, and creates
/// neither the parent directory nor the database file.
#[test]
pub(crate) fn version_with_unusable_db_path_is_db_free() {
    let Some(blocked) = unwritable_db_path_fixture("AC1") else {
        return;
    };
    let nested = blocked.path().join("nested");
    let db_path = nested.join("x.db");
    let db_path_str = db_path.to_string_lossy().to_string();

    let (code, stdout, stderr) = run_child(&["version"], &[("VIPUNE_DATABASE_PATH", &db_path_str)]);

    assert_exit(0, code, &stdout, &stderr, "AC1 version");
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
}

/// AC1, second half of the "creates" bug: with a *writable* but absent parent,
/// `version` must not create the directory tree or the database either. (Before
/// the fix this fixture produced both, silently, while still exiting 0 — so the
/// exit code alone did not pin it.)
#[test]
fn version_does_not_create_a_writable_db_path() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let nested = dir.path().join("deep").join("nested");
    let db_path = nested.join("x.db");
    let db_path_str = db_path.to_string_lossy().to_string();

    let (code, stdout, stderr) = run_child(&["version"], &[("VIPUNE_DATABASE_PATH", &db_path_str)]);

    assert_exit(0, code, &stdout, &stderr, "AC1 writable-path version");
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
/// parsed shape, per the criterion, decoded as a `serde_json::Value` (this is
/// output the CLI *prints*, not a DTO the CLI consumes, so there is no type to
/// decode it through).
#[test]
pub(crate) fn json_version_with_unusable_db_path() {
    let Some(blocked) = unwritable_db_path_fixture("AC2") else {
        return;
    };
    // The premise, stated rather than re-checked: `Some(blocked)` already proves
    // the parent refuses `mkdir` *and* `open(O_CREAT)` — `BlockedParent::new`
    // returns `None` if either probe succeeded. `MemoryStore::new`'s side of the
    // prologue (the sqlite file itself) is the half the `nested/` directory alone
    // only breaks for `ensure_directories`.
    let db_path = blocked.path().join("nested").join("x.db");
    let db_path_str = db_path.to_string_lossy().to_string();

    let (code, stdout, stderr) = run_child(
        &["--json", "version"],
        &[("VIPUNE_DATABASE_PATH", &db_path_str)],
    );

    assert_exit(0, code, &stdout, &stderr, "AC2 --json version");
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json version stdout must parse as JSON");
    let obj = parsed.as_object().expect("JSON object");
    let expected = version_output();
    assert_eq!(obj["version"].as_str(), Some(expected.version.as_str()));
    assert_eq!(obj["name"].as_str(), Some(expected.name.as_str()));
}

/// AC3: `vipune version` exits 0 against a database file whose `user_version`
/// is one above this binary's ceiling, and both sides of that boundary are
/// asserted here — `Database::open` accepts the file *at* the ceiling and
/// refuses the fixture above it — so the test proves `version` skipped the open
/// rather than that the file was never unusual. The boundary is not left solely
/// to the fixture's premise assert, which could be deleted without turning the
/// test red.
///
/// Deliberately *no* claim that another command fails on this fixture: the
/// schema is fully migrated, so a data command like `list` succeeds on it.
#[test]
fn version_with_future_schema_exits_zero() {
    let dir = tempfile::TempDir::new().expect("temp dir");

    // The boundary itself, asserted in the test body.
    let at_ceiling = dir.path().join("at-ceiling.db");
    assert!(
        Database::open(&at_ceiling).is_ok(),
        "premise: a database at the ceiling must open"
    );
    let db_path = super::child::future_schema_db(&dir);

    let db_path_str = db_path.to_string_lossy().to_string();
    let (code, stdout, stderr) = run_child(&["version"], &[("VIPUNE_DATABASE_PATH", &db_path_str)]);

    assert_exit(0, code, &stdout, &stderr, "AC3 version");
    assert_eq!(stdout, expected_version_line());
    assert!(stderr.is_empty(), "stderr must be empty, got: {stderr:?}");
}

/// AC4: `vipune version` exits 0 with a malformed `VIPUNE_*` value, an
/// unrecognised `VIPUNE_EMBEDDING_MODEL`, and an unparseable `config.toml` —
/// all three at once, since each independently kills `Config::load()`.
///
/// The database path here is a plain temp path: the unusable-path half is
/// pinned by the AC1/AC2 fixtures, and this test's subject is the config/env
/// tier, which reads no database at all.
///
/// `VIPUNE_EMBEDDING_MODEL` is a *validating* knob (an unknown id is refused by
/// `profile_for`) but not a *fatal* one at load time unless the model is
/// resolved, so it is pinned separately below; the two genuinely fatal knobs —
/// a bad float and a bad TOML file — are pinned together here. Each of the three
/// is asserted to be fatal for a data command in
/// [`data_commands_keep_failing_on_bad_config`], which is what makes "version
/// exits 0" a statement about the short-circuit rather than about inputs that
/// nothing would have rejected.
#[test]
fn version_survives_malformed_env_and_config_file() {
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

    let db_path = home.path().join("absent").join("x.db");
    let db_path_str = db_path.to_string_lossy().to_string();
    let home_str = home.path().to_string_lossy().to_string();
    let xdg_str = home.path().join("config").to_string_lossy().to_string();

    let (code, stdout, stderr) = run_child(
        &["version"],
        &[
            ("VIPUNE_DATABASE_PATH", &db_path_str),
            ("VIPUNE_SIMILARITY_THRESHOLD", "-1"),
            ("VIPUNE_EMBEDDING_MODEL", "nope/not-a-builtin"),
            ("HOME", &home_str),
            ("XDG_CONFIG_HOME", &xdg_str),
        ],
    );

    assert_exit(
        0,
        code,
        &stdout,
        &stderr,
        "AC4 version (bad env value, unknown model, unparseable config.toml)",
    );
    assert_eq!(stdout, expected_version_line());
    assert!(stderr.is_empty(), "stderr must be empty, got: {stderr:?}");
}

/// AC5: `vipune --help` still exits 0 with help text on stdout — the
/// non-regression half of the reordering (clap short-circuits `--help` in
/// `try_parse`, so it never reached the prologue, and no subprocess pin for it
/// existed).
#[test]
fn help_still_exits_zero_with_help_on_stdout() {
    let (code, stdout, stderr) = run_child(&["--help"], &[]);

    assert_exit(0, code, &stdout, &stderr, "AC5 --help");
    assert!(
        stdout.contains("vipune"),
        "help text must be printed on stdout, got: {stdout:?}"
    );
    // A marker clap emits only for a subcommand roster, plus the two
    // subcommands this module cares about: `contains("version")` alone would be
    // satisfied by the usage line and pin nothing.
    assert!(
        stdout.contains("Commands:") && stdout.contains("  version") && stdout.contains("  list"),
        "the subcommand roster must list `version` and `list`, got: {stdout:?}"
    );
    assert!(stderr.is_empty(), "--help must write nothing to stderr");
}

/// AC6: the data commands keep failing exactly as before with an unusable
/// database path — exit 1, the existing `Configuration error` text on stderr,
/// and the JSON error on stdout under `--json`.
#[test]
pub(crate) fn data_commands_still_fail_on_unusable_db_path() {
    let Some(blocked) = unwritable_db_path_fixture("AC6") else {
        return;
    };
    // The premise, stated rather than re-checked: `Some(blocked)` already proves
    // the parent refuses both `mkdir` and `open(O_CREAT)` (see
    // `BlockedParent::new`), which is what these assertions rest on —
    // `list`/`search`/`add` fail here through `ensure_directories` (the
    // `nested/` create) and, if that were somehow allowed, through
    // `MemoryStore::new` opening the file.
    let db_path = blocked.path().join("nested").join("x.db");
    let db_path_str = db_path.to_string_lossy().to_string();

    for (args, expected_word) in [
        (["list"].as_slice(), "list"),
        (["search", "query"].as_slice(), "search"),
        (["add", "some content"].as_slice(), "add"),
    ] {
        assert_fails_human(args, expected_word, &db_path_str);
        assert_fails_json(args, expected_word, &db_path_str);
    }
}

/// AC6, second premise: the malformed inputs AC4 pins are genuinely fatal for
/// the *other* commands, so AC4's exit 0 comes from the short-circuit and not
/// from inputs nothing would have rejected. Same knobs, driven through `list`
/// instead of `version`.
///
/// The per-knob "version survives it too" half is *not* re-pinned here for all
/// three knobs: AC4's single combined child already asserts exit 0 plus the
/// exact line for all three at once, which strictly implies each one alone
/// (nothing in `version` reads any of them). The unknown-model case is kept
/// individually as the least likely of the three to be covered by a later
/// env-parse regression.
#[test]
fn data_commands_keep_failing_on_bad_config() {
    let home = tempfile::TempDir::new().expect("temp home");
    let config_dir = home.path().join("config").join("vipune");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(
        config_dir.join("config.toml"),
        "this is not valid toml [[[\n",
    )
    .expect("write malformed config.toml");

    // An unknown model id alone: fatal through the normal path (validate →
    // profile_for) but not something `version` can see. Pinned on both sides:
    // `list` fails, `version` succeeds.
    let (code, stdout, stderr) = run_child(
        &["list"],
        &[("VIPUNE_EMBEDDING_MODEL", "nope/not-a-builtin")],
    );
    assert_exit(
        1,
        code,
        &stdout,
        &stderr,
        "list with an unknown embedding model",
    );
    assert!(
        stderr.contains("Unknown embedding model"),
        "expected the unknown-model error, got: {stderr:?}"
    );
    let (code, stdout, stderr) = run_child(
        &["version"],
        &[("VIPUNE_EMBEDDING_MODEL", "nope/not-a-builtin")],
    );
    assert_exit(0, code, &stdout, &stderr, "version with an unknown model");
    assert_eq!(stdout, expected_version_line());

    // A malformed float alone.
    let (code, stdout, stderr) = run_child(&["list"], &[("VIPUNE_SIMILARITY_THRESHOLD", "-1")]);
    assert_exit(1, code, &stdout, &stderr, "list with a malformed threshold");
    assert!(
        stderr.contains("Invalid similarity threshold"),
        "expected the threshold error, got: {stderr:?}"
    );

    // An unparseable config.toml alone.
    let home_str = home.path().to_string_lossy().to_string();
    let xdg_str = home.path().join("config").to_string_lossy().to_string();
    let (code, stdout, stderr) = run_child(
        &["list"],
        &[("HOME", &home_str), ("XDG_CONFIG_HOME", &xdg_str)],
    );
    assert_exit(
        1,
        code,
        &stdout,
        &stderr,
        "list with an unparseable config.toml",
    );
    assert!(
        stderr.contains("Failed to parse config file"),
        "expected the config-parse error, got: {stderr:?}"
    );
}
