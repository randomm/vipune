# Testing Guide

This guide explains how to write and run tests for vipune.

## Running Tests

Run all tests:

```bash
cargo test
```

Run tests with output (useful for debugging):

```bash
cargo test -- --nocapture
```

Run a specific test function:

```bash
cargo test test_function_name
```

Run tests in a specific file:

```bash
cargo test --test lib_integration
```

## Test Organization

vipune uses Rust's standard test organization:

- **Unit tests**: At the bottom of each source file in `#[cfg(test)]` modules
- **Integration tests**: In the `tests/` directory
- **Example locations**:
  - Unit tests: `src/embedding/engine.rs`, `src/config/mod.rs`, `src/commands/handlers.rs`
  - Integration tests: `tests/lib_integration.rs`

## Test Utilities

### Environment Variable Management

libtest runs the tests of one target on threads of a **single process**, and
`std::env` is process-global. Any test that reads or writes a `VIPUNE_*`
variable production code consults therefore shares state with every other such
test in that target — including tests in other modules that reach it through
the same production code.

Every test that **writes** a `VIPUNE_*` knob in an in-crate unit-test target
takes the **one** guard for that target, `config::tests_utils::ENV_MUTEX`
(declared `pub(crate)` so modules outside `config` can reach it), through the
RAII helper `ConfigEnvGuard`. It is a `#[cfg(test)]` item of the crate, so
integration tests under `tests/` — separate binaries, separate processes —
cannot reach it:

```rust
use crate::config::tests_utils::ConfigEnvGuard;

#[test]
fn test_something_with_env_vars() {
    // takes ENV_MUTEX (poison-recovered) and clears every knob in
    // tests_utils::CONFIG_ENV_VARS, on entry and again on Drop
    let _guard = ConfigEnvGuard::acquire();
    // Safe to modify env vars here
    unsafe { std::env::set_var("VIPUNE_DATABASE_PATH", "/tmp/test.db"); }
    // ... test code, assertions included, all under the guard ...
}
```

The knob list itself, `tests_utils::CONFIG_ENV_VARS`, has exactly one copy: the
config tests clear it, the helpers that wrap `Config::load()` clear it, and the
CLI tests that spawn the binary remove it from the child (plus the two
project-detection knobs on top). `test_config_env_vars_match_env_parser` fails
the build if the list drifts from the parser: it compares the set of `VIPUNE_*`
names spelled in `config/env_parser.rs` against `CONFIG_ENV_VARS` and checks the
`apply_*_override` functions 1:1, so a new knob cannot be silently left out of
the clearing. If your test writes a knob outside that list (or wants the lock
without the bulk clear — e.g. to serialise a child-process launch against
writers), use `tests_utils::lock_env_mutex()` and
`tests_utils::cleanup_env_vars(&["YOUR_VAR"])` explicitly.

A writer-only rule is not enough, because a lock serialises only against
other holders. Tests that **read** the environment indirectly are easy to miss
— they contain no `set_var` — and there are two shapes of it:

- `Config::load()` parses every `VIPUNE_*` knob and returns `Err` on a
  malformed value, so a test calling it — directly or through a helper —
  panics on a sibling writer's `-1` / `abc`.
- `std::process::Command` reads the parent's environment when it spawns the
  child; a sibling's malformed knob makes the child's `Config::load()` fail
  and a clean pass exit non-zero.

Prefer not to read at all. A fixture that only needs a `Config` should build
one with `Config::default()` — it reads no `VIPUNE_*` knob (only `HOME`, through
`dirs::home_dir()`), so it needs no lock and no ordering to lose. That is a
statement about the knobs, not about the environment: `Config::default()` still
calls `getenv`, so it is a concurrent reader like any other.
A spawned child is neutralised the same way, with `Command::env_remove` per
knob, which touches no process-global state.

Where the read is genuinely wanted, a reader follows the same convention as a
writer: take the shared guard with the first statement and hold it for the
whole test body, so the entry-clear guarantees it observes a clean environment.

Three rules for writers and knob-readers alike, all of which exist because of
an observed flake (issue #233, where one test file kept its own mutex while
tests in another module moved the same variable underneath it):

- **One guard per target.** A file-local `static Mutex` only serialises within
  that file; a reader in another module writes the same variable freely. Reuse
  `config::tests_utils::ENV_MUTEX` rather than declaring a new one.
- **Hold it for the whole body**, from the first statement, so seeding, the
  `set_var` and the production code's own `env::var` read sit in one critical
  section.
- **Clear the variable on entry and on exit.** Every test that writes clears
  its own variables first so it cannot inherit an ambient value, and clears
  again on exit — via an RAII guard's `Drop`, as `ConfigEnvGuard` does — because
  a panicking assertion skips a trailing `remove_var`. That leak is precisely
  how one genuine assertion failure in `config::overrides` used to become two or
  three unrelated `Config::load().unwrap()` panics elsewhere in the target. Do
  not treat a trailing `clean_env()` as the exit-clear. Recover from poisoning
  at **every** acquisition — a genuine assertion failure in one test should not
  make every sibling panic with `PoisonError` on top of it. In-crate tests get
  this from one helper, `tests_utils::lock_env_mutex()`: both
  `ConfigEnvGuard::acquire` and any explicit acquisition go through it, so the
  poison-recovery incantation exists in exactly one place. A `tests/` file that
  owns its own mutex recovers there the same way. Recovery is required at
  **every** acquisition of the shared mutex, not only in
  the file that most recently changed: poisoning is a property of the mutex, so
  one unrecovered `.lock().unwrap()` in any module turns a single real failure
  into a target-wide one.

The rule is about the **call**, not the variable's name. `set_var`/`remove_var`
racing a concurrent `getenv` is the hazard — it is why Rust 2024 made those
methods `unsafe` — and a sibling reader in this target calls `getenv` for
unrelated reasons (`Config::default()` → `dirs::home_dir()` → `getenv("HOME")`,
`detect_project` → `env::var("VIPUNE_PROJECT"/"VIPUNE_DEBUG")`). So a test that
mutates a variable nothing else reads — `VIPUNE_TEST_VAR`, used only to exercise
`cleanup_env_vars` itself — still holds the mutex; what it may skip is the
shared-list clear, since it is not a config knob.

Integration tests under `tests/` cannot reach the crate's `#[cfg(test)]`
mutex, but they are one process each and the same `set_var`-vs-`getenv` hazard
applies inside them. A test there that mutates the environment takes a
file-local `static Mutex` (declared at the top of that test file) and holds it
for the whole body. A writer-side lock serialises only against other holders
of *that* mutex — every env-touching test in the file must take it, or the
lock protects nothing. `tests/lib_integration.rs` declares no such mutex today:
no test in it writes the environment (`Config::default()` reads no `VIPUNE_*`
knob), so the four `remove_var` calls it used to make before asserting on the
defaults protected nothing.

Do not paper over a shared-env race with `--test-threads=1` or
`RUST_TEST_THREADS`; make the test self-isolating instead.

### Temporary Directories

Integration tests create temporary databases:

```rust
use std::env;

#[test]
fn test_memory_operations() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // ... test code ...

    std::fs::remove_file(db_path).ok();
}
```

### Integration Test Patterns

See `tests/lib_integration.rs` for comprehensive examples:

```rust
// Test add-then-search workflow
#[test]
fn test_memory_store_add_then_search_returns_matching_memory() {
    // 1. Create temporary database
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    // 2. Initialize store
    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // 3. Add memory
    let project_id = "test-project";
    let memory_id = match store
        .add_with_conflict(project_id, "Alice works at Microsoft", None, false)
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // 4. Search
    let results = store
        .search(project_id, "where does alice work", 10, 0.0)
        .expect("Failed to search");

    // 5. Verify
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "Alice works at Microsoft");

    // 6. Clean up
    std::fs::remove_file(db_path).ok();
}
```

## What to Test

When adding a new feature, test:

- [ ] **Happy path**: Expected behavior with valid inputs
- [ ] **Error paths**: Invalid inputs, empty strings, oversized inputs
- [ ] **Edge cases**: Boundary values (exactly MAX_INPUT_LENGTH)
- [ ] **Public API behavior**: What users call, not internal implementation

### Checklist for New Features

- [ ] Write test before implementation (TDD preferred)
- [ ] Test returns correct values for valid inputs
- [ ] Test returns appropriate errors for invalid inputs
- [ ] Test boundary conditions (empty, max length, zero, negative)
- [ ] Test idempotency where applicable (calling multiple times)
- [ ] Clean up resources (temp files, env vars)

## What NOT to Test

Avoid testing implementation details that change:

- ❌ Internal function names and structure
- ❌ Exact error messages (test error types/variants instead)
- ❌ Database schema internals
- ❌ Embedding generation internals (test the API, not the math)
- ❌ Specific embedding scores (test similarity ordering, not exact values)

**Test behavior, not implementation.**

## Golden Embedding Fixture

`src/embedding/tests/golden.rs` pins the **numbers** the ONNX pipeline produces:
it re-embeds a fixed corpus through `embed_passage` and compares element-wise
against the committed `golden_bge_small.txt`, failing above a 1e-4 max absolute
difference (the `#[ignore]`d `test_golden_embeddings_match_fixture`, run by the
nightly model lane). Regenerate only when the model id/revision legitimately
changes, with `VIPUNE_REGENERATE_GOLDEN=1 cargo test golden_fixture_regenerate
-- --ignored --nocapture` (the `--nocapture` is what surfaces the per-text
diffs and the summary line the test prints) — **never to make a drift failure
pass**, which would delete the
very signal the test exists to produce.

## Coverage Expectations

- **New code**: 80%+ test coverage per project standards
- **Critical paths**: Conflict detection, database I/O, embedding generation
- **Error handling**: All error branches should have tests

## Example Test Patterns

### Testing Error Cases

```rust
#[test]
fn test_add_with_empty_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let result = store.add_with_conflict("test", "", None, false);
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    std::fs::remove_file(db_path).ok();
}
```

### Testing Boundary Values

```rust
#[test]
fn test_add_at_exactly_max_input_length_returns_success() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let exact_text = "x".repeat(MAX_INPUT_LENGTH);
    let result = store.add_with_conflict("test", &exact_text, None, false);
    assert!(result.is_ok(), "Should accept input at exactly MAX_INPUT_LENGTH");

    std::fs::remove_file(db_path).ok();
}
```

### Testing Public API Constants

```rust
#[test]
fn test_constant_max_search_limit_is_accessible() {
    assert_eq!(MAX_SEARCH_LIMIT, 10_000);
}
```

## Running Tests Before Pushing

Always run all tests before pushing:

```bash
cargo test
```

If any test fails, fix it locally. CI verifies, but does not discover.

## Dependency Advisory Gate

CI also runs [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) **0.20.2** against the
committed `Cargo.lock`, on pull requests, pushes to `main` and the nightly schedule. The version is
pinned by the `tool: cargo-deny@0.20.2` line of the `deny` job in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) so the gate's verdict cannot change
mid-PR; `test_ci_deny_pin_matches_the_docs` fails if the two disagree. It is not a test suite
— it fails the build for a dependency that has a RustSec advisory, a yanked release, or a denied
feature. Configuration lives in [`deny.toml`](../deny.toml) at the repo root.

Run the same gate locally (install once with `cargo install cargo-deny`):

```bash
cargo deny --locked check advisories bans
```

What it covers, and what it deliberately does not:

- **Advisories** — RustSec advisories for the crates cargo-deny resolves from the committed
  `Cargo.lock`. The `mcp` feature is covered because it is a default feature, and dev/bench deps
  (criterion, tempfile, tokio) are included by default.
- **Yanked releases** — `yanked = "deny"` fails the build for a dependency release that has been
  withdrawn from crates.io.
- **Advisory database freshness** — CI runs online, so cargo-deny re-fetches the RustSec
  database on every run and a fetch that fails is reported as an error (`git.rs:126` returns
  `Err` on a failed clone/fetch, and `check.rs:346` propagates it). `maximum-db-staleness = "P2D"`
  is a *different* mechanism: cargo-deny only consults it when fetching is disabled — the
  `Fetch::Disallow` branch selected by `--offline`/`--frozen` (`check.rs:313`, `db.rs:62`) — where
  it aborts if the local clone's last fetch is older than two days. It does not affect the online
  CI run.
- **Unmaintained / unsound** — reported for direct dependencies only (`"workspace"`). Transitive
  ones (paste via tokenizers, rand via hf-hub) would redden PRs for upstream churn this repo cannot
  fix. No current direct dependency carries either kind of advisory: dirs' RUSTSEC-2020-0053 was
  withdrawn in 2021 and is skipped, and the tokio unsound advisories are all patched at the
  resolved version.
- **Triage rows** — each ignored advisory carries an `id`, a `reason` and the upstream blocker. An
  `ignore` row whose advisory stops firing is reported by cargo-deny's `advisory-not-detected`
  lint. In 0.20.2 that lint defaults to **warn**, which would let a stale row rot silently, so
  `deny.toml` sets `unused-ignored-advisory = "deny"` explicitly and a fabricated ignore id fails
  the gate. (A row naming an id that is absent from the database *itself* is `unknown-advisory`,
  which is a warning and does not fail the run.)
- **Denied features** — `base64` 0.23 may not resolve with `simd-unsafe`, because that crate
  decodes untrusted import files and feature unification would otherwise re-enable it. `ort` and
  `ort-sys` may not resolve with `tls-native` or `tls-native-vendored` (native-tls/OpenSSL);
  `Cargo.toml` chooses `tls-rustls`, and `tests/deny_feature_ban_lock.rs` fails if such a row
  names a crate@version the lockfile does not resolve — cargo-deny ignores a dead row silently.
- **Not configured** — `[licenses]` and `[sources]` are intentionally absent; enabling them is a
  separate decision, and adding an empty section rejects every license in the graph.

### Triaging a gate that flips without a repo change

Because the nightly schedule refetches the advisory database, a build that was green can go red
with no commit in between. That is the gate working: new RustSec data, not a broken merge. Read
the diagnostic the failure prints — the crate, the version, and the advisory ID and URL are all in
it — then:

1. Open the advisory URL and check the patched version range against `Cargo.lock`.
2. If a compatible update exists, `cargo update -p <crate>` and commit it with the same message as
   the triage; the gate must pass against the committed lockfile.
3. If it does not (an upstream breaking change, or the advisory does not apply to how vipune uses
   the crate), add an `ignore` row to `deny.toml` with the ID, the reason, and the upstream
   blocker, and file an issue for the migration.
4. If the failure says the advisory database is stale, it came from an `--offline`/`--frozen` run
   (the only mode that reads `maximum-db-staleness`): delete the clone under
   `~/.cargo/advisory-dbs/`, rerun online, and let the fetch refresh it. An online run that cannot
   reach the database fails with a fetch error instead.

### Triaging a yanked release

A yanked release is allowlisted by an `ignore` row in `deny.toml` written in **spec-string** form,
`"crate@x.y.z"`, never in the `{ crate, version }` table form. cargo-deny sorts `[advisories]`
`ignore` rows into advisory-id rows and yanked rows by shape (`advisories/cfg.rs:163-232`), and the
string branch splits on `@` into a name plus an exact version requirement while the table branch
reads `crate`, `name` and `reason` and drops `version` entirely (`cfg/package_spec.rs:66-96`), so a
table row matches by name alone and mutes every release of that crate. Reproduced on 0.20.2 in a
scratch copy with `ppv-lite86 0.2.11` marked yanked in `Cargo.lock` — the yank has to exist for the
comparison to mean anything, since on a graph with no yanked release *both* forms are reported as
`yanked-not-detected` and fail: the table row `{ crate = "ppv-lite86", version = "=0.2.10", reason
= ... }` silences the yank (the `version` key is discarded, so the row matches the yanked 0.2.11),
while the spec string `"ppv-lite86@0.2.10"` leaves `error[yanked]` firing. The spec string also
keeps `unused-ignored-advisory = "deny"` meaningful for yank rows: a pin that matches no yanked
release is reported as `yanked-not-detected`, which a name-only table row cannot do — it matches
whatever is yanked.

## See Also

- [Architecture documentation](architecture.md) for module structure
- [CONTRIBUTING.md](../CONTRIBUTING.md) for quality gates
- [`deny.toml`](../deny.toml) for the dependency advisory gate configuration
- [Main README](../README.md) for project overview