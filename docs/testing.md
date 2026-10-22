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
  - Unit tests: `src/embedding.rs`, `src/config/mod.rs`, `src/commands.rs`
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

The knob list itself, `tests_utils::CONFIG_ENV_VARS`, has exactly one copy: it
is what `config::overrides` tests clear, what `config::tests` clears around
`Config::load()`, and what `commands::reindex_cli_tests` removes from spawned
children (as `CONFIG_ENV_VARS_EXCLUDED`, with `PROJECT_ENV_VARS_EXCLUDED` adding
the two `project.rs` knobs on top).
`test_config_env_vars_match_env_parser` fails the build if it drifts from the
`apply_*_override` functions in `config/env_parser.rs`, so a new knob cannot be
silently left out of the clearing. If your test writes a knob outside that list
(or wants the lock without the bulk clear — e.g. a slow spawn that must not
hold the mutex while it waits on a child), use `tests_utils::lock_env_mutex()`
and `tests_utils::cleanup_env_vars(&["YOUR_VAR"])` explicitly.

A writer-only rule is not enough, because a lock serialises only against
other holders. Tests that **read** the environment indirectly are easy to miss
— they contain no `set_var` — and there are two shapes of it:

- `Config::load()` parses every `VIPUNE_*` knob and returns `Err` on a
  malformed value, so a test calling it — directly or through a helper —
  panics on a sibling writer's `-1` / `abc`.
- `std::process::Command` hands the child the parent's environment at `spawn`
  time (see `capture_force_output` in `src/commands/reindex_cli_tests.rs`);
  a sibling's malformed knob makes the child's `Config::load()` fail and a
  clean pass exit non-zero.

Prefer not to read at all. A fixture that only needs a `Config` should build
one with `Config::default()` — no environment, no lock, no ordering to lose
(`migration_tests::test_config`). A spawned child is neutralised the same way,
with `Command::env_remove` per knob, which touches no process-global state.

Where the read is genuinely wanted, hold the same guard for the duration of
the read and clear the knobs on entry first, as narrow as the call itself
rather than the whole test body — a reader holding the shared mutex across a
multi-second model migration serialises every writer in the target behind it.

Three rules for writers, all of which exist because of an observed flake
(issue #233, where `promote_tests` kept its own mutex and `config::overrides`
tests could still move `VIPUNE_PROMOTION_THRESHOLD` underneath it):

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
  with `unwrap_or_else(|e| e.into_inner())` (or just call
  `tests_utils::lock_env_mutex()`, which does) — a genuine assertion failure in
  one test should not make every sibling panic with `PoisonError` on top of it.
  Recovery is required at **every** acquisition of the shared mutex, not only in
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
for the whole body — as `tests/lib_integration.rs` does for its one env-writing
test.

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

## See Also

- [Architecture documentation](architecture.md) for module structure
- [CONTRIBUTING.md](../CONTRIBUTING.md) for quality gates
- [Main README](../README.md) for project overview