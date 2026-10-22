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
takes the **one** guard for that target,
`config::tests_utils::ENV_MUTEX` (declared `pub(crate)` so modules outside
`config` can reach it). It is a `#[cfg(test)]` item of the crate, so integration
tests under `tests/` — separate binaries, separate processes — cannot reach it
and are outside this rule:

```rust
use crate::config::tests_utils::{ENV_MUTEX, cleanup_env_vars};

#[test]
fn test_something_with_env_vars() {
    let _guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cleanup_env_vars(&["VIPUNE_PROMOTION_THRESHOLD"]); // clear inherited state
    // Safe to modify env vars here
    unsafe { std::env::set_var("VIPUNE_DATABASE_PATH", "/tmp/test.db"); }
    // ... test code, assertions included, all under the guard ...
}
```

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
  again on exit (an RAII guard's `Drop`, as in `src/commands/promote_tests.rs`)
  because a panicking assertion skips a trailing `remove_var`. Recover from
  poisoning with `unwrap_or_else(|e| e.into_inner())` — a genuine assertion
  failure in one test should not make every sibling panic with `PoisonError`
  on top of it. Recovery is required at **every** acquisition of the shared
  mutex, not only in the file that most recently changed: poisoning is a
  property of the mutex, so one unrecovered `.lock().unwrap()` in any module
  turns a single real failure into a target-wide one.

Variables no production code reads — `VIPUNE_TEST_VAR`, used only to exercise
`cleanup_env_vars` itself — are outside this rule; nothing else observes them.

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