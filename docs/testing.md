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

Tests that modify environment variables use `ENV_MUTEX` to prevent race conditions:

```rust
use crate::config::tests_utils::ENV_MUTEX;

#[test]
fn test_something_with_env_vars() {
    let _guard = ENV_MUTEX.lock().unwrap();
    // Safe to modify env vars here
    unsafe { std::env::set_var("VIPUNE_DATABASE_PATH", "/tmp/test.db"); }
    // ... test code ...
}
```

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