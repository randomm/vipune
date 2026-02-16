# vipune Agent Team Configuration

## Project Overview

**vipune** is a minimal memory layer for AI agents, written entirely in Rust. It provides semantic search capabilities using SQLite and ONNX embeddings (bge-small-en-v1.5, 384 dimensions).

**Key characteristics:**
- Single Rust binary CLI
- Fully synchronous - no tokio, no async, no .await
- No network dependencies at runtime
- SQLite + ONNX embeddings for semantic memory storage
- Apache-2.0 license
- Target: ~2,500 lines of Rust code total

**Reference material:** `_reference/remory/` contains the original remory codebase as a git submodule (temporary). Use it to understand patterns, NOT to copy-paste. All code must be written fresh for vipune using a clean-room approach.

---

## 🚨 CRITICAL: context7 Usage Protocol 🚨

**BEFORE doing ANYTHING programming-related, you MUST use context7.**

context7 is an MCP (Model Context Protocol) server that provides instant access to:
- **Rust documentation** - API guidelines, idioms, best practices
- **Library documentation** (rusqlite, ort, clap, etc.)
- **Configuration syntax and options**
- **Testing patterns and frameworks**

### Why context7 is MANDATORY:
1. **Authoritative**: Official, up-to-date documentation directly from the source
2. **Accurate**: Eliminates guessing, hallucinations, and outdated information
3. **Efficient**: Instant access without web searches or manual lookups
4. **Complete**: Comprehensive coverage of Rust ecosystem

### When to use context7 (SHORT ANSWER: ALWAYS):
- ✅ Before writing ANY code
- ✅ Before implementing ANY feature
- ✅ When unsure about API syntax
- ✅ When choosing between approaches
- ✅ When configuring tools (clippy, fmt, test frameworks)
- ✅ When writing tests
- ✅ When handling errors
- ✅ When making architectural decisions

**If you write code without checking context7 first, you are doing it WRONG.**

---

## Architecture

vipune is a **single crate** project with no workspace complexity.

### Source Files

```
src/
├── main.rs       # CLI entry point (~300 lines)
├── memory.rs     # Core CRUD + conflict detection (~500 lines)
├── sqlite.rs     # SQLite backend (~400 lines)
├── embedding.rs  # ONNX embeddings (~200 lines)
├── project.rs    # Git project detection (~100 lines)
├── config.rs     # Configuration (~150 lines)
└── errors.rs     # Error types (~100 lines)
```

**Target:** ~2,500 lines total (excluding tests)

### Key Constraints

- ❌ **No async code** - No tokio, no async fn, no .await
- ❌ **No daemon** - CLI tool only, runs and exits
- ❌ **No protocol** - Direct SQLite access, no network protocol
- ❌ **No Python bindings** - Pure Rust CLI
- ✅ **Fully synchronous** - All operations block until complete

### Model Configuration

- **Model:** bge-small-en-v1.5 from HuggingFace
- **Embedding dimensions:** 384 × f32 little-endian = exactly 1,536 bytes per embedding BLOB
- **Model cache:** `~/.cache/vipune/models/`
- **Cosine similarity:** Computed in Rust (not via SQL extension)

### Database Location

- **Path:** `~/.local/share/vipune/memories.db`
- **Format:** SQLite with bundled rusqlite (no external SQLite installation required)

---

## Agent Hierarchy & Delegation Rules

- **Primary Agent**: Project Manager (orchestrator only - no execution)
- **Specialist Agents**: Domain experts (Rust, Code Review, Git, etc.)
- **Support Agents**: Git operations, Code Review
- **Tool Restrictions**: Project Manager has read-only tools + delegation only
- **Execution Rule**: Specialists execute, Project Manager coordinates

### Delegation Protocol

- **Project Manager**: ONLY entity that delegates to specialists - maintains scope control
- **All Specialists**: Report discoveries to project manager (NO direct specialist-to-specialist delegation)
- **Exception**: Git operations only - all agents can delegate git work to @ops

---

## Universal Quality Standards

- **Issue-Driven Development**: All work must match GitHub issue content exactly
- **Zero Quality Bypasses**: No `#[allow(...)]` suppressions without justification
- **Zero Technical Debt**: No TODO, FIXME, HACK, or incomplete implementations in source code
- **Module/File Size Limits**: Maximum 500 lines per file (exceptions require justification)
- **Local-First Quality Gates**: ALL quality checks (format, lint, test, coverage) MUST pass locally BEFORE pushing
- **Delegation Protocol**: Project Manager orchestrates, specialists execute
- **Scope Control**: No work beyond what's explicitly listed in GitHub issues

---

## Minimalist Engineering Philosophy

- **LESS IS MORE**: Every line of code is a liability - question necessity before creation
- **Challenge Everything**: Ask "Is this truly needed?" before implementing anything
- **Minimal Viable Solution**: Build the simplest thing that fully solves the problem
- **No Speculative Features**: Don't build for "future needs" - solve today's problem
- **Prefer Existing**: Reuse existing code/tools before creating new ones
- **One Purpose Per Component**: Each function/struct/file should do one thing well

---

## Module Size & Organization

### File Size Limits

- **Hard limit**: 500 lines per module (exceptions require justification)
- **Ideal target**: 300 lines or fewer
- **Test files**: 800 lines max (comprehensive test suites only)
- **Configuration files**: 200 lines max
- **Why**: Cognitive load, maintainability, testability, merge conflicts

### Refactoring Triggers

- File exceeds 500 lines → Immediate refactoring required
- File has 3+ distinct responsibilities → Violates Single Responsibility Principle

### Rust Module Organization

```rust
// src/main.rs - CLI entry point and public API
mod memory;
mod sqlite;
mod embedding;
mod project;
mod config;
mod errors;

use clap::Parser;
use memory::MemoryStore;

// src/memory.rs - Core logic, public interface
pub struct MemoryStore { ... }

// src/sqlite.rs - SQLite backend implementation
pub struct SqliteBackend { ... }

// src/embedding.rs - ONNX embedding computation
pub struct EmbeddingModel { ... }
```

---

## Technical Debt Prevention

**Philosophy**: Every TODO is a promise easily broken. Every stub is a bug waiting to happen.

### Zero Technical Debt Policy

**Forbidden in src/ (source code):**
- ❌ TODO, FIXME, HACK, XXX, TEMP, WORKAROUND comments
- ❌ `unimplemented!()` or `todo!()` macros in source code
- ❌ Stub functions that return empty values without implementation
- ❌ Incomplete error handling (bare `_` patterns that swallow errors)
- ❌ Missing edge case validation

**Allowed in tests/ only:**
- ✅ TODO comments for future test enhancements

### If You Need to Defer Work

1. Create a GitHub issue documenting the work
2. Implement what's required for the current issue
3. Do NOT add TODO comments - the GitHub issue is the TODO
4. If the feature can't be completed: mark issue as blocked, explain blocker

### Rust-Specific Anti-Patterns

```rust
// ❌ BAD: Using unimplemented! or todo! in production code
pub fn process_data(data: &[u8]) -> Result<String, Error> {
    todo!("Implement data processing")
}

// ✅ GOOD: Implement or return appropriate error
pub fn process_data(data: &[u8]) -> Result<String, Error> {
    if data.is_empty() {
        return Err(Error::EmptyInput);
    }
    process_inner(data)
}

// ❌ BAD: Swallowing errors with bare _
let result = some_operation().unwrap(); // May panic

// ✅ GOOD: Propagate errors with ? operator
let result = some_operation()?;

// ❌ BAD: Suppressing clippy warnings without justification
#[allow(clippy::unnecessary unwrap_used)]
let x = result.unwrap();

// ✅ GOOD: Fix the underlying issue or justify the suppression
#[allow(dead_code)] // Dead code justified: public API for semver stability
const DEPRECATED_CONSTANT: u32 = 42;
```

---

## Pre-Creation Challenge Protocol (MANDATORY)

Before creating ANY code, file, or component, agents MUST ask:
- **🚨 Have you checked context7?** Look up best practices, existing patterns, and recommended approaches first
- **Is this explicitly required** by the GitHub issue?
- **Can existing code/tools** solve this instead?
- **What's the SIMPLEST** way to meet the requirement?
- **Will removing this** break the core functionality?
- **Am I building for hypothetical** future needs?
- **Does context7 show a better approach?** Check documentation for idiomatic solutions
- **Is this implementation complete and production-ready?** No TODOs, stubs, or partial solutions

**If you cannot justify the necessity, DO NOT CREATE IT.**

---

## GitHub Issue Quality Template (Mandatory)

Every development issue MUST include these checkboxes:
- [ ] **Tests written**: All new code includes tests (TDD preferred)
- [ ] **Coverage**: 80%+ test coverage for new code
- [ ] **Linting**: All code passes `cargo clippy -- -D warnings`
- [ ] **Formatting**: All code passes `cargo fmt --check`
- [ ] **Local Verification**: Tests pass before completion

---

## Scope Control Protocol

- **READ**: `gh issue view #123` for complete requirements
- **VALIDATE**: All work matches issue content exactly
- **REFUSE**: Any work not explicitly listed in issue
- **EXPAND**: Update issue before adding scope
- **COMPLETE**: Only when all issue checkboxes are done

---

## Rust-Specific Code Style Guidelines

### 🚨 Use context7

Before writing any Rust code, check context7 for:
- Rust API Guidelines
- Error handling patterns (Result types)
- Idiomatic patterns and best practices
- Library-specific documentation (rusqlite, ort, clap, etc.)

### Code Style

- Follow Rust API Guidelines (enforced by clippy)
- Use `Result<T, E>` for fallible operations (not bool or Option for error cases)
- Error types must implement `std::error::Error` and `std::fmt::Display`
- Documentation comments for public APIs (use `///` for items, `//!` for modules)
- Imports: sorted by crate, then items within crate
- Naming conventions:
  - Functions/variables: `snake_case`
  - Types/structs/enums: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Error Handling

```rust
// ✅ GOOD: Use Result for fallible operations
pub fn load_config(path: &Path) -> Result<Config, Error> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

// ✅ GOOD: Explicit early returns for error conditions
fn process_data(data: &[u8]) -> Result<String, Error> {
    if data.is_empty() {
        return Err(Error::EmptyInput);
    }
    Ok(serde_json::to_string(data)?)
}

// ❌ BAD: Using bool for error cases (use Result instead)
pub fn try_parse(s: &str) -> (bool, Option<i32>) {
    match s.parse::<i32>() {
        Ok(n) => (true, Some(n)),
        Err(_) => (false, None),
    }
}
```

### No Async Policy

🚨 **CRITICAL:** vipune is a fully synchronous project. No async code allowed.

```rust
// ❌ BAD: Async code (forbidden)
async fn store_memory(&mut self, content: &str) -> Result<String, Error>;
tokio::runtime::Runtime::new()?;

// ❌ BAD: Using .await (forbidden)
let result = some_async_operation().await?;

// ✅ GOOD: Synchronous code
pub fn store_memory(&mut self, content: &str) -> Result<String, Error>;
let result = some_operation()?;
```

---

## Development Commands

### Build and Run

```bash
# Build debug version
cargo build

# Build release version (optimized)
cargo build --release

# Run CLI
cargo run -- --help

# Run from release binary
./target/release/vipune --help
```

### Quality Checks

```bash
# Format code (check mode - fails if not formatted)
cargo fmt --check

# Format code (apply formatting)
cargo fmt

# Lint code (all warnings are errors)
cargo clippy -- -D warnings

# Lint code (show warnings without failing)
cargo clippy

# Run tests
cargo test

# Build release (optimization check)
cargo build --release
```

### Full Validation Before Push

```bash
# Run all checks in sequence (all must pass)
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

---

## Quality Standards

### Code Quality

- **Formatting**: `cargo fmt --check` — All code must pass Rust formatting
- **Linting**: `cargo clippy -- -D warnings` — All warnings are errors
- **Testing**: `cargo test` — All tests must pass
- **Coverage**: 80%+ test coverage for new code
- **Build**: `cargo build --release` — Must compile in release mode

### Code Style Rules

- ❌ **No `#[allow(...)]` suppressions** without justification in code comments
- ❌ **No TODO/FIXME/HACK** in source code (use GitHub issues instead)
- ⚡ **Suppression example (with justification):**
  ```rust
  #[allow(dead_code)] // Dead code justified: public API for semver stability
  const DEPRECATED_CONSTANT: u32 = 42;
  ```

### File Size Limits

- **Hard limit**: 500 lines per file
- **Ideal target**: 300 lines or fewer per file
- **Refactoring required**: File exceeds 500 lines or has 3+ distinct responsibilities

---

## Commit Convention

vipune uses [Conventional Commits](https://www.conventionalcommits.org/) to drive automated versioning.

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types and Version Impact

| Type | Description | Version Bump |
|------|-------------|--------------|
| `feat` | New feature | Minor (0.1.0 → 0.2.0) |
| `fix` | Bug fix | Patch (0.1.0 → 0.1.1) |
| `docs` | Documentation only | None |
| `style` | Code style (formatting, semicolons) | None |
| `refactor` | Code change that neither fixes nor adds | None |
| `perf` | Performance improvement | Patch |
| `test` | Adding or correcting tests | None |
| `chore` | Maintenance tasks | None |
| `ci` | CI/CD changes | None |

### Breaking Changes

For breaking changes, either:
- Add `!` after type: `feat!: remove deprecated API`
- Add `BREAKING CHANGE:` in footer

Breaking changes trigger a **major** version bump (0.1.0 → 1.0.0).

### Examples

```bash
# Patch release
git commit -m "fix: correct embedding BLOB size validation"

# Minor release
git commit -m "feat: add project auto-detection via .git directory"

# Major release
git commit -m "feat!: redesign CLI output format for JSON mode"

# With scope and issue reference
git commit -m "feat(#2): implement SQLite backend with bundled rusqlite"

# With body
git commit -m "fix(sqlite): handle database upgrade from v1 to v2

Migration adds embedding table with proper BLOB column (1536 bytes)"
```

---

## Dependencies Policy

### Forbidden Dependencies

Do not add these dependencies under any circumstances:
- ❌ `tokio` — Async runtime (project is synchronous only)
- ❌ `reqwest` — Async HTTP client (no network dependencies at runtime)
- ❌ `pyo3` — Python bindings (Rust-only project)
- ❌ `sqlx` — Async database toolkit (use rusqlite instead)
- ❌ `async-trait` — Async traits (project is synchronous only)

### Required Dependencies (from Issue #1)

These dependencies are already planned and justified:
- `clap` — CLI argument parsing
- `rusqlite` — SQLite backend (bundled feature)
- `ort` v2 — ONNX runtime for embeddings
- `tokenizers` — Text tokenization (HF tokenizers)
- `hf-hub` — HuggingFace Hub model download
- `serde` / `serde_json` — Serialization
- `uuid` — ID generation
- `chrono` — Date/time handling
- `thiserror` — Error handling
- `toml` — Configuration parsing
- `dirs` — XDG directory paths

### Adding New Dependencies

New dependencies require justification:
1. Check context7 for existing solutions using current dependencies
2. Propose in GitHub issue before implementation
3. Explain why existing dependencies cannot solve the problem
4. Consider dependency size and build time impact

---

## File Structure

### Source Tree

```
src/
├── main.rs       # CLI entry point (~300 lines)
├── memory.rs     # Core CRUD + conflict detection (~500 lines)
├── sqlite.rs     # SQLite backend (~400 lines)
├── embedding.rs  # ONNX embeddings (~200 lines)
├── project.rs    # Git project detection (~100 lines)
├── config.rs     # Configuration (~150 lines)
└── errors.rs     # Error types (~100 lines)
```

### Test Structure

```
tests/
├── integration.rs    # Integration tests
└── fixtures/         # Test data/fixtures (if needed)
```

### Configuration

```
Cargo.toml             # Project dependencies
Cargo.lock             # Pinned versions (commit this)
.github/
└── workflows/
    └── ci.yml         # CI configuration
```

---

## Testing Best Practices

### Rust Testing Framework

vipune uses Rust's built-in test framework. No external test frameworks required.

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimensions() {
        assert_eq!(EMBEDDING_DIMS, 384);
    }

    #[test]
    fn test_blob_size() {
        let embedding = vec![0.0f32; 384];
        let bytes: Vec<u8> = bytemuck::cast_vec(embedding);
        assert_eq!(bytes.len(), 1536); // 384 × 4 bytes
    }
}
```

### Integration Tests

```rust
// tests/integration.rs
use vipune::memory::MemoryStore;

#[test]
fn test_memory_add_and_search_integration() {
    let store = MemoryStore::new("test.db").unwrap();

    store.add("test memory", None).unwrap();
    let results = store.search("test", 5).unwrap();

    assert!(!results.is_empty());
}
```

### Error Testing

```rust
#[test]
fn test_empty_input_error() {
    let result = process_data(&[]);
    assert!(matches!(result, Err(Error::EmptyInput)));
}
```

### Coverage

- Target: 80%+ test coverage for new code
- Measure: Use `cargo tarpaulin` (if available) or manual review
- Focus: Test error paths and edge cases, not just happy paths

---

## 🚨 MANDATORY: Local-First Quality Gates

**CRITICAL PRINCIPLE: CI is for VERIFICATION, not DISCOVERY.**

All quality gates MUST pass locally BEFORE pushing to remote.

### Pre-Push Checklist (BLOCKING)

Before ANY `git push`, ALL of these must pass locally:

```bash
cargo fmt --check              # Formatting check
cargo clippy -- -D warnings     # Linting (all warnings as errors)
cargo test                      # All tests pass
cargo build --release           # Release build succeeds
```

**If ANY check fails locally → FIX IT before pushing.**

### Enforcement Rules

1. **No "fix in next commit" pattern**: Fix issues in the CURRENT commit
2. **No "let CI catch it" mindset**: CI failures from local issues are workflow violations
3. **Amend commits when possible**: If you catch an issue before push, amend the commit
4. **Run full suite before PR**: Before creating or updating a PR, run complete quality gates

### Quick Validation Command

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test && echo "✅ All checks passed"
```

### CI Role

CI exists to:
- ✅ Verify that local checks were actually run
- ✅ Catch environment-specific issues (different OS, Rust version)
- ✅ Test on multiple platforms (Linux, macOS, Windows)
- ✅ Generate coverage reports and badges

CI does NOT exist to:
- ❌ Discover linting errors
- ❌ Find build failures
- ❌ Catch test failures

---

## Merge Policy

### Branch Protection

- ❌ **NO direct commits to main branch** - All work must go through feature branches
- ❌ **NO direct pushes to main branch** - Must go through PR review
- ✅ **Feature branches only** - All work on `feature/issue-{NUMBER}-*` branches
- ✅ **PR required** - All changes must go through pull requests

### PR Requirements

Before merging a PR to main:
- [ ] **CI green** - All CI checks pass
- [ ] **Code review complete** - At least one human review approved
- [ ] **Tests pass locally** - All quality gates verified locally
- [ ] **Coverage adequate** - New code has 80%+ test coverage

### Human Approval Required

Agents may NOT merge to main without explicit human approval:
- Automated agents: Create PRs only, do not merge
- Code review agents: Review and comment, do not merge
- Project manager: Coordinate, delegate, do not merge
- Only human developers: Can approve and merge PRs

---

## vipune-Specific Technical Details

### Embedding BLOB Contract

**Critical specification for embedding storage:**

- **Dimensions:** 384 f32 values
- **Endianness:** Little-endian (standard for Intel/ARM)
- **Total bytes:** 384 × 4 = 1,536 bytes per embedding
- **Data type:** BYTES BLOB column in SQLite

```rust
// Correct transformation
let embedding: Vec<f32> = model.encode(text)?;
let bytes: Vec<u8> = bytemuck::cast_vec(embedding); // f32 → [u8; 4]
assert_eq!(bytes.len(), 1536);
```

### Cosine Similarity

Cosine similarity is computed in Rust (NOT via SQL extension):

```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
```

### Model Configuration

- **Model:** bge-small-en-v1.5
- **Source:** HuggingFace Hub
- **Cache location:** `~/.cache/vipune/models/`
- **Download:** Automatic on first use via `hf-hub`

### Database Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- 1536 bytes (384 × f32 little-endian)
    metadata TEXT,            -- JSON metadata
    created_at TEXT NOT NULL, -- ISO 8601 timestamp
    updated_at TEXT NOT NULL  -- ISO 8601 timestamp
);

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE
);
```

### CLI Exit Codes

- **0**: Success
- **1**: Error (invalid input, file not found, etc.)
- **2**: Conflicts (duplicate memory content detected)

### JSON Output Mode

All CLI commands support `--output json` for programmatic parsing:

```bash
vipune add "test memory" --output json
# {"id":"uuid-...","content":"test memory","status":"added"}

vipune search "test" --output json
# {"results":[{"id":"uuid-...","content":"test memory","score":0.95}]}
```

---

## Quality Gates Checklist

### Before Claiming Task Complete

- [ ] **Tests pass**: `cargo test`
- [ ] **Coverage ≥ 80%**: Manual review or `cargo tarpaulin` (if available)
- [ ] **Linting passes**: `cargo clippy -- -D warnings`
- [ ] **Code formatted**: `cargo fmt --check`
- [ ] **No quality suppressions**: No `#[allow(...)]` without justification
- [ ] **No technical debt**: No TODO/FIXME/HACK in src/, no `unimplemented!()` or `todo!()`
- [ ] **No async code**: No tokio, no async fn, no .await anywhere
- [ ] **Complete implementation**: All functions fully implemented with error handling
- [ ] **File size compliance**: No files over 500 lines (or justification in PR)
- [ ] **Documentation updated**: As specified in issue requirements
- [ ] **Embedding BLOB correct**: 1,536 bytes (384 × f32 little-endian)
- [ ] **Issue requirements met**: All checkboxes in GitHub issue completed

---

## Agent Tool Restrictions

- **Project Manager**: Read-only tools + delegation only (no bash, write, edit)
- **Specialists**: Full tool access within their domain
- **Git Agent**: Version control operations only
- **Code Review Agent**: Read-only analysis tools only

---

## Quality Enforcement Flow

1. Project Manager receives user request
2. Project Manager delegates to @git-agent for issue creation with quality template
3. **🚨 CRITICAL: Git agent enforces main branch protection - blocks all operations on main, auto-creates feature branch**
4. Project Manager delegates to @developer with issue number
5. Developer reads issue, checks context7, completes ALL checkboxes exactly
6. Developer refuses any work not listed in issue
7. Work complete only when all quality gates passed (verified locally)
8. **🚨 CRITICAL: Spawn @adversarial-developer with `sync: true` BEFORE claiming task complete**
9. If adversarial APPROVED → Report completion
10. If adversarial finds issues → Fix and re-spawn

---

## Essential Commands Quick Reference

```bash
# Development
cargo build                    # Build debug
cargo build --release          # Build optimized
cargo test                     # Run tests
cargo clippy -- -D warnings    # Lint (all warnings as errors)
cargo fmt                      # Format code
cargo fmt --check              # Check formatting

# Full validation
cargo fmt --check && cargo clippy -- -D warnings && cargo test

# Git operations (feature/issue-12-example)
git branch --show-current      # Check current branch
git status                     # Check changes
git add AGENTS.md
git commit -m "docs(#12): create AGENTS.md for vipune project governance"
git push -u origin feature/issue-12-example

# Issue management
gh issue view #12              # View issue requirements
gh pr create                    # Create pull request
gh run watch <run-id>         # Watch CI in real-time
```

---

## For Detailed Workflows

See:
- **Issue tracking** - All work requirements are in GitHub issues
- **CI configuration** - `.github/workflows/ci.yml` for automated checks
- **Reference material** - `_reference/remory/` for patterns (temporary git submodule)