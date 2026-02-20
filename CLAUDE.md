# CLAUDE.md — vipune for Claude Code Agents

This document helps AI agents working on vipune understand the codebase and how to use vipune itself for memory.

## Quality Gates

Before pushing any changes:

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

All three checks must pass. No exceptions, no bypasses (`--no-verify`, suppressions, etc.).

## Architecture

**Single Binary, Synchronous Rust:**
- All code compiles to one CLI executable
- No async/await, no tokio, no concurrent runtime
- Fully synchronous Rust (stdlib-first philosophy)
- Max 500 lines per source file (refactor if exceeded)

**Key Modules:**
- `main.rs` — CLI entry point, command parsing
- `memory/` — Core add/search/get/delete operations
- `embedding/` — ONNX model loading and vector inference
- `db/` — SQLite backend with vector search
- `config/` — Environment/TOML config handling
- `project/` — Git auto-detection for project scoping

## Using vipune as Memory Backend

An agent working ON this repo can also USE vipune for its own memory:

### Add findings to memory:
```bash
vipune add "Query semantics work best with [domain] [component] patterns - tested across 7 categories with bge-small-en-v1.5"
vipune add "Maximum 500 lines per file enforced in AGENTS.md - refactor into separate modules when approaching limit"
```

### Search by meaning:
```bash
vipune search "query patterns effective"
vipune search "file size limits architecture"
```

This creates persistent memory across tasks — store patterns, conventions, gotchas, and decisions as you work.

## Development Workflow

1. Read the GitHub issue fully
2. Search memory for related prior work
3. Implement with tests first
4. Run quality gates locally before push
5. Store learnings in memory for future tasks

## Quick Commands

| Task | Command |
|------|---------|
| Run tests | `cargo test` |
| Lint | `cargo clippy -- -D warnings` |
| Format check | `cargo fmt --check` |
| Format code | `cargo fmt` |
| All gates | `cargo fmt --check && cargo clippy -- -D warnings && cargo test` |
| Build release | `cargo build --release` |

## Key Constraints

- **No premature optimization** — profile first
- **No TODO comments** — create GitHub issues instead
- **No `# noqa` or `@ts-ignore`** — fix the actual issue
- **Issue-driven only** — all work matches GitHub issue scope exactly

See `AGENTS.md` for complete project guidelines.
