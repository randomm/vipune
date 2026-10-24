# AGENTS.md

Instructions for AI coding agents working in this repo.

**vipune** is a single-binary, project-scoped semantic memory store in Rust:
add memories, search them by meaning (ONNX embeddings + SQLite FTS5 with
reciprocal-rank fusion), and expose the same store through an MCP server.
The human-facing workflow (setup, testing, release process, debugging tips)
lives in `CONTRIBUTING.md` — this file is agent-only doctrine.

Last updated: 2026-09-23

## Context7 Protocol

Before writing code that touches an external library or crate API, check
Context7 for current documentation. Training data is months out of date;
Context7 has the authoritative current shape of each crate.

This repo's real external surface (from `Cargo.toml`):

- `ort` / `ort-sys` — ONNX Runtime, pinned together at `=2.0.0-rc.9`
  (originally for gline-rs compatibility, issue #53; gline-rs is no longer in
  vipune's dependency tree — lifting the pin is tracked in issue #254);
  `download-binaries` feature
- `tokenizers` — HuggingFace tokenizer for the embedding pipeline
- `hf-hub` — model download from the HuggingFace Hub (`ureq` feature)
- `rusqlite` — bundled + `backup` features; SQLite is the only storage
- `clap` — CLI argument parsing (derive)
- `rmcp` / `tokio` / `schemars` — MCP server, feature-gated (`mcp`)
- `thiserror` — error types; `serde`/`serde_json`, `chrono`, `uuid`,
  `base64`, `dirs`, `toml` — utilities

## Minimalist Engineering Philosophy

**Every line of code is a liability.** Before creating anything:

- **LESS IS MORE**: Question necessity before creation
- **Challenge Everything**: Ask "Is this truly needed?" before implementing
- **Minimal Viable Solution**: Build the simplest thing that fully solves the problem
- **No Speculative Features**: Don't build for "future needs" — solve today's problem
- **Prefer Existing**: Reuse stdlib and existing code before adding abstractions
- **One Purpose Per Component**: Each function/module should do one thing well

### Pre-Creation Challenge (MANDATORY)

Before creating ANY code, ask:

1. Is this explicitly required by the GitHub issue?
2. Can existing code/tools solve this instead?
3. What's the SIMPLEST way to meet the requirement?
4. Will removing this break core functionality?
5. Am I building for hypothetical future needs?

**If you cannot justify the necessity, DO NOT CREATE IT.**

---

## Pre-Push Quality Gates

**CI is for VERIFICATION, not DISCOVERY.** All gates pass locally before
`git push`. Never push to "see if CI catches anything."

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

Do NOT:

- ❌ Use `--no-verify` to skip hooks
- ❌ Add `#[allow(...)]`, `#[rustfmt::skip]`, or `#[expect(...)]` suppressions
  to bypass a lint — fix the code
- ❌ Push with failing format, lint, or tests
- ❌ Run only the test that exercises your change — run the full suite

---

## Testing Standards

- **TDD preferred**: write the failing test first when behavior is well-specified
- **80%+ coverage for new code** (project convention from
  `docs/testing.md`); no trivial assertions (`assert!(true)`)
- Unit tests in-module (`#[cfg(test)]`) for pure logic; integration tests in
  `tests/` for CLI and DB roundtrips
- **Real-model tests are `#[ignore]d`** — tests that download or run the ONNX
  embedding model (marked with `#[ignore]` in
  `src/embedding/tests/real_model.rs` etc.) are excluded from the normal run
  by design; they cost real downloads. Run them explicitly
  (`cargo test -- --ignored`) only when you changed model-pipeline code
- No skipped or commented-out tests in committed code

---

## Code Style & Conventions

- **500-line hard cap per source file** — refactor into submodules past the
  limit (the `memory/`, `sqlite/`, `config/`, `hook/` splits exist because of
  this)
- **Synchronous library core** — `src/lib.rs` is fully synchronous,
  stdlib-first; the async runtime (`tokio`) exists only inside the feature-gated
  MCP server
- **Stdlib-first** — do not add a dependency without a concrete, vetted reason
- No `TODO` / `FIXME` / `HACK` comments — the GitHub issue is the TODO; file
  an issue instead
- Comments explain **WHY**, not WHAT
- Errors via `thiserror`; no `unwrap()` in library code outside tests

---

## Git Workflow

- **Conventional commits** with a scope-free or **alphabetic** scope:
  `fix: correct BLOB size validation`, `feat(sqlite): add prune`.
- **NEVER an issue-number scope in commit subjects or PR titles** — the
  `fix(#173):` form. This repo squash-merges, so the PR title becomes the
  commit subject; release-plz's default preprocessor rewrites an issue-number
  scope into `fix([#173](url))`, which git-cliff **silently drops from the
  changelog** while still counting the commit toward the version bump — the
  loss is invisible. Root cause and full analysis: issue [#155](https://github.com/randomm/vipune/issues/155).
  Enforced by `.github/workflows/pr-title-check.yml`, which rejects any PR
  title matching `type(<scope with #NNN>):` and explains the rule on rejection.
- **Issue linkage goes in the PR body**: `Closes #N` (or `Fixes #N`) — that
  line is what keeps the issue linked and auto-closes it on merge, since the
  changelog entry only shows the auto-appended PR number.
- **Branch naming**: `feature/issue-N-short-slug`, `fix/issue-N-short-slug`,
  `chore/issue-N-short-slug` (issue numbers in branch names are fine — the
  rule is about commit subjects and PR titles only).
- Squash-merge by default. No force-push to `main`.
- **Release PRs**: release-plz opens `chore: release vX.Y.Z` PRs via
  `.github/workflows/release-plz.yml`; a release PR is armed — merging it
  publishes to crates.io irreversibly. If one must be abandoned, close it via
  the GitHub UI, and never leave a release PR open across subsequent merges to
  main (it can go stale, issue #155).
- **This file overrides the generic pi-ensemble instruction files**
  (`AGENTS.md` templates, `modules/workflows/git-workflow.md`, etc.) that
  still prescribe the issue-number-scope form.

---

## Merge Authority

### Merging

LLMs are allowed to squash merge PRs if all project quality gates have been met (code reviews, CI, linters, type checks).

This grant covers PRs for issues filed by the operator's team, including the release-plz release PRs (`chore: release vX.Y.Z`) opened by .github/workflows/release-plz.yml.

The operator has granted this in writing (2026-09-23).

**This grant sentence is load-bearing, not decorative.** The `/work` driver's
policy judge reads this file and quotes the grant sentence verbatim; deleting
or softening the first sentence above turns auto-merge off for this repo.
Keep the sentence verbatim if you edit this section.

If a gate is not met, stop at PR creation, apply the needs-human-attention
label, and notify the operator.

---

## Issue-Driven Development

- **No work without a GitHub issue** — every code change links to an issue via
  `Closes #N` in the PR body
- Research/analysis tasks are exempt from the issue requirement
- Scope discipline: match the issue scope exactly; out-of-scope findings
  become new issues, not drive-by commits
- PRs are small: one issue per PR, squash-merged

---

## Architecture Notes

Orientation for agents new to the codebase (verify against the tree — the
`Memory` struct and these paths are load-bearing):

| Module | Responsibility |
|--------|---------------|
| `src/main.rs` | CLI entry; `to_lib_config()` maps the CLI `config::Config` **totally** onto the library `vipune::Config` — a test asserts every field maps, keep it exhaustive |
| `src/lib.rs` | Synchronous library core; public API surface (`MemoryStore`, `MemoryType`, `MemoryStatus`, `Config`) |
| `src/memory/` | Domain layer: `MemoryStore`, `crud`, `search`, `batch`, `lifecycle` (types/status) |
| `src/sqlite/` | Storage: `Database`, `Memory` struct, `search`, `fts` (FTS5), `migrations`, `supersede`, `import`/`export_scan` |
| `src/embedding/` | ONNX pipeline; `EMBED_MODEL_ID` / `EMBED_MODEL_REVISION` pin the model |
| `src/config/` | Env + TOML loading, validation, paths |
| `src/project.rs` | Git-root project detection/scoping |
| `src/rrf.rs` | Reciprocal-rank fusion of dense + FTS results |
| `src/temporal.rs` | Recency/decay scoring |
| `src/hook/` | Hook install/run (embedding hook payloads) |
| `src/commands/` | Subcommand handlers: add/search/get/delete/promote/prune/merge/reindex/import/export/backup/doctor |
| `src/mcp/` | Feature-gated MCP server (`rmcp` + `tokio`) |
| `src/errors.rs`, `src/output.rs` | `thiserror` errors; JSON/text output shaping |

**Key invariants:**

- **`Memory` read fan-in** — the `Memory` struct in `src/sqlite/` is the
  single row-level type every command, store, and output path shares;
  changes to its fields ripple across `memory/`, `commands/`, `output.rs`,
  and `mcp/`.
- **Migrations use `PRAGMA user_version`** — version is bumped only after a
  successful migration (the pragma is not transactional); tests assert
  `user_version == total_migrations()` and that a failing migration leaves the
  version unchanged. Never edit an applied migration — append a new one.
- **Embedding pinned revision** — the default model downloads at the exact
  `EMBED_MODEL_REVISION` commit for reproducibility. The test
  `test_air_gapped_docs_and_ci_cache_match_embed_revision` pins that the
  README's air-gapped instructions **and** the CI `HF_CACHE_KEY`
  (`hf-hub-bge-small-en-v1.5-<sha-prefix>` in `ci.yml`) reference the same
  revision — if you bump the revision, update both or the drift test fails.
- **Release config**: `release-plz.toml` sets `features_always_increment_minor
  = true` (0.x crate, Conventional-Commits reading: `feat` → minor) and
  publishes to crates.io with trusted publishing; tags follow `vX.Y.Z` to
  trigger cargo-dist.

---

## Using vipune for memory

An agent working ON this repo can USE vipune for its own memory (project
scoping is automatic via git root):

```bash
# Store a durable finding
vipune add 'query semantics work best with [domain] [component] patterns'

# Search by meaning
vipune search "query patterns effective"
```

Store conventions, gotchas, and decisions as you work so the next session
starts smarter. One claim per memory. Never store secrets (tokens,
passwords) — the store is plain SQLite.

---

## Documentation Policy

### The 200-PR Test

Before creating documentation, ask: "Will this be true in 200 PRs?"

- **YES** (principle that endures) → document the principle (WHY)
- **NO** (implementation detail) → skip, or use a code comment

### Forbidden Documentation

- ❌ ALL_CAPS scratch files (`PLAN.md`, `RESEARCH.md`, `ANALYSIS.md`,
  `SUMMARY.md`, `IMPLEMENTATION_PLAN.md`, `TODO.md`, …) — agent work
  artifacts stay out of git. **The issue IS the TODO.**
- ❌ Manually editing released `CHANGELOG.md` entries — release-plz owns that
  file
- ❌ Manually creating git tags — release-plz handles those

### Documentation Locations

- Root: `README.md` (user-facing), `AGENTS.md` (agent doctrine)
- `docs/` with lowercase-hyphenated filenames (`architecture.md`,
  `testing.md`, `search.md`, …)

---

## Commands Reference

| Task | Command |
|------|---------|
| All quality gates (pre-push) | `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test` |
| Format (write) | `cargo fmt` |
| Lint | `cargo clippy --all-targets -- -D warnings` |
| All tests | `cargo test` |
| A single test | `cargo test <name>` |
| Ignored real-model tests | `cargo test -- --ignored` |
| Release build | `cargo build --release` |
| Run the CLI locally | `cargo run -- add "test memory"` |
| Query semantics docs | see `docs/search.md`, `docs/vipune-query-guide.md` |

---

## Operator choices

- **Review-blocking severity**: MEDIUM and above blocks a merge —
  CRITICAL, HIGH, and MEDIUM findings must be resolved or explicitly
  dispositioned before the merge step; LOW findings may be noted and carried
  forward.
- **Coverage threshold**: 80%+ for new code (see Testing Standards).
