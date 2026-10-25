# AGENTS.md

Instructions for AI coding agents working in this repo.

**vipune** is a single-binary, project-scoped semantic memory store in Rust:
add memories, search them by meaning (ONNX embeddings + SQLite FTS5 with
reciprocal-rank fusion), and expose the same store through an MCP server.
The human-facing workflow (setup, testing, release process, debugging tips)
lives in `CONTRIBUTING.md` — this file is agent-only doctrine.

Last updated: 2026-10-24

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
- `rmcp` / `tokio` / `schemars` — MCP server, behind the `mcp` feature, which is
  a **default** feature (`Cargo.toml`: `default = ["mcp"]`) — the default build
  compiles them; use `--no-default-features` for a sync-only build
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

The per-gate commands are below. The same chain is documented for humans in
`CONTRIBUTING.md` (Quality Gates).

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features
```

`--all-targets` is what CI runs, and it is what makes clippy compile
`benches/db_search.rs` (a `test = false` bench target) — without it that target
can rot with a green build. Run both pairs — `default = ["mcp"]`, so the
`--no-default-features` tree is a real second build, not an edge case. CI's
per-push job is this chain; the `#[ignore]`d real-model lane
(`cargo test -- --ignored`) runs only on the nightly schedule.

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
  this). Several tracked files are already over it — check with
  `find src -name '*.rs' | xargs wc -l | sort -n` (`src/**/*.rs` under default
  bash silently drops top-level `src/*.rs`, including `src/main.rs`) — the cap
  is the target for new and touched code, not a claim about the current tree.
- **Synchronous library core** — `src/lib.rs` is fully synchronous,
  stdlib-first; the async runtime (`tokio`) exists only inside the MCP server,
  which is a **default** Cargo feature (`default = ["mcp"]`) — build with
  `--no-default-features` for a sync-only, tokio-free tree
- **Stdlib-first** — do not add a dependency without a concrete, vetted reason
- No `TODO` / `FIXME` / `HACK` comments — the GitHub issue is the TODO; file
  an issue instead
- Comments explain **WHY**, not WHAT
- Errors via `thiserror` for new error types; no `unwrap()` or `expect()` on
  fallible results in library code outside tests. `Mutex::lock()` poisoning is
  exempt from the rule, but it is handled three ways in this tree, not one:
  recover with `into_inner()` where a panic would take down sibling work (see
  `lock_env_mutex`, `src/config/tests_utils.rs`); map the poison to a typed
  `rmcp::ErrorData::INTERNAL_ERROR` at a request boundary (see `StoreWrapper`,
  `src/mcp/store_wrapper.rs`); `unwrap()` is also present at eight acquisitions
  in `src/mcp/store_wrapper.rs` — in a long-lived process that turns one poison
  into every later request, so prefer the first two forms for new code.

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
- **After a PR lands**: delete its branch (remote and local) and remove its
  worktree. Never leave stale branches or worktrees behind. For release-plz
  PRs see `CONTRIBUTING.md` (Release Process hard rules) — including deleting
  the branch of a superseded release PR, remote and local, so it cannot be
  merged after later merges to main.
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

The authoritative module map is `docs/architecture.md` (Module Map). This
section carries only the invariants an agent cannot recover from the compiler —
verify against the tree; the `Memory` struct and these paths are load-bearing:

| Module | Responsibility |
|--------|---------------|
| `src/main.rs` | CLI entry; `to_lib_config()` maps the CLI `config::Config` onto the library `vipune::Config` and feeds **only** `run_mcp`, so an unmapped field silently takes its `Default` in MCP sessions while the CLI honours the configured value — `vipune::Config` derives `Default`, so there is no compile or runtime signal. The guard test is `#[cfg(feature = "mcp")]` and its asserts are hand-written (verify both against the tree before trusting them) |
| `src/lib.rs` | Synchronous library core; public API surface (`MemoryStore`, `MemoryType`, `MemoryStatus`, `Config`). `mod sqlite` is **private** — `Memory` escapes only through the `pub use sqlite::Memory` re-export, so the type's public name is `vipune::Memory` |
| `src/memory/` | Domain layer: `store` (`MemoryStore`), `add` (`NewMemory`, `MemoryStore::add_impl`/`insert_row` — the *validating* write edge; `hook/run.rs` reaches the insert sinks directly and bypasses it), `crud` (`UpdateParams`, the enum-typed update boundary), `search`, `batch`, `lifecycle` (types/status) |
| `src/mcp/` | MCP server (`rmcp` + `tokio`) behind the **default** `mcp` feature. Long-lived process over a `Mutex<MemoryStore>`: a poisoned lock must surface as a per-request `INTERNAL_ERROR`, never abort the server (precedent: the four `map_err` sites in `store_wrapper.rs`) |
| `src/errors.rs`, `src/output.rs` | `thiserror` errors; JSON/text response DTOs (`SearchResultItem`, `GetResponse`, …) |

**Key invariants:**

- **`Memory` read fan-in** — the `Memory` struct (`src/sqlite/mod.rs`) is the
  single row-level type the read paths share — e.g. `src/sqlite/query_mod.rs`,
  `src/sqlite/list.rs`, `src/sqlite/fts.rs`, `src/sqlite/export_scan.rs`,
  `src/commands/prune.rs`, `src/memory/crud.rs`, `src/rrf.rs`,
  `src/mcp/helpers.rs`, `src/mcp/store_wrapper.rs` (not exhaustive;
  `rg 'sqlite::Memory|crate::Memory|vipune::Memory'` is — `sqlite` is a private
  module, so the public name is `vipune::Memory`).
  `map_row_to_memory` (`src/sqlite/query_mod.rs`) is the **canonical** decoder —
  used by `query_mod.rs`, `sqlite/mod.rs` (`Database::get`), `sqlite/list.rs`
  and `commands/prune.rs`. Three further readers bypass the mapper and decode
  by literal position in their own closure, so they must be edited in lockstep
  with the projection: `sqlite/search.rs`, `sqlite/fts.rs` and
  `sqlite/export_scan.rs` (`scan_all_rows`). The 13-column projection is
  duplicated verbatim in `query_mod.rs`, `list.rs`, `search.rs`,
  `export_scan.rs`, `prune.rs` and `mod.rs` (`Database::get`) — find them with
  `rg 'SELECT id, project_id, content, metadata, embedding' src/`; `fts.rs`
  carries the same 13 columns `m.`-qualified plus a trailing `bm25_score`, so
  it is positional-compatible but not verbatim. A column reorder therefore has
  to be fixed in four decoders and silently mis-maps fields if it is not. The
  row shape is enforced only by those hard-coded 0–12 offsets; the
  `EMBEDDING_COLUMN` constant in `src/sqlite/mod.rs` covers column 4 alone and
  shares no guard with the rest. The projection is also **unbounded** —
  `sqlite/search.rs` fetches every row of a project and applies `limit` after
  fetch (capped only by `MAX_SEARCH_LIMIT = 10_000`, `src/memory/store.rs`),
  so every added column is paid per row on the search hot path.
  `src/output.rs` does **not** import `Memory` — it declares its own response
  DTOs, which `src/commands/handlers.rs` fills by projecting `Memory` fields.
  So a field rename/retype does break those projections, but the compiler
  catches it in `handlers.rs`, not in `output.rs`.
- **`Memory` stores `memory_type`/`status`/`importance` as `String`**, not as
  enums (`src/sqlite/mod.rs`) — no type and no CHECK constraint backs any of
  them. Validation happens at the write boundaries only — `add`
  (`src/commands/handlers.rs`, `src/commands/mod.rs`) via
  `MemoryType`/`MemoryStatus::from_str` and `MemoryImportance::from_str`
  (`src/memory/lifecycle.rs`), plus `run_import`'s `parse_row`
  (`src/commands/import.rs`), which validates `importance` only; `src/mcp/tools.rs`
  validates type and status only. No read path validates: the same `String`s
  are projected straight into `output.rs` DTOs by `src/commands/handlers.rs`
  (three sites) and `src/main.rs`, so a bad value is both persisted and
  echoed. Writers that bypass it: `src/hook/run.rs` binds
  `"observation"`/`"candidate"` literals through `Database::insert_with_hash`
  (`&str` params, `src/sqlite/mod.rs`, which also binds `"medium"` for
  `importance`); `run_import` binds `ImportRow.memory_type`/`status` verbatim
  from JSON, and its `content`/`metadata` are written with no content-level
  check — importing an untrusted JSONL file is the one unvalidated secret sink;
  `parse_row` canonicalises `importance` before the INSERT binds it blind; the
  update path takes `Option<&str>` (`UpdateOptions`, `src/sqlite/update.rs`)
  and `src/commands/promote.rs` writes `status: Some("active")` unvalidated, as
  does the `"active"` literal in `build_filters` (`src/sqlite/query_mod.rs`).
  `UpdateOptions` is the `&str`-typed twin of the enum-typed `UpdateParams`
  (`src/memory/crud.rs`) — reaching for the storage-layer struct compiles with
  any string, so prefer the `MemoryStore`/`UpdateParams` path.
  The sinks (`src/sqlite/insert.rs`, `supersede.rs`, and `sqlite/import.rs`'
  `insert_with_id`, which binds only **12** columns and leaves `importance` to
  the column DEFAULT) take `&str` for the columns they accept, so a new entry
  point writes unvalidated strings unless it validates itself — and those same
  input boundaries are the only place a secret can be refused, since
  `content`/`metadata` land unredacted in a plain SQLite file that any local
  process can read. (`src/hook/extract.rs`'s `CREDENTIAL_PATTERNS` is a
  hook-path-only screen; `add`/`update`/MCP `store_memory` enforce nothing.)
- **`importance` has a second, independent vocabulary on the read side** —
  `temporal::ImportanceLevel::parse` plus the raw `'high'`/`'critical'` SQL
  literals in `src/commands/prune.rs`. Nothing structural links the two enums:
  only `test_importance_vocabularies_agree_with_the_read_side`
  (`src/memory/lifecycle.rs`), which pins *rejection* agreement over 11 words
  (case and whitespace variants included) and asserts every `MemoryImportance`
  variant parses on the read side — so an added level is caught by the pre-push
  chain. The silent path is a value that reaches the column through one of the
  non-validating writers above, and the two read sides fail in opposite
  directions: `ImportanceLevel::parse` at `src/memory/search.rs:95`/`:204`
  returns `Error::InvalidInput`, so one bad row aborts every recency-weighted
  search, while prune's raw literals fail soft, protecting nothing and demoting
  rows that should have been kept.
- **`Database::conn()` is a leaky storage boundary** (`src/sqlite/mod.rs`) — it
  hands out the raw `rusqlite::Connection`, and `src/commands/import.rs`,
  `src/commands/export.rs`, `src/commands/backup.rs`, `src/commands/promote.rs`,
  `src/commands/reindex.rs`, `src/commands/prune.rs` and `src/migration.rs`
  bypass `Database` to issue raw SQL against it, as do the identity gates in
  `src/memory/store.rs:138` and `src/hook/` (not exhaustive; `rg 'db\.conn\(\)'`
  is). New queries belong on `Database`
  unless they need a handle only it can give (e.g. the Online Backup API in
  `backup.rs`).
- **Migrations use `PRAGMA user_version`** — version is bumped only after a
  successful migration (the pragma is not transactional); tests assert
  `user_version == total_migrations()` and that a failing migration leaves the
  version unchanged. Each migration runs inside a manual `BEGIN EXCLUSIVE` …
  `COMMIT` (with `ROLLBACK` on error) and `user_version` is written afterwards
  via `pragma_update`. `migrations()` is a positional `Vec`, so **version ==
  index + 1**: never renumber, insert before, or remove an applied migration —
  append a new one. Two branches are untested and are where a version/file
  mismatch would appear: a failing `ROLLBACK` replaces the migration's error
  with the rollback's, and a failing `COMMIT` (e.g. `SQLITE_BUSY`) propagates
  with no rollback. `run_migrations` runs on every `Database::open`, so a
  migration that scans or rewrites rows (v4's backfill + windowed `DELETE`)
  holds `BEGIN EXCLUSIVE` and stalls every concurrent vipune/mcp process — keep
  new migrations DDL-only or bound the work.
- **Embedding pinned revision** — the default model downloads at the exact
  `EMBED_MODEL_REVISION` commit for reproducibility. The test
  `test_air_gapped_docs_and_ci_cache_match_embed_revision` pins that the
  README's air-gapped instructions **and** the CI `HF_CACHE_KEY`
  (`hf-hub-bge-small-en-v1.5-<sha-prefix>` in `ci.yml`) reference the same
  revision — if you bump the revision, update both or the drift test fails.
  The same test also pins the **e5 profile**: `HF_CACHE_KEY_E5`
  (`hf-hub-multilingual-e5-small-<sha-prefix>`) and the README `--revision` line
  for `intfloat/multilingual-e5-small`, so bumping either profile's revision
  means updating both cache keys plus the README.
- **Release config**: `release-plz.toml` sets `features_always_increment_minor
  = true` (0.x crate, Conventional-Commits reading: `feat` → minor) and
  publishes to crates.io using the `CARGO_REGISTRY_TOKEN` repository secret
  (see `.github/workflows/release-plz.yml`); tags follow `vX.Y.Z` to trigger
  cargo-dist.

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

`vipune` is a **bash binary on `$PATH`** — invoke it via bash
(`vipune search "…"`, `vipune add "…"`); a structured tool call named `vipune`
has no handler and fails. Tool-integration mechanics (MCP config, Claude
Desktop, absolute binary paths) live in
[`docs/agent-integration.md`](docs/agent-integration.md).

Store conventions, gotchas, and decisions as you work so the next session
starts smarter. One claim per memory.

### Never store secrets

Never store secrets (tokens, passwords) — the store is plain SQLite, and
nothing enforces it: the only credential screen is the hook path's
`CREDENTIAL_PATTERNS` (`src/hook/extract.rs`), so `add`, `update` and the MCP
`store_memory` tool write whatever they are given, and a secret that lands is
readable by any local process.

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
  `testing.md`, `search.md`, `agent-integration.md`, …)

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
| No-default-features lint + tests (CI runs both; see Pre-Push Quality Gates) | `cargo clippy --no-default-features --all-targets -- -D warnings` then `cargo test --no-default-features` |
| Benchmarks (criterion target `db_search`, `benches/db_search.rs`) | see README (Benchmarks) — `cargo bench -- --save-baseline main` records the baseline; the corpus re-seeds on every run |
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
