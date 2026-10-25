# AGENTS.md

Instructions for AI coding agents working in this repo.

**vipune** is a single-binary, project-scoped semantic memory store in Rust:
add memories, search them by meaning (ONNX embeddings + SQLite FTS5 with
reciprocal-rank fusion), and expose the same store through an MCP server.
The human-facing workflow (setup, testing, release process, debugging tips)
lives in `CONTRIBUTING.md` — this file is agent-only doctrine.

Last updated: 2026-10-25

## Context7 Protocol

Before writing code that touches an external library or crate API, check
Context7 for current documentation. Training data is months out of date;
Context7 has the authoritative current shape of each crate.

This repo's real external surface (from `Cargo.toml`):

- `ort` / `ort-sys` — ONNX Runtime, pinned together at `=2.0.0-rc.9`;
  `download-binaries` feature. The pin is still justified in `Cargo.toml` by
  "compatibility with gline-rs", but gline-rs is not in vipune's dependency
  tree (not in `Cargo.toml`, not in `Cargo.lock`) — the pin is kept at rc.9
  for now and moving to rc.15 with a golden-embedding drift test is tracked in
  issue [#265](https://github.com/randomm/vipune/issues/265)
- `tokenizers` — HuggingFace tokenizer for the embedding pipeline
- `hf-hub` — model download from the HuggingFace Hub (`ureq` feature)
- `rusqlite` — bundled + `backup` features; SQLite is the only storage
- `clap` — CLI argument parsing (derive)
- `rmcp` / `tokio` / `schemars` — MCP server, behind the `mcp` feature, which
  is a **default** feature (`Cargo.toml`: `default = ["mcp"]`), so the default
  build compiles them; `--no-default-features` gives a sync-only tree
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

`CONTRIBUTING.md` (Quality Gates, Dependency advisories) is the human-facing
copy of this chain — it omits `--all-targets` and the `--no-default-features`
pair, so this file keeps its own full chain rather than pointing at it.

Each gate is run separately by CI (`cargo deny` on push and nightly, the
`#[ignore]`d real-model lane on the nightly schedule only).

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo deny --locked check advisories bans
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features
```

`--all-targets` is what CI runs and it is what makes clippy lint the
`[[bench]]` target (`benches/db_search.rs`): that target is `test = false` in
`Cargo.toml` (so plain `cargo test` does not run it) and clippy does not lint
it without `--all-targets` — so without that flag the target can rot with a
green build. The `--no-default-features` pair is a real second build, not an
edge case, because `mcp` is a default feature.

`cargo deny` needs `cargo install cargo-deny` once. It checks the committed `Cargo.lock`
against RustSec (`--locked` is what makes the gate describe the graph we actually ship) and
fails if `base64` 0.23 resolves with `simd-unsafe` — the feature Cargo.toml opts out of because
import files are untrusted input. Config in `deny.toml`; triage policy in
`docs/testing.md`.

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
  this). Several tracked files are already over it (`src/main.rs`,
  `src/sqlite/migrations.rs`, `src/commands/prune.rs`,
  `src/commands/hook_install.rs`), so the cap is the target for new and touched
  code, not a claim about the current tree.
- **Synchronous library core** — `src/lib.rs` is synchronous apart from the
  async runtime (`tokio`), which appears only inside the MCP server — and that
  server is a **default** Cargo feature (`default = ["mcp"]`), so the default
  build *does* compile tokio/rmcp/schemars; build with `--no-default-features`
  for a tokio-free tree
- **Stdlib-first** — do not add a dependency without a concrete, vetted reason
- No `TODO` / `FIXME` / `HACK` comments — the GitHub issue is the TODO; file
  an issue instead
- Comments explain **WHY**, not WHAT
- Errors via `thiserror` for new error types; no `unwrap()` or `expect()` on
  fallible results in library code outside tests. `Mutex::lock()` poisoning is
  the one accepted exception — it is still worth handling rather than
  unwrapping in a long-lived process (the MCP server turns one poison into
  every later request), and `src/mcp/store_wrapper.rs` does both today: four
  acquisitions answer the poison as `INTERNAL_ERROR`, eight bare `unwrap()`s
  do not

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
  worktree — never leave stale branches or worktrees behind, because a stale
  branch is one careless merge away from main.
- **Release PRs**: release-plz opens `chore: release vX.Y.Z` PRs via
  `.github/workflows/release-plz.yml`; a release PR is armed — merging it
  publishes to crates.io irreversibly. If one must be abandoned, close it via
  the GitHub UI **and delete its branch** (remote and local), and never leave a
  release PR open across subsequent merges to main (it can go stale and be
  merged as-is, issue #155).
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
| `src/main.rs` | CLI entry; `to_lib_config()` maps the CLI `config::Config` onto the library `vipune::Config` and feeds **only** `run_mcp` (`main.rs`, `#[cfg(feature = "mcp")]`), so an unmapped field silently takes its `Default` in MCP sessions while the CLI honours the configured value — `vipune::Config` has a hand-written `Default` (`src/config/mod.rs`), so there is no compile-time signal. The guard test is `test_to_lib_config_maps_all_fields_non_default` (`src/main.rs`), itself `#[cfg(feature = "mcp")]`: it asserts each field of **today's 9-field** shape by hand, so a field added to either struct compiles without failing it |
| `src/lib.rs` | Synchronous library core; public API surface (`MemoryStore`, `MemoryType`, `MemoryStatus`, `Config`). `mod sqlite` is **private** — `Memory` escapes only through the `pub use sqlite::Memory` re-export, so the type's public name is `vipune::Memory` |
| `src/memory/` | Domain layer: `store` (`MemoryStore`), `add` (`NewMemory`, `MemoryStore::add_impl`/`insert_row` — the validating write edge that funnels every `vipune add`), `crud` (`UpdateParams`, the enum-typed update boundary), `search`, `batch`, `lifecycle` (types/status) |
| `src/sqlite/` | Storage: `Database`, `Memory` struct, `search`, `fts` (FTS5), `migrations`, `supersede`, `insert`, `update`, `import`/`export_scan` |
| `src/embedding/` | ONNX pipeline; `EMBED_MODEL_ID` / `EMBED_MODEL_REVISION` pin the model |
| `src/config/` | Env + TOML loading, validation, paths |
| `src/project.rs` | Git-root project detection/scoping |
| `src/rrf.rs` | Reciprocal-rank fusion of dense + FTS results |
| `src/temporal.rs` | Recency/decay scoring |
| `src/hook/` | Hook *event* machinery only — `payload.rs` (`HookPayload`, `parse_hook_payload`, `HookEvent`), `extract.rs` (`extract_candidate`, `CREDENTIAL_PATTERNS`), `embedding.rs` (`placeholder_embedding`, `ensure_hook_identity_ok`), `run.rs` (`run_hook_event`, which inserts via `Database::insert_with_hash`). **Not** install/uninstall — those live in `src/commands/hook_install.rs` (`handle_hook_install`/`handle_hook_uninstall`), with event dispatch in `src/commands/hook_run.rs` |
| `src/commands/` | Subcommand handlers (add/search/get/delete/promote/prune/merge/reindex/import/export/backup/doctor/validate/list/update/version). The clap surface is three enums in `mod.rs` — `Commands`, `ProjectCommands` (`Merge`) and `HookCommands` (the five events + `Install`/`Uninstall`); `handlers.rs` projects `Memory` into the `output.rs` DTOs |
| `src/migration.rs` | Model-revision migration: `migrate_model` plus the marker-first crash-safe lifecycle (`write_migration_marker`, `record_identity_and_clear_marker`); reaches for `Database::conn()` directly |
| `src/sqlite/identity.rs` | `ModelIdentity` and the read side of the identity marker (`read_identity`, `is_migrating`, `assert_identity_ok`) — the gate every embedding-capable path must pass |
| `src/embedding_profiles.rs` | `BUILTIN_PROFILES` (bge + `intfloat/multilingual-e5-small`), `profile_for`, and the per-model query/passage prefixes |
| `src/memory_types.rs` | Cross-layer result types (not storage): `AddResult`, `ConflictMemory`, `IngestPolicy`, the batch-ingest types |
| `src/mcp/` | MCP server (`rmcp` + `tokio`) behind the **default** `mcp` feature. Long-lived process over a `Mutex<MemoryStore>`: a poisoned lock must surface as a per-request error, never abort the server |
| `src/errors.rs`, `src/output.rs` | `thiserror` errors; JSON/text response DTOs (`SearchResultItem`, `GetResponse`, …) — `output.rs` declares its **own** DTOs and never imports `Memory` |

**Key invariants:**

- **`Memory` read fan-in** — the `Memory` struct in `src/sqlite/` is the
  single row-level type the read paths share (`src/rrf.rs`, `src/memory/crud.rs`,
  `src/memory/search.rs`, `src/mcp/helpers.rs`,
  `src/sqlite/{list,search,fts,export_scan}.rs`, `src/commands/prune.rs`;
  not exhaustive — `mod sqlite` is private so the public name is
  `vipune::Memory`). `map_row_to_memory`
  (`src/sqlite/query_mod.rs`) is the canonical decoder, but **three readers
  bypass it and decode by literal position** — `sqlite/search.rs`,
  `sqlite/fts.rs` and `sqlite/export_scan.rs` (`scan_all_rows`) — and the
  13-column projection is copy-pasted across `query_mod.rs`, `list.rs`,
  `search.rs`, `export_scan.rs`, `prune.rs` and `Database::get`. Nothing guards
  those offsets (the `EMBEDDING_COLUMN` constant covers column 4 alone), so a
  column reorder silently mis-maps fields unless the mapper *and* the three
  positional decoders are edited together. Find them with
  `rg 'SELECT id, project_id, content, metadata, embedding' src/`.
  `src/output.rs` does **not** import `Memory` — it declares its own DTOs,
  which `src/commands/handlers.rs` fills by projecting `Memory` fields, so a
  rename/retype does break those projections but the compiler catches it in
  `handlers.rs`, not in `output.rs`.
- **`Memory` stores `memory_type`/`status`/`importance` as `String`** with no
  CHECK constraint — validated only at the write boundaries
  (`MemoryType`/`MemoryStatus`/`MemoryImportance::from_str` in
  `src/memory/lifecycle.rs`, called from `src/commands/handlers.rs`,
  `src/commands/mod.rs`, `src/mcp/tools.rs`, and `parse_row` in
  `src/commands/import.rs`). Neither `memory_type` nor `status` is validated on
  any read path, so a bad value that gets in is persisted *and* echoed back to
  the user. `importance` is the exception: the search path rejects it —
  `ImportanceLevel::parse(&memory.importance).map_err(Error::InvalidInput)?`
  (`src/memory/search.rs:95` and `:204`) — so a corrupt value surfaces as a
  search error rather than an echo.
- **`importance` has two independent vocabularies** — the write-side
  `MemoryImportance` (`src/memory/lifecycle.rs`) and the read-side
  `temporal::ImportanceLevel::parse` used by recency scoring, plus `prune`'s
  raw `'high'`/`'critical'` SQL literals. Nothing structural links them: only
  `test_importance_vocabularies_agree_with_the_read_side`
  (`src/memory/lifecycle.rs`) does, and it enumerates the variants **by hand**,
  so it catches a new level only if someone adds it to the test's lists too.
- **`Database::conn()` (`src/sqlite/mod.rs`) is a leaky storage boundary** — it
  hands out the raw `rusqlite::Connection`, and `src/commands/{import,export,
  backup,promote,reindex,prune}.rs`, `src/migration.rs`, `src/hook/` and
  `src/memory/store.rs` bypass `Database` to issue raw SQL against it (not
  exhaustive — `rg 'db\.conn\(\)'` is). New queries belong on `Database`
  unless they need a handle only it can give (the Online Backup API in
  `backup.rs`).
- **Migrations use `PRAGMA user_version`** — version is bumped only after a
  successful migration (the pragma is not transactional); tests assert
  `user_version == total_migrations()` and that a failing migration leaves the
  version unchanged. Each migration runs in a manual `BEGIN EXCLUSIVE` …
  `COMMIT`, and `user_version` is written afterwards via `pragma_update`
  (`src/sqlite/migrations.rs`; the same function issues a `ROLLBACK` on the
  error path, so a migration must not rely on its own partial writes). The
  error path is not tested, so do not lean on it. `migrations()` is a
  positional `Vec`, so **version == index + 1**: never renumber, insert before,
  or remove an applied migration — append a new one. `run_migrations` runs on
  every `Database::open`, so a migration that scans or rewrites rows holds
  `BEGIN EXCLUSIVE` and stalls every concurrent vipune/mcp process — keep new
  migrations DDL-only or bound the work.
- **Embedding pinned revision** — the default model downloads at the exact
  `EMBED_MODEL_REVISION` commit for reproducibility. The test
  `test_air_gapped_docs_and_ci_cache_match_embed_revision` pins that the
  README's air-gapped instructions **and** the CI `HF_CACHE_KEY`
  (`hf-hub-bge-small-en-v1.5-<sha-prefix>` in `ci.yml`) reference the same
  revision — if you bump the revision, update both or the drift test fails.
  The same test pins the **e5** profile too: `HF_CACHE_KEY_E5`
  (`hf-hub-multilingual-e5-small-<sha-prefix>`) and the README `--revision`
  line for `intfloat/multilingual-e5-small`.
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
| All quality gates (pre-push) | `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo deny --locked check advisories bans` |
| Format (write) | `cargo fmt` |
| Lint | `cargo clippy --all-targets -- -D warnings` |
| All tests | `cargo test` |
| A single test | `cargo test <name>` |
| Ignored real-model tests | `cargo test -- --ignored` |
| No-default-features lint + tests (CI job `test-no-default-features`) | `cargo clippy --no-default-features --all-targets -- -D warnings` then `cargo test --no-default-features` |
| Benchmarks (criterion target `db_search`, `benches/db_search.rs`) | `cargo bench` — see README (Benchmarks) for the baseline flags |
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
