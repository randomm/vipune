# AGENTS.md

Instructions for AI coding agents working in this repo.

**vipune** is a single-binary, project-scoped semantic memory store in Rust:
add memories, search them by meaning (ONNX embeddings + SQLite FTS5 with
reciprocal-rank fusion), and expose the same store through an MCP server.
The human-facing workflow (setup, testing, release process, debugging tips)
lives in `CONTRIBUTING.md` — this file is agent-only doctrine.

Last updated: 2026-10-26

## Context7 Protocol

Before writing code that touches an external library or crate API, check
Context7 for current documentation. Training data is months out of date;
Context7 has the authoritative current shape of each crate.

This repo's real external surface (from `Cargo.toml`):

- `ort` / `ort-sys` — ONNX Runtime, pinned together at `=2.0.0-rc.15`
  (`ndarray` + `download-binaries` features). The rc pin is kept for the
  memory-safety/UB fixes rc.15 carries; stable 2.0.0 is unreleased, and
  `ort` declares exactly the matching `ort-sys`, so the two always move together
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
copy of this chain, but a weaker one: it runs `cargo clippy -- -D warnings`
without `--all-targets` and has no `--no-default-features` lane, so this file
keeps its own full chain rather than pointing at it.

Each gate is run separately by CI. `cargo deny` runs on PR, on push to `main`,
and nightly; the `#[ignore]`d real-model lane runs only on the schedule
(`if: github.event_name == 'schedule'`).

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
edge case, because `mcp` is a default feature (see the dependency list above).

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
  this). Some tracked files are already over it — find them with
  `wc -l src/**/*.rs` — the cap is the target for new and touched code, not a
  claim about the current tree.
- **Synchronous library core** — `src/lib.rs` is synchronous apart from the
  async runtime (`tokio`), which appears only inside the MCP server (see the
  `mcp` feature note above: default on, `--no-default-features` for the
  tokio-free tree)
- **Stdlib-first** — do not add a dependency without a concrete, vetted reason
- No `TODO` / `FIXME` / `HACK` comments — the GitHub issue is the TODO; file
  an issue instead
- Comments explain **WHY**, not WHAT
- Errors via `thiserror` for new error types; no `unwrap()` or `expect()` on
  fallible results in library code outside tests. `Mutex::lock()` poisoning is
  the one accepted exception — it is still worth handling rather than
  unwrapping in a long-lived process, where one poison would fail every later
  request — `src/mcp/store_wrapper.rs` handles some lock acquisitions as
  `INTERNAL_ERROR` and `unwrap()`s the rest; no test asserts either behaviour

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
- **Disk hygiene** — scratch work is the disk-filling habit, so it has rules.
  They are stated as principles (the 200-PR test): they hold on any host, and
  this file names no host path, mount point or quota size for that reason.
  - **Leave `TMPDIR` as the environment sets it.** Never redirect temp or
    scratch output into a shared system directory to get around a full disk
    or a full quota — those directories carry no per-user limit and age out
    only slowly, so a runaway task fills a filesystem other people share.
    Report the full disk instead of routing around it.
  - **Work in a git worktree, not a copy.** A scratch or review build is a
    worktree of this repo (`.worktrees/` is gitignored) sharing one
    `CARGO_TARGET_DIR` — for example the main checkout's `target/` directory —
    never a full copy of the repo, which compiles a fresh target directory of
    its own. Concurrent builds on the shared target directory wait on cargo's
    build lock; that wait is acceptable and far cheaper than duplicate target
    directories. The shared target directory belongs to the main checkout and
    is never deleted by task or post-merge cleanup.
  - **Reviewers: don't rebuild what CI already verified.** When CI is green on
    the exact head under review, do not re-run full builds or test suites. If
    the review question genuinely needs a build (a reproducer, a
    platform-specific failure), build once in the shared target directory.
    This is a reviewer-side rule only: the author of a change still runs the
    whole Pre-Push Quality Gates chain locally before pushing.
  - **A task cleans up after itself.** Any scratch a task creates — worktrees,
    repo copies, probe files, temp directories, private build directories — is
    removed by that task before it reports, and only paths it created itself:
    never a shared or pre-existing directory it merely used.
- **After a PR lands**: delete its branch (remote and local) and remove its
  worktree **including its build directory** — meaning a build directory
  private to that worktree, which exists only if the worktree did not use the
  shared target directory, never the shared one — along with any stash created for
  that branch — drop them one at a time only after confirming the stash message
  names the branch, because `git stash list` shares one stack across branches
  and renumbers on every drop. Never leave stale branches, worktrees or their
  build output behind, because a stale branch is one careless merge away from
  main and a stale worktree keeps a whole `target/` directory alive.
- **Release PRs**: release-plz opens `chore: release vX.Y.Z` PRs via
  `.github/workflows/release-plz.yml`; a release PR is armed — merging it
  publishes to crates.io irreversibly. If one must be abandoned, close it via
  the GitHub UI (never leave a release PR open across subsequent merges to
  main — it can go stale and be merged as-is, issue #155).
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
| `src/main.rs` | CLI entry; declares **no** modules and shares the library's one `Config` type (issue #267), so it feeds the loaded `Config` straight to `run_mcp` (`#[cfg(feature = "mcp")]`) — there is no field-by-field mapping to drift |
| `src/lib.rs` | Synchronous library core; public API surface (`MemoryStore`, the lifecycle enums, `Config`) |
| `src/memory/` | Domain layer: store, the validating add edge that funnels every `vipune add`, the enum-typed update boundary, search, batch, and the lifecycle types |
| `src/sqlite/` | Storage: `Database`, `Memory` struct, search, FTS5, migrations, and the supersede/insert/update/import paths |
| `src/embedding/` | ONNX pipeline; `EMBED_MODEL_ID` / `EMBED_MODEL_REVISION` pin the model |
| `src/config/` | Env + TOML loading, validation, paths |
| `src/project.rs` | Git-root project detection/scoping |
| `src/rrf.rs` | Reciprocal-rank fusion of dense + FTS results |
| `src/temporal.rs` | Recency/decay scoring |
| `src/hook/` | Hook *event* machinery only (payload parsing, candidate extraction, placeholder embeddings, `run_hook_event`). **Not** install/uninstall — those live in `src/commands/hook_install.rs` |
| `src/commands/` | Subcommand handlers **and the CLI's JSON DTOs** (`output.rs`); the crate is the shared surface the binary consumes (issue #267 — `src/main.rs` declares no modules). The clap surface is the `Commands` / `ProjectCommands` / `HookCommands` enums in `mod.rs`, and `handlers.rs` projects `Memory` into the `output.rs` DTOs |
| `src/migration.rs` | Model-revision migration: `migrate_model` plus the marker-first crash-safe lifecycle; reaches for `Database::conn()` directly |
| `src/sqlite/identity.rs` | `ModelIdentity` and the read side of the identity marker — the gate every embedding-capable path must pass |
| `src/embedding_profiles.rs` | `BUILTIN_PROFILES` (bge + `intfloat/multilingual-e5-small`), `profile_for`, and the per-model query/passage prefixes |
| `src/memory_types.rs` | Cross-layer result types (not storage): conflict and batch-ingest results |
| `src/mcp/` | MCP server (`rmcp` + `tokio`) behind the **default** `mcp` feature. Long-lived process over a `Mutex<MemoryStore>`: a poisoned lock must surface as a per-request error, never abort the server |
| `src/errors.rs`, `src/commands/output.rs` | `thiserror` errors; JSON/text response DTOs — `output.rs` declares its **own** DTOs and never imports `Memory` |

**Key invariants:**

- **`Memory` read fan-in** — `Memory` (`src/sqlite/mod.rs`) is the single
  row-level type the read paths share (`mod sqlite` is **private** in
  `src/lib.rs`, so inside the crate it is `crate::sqlite::Memory` and
  `vipune::Memory` is the public name). Discover consumers with
  `rg 'sqlite::Memory' src/`. Two hazards:
  - **Positional decoders that bypass `map_row_to_memory`**: the search and FTS
    rows construct `Memory { .. }` from literal `row.get(N)` offsets, and
    `sqlite/export_scan.rs` (`scan_all_rows`) decodes the same offsets into
    `ExportRow`. The roster is representative, not complete — other raw
    positional reads exist (e.g.
    `Database::list_all_rows_for_project` decodes `id`/`content`/`embedding`
    by offset). Discover with `rg 'row\.get(' src/` plus
    `rg 'map_row_to_memory' src/` (who re-uses the mapper).
  - **Callers that reuse the projection and decode via `map_row_to_memory`**,
    re-selecting the 13-column projection themselves (e.g. `sqlite/list.rs`,
    `commands/prune.rs`, `Database::get`).

  Nothing guards those offsets (the `EMBEDDING_COLUMN` constant covers the
  embedding column alone), so a column reorder silently mis-maps fields unless
  the mapper *and* every positional decoder are edited together. The
  "12-column" comments in `src/sqlite/query_mod.rs` and `src/sqlite/mod.rs` are
  **stale** — the shape is 13 columns.

  `src/commands/output.rs` does **not** import `Memory` — it declares its own
  DTOs, which `src/commands/handlers.rs` fills by projecting `Memory` fields, so
  a rename/retype does break those projections but the compiler catches it in
  `handlers.rs`, not in `output.rs`.
- **`Memory` stores `memory_type`/`status`/`importance` as `String`** with no
  CHECK constraint, and validation is per column — the enum `from_str` helpers
  in `src/memory/lifecycle.rs` name the *validating edges*, not every write.
  Find the callers with `rg 'Memory(Type|Status|Importance)::from_str' src/`;
  `commands/import.rs` `parse_row` validates `importance` **only**, so imported
  `memory_type`/`status` are inserted unvalidated. The MCP update tool parses
  but the write itself goes through `store_wrapper`/`UpdateParams` (enum-typed,
  `src/memory/crud.rs`). The storage layer takes all three as `&str` and
  validates nothing: `Database::insert*`/`update` and the hook path
  (`src/hook/run.rs` → `Database::insert_with_hash`).
  On the **read** side: `memory_type` is never validated; `status` is only
  *compared* (`prune` compares against `MemoryStatus::Candidate.as_str()`);
  `importance` is parsed by `temporal::ImportanceLevel::parse` only inside the
  `if recency_weight > 0.0` block in `src/memory/search.rs`. The default
  `recency_weight` is **0.3** (`Config::default()`, `src/config/mod.rs`; the
  CLI's `--recency` and the MCP `recency_weight` param both fall back to it), so
  a corrupt `importance` makes search **fail** by default and is echoed back
  unvalidated only when recency is explicitly set to `0`.
- **`importance` has two independent vocabularies** — the write-side
  `MemoryImportance` (`src/memory/lifecycle.rs`) and the read-side
  `temporal::ImportanceLevel::parse` used by recency scoring, plus `prune`'s
  raw `'high'`/`'critical'` SQL literals. Nothing structural links them: only
  `test_importance_vocabularies_agree_with_the_read_side`
  (`src/memory/lifecycle.rs`) does, and it enumerates the variants **by hand**,
  so it catches a new level only if someone adds it to the test's lists too.
- **`Database::conn()` (`src/sqlite/mod.rs`) is a leaky storage boundary** — it
  hands out `&rusqlite::Connection`, which can **write**: several callers
  `execute` raw SQL through it (`rg '\.conn\(\)' src/` for the current set).
  New queries belong on `Database` unless they genuinely need connection-level
  access — the Online Backup API, or `PRAGMA`s such as `busy_timeout`.
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
  every `Database::open` — reached by the CLI via `MemoryStore::new` and the
  per-command `Database::open` calls, and by the MCP server via
  `MemoryStore::new` in `run_mcp` — and each migration holds `BEGIN EXCLUSIVE`
  until it commits, so a migration that scans or rewrites rows blocks every
  concurrent vipune/mcp process for that whole window: keep new migrations
  DDL-only or bound the work. The model-revision re-embed in `src/migration.rs`
  is *not* a schema migration: its per-project `reembed_project` transactions
  commit before the failure is reported, so `MigrationIncomplete` means rows
  are already rewritten while the marker is retained.
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
| Benchmarks (criterion target `db_search`, `benches/db_search.rs`) | `cargo bench` — **not a CI gate**; see README (Benchmarks) for the baseline flags |
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
