---
name: vipune
description: Mandates the use of the vipune memory CLI as the primary persistent memory store for software development agents. Search vipune at the start of any task to discover prior decisions, conventions, pitfalls, and validated patterns. Save new findings as typed memories—code conventions, architectural decisions, failed approaches, quality-gate gotchas, and ADR-lite rationale—at task boundaries. Covers issue/PR linkage, experiment tracking, pre-flight gotcha checks, dev-loop phase mapping, lightweight ADR capture, the 5-type memory taxonomy (fact/preference/procedure/guard/observation), active/candidate split, mandatory freshness verification, contradiction handling via --supersedes, and the hard prohibition on storing secrets.
---

# vipune memory skill for software development

This skill makes **vipune** — a CLI semantic memory store with project scoping, hybrid search, and typed memories — the authoritative persistent memory layer for software development agents working in codebases.

Codebases evolve, patterns accumulate, quality-gate gotchas emerge, and the agent learns which approaches fail and why. None of this should be re-learned session by session. vipune is the durable layer that prevents repeating mistakes, re-paving ground already walked, and losing the rationale behind decisions.

> **The mandate**: search vipune at the start of every task, save vipune memories at task boundaries. This is not optional. It is the protocol.

## Why vipune over file-based notes

- **Semantic search** finds memories by meaning, not exact keywords
- **Project scoping** is automatic via git remote — no cross-contamination between codebases
- **Typed memories** route by purpose (fact / preference / procedure / guard / observation)
- **Conflict detection** at 0.85 similarity flags duplicate memories before write
- **Supersede** atomically replaces outdated memories — no orphaned contradictions
- **Status field** (active/candidate) gives a long-term/short-term split
- **Recency-weighted search** gives implicit decay without losing data
- **Metadata support** for issue/PR linkage and experiment tracking
- **Single binary, local SQLite** — no API calls, no external service, no network dep

## The protocol — two bookends and a watchpoint

### 1. Session open / new task — SEARCH FIRST

Before reading code, before running anything, before forming a plan:

```bash
vipune search "<topic the user mentioned>" --hybrid --recency 0.3 --limit 5
```

Read the returned memories. They tell you:

- The project's conventions (what tools, what style, what to NOT do)
- Past user corrections (don't repeat past mistakes)
- Validated workflows (recipes that already work)
- Known pitfalls and dead-ends
- Architecture decisions and their rationale
- Issue/PR context from previous work
- In-progress context that crossed a session boundary

For software development tasks, run **3–4 targeted searches**:
1. **Domain/gotcha search**: `vipune search "<domain> gotcha" --memory-type guard` — what traps exist?
2. **Task-specific search**: `vipune search "<task topic>" --hybrid` — what do we know?
3. **Decision rationale search**: `vipune search "<approach> why" --memory-type procedure` — WHY was this chosen?
4. **Issue/PR recall** (if issue number known): search gets IDs, then fetch details: `vipune get <id> --json | jq '.metadata'` — what was decided?

Run searches across 2–3 angles before starting non-trivial work. Single-term queries score highest (0.81+); compound `[domain] [component] [action]` queries also work well. See "Search recipes" below.

### 2. Mid-task — SEARCH ON SURPRISE

If the codebase contradicts a recalled memory, or you encounter unexpected behavior, search again:

```bash
vipune search "<symptom> <component>" --hybrid --limit 5
```

The memory may explain the surprise (known gotcha), or you need to update/supersede a stale memory.

### 3. Task close / session end — SAVE WHAT'S WORTH KEEPING

At the natural end of a unit of work, distill the findings into typed memories:

```bash
# A build convention discovered
vipune add "Build runs via 'cargo test --release' not 'cargo test' (release tests reveal UB)" \
  --memory-type procedure

# A user correction — high importance
vipune add "Operator wants error messages to reference actual model_id from DB, not hardcoded MODEL_ID const" \
  --memory-type guard

# A quality-gate gotcha — guard against repeating
vipune add "Clippy::too_many_arguments triggers for wrapper functions, disable locally for thin wrappers" \
  --memory-type guard

# An architecture decision with rationale — ADR-lite
vipune add "Chose bge-small-en-v1.5 over bge-base-en-v1.5: 2.5x faster inference with <3% accuracy loss for code queries" \
  --memory-type procedure

# A failed approach — observation with metadata for experiment tracking
vipune add "tried async sqlite (rusqlite with tokio) — failed because model loading is CPU-bound, not I/O-bound" \
  --memory-type observation --metadata '{"experiment":"async-db","failed":true}'

# A linked memory (attaching to issue/PR context)
vipune add "CLI --local-dir flag uses HF Hub symlink layout, not direct model path" \
  --memory-type fact --metadata '{"issue":135}'
```

**Never save mid-debug.** Only after a conclusion is reached. Mid-debug hypotheses are exactly the kind of false-positive context that becomes "memory rot."

## Memory types — when to use which

vipune ships with 5 types. Each routes by purpose:

| Type | What it captures | Examples |
|---|---|---|
| `fact` (default) | Objective truths about the project, codebase, or environment | "Project uses ONNX Runtime, not PyTorch" / "Embedding model pinned to bge-small-en-v1.5 revision main" / "Database path: ~/.vipune/memories.db" |
| `preference` | How the user wants things done | "User prefers one-command test runs: cargo test over pytest-style suites" / "Always confirm before cargo clippy --fix" |
| `procedure` | Validated step-by-step recipes AND decision rationale | "Quality gate: cargo fmt --check && cargo clippy -- -D warnings && cargo test" / "Chose zero-copy embeddings: avoids 40% memory overhead" |
| `guard` | Things the agent MUST NOT do | "Never commit with --no-verify" / "Never store secrets in vipune" / "Never use MODEL_ID const in error messages; use actual model_id from DB" |
| `observation` | Notable but not yet load-bearing context, including failed experiments | "Embedding inference ~20ms/cpu for bge-small on M1" / "tried async sqlite — failed because model loading is CPU-bound" / "cargo test release mode triggers UB in float parsing" |

Default to `fact` only when nothing more specific fits. **Be aggressive about typing** — typed memories filter better, decay better, and survive consolidation passes.

**Failed approaches and experiments**: Store as `observation` with appropriate metadata (e.g., `{"experiment":"async-db","failed":true}` or `{"issue":147,"attempt":"refactor"}`). This distinguishes "we tried X and it failed" from "never do X" — the former is provisional learning, the latter is a hard constraint.

## Active vs candidate — the long-term / short-term mechanism

Vipune has no built-in TTL. Use `--status` instead:

| Status | Behavior | Use for |
|---|---|---|
| `active` (default) | Surfaces in default searches | Validated, durable knowledge |
| `candidate` | Hidden unless `--include-candidates` is passed | Provisional observations, hypotheses, things that may not outlast this session |

```bash
# Provisional — won't show in default searches until promoted
vipune add "cargo build --release takes ~30s for incremental changes" \
  --memory-type observation --status candidate --metadata '{"timing": "release-build"}'

# Promote to active once validated across multiple sessions
vipune update <id> --status active
```

**Promote candidates only after the fact survives a second confirmation.** Otherwise, let them sit — they're inert until explicitly queried. This is the short-term tier.

## Issue/PR linkage — connecting memories to work

Use the `--metadata` flag to link memories to GitHub issues or PRs:

```bash
# Store a decision linked to an issue
vipune add "Error messages must reference actual model_id from DB, not MODEL_ID const" \
  --memory-type guard --metadata '{"issue":135}'

# Store a failed approach from an issue investigation
vipune add "tried parsing HF Hub cache directly -- failed because symlink layout differs from --local-dir layout" \
  --memory-type observation --metadata '{"issue":138,"experiment":"hf-cache-parsing","failed":true}'

# Store implementation notes linked to a PR
vipune add "Supersede requires similarity check before accepting replacement" \
  --memory-type procedure --metadata '{"pr":84}'
```

**Recall by issue/PR context**:

Note: Search results include ID and content, not metadata. To get issue/PR-linked memories:

```bash
# Search for memories mentioning an issue, then fetch details
vipune search "issue 135" --json | jq -r '.results[].id' | xargs -I {} vipune get {} --json | jq 'select((.metadata | fromjson? // {} | .issue) == 135)'

# For failed experiments, search for "failed" then inspect individually
vipune search "failed" --memory-type observation
# Then fetch full details for each with: vipune get <id> --json
```

This pattern creates traceability: future agents can see exactly what was decided and why, linked to the original issue context.

## Freshness verification — REQUIRED before acting

Every memory is a snapshot of when it was written. Code moves, files rename, decisions reverse. **Before acting on a retrieved memory, verify it against current state.** This is the single biggest gap in most agentic memory implementations.

| Memory claims | Verification step |
|---|---|
| A file exists at `path/src/foo.rs` | `ls path/src/foo.rs` |
| A function/symbol exists | `grep -r "function_name" .` or `rg "function_name"` |
| A CLI flag is supported | Check `--help` or the actual command surface |
| A version is pinned at X | Read the lockfile / image tag / Cargo.toml |
| A user preference is current | If context suggests it may have changed, ask |
| An issue/PR still exists | Search the GitHub repo or check `git log` |

If verification fails: **update or delete the stale memory immediately.** Don't just ignore it — the next session will hit the same trap.

```bash
# Memory was wrong — supersede atomically, link to issue if relevant
vipune add "New corrected statement" --supersedes <old-id> --memory-type fact --metadata '{"supersedes_issue":135}'
```

## Contradiction handling — supersede, don't duplicate

When you save a memory similar to an existing one (≥ 0.85 similarity), vipune flags a conflict. Three responses:

1. **The new is correct, the old is wrong** → `vipune add "..." --supersedes <old-id>` (atomic replacement)
2. **Both are true but address different angles** → `vipune add "..." --force` (intentional coexistence — rare)
3. **The old is fine, no new memory needed** → don't add

Default to **option 1** if the facts contradict. Never let two contradictory memories coexist — that's the leading cause of "memory rot" where agents confidently cite stale info while also citing the correction.

## Pre-flight gotcha check — before quality gates

Before running tests, linting, or other quality gates, search for known gotchas:

```bash
# Search for domain-specific gotchas
vipune search "<domain> gotcha" --memory-type guard

# Examples from real codebases:
vipune search "clippy gotcha" --memory-type guard          # reveals: "Clippy::too_many_arguments triggers for wrappers"
vipune search "test gotcha" --memory-type guard            # reveals: "cargo test --release reveals UB not seen in debug"
vipune search "embedding gotcha" --memory-type guard       # reveals: "validate before adding >512-token content"
```

If relevant gotchas surface, address them before running quality gates. This pattern prevents wasted CI runs and repeated failures from known issues.

## Dev-loop phase → action mapping

Map vipune actions to the software development loop:

| Phase | vipune search | vipune save |
|---|---|---|
| **Read issue** | Search for prior context: `vipune search "issue <number>"`, `vipune search "<task domain>" --hybrid` | Store initial understanding if novel: `vipune add "Issue scope: ..." --memory-type fact --metadata '{"issue":N}'` |
| **Plan** | Search for patterns: `vipune search "<approach> pattern"`, `vipune search "<domain> gotcha" --memory-type guard` | Store chosen approach with rationale: `vipune add "Choosing X over Y because: ..." --memory-type procedure --metadata '{"issue":N}'` |
| **Implement** | Search mid-stream on surprises: `vipune search "<symptom> <component>" --hybrid` | Store discovered patterns mid-task as candidate: `vipune add "..." --memory-type observation --status candidate` |
| **Test** | Pre-flight gotcha check: `vipune search "<domain> gotcha" --memory-type guard` | Store test/lint findings: `vipune add "Test revealed X" --memory-type guard` |
| **Commit** | Search for commit conventions: `vipune search "commit message" --memory-type preference` | Store completion + link to PR: `vipune add "Completed issue N: ..." --memory-type fact --metadata '{"issue":N,"pr":M}'` |

## Lightweight ADR — capturing decision rationale

Store WHY an approach was chosen or rejected, not just WHAT. This creates an ADR-lite layer that survives rotation:

```bash
# Architecture decision with rationale
vipune add "Chose bge-small-en-v1.5 over bge-base-en-v1.5: 2.5x faster inference with <3% accuracy loss for code queries; tested on 500 sample queries" \
  --memory-type procedure --metadata '{"adr":"embed-model-choice"}'

# Rejection rationale
vipune add "Rejected async sqlite (rusqlite with tokio): model loading is CPU-bound, not I/O-bound; async adds complexity without benefit; verified via profiler" \
  --memory-type observation --metadata '{"adr":"rejected-async-db"}'

# Tradeoff captured for future review
vipune add "Using 0.85 as conflict threshold: balances precision/recall for 500+ memories; 0.90 too strict, 0.80 creates noise" \
  --memory-type fact --metadata '{"adr":"conflict-threshold"}'
```

Future agents searching for `"<approach> why"` or `"rejected` will surface the rationale, preventing circular re-decisions.

## Search recipes — what actually works

From vipune's own query-guide scoring data:

| Pattern | Example | Top score |
|---|---|---|
| Single technical term | `vipune search "Caddyfile"` | 0.81 |
| `[Domain] [Component]` | `vipune search "embed model"` | 0.80 |
| `[Feature] [Implementation]` | `vipune search "hybrid search"` | 0.79 |
| `[Component] [Action] [Concept]` | `vipune search "conflict detection threshold"` | 0.78–0.81 |
| `[Problem] [Technical Context]` | `vipune search "clippy too_many_arguments"` | 0.80 |
| `[Approach] [Rationale]` | `vipune search "why bge-small"` | 0.76–0.80 |
| Natural-language question | `vipune search "how do we detect conflicts"` | ~0.70 (still useful) |

Hybrid mode helps short / proper-noun queries:

```bash
vipune search "MCP" --hybrid          # 1–3 word query → hybrid wins
vipune search "ConfigClass" --hybrid  # exact identifier
```

Recency-weighted for "what changed recently":

```bash
vipune search "deployment" --recency 0.7 --limit 10
```

Type-filtered for narrow recall:

```bash
vipune search "do not" --memory-type guard         # only guards
vipune search "test" --memory-type guard           # quality-gate gotchas
vipune search "chose" --memory-type procedure      # ADR-lite rationale
vipune search "procedure" --memory-type procedure  # validated recipes
```

For issue/PR context, search by content and inspect metadata per-result:

```bash
vipune search "memory" --json | jq -r '.results[] | "\(.id): \(.content)"'
# Then fetch individual details: vipune get <id> --json | jq '.metadata' (metadata is a JSON string; use | fromjson to parse fields)
```

Score thresholds:

- **0.80+** — Use the result directly (verify freshness first)
- **0.70–0.79** — Cross-check top 2–3
- **0.60–0.69** — Refine the query
- **< 0.60** — Probably nothing relevant; don't act on it

## What to ALWAYS save vs NEVER save

**ALWAYS** (long-term, `active`):

- Project conventions (build tool, package manager, lint config, CI pipeline)
- Quality-gate gotchas (test quirks, linting exceptions, known failures)
- User preferences and corrections (style, format, tone, depth)
- Architectural decisions and the *reason* behind them (ADR-lite)
- Validated workflows (the exact commands that work)
- Known pitfalls + their workarounds
- Non-obvious constraints (compliance, performance, security)
- Issue/PR decisions and their rationale
- Failed approaches / experiments (with metadata)

**NEVER** (defeats the purpose, causes bloat or actual harm):

- **Secrets of any kind** — API tokens, passwords, SSH private keys, .env contents. Vipune stores in plaintext SQLite. **This is a hard line.** If you discover a secret in memory, `vipune delete` immediately and rotate the credential.
- Raw conversation turns
- One-off file paths likely to move
- Intermediate debugging hypotheses
- Ephemeral stack traces
- Anything reconstructible cheaply from `git log` or current code state

**CONDITIONAL** (use `--status candidate`, promote only if it survives):

- Tentative explanations for observed behavior
- Performance numbers (they drift)
- "I think this is how X works" claims
- Process state mid-task

## Periodic consolidation — the reflection pass

Every ~10–15 sessions on a project, run:

```bash
vipune list --limit 50 --json | jq .
```

Look for:

- **Duplicates** (≥ 2 memories saying nearly the same thing) → supersede the older ones
- **Stale facts** (still `active` but contradicted by current code) → delete or supersede
- **Drift** (multiple `observation`-type memories on the same theme) → consolidate into one `fact` or `procedure`, then delete the originals
- **Candidates** that have survived 5+ sessions → promote to `active`
- **Candidates** untouched for months → delete
- **Orphaned issue references** → verify issues still open/relevant, update or delete

This is the reflection pattern from the agentic memory literature (Park et al. 2023, A-MEM 2025, MemoryBank). It prevents the bloat-driven degradation that hits unbounded memory stores around 50+ sessions.

## Failure modes to avoid

Real failure patterns documented in agentic memory systems — don't repeat them:

1. **Stale-fact citation** — Quoting a memory that no longer matches reality. *Mitigation*: freshness verification before acting (above).
2. **Contradiction accumulation** — Two memories that disagree, both surfacing in search. *Mitigation*: `--supersedes`, never `--force` unless they truly address different angles.
3. **Context bloat** — So many memories surface that the context window saturates and reasoning quality drops. *Mitigation*: aggressive pruning, candidate status, tight `--limit`.
4. **Ephemeral leakage** — Persisting raw conversation-context as memory. *Mitigation*: only save *distilled* conclusions, never raw turns.
5. **Premature commit** — Saving a hypothesis mid-debug. *Mitigation*: save only at task close, after the conclusion holds.
6. **Secret leakage** — A single token or password committed to memory is a breach. *Mitigation*: the explicit NEVER list above + immediate delete + rotate if found.
7. **Issue-link rot** — Metadata referencing resolved/closed issues without context. *Mitigation*: periodic consolidation checks for orphaned references.

## Quick reference

```bash
# SEARCH (start every task here)
vipune search "<technical term or compound>" --hybrid --recency 0.3 --limit 5

# PRE-FLIGHT GOTCHA CHECK (before quality gates)
vipune search "<domain> gotcha" --memory-type guard

# SEARCH BY ISSUE/PR (search by content, then fetch metadata)
vipune search "issue 135" --json | jq -r '.results[].id' | xargs -I {} vipune get {} --json | jq -r '"\(.id): \(.metadata // "(no metadata)")"'
# Note: Best for small result sets (<50 results; for larger sets, narrow the search first)

# SAVE (at task boundaries only)
vipune add "<distilled fact/preference/procedure/guard/observation>" \
  --memory-type <type> [--status candidate] [--supersedes <old-id>] [--metadata '{"issue":N}']

# LIST (for the consolidation pass)
vipune list --limit 50

# UPDATE / SUPERSEDE (when a memory drifts)
vipune update <id> --text "<new content>"
vipune add "<new content>" --supersedes <old-id> --memory-type <type>

# DELETE (when fully obsolete)
vipune delete <id>

# VALIDATE (check before adding long content)
vipune validate "<text>"           # exits 3 if over 512-token embedding limit

# JSON OUTPUT (for scripting / consolidation passes / metadata filtering)
vipune list --json | jq .
vipune search "<q>" --json | jq '.results[]'
# For metadata: vipune get <id> --json | jq '.metadata'
```

## Project scoping

vipune auto-detects the project from the git remote. Override only when intentionally cross-cutting:

```bash
vipune search "<q>" -p some-other-project   # search outside the current project
```

Default behavior is correct ~99% of the time — don't override unless you know why.

## When to NOT use vipune

This skill is opinionated, but there are legitimate non-vipune paths:

- **One-shot session info** that won't outlast the conversation → keep in context, don't persist
- **Code-level documentation** that belongs in comments / docstrings / READMEs → put it there, not in vipune
- **User-cross-project preferences** that belong in user-wide memory (e.g., the file-based `MEMORY.md` system in `~/.claude/projects/.../memory/`) → those go there; vipune is project-scoped
- **Anything sensitive** → no memory tool; use a real secrets manager (1Password, op, vault)

vipune is for **project-scoped, persistent, semantic-recall-worthy** knowledge. Use it for that. Don't use it as a notes file.

## Pre-flight checklist (start of any software development task)

- [ ] `vipune search "<task domain>" --hybrid --recency 0.3` — what do we already know?
- [ ] `vipune search "<domain> gotcha" --memory-type guard` — what traps exist?
- [ ] `vipune search "<approach> why" --memory-type procedure` — WHY was this chosen before?
- [ ] Read top 3–5 results; verify any I plan to act on against current state
- [ ] If contradictions surface: `--supersedes` the wrong one before continuing
- [ ] Note candidate observations mentally; save after task closes
- [ ] If issue number known: search for prior context via metadata filtering

## Closing checklist (end of any software development task)

- [ ] What code patterns did we learn that weren't in vipune already?
- [ ] What did the user correct me on?
- [ ] What quality-gate gotchas did we hit?
- [ ] What workflow worked that's worth recording?
- [ ] What approach failed and why?
- [ ] Save each as `vipune add ... --memory-type <type>`, with `--status candidate` if uncertain
- [ ] Link to issue/PR via `--metadata` if relevant: `'{"issue":N}'` or `'{"pr":M}'`
- [ ] Store decision rationale (why), not just what (ADR-lite)
