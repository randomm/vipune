# Retrieval-Quality Evaluation Harness — Scope Document

This document is the scope deliverable for the retrieval-quality evaluation
spike (issue #192). It records exactly four decisions (benchmark, licensing
posture, scoring protocol, and retrieval path measured), specifies the
evaluation corpus, outlines the seeding script a follow-up ticket (~0.5–1
day) will implement, and lists the guardrails and prior art the follow-up
work must respect.

The baseline recall number is NOT computed here; this document ships no
harness code, no vendored data, no CI step, and no new dev-dependencies. The
seeding-script outline, its entry point, and expected runtime are recorded so
the follow-up ticket's first milestone is producing that number.

## Section 1: The Four Decisions

### Decision 1: Benchmark

**LongMemEval** (`xiaowu0162/longmemeval`, MIT) is the single in-repo
benchmark; its Hugging Face mirror `longmemeval-v2` (apache-2.0) is the data
source used for seeding.

LongMemEval is a purpose-built long-term memory benchmark: it pairs questions
against multi-session conversation histories with gold labels identifying the
relevant sessions, which is exactly the retrieval task vipune performs. Its
MIT license (and the apache-2.0 HF mirror) aligns with vipune's MIT OSS
posture, so the data can be fetched, cached, and referenced without
licensing friction. Every other candidate either carries a non-commercial
license or does not structure its labels as retrieval targets, leaving
LongMemEval as the only candidate that is both task-faithful and
license-compatible.

### Decision 2: Licensing Posture

The cc-by-nc-4.0 ruling on **LoCoMo** covers **committing the data to the
OSS repo** — i.e. vendoring LoCoMo passages into the vipune repository is
prohibited for this project. It does **not** rule out downloading the data
at eval time or running it internally on a developer's machine: those
activities do not redistribute the data under vipune's MIT license, so they
fall outside the CC-BY-NC restriction. The decision is therefore
narrow and concrete: **no LoCoMo data may be committed to this repository**;
eval-time download or internal runs of LoCoMo would be legally permissible
but are out of scope by choice, not by license. LoCoMo appears nowhere
else in this document.

### Decision 3: Scoring Protocol

Scoring is **model-free recall@k with k ∈ {1, 5, 10}**: for each pair, take
the top-k retrieved memories and compute the fraction of the pair's gold
targets that appear in that top-k, then average across pairs.

This is deliberately NOT LLM-judged answer correctness. Both LongMemEval's
and LoCoMo's headline numbers conflate retrieval with answer generation: a
system that retrieves perfectly but answers poorly scores low, and one that
answers well by luck with weak retrieval scores high. Since vipune is a
retrieval layer, not an answerer, measuring it requires a score that isolates
retrieval alone. Recall@k is deterministic, reproducible, and requires no
second LLM to judge — it depends only on the gold labels and the retrieved
rank list, making the baseline number stable across runs and free of
grader-variance noise.

### Decision 4: Retrieval Path Measured

The retrieval path under test is **`MemoryStore::search_hybrid` with
`recency_weight=0` and RRF k=25** — the full hybrid path that fuses semantic
and BM25 results via RRF, with recency weighting disabled so the score is
pure retrieval quality.

`search_hybrid` calls `get_embedding` internally, so running it requires the
real bge-small model to be available; it belongs in the **model-dependent
execution lane**. By contrast, `Database::search` accepts a **precomputed
embedding vector** and performs only the cosine scan — it is the
**model-free seam** that runs in the **model-free lane**, where the query
vector is produced once at prep time (with the same bge-small model the
corpus is embedded with) and cached. The two lanes matter for cost: the
model-dependent lane bears the per-run embedding call, while the model-free
lane isolates the search-quality measurement from any embedding-time
variance. The primary metric is reported against the model-dependent lane
(the real user-facing path); the model-free lane exists so the search
component can be regression-tested without re-running the model.

## Section 2: Corpus Specification

This section specifies the evaluation corpus: the exact shape of a scored
"pair", the LongMemEval question-type categories, the raw and usable pair
counts, the gold-label-level fallback, and the corpus bracketing rule when a
persona's corpus grows past `MAX_SEARCH_LIMIT`.

The dataset under test is LongMemEval, file `longmemeval_m.json` (500
instances, ~500 history sessions per instance). The sibling files
`longmemeval_s.json` (~40 sessions) and `longmemeval_oracle.json` (evidence
sessions only) exist for context but are not the harness's corpus.

### Definition of a pair

A **pair** is one scored retrieval instance. It binds:

- a **question** (`question` string, plus `question_date`); and
- a **gold session id set**: the `answer_session_ids` list — the set of
  `haystack_session_ids` entries whose turns contain the required evidence.

The pair is *not* defined at the turn level. The retrieval unit is the session:
each session in the persona's history is stored as one memory row
(concatenated turn text, `role` and `content` fields joined), keyed by its
`haystack_session_id`, so a pair's gold set is a set of session ids, not a set
of turn indices.

This choice is deliberate. Model-free `recall@k` scores a pair by asking
whether the retrieved rows cover the pair's gold rows, and the rows that exist
in the store are sessions (see the bracketing rule below for why sessions are
the natural unit here). If the store instead indexed individual turns, the gold
set would be the `has_answer == true` turn indices within the evidence
sessions — that is the fallback, described in the next-but-one section — and
not the default.

### LongMemEval question-type categories

The `question_type` field takes exactly these values, named as they appear in
the dataset (card and file):

- `single-session-user`
- `single-session-assistant`
- `single-session-preference`
- `temporal-reasoning`
- `knowledge-update`
- `multi-session`
- `abstention` — the label the dataset card assigns to any instance whose
  `question_id` ends with `_abs`. These are not a seventh value of
  `question_type` in the raw file; the card derives them from the `_abs`
  suffix, and the harness must apply the same derivation.

### Raw pair count, filter, and usable count

The raw pair count is **500 instances** in `longmemeval_m.json`.

The filter applied: **drop the 30 abstention instances** (those whose
`question_id` ends with `_abs`). These instances are excluded because they
reference non-existent events and carry no meaningful retrieval target — the
gold `answer_session_ids` is present but does not correspond to a real
evidence location the retriever should find, and LongMemEval's own retrieval
evaluation ("To evaluating the retrieval, we always skip the 30 abstention
instances. This is because these instances generally refer to non-existing
events and do not have a ground truth answer location.") skips them for the
same reason.

The resulting **usable count is 470 pairs**.

Per-category usable counts (abstention instances removed from their
`question_type`):

| Category                 | Raw | Usable |
|--------------------------|-----|--------|
| multi-session            | 133 | 133    |
| temporal-reasoning       | 133 | 133    |
| knowledge-update         | 78  | 78     |
| single-session-user      | 70  | 70     |
| single-session-assistant | 56  | 56     |
| single-session-preference| 30  | 30     |
| **Total**                | 500 | 470    |

The 30 abstention instances are distributed across the six named
`question_type` values; no category is fully abstention-only.

### Fallback if gold labels are turn-level only

The default pair definition above depends on `answer_session_ids` — the
session-level gold label — being present and non-empty. In the current
`longmemeval_m.json` release it is: every instance (including the 30
abstention ones) carries a non-empty `answer_session_ids` list (1–6 session
ids per instance; 948 across all 500), and every evidence session's turns
carry the per-turn `has_answer: true` marker. Both label granularities
coexist in the file.

If a future revision of the dataset ships **turn-level golds only** (the
`has_answer` markers on turns, without the `answer_session_ids` list), the
pair definition in the section above no longer computes, because there is no
session-level gold set to score against. The fallback is:

1. **Aggregate turn labels to sessions.** For each session in the persona's
   history, compute `has_evidence = any(turn.has_answer for turn in session)`.
   The gold session id set for the pair becomes `{ session_id : has_evidence
   }` — i.e. the set of session ids that contain at least one
   `has_answer: true` turn.
2. **Score with the same `recall@k` formula**, but against the aggregated gold
   set instead of the `answer_session_ids` list. The pair definition, the
   `k` values (1/5/10), and the scoring formula are unchanged; only the
   derivation of the gold set moves from a direct field read to the
   aggregation in step 1.

This fallback is sound only if the `has_answer` markers are themselves
session-consistent (i.e. if a session has any evidence, all its evidence is
captured by `has_answer`). If the markers are sparse or inconsistent, the
aggregated gold set is an approximation of the true evidence sessions and the
`recall@k` number should be reported with that caveat. The fallback is
recorded here so the harness does not silently assume session-level golds
exist; if they don't, this is the rule to apply rather than a hard stop.

### Bracketing a persona corpus over `MAX_SEARCH_LIMIT`

`MAX_SEARCH_LIMIT` (10 000, `src/memory/store.rs:13`) bounds the `limit`
argument of `MemoryStore::search` / `search_hybrid`. It does **not** bound the
number of rows in the database: seeding a persona's full `longmemeval_m.json`
corpus of ~500 sessions fits a single store instance comfortably, and no
bracketing is needed for the current corpus size.

The rule is stated here because the corpus can grow: if a persona's usable
corpus exceeds 10 000 rows (e.g. because the harness is later pointed at a
larger LongMemEval variant, or because turns — not sessions — become the
indexing unit, multiplying the row count by the average turns-per-session),
then a single `search` call with `limit > 10 000` is rejected by
`validate_limit`. The harness must then bracket the corpus:

- **Per-persona store partitioning.** Each persona (one `question_id`'s full
  history) gets its own store instance (or its own `project` scope within one
  instance). The `limit` on each individual search call stays at 10 000 or
  below; the *number of calls* is not what `MAX_SEARCH_LIMIT` bounds — only
  the per-call `limit` is.
- **No single call may request more than 10 000 rows.** If the harness
  genuinely needs to retrieve more than 10 000 rows for one query (which the
  current corpus size does not), it must either (a) split the corpus across
  store instances and issue one call per instance, or (b) raise
  `MAX_SEARCH_LIMIT` in code (out of scope for this spike; noted here as the
  alternative the harness authors would choose if they hit the ceiling).

For the current `longmemeval_m.json` corpus (~500 rows per persona), this rule
is a no-op: every persona's corpus fits well under the limit, and each
per-persona search call uses `limit = 10` (for `k=10`) or lower.

## Section 3: Seeding-Script Outline

The follow-up ticket's **first milestone** is producing the baseline
recall@k number using the seeding script described here. The baseline number
itself is not computed by this spike.

### Entry point

`scripts/retrieval-eval/main.rs` — a standalone `main` under `scripts/`
(buildable via `cargo run --manifest-path scripts/retrieval-eval/Cargo.toml`
or as an example target once the harness crate lands). Two subcommands:

- `scripts/retrieval-eval <data> --prep <cache-dir>` — the one-time
  **prep** phase: load the corpus, embed every passage once with the real
  bge-small model, and write the cache (see below).
- `scripts/retrieval-eval <data> --run <cache-dir>` — the **per-run** phase:
  load the cached vectors, seed a throwaway database, run every pair's
  retrieval, and report recall@k (k = 1/5/10). No embedding, no model
  download.

### Outline — reusing the public-API seeding pattern from `benches/db_search.rs` (#185)

The bench seeds through public API only (`Database::open` on a `tempfile::TempDir`
path, then `Database::insert(project_id, content, vector, …)`) with
precomputed vectors, because `#[cfg(test)]` helpers (e.g.
`insert_with_time`) are invisible to non-test targets. The seeding script
reuses the same shape:

1. **Load corpus + cache.** Read the LongMemEval corpus for the target
   persona and the precomputed 384-dim bge-small vectors from the
   `--prep` cache (one vector per passage). No model, no network.
2. **Seed a throwaway database.** `tempfile::TempDir` →
   `vipune::Database::open(path)`, then one `Database::insert` per
   passage — the same call sequence as `benches/db_search.rs::seed_db`.
   (When the retrieval-path decision's `MemoryStore::search_hybrid` lane is
   exercised, seed through `MemoryStore::ingest` with pre-seeded vectors
   from the same cache instead; the corpus, cache, and pair loop are
   unchanged.)
3. **Loop over pairs.** For each pair (definition in the corpus
   specification section): embed the query *from the cache* (prep already
   embedded every question — or embed the single query vector on demand
   from the same cache file), call the measured retrieval path, and collect
   the top-k result ids for k = 1, 5, 10.
4. **Score model-free.** Recall@k = |gold set ∩ top-k| / |gold set|,
   averaged over pairs. No LLM judging, no answer generation.
5. **Report.** Emit per-k recall (plus per-category breakdown) as JSON to
   stdout; that output is the baseline number the follow-up ticket records
   as its first milestone.

### Expected runtime

- **Prep (one-time, excluded from per-run runtime).** Every corpus passage
  and question is embedded **once** with bge-small (BAAI/bge-small-en-v1.5,
  384-dim, ~66 MB model) and the vectors are cached to disk
  (`<cache-dir>/vectors.bin` or a small SQLite/JSON sidecar). This phase
  carries the model download (first machine only) and all embedding cost;
  it runs once and the cache is reused.
- **Per run (what gets reported).** Corpus load from the cache, `TempDir`
  + `Database::open` + N inserts of precomputed vectors, then the
  pair loop of `search`/`search_hybrid` calls. **Per-run runtime excludes
  model download and per-run embedding cost by construction** — this is the
  direct contrast to the #185 bench's two gotchas (hardcoded
  `save_baseline("main")` making plain `cargo bench` self-referential, and
  the synthetic normalised-corpus-mean query whose cosine ≈ 0.051
  everywhere makes the sort a near-tie no-op). This harness uses real
  bge-small vectors on real data, reports rather than asserts, and writes
  no baseline that later runs compare against.
- **Rough budget (to be confirmed at baseline milestone).** Per-run on the
  order of seconds-to-a-minute: DB seeding of a few thousand 384-dim rows is
  the dominant cost (the #185 bench seeds 10k rows in well under a second),
  and each pair is one in-memory/SQLite vector scan over at most
  `MAX_SEARCH_LIMIT = 10_000` rows. Prep dominates by comparison (full
  corpus + questions through the ONNX model) but is explicitly one-time.

### What this is not

- Not a `criterion` benchmark and not a `#[cfg(test)]` module: it is a
  plain entry point whose output is a quality number, not a timing
  regression, so it must not inherit the `save_baseline("main")`
  self-reference or the synthetic-corpus near-tie of #185.
- Not a timing measurement with thresholds: report, never assert (same
  policy as the bench — thresholds go flaky on shared runners).

## Section 4: Guardrails & Prior Art

### BEAM is out of scope

BEAM is out of scope entirely for this spike and for the follow-up harness
ticket. It is not a candidate benchmark, and no decision in this document
attributes any scope to it.

### LoCoMo exclusion

LoCoMo must not be scoped into the harness. It appears only in the
one-paragraph licensing-comparison note (owned by the licensing decision
section of this document) and carries no gold-label analysis, no embedding
budget, and no harness scope. The claim "LoCoMo ~9K conversations x ~300
turns" is wrong by orders of magnitude — LoCoMo is roughly 10 very long
multi-session dialogues — and must not be propagated in any follow-up work.

### Issue #186 is not a gate

Issue #186's fix (commit 5622a36 via PR #188) is already merged and must not
be listed as a blocking precondition or gate for this spike or for the
follow-up harness ticket.

### Existing test surface to anchor against

PR #185's criterion latency benchmark for `Database::search` is the existing
test surface to anchor against: latency is measured, relevance is not, so
the follow-up ticket adds the retrieval-quality eval. The new eval must not
inherit the two known gotchas of that benchmark:

1. The hardcoded `save_baseline("main")` makes plain `cargo bench`
   self-referential.
2. The synthetic query (the normalised corpus mean) yields cosine ~0.051 for
   every row, making the sort a near-tie no-op.

Both are exactly why the eval needs real bge-small vectors on real data.
