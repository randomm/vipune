# Retrieval-Quality Evaluation — Scope & Decisions

This document is the scope deliverable for the retrieval-quality evaluation
spike (issue #192). It records exactly four decisions, each with a
one-paragraph rationale: the benchmark, the licensing posture, the scoring
protocol, and the retrieval path measured. The baseline recall number is NOT
computed here; the seeding-script outline, its entry point, and expected
runtime are recorded below so the follow-up ticket's first milestone is
producing that number.

## Decision 1: Benchmark

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

## Decision 2: Licensing Posture

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

## Decision 3: Scoring Protocol

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

## Decision 4: Retrieval Path Measured

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

---

*Remaining sections of this scope document (corpus specification, seeding
script outline, guardrails) are populated by the sibling workstreams
(`task-b`, `task-c`, `task-d`) and are intentionally not written here.*
