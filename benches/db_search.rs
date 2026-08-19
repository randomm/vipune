//! Criterion baseline for the `Database::search` path (issue #183).
//!
//! Benchmarks the post-embedding portion of semantic search — SQLite row fetch,
//! BLOB deserialization, and per-row cosine scoring — by feeding
//! [`Database::search`] a precomputed 384-dim query vector. The ONNX embedding
//! pass is deliberately not measured here (it needs the real ~66MB model); that
//! belongs in the scheduled/nightly lane.
//!
//! **Runs with no ONNX model and no network access.** All seeding goes through
//! the public API (`Database::open`, `Database::insert`) with synthetic vectors;
//! no `#[cfg(test)]` helpers are used, because they are invisible to bench
//! targets.
//!
//! **Corpus** — fixed, reproducible:
//! - 384-dim vectors from a fixed seed (20260819) via a small deterministic
//!   SplitMix64 step. Every row is unit-normalised, so similarities cluster in
//!   a narrow, non-degenerate band near 1.0 (realistic for same-model
//!   embeddings).
//! - Row counts: 1k and 10k, bracketing today's real corpus (~2,300 rows
//!   across all projects). 10k is the ceiling because
//!   `MAX_SEARCH_LIMIT = 10_000` caps any single `search` call.
//! - Query vector: the unit-normalised mean of the corpus (recomputed
//!   deterministically each run), so the bench has no magic-number dependency.
//! - `limit` is the maximum legal value (10,000) for both benchmarks: this
//!   exercises the worst-case result allocation and keeps the two scales
//!   comparable.
//!
//! **Known deviation from production data — `created_at` near-degeneracy.**
//! `Database::insert` stamps `Utc::now().to_rfc3339()` (second precision), and
//! the explicit-timestamp variant `insert_with_time` is `#[cfg(test)]
//! pub(crate)`, invisible to bench targets. A tight seed loop therefore
//! produces rows whose `created_at` values are often identical (at most a few
//! distinct values for a 10k seed). `Database::search` fetches rows
//! `ORDER BY created_at DESC`, so that sort is partly degenerate here and does
//! not represent production data, where timestamps are spread over time.
//! The harness documents this rather than asserting distinctness — it has no
//! public way to verify it, since `list_all_rows_for_project` returns
//! id/content/embedding only, with no `created_at`.
//!
//! **Timings only; no timing thresholds are asserted.** Report, never assert —
//! thresholds go flaky on shared runners.
//!
//! **How to run and compare against the baseline** (README note lands in the
//! task-b workstream):
//! ```bash
//! cargo bench --bench db_search                       # run the benchmark
//! # save the current run's samples as the "main" baseline:
//! cargo bench --bench db_search -- --save-baseline main
//! # after a performance-shaped change, diff against it:
//! cargo bench --bench db_search -- --baseline main            # hard gate
//! cargo bench --bench db_search -- --baseline-lenient main    # report-only
//! ```
//! Criterion stores baselines under `target/criterion/db_search/baselines/`.
//! No committed baseline is bundled: baselines are machine-specific, so each
//! machine records its own "main" baseline, and later runs diff against it
//! numerically.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use vipune::Database;

/// Embedding dimensionality — matches the model used in production (384-dim).
const DIM: usize = 384;

/// Corpus seed. Fixed so the corpus is reproducible across runs and machines.
const CORPUS_SEED: u64 = 20260819;

/// Maximum `search` limit — `MAX_SEARCH_LIMIT` in `src/memory/store.rs`.
const SEARCH_LIMIT: usize = 10_000;

/// Project id used for all seeded rows.
const PROJECT_ID: &str = "bench-proj";

/// Deterministic 384-dim vector generator: a SplitMix64 step. No external RNG
/// crate — the bench stays dependency-minimal per the stdlib-first philosophy.
struct Corpus {
    state: u64,
}

impl Corpus {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next vector in the corpus (unit-normalised; fixed seed → fixed sequence).
    fn next_vector(&mut self) -> Vec<f32> {
        let mut v = Vec::with_capacity(DIM);
        for _ in 0..DIM {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z = z ^ (z >> 31);
            // Map to roughly [-1, 1).
            let frac = (z >> 11) as f64 * (2.0 / (1u64 << 53) as f64);
            v.push(frac as f32);
        }
        normalise(&v)
    }
}

/// Returns `v / ||v||` so all corpus rows sit on the unit sphere, mirroring
/// real (normalised) embeddings and giving a non-degenerate similarity band.
fn normalise(v: &Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.clone();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Build a corpus of `n` row vectors plus a query vector (their unit-normalised
/// mean), computed deterministically from `CORPUS_SEED`.
fn build_corpus(n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut corpus = Corpus::new(CORPUS_SEED);
    let rows: Vec<Vec<f32>> = (0..n).map(|_| corpus.next_vector()).collect();

    let mut query = vec![0.0f32; DIM];
    for row in &rows {
        for (q, r) in query.iter_mut().zip(row) {
            *q += r;
        }
    }
    for q in query.iter_mut() {
        *q /= n as f32;
    }
    let query = normalise(&query);
    (rows, query)
}

/// Unique temp path for this bench process + row count.
fn db_path(n: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vipune-bench-db_search-{}-{}.db",
        std::process::id(),
        n
    ))
}

/// Open a throwaway database, seed `n` rows via the public API only, and return
/// the (db, path, query) tuple the bench searches against.
fn seed_db(n: usize) -> (Database, PathBuf, Vec<f32>) {
    let (rows, query) = build_corpus(n);
    let path = db_path(n);
    // Best-effort cleanup of any stale file from a crashed prior run.
    let _ = std::fs::remove_file(&path);

    let db = Database::open(&path).expect("bench: open temp db");
    for row in &rows {
        db.insert(PROJECT_ID, "bench row", row, None, "fact", "active")
            .expect("bench: insert row");
    }
    (db, path, query)
}

/// Bench `Database::search` at 1k and 10k rows.
///
/// Each `search` call inside `bench.iter` exercises the full path:
/// SQL fetch (with the partly-degenerate `created_at` sort, see module docs),
/// BLOB deserialisation, per-row cosine scoring, similarity sort, and result
/// allocation. The query vector and seeded rows are precomputed outside the
/// timed region; the input is passed through `black_box` by criterion, which
/// also prevents elision of `search` itself.
fn db_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("db_search");

    for n in [1_000usize, 10_000] {
        let (db, path, query) = seed_db(n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rows-{}", n)),
            &query,
            |bench, query| {
                bench.iter(|| {
                    let results = db
                        .search(PROJECT_ID, query, SEARCH_LIMIT, None, None)
                        .expect("bench: search");
                    // Consume the result so the allocator churn is real.
                    let _ = results;
                });
            },
        );

        drop(db);
        let _ = std::fs::remove_file(&path);
    }

    group.finish();
}

/// Criterion config: saves the current run's samples into the "main" baseline
/// store on every run (criterion's `--save-baseline main` equivalent) so a
/// later `cargo bench` can be diffed numerically with `--baseline main`.
/// This is the "save baseline" behaviour the issue asks for; it persists the
/// run's samples into the baseline store without adding any timing assertion.
fn custom_criterion() -> Criterion {
    Criterion::default().save_baseline("main".to_string())
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = db_search
}
criterion_main!(benches);
