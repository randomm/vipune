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
//!   SplitMix64 step, mapped to roughly [-1, 1) per component, then
//!   unit-normalised — mirroring real (normalised) embeddings.
//! - Row counts: 1k and 10k, bracketing today's real corpus (~2,300 rows
//!   across all projects). 10k is the ceiling because
//!   `MAX_SEARCH_LIMIT = 10_000` caps any single `search` call.
//! - Query vector: the *next* vector in the same deterministic sequence —
//!   i.e. the vector that would be row `n+1` — and is therefore **not a
//!   member of the seeded corpus**. No row is a duplicate of the query,
//!   so no score can be 1.0 by construction. With the sign-correct
//!   generator (components genuinely spanning [-1, 1)), the query is a
//!   random direction independent of every corpus row, and the cosine
//!   distribution is what one expects for random unit vectors in 384
//!   dimensions: tightly concentrated around 0. Measured this round
//!   (f32, through the real `Database::search` path including the BLOB
//!   round-trip, fixed seed, identical to an f64 recompute from the raw
//!   vectors):
//!   - 1k rows: min ≈ -0.1807, p01 ≈ -0.1145, p50 ≈ 0.0019, mean ≈ 0.0004,
//!     p99 ≈ 0.1096, max ≈ 0.1366
//!   - 10k rows: min ≈ -0.2123, p01 ≈ -0.1171, p50 ≈ -0.0006, mean ≈ -0.0005,
//!     p99 ≈ 0.1156, max ≈ 0.1839
//!
//! **Sign-bug note.** An earlier revision of this bench mapped the raw
//! SplitMix64 output to [0, 2) — entirely non-negative — despite a comment
//! claiming [-1, 1). That put every vector in the positive orthant and
//! inflated the mean cosine to ≈ 0.75 (E[cosine] ≈ 1/√2 for positive-
//! orthant random directions); it is fixed here, and those numbers are void.
//!
//! **How this corpus differs from production.** Production bge-small-
//! en-v1.5 embeddings give cosine ≈ 0.3–0.7 between unrelated texts and
//! ≥ 0.85 for paraphrases. This corpus is random unit directions, so its
//! scores cluster near 0.0 — *more* random than production, not less.
//! That is deliberate: the goal is a stable, honest synthetic workload
//! for the fetch/decode/score/sort path, not a simulation of semantic
//! similarity. Accuracy over realism — do not read these cosine values
//! as representative of production search quality.
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
//! thresholds go flaky on shared runners. For the run/compare workflow
//! (recording and diffing against the `main` baseline), see the README's
//! Benchmarks section.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tempfile::TempDir;
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
            // Map to roughly [-1, 1): (z >> 11) is a 53-bit value in [0, 2^53),
            // so (z >> 11) / 2^53 is uniform in [0, 1); subtract 0.5 and scale
            // by 2 to centre on zero. (The pre-fix expression omitted the
            // centring and produced [0, 2) — all-positive components.)
            let frac = ((z >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5) * 2.0;
            v.push(frac as f32);
        }
        normalise(&v)
    }
}

/// Returns `v / ||v||` so all corpus rows sit on the unit sphere, mirroring
/// real (normalised) embeddings.
fn normalise(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Build a corpus of `n` row vectors, plus the query vector. The query is
/// the *next* vector in the same deterministic sequence (what would be row
/// `n+1` if seeding continued) — deliberately **not** one of the seeded
/// rows, so no search result can score 1.0: there is no anchor to hide
/// behind, and every ranked hit is a genuinely independent random
/// direction. Deterministic: fixed seed, fixed generator, fixed position.
fn build_corpus(n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut corpus = Corpus::new(CORPUS_SEED);
    let rows: Vec<Vec<f32>> = (0..n).map(|_| corpus.next_vector()).collect();
    let query = corpus.next_vector();
    (rows, query)
}

/// Open a throwaway database in a fresh temp dir (cleaned up on drop), seed
/// `n` rows via the public API only, and return the (db, temp dir, query)
/// triple the bench searches against.
fn seed_db(n: usize) -> (Database, TempDir, Vec<f32>) {
    let (rows, query) = build_corpus(n);
    let tmp = TempDir::new().expect("bench: create temp dir");
    let path = tmp.path().join(format!("vipune-bench-db_search-{n}.db"));

    let db = Database::open(&path).expect("bench: open temp db");
    for row in &rows {
        db.insert(PROJECT_ID, "bench row", row, None, "fact", "active")
            .expect("bench: insert row");
    }
    (db, tmp, query)
}

/// Bench `Database::search` at 1k and 10k rows.
///
/// Each `search` call inside `bench.iter` exercises the full path:
/// SQL fetch (with the partly-degenerate `created_at` sort, see module docs),
/// BLOB deserialisation, per-row cosine scoring, similarity sort, and result
/// allocation. The query vector and seeded rows are precomputed outside the
/// timed region; the search result is *returned* from the closure, so
/// criterion's `black_box` on the output keeps the whole call (including its
/// allocation churn) in the timed region.
fn db_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("db_search");

    for n in [1_000usize, 10_000] {
        let (db, _tmp, query) = seed_db(n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rows-{n}")),
            &query,
            |bench, query| {
                bench.iter(|| {
                    db.search(PROJECT_ID, query, SEARCH_LIMIT, None, None)
                        .expect("bench: search")
                })
            },
        );

        drop(db);
        // _tmp drop cleans the directory (and the .db inside it) on scope exit.
    }

    group.finish();
}

/// Criterion config: compares each run against the stored `main` baseline
/// *without ever overwriting it* (`retain_baseline`, lenient mode).
///
/// `retain_baseline` with `strict = false` is deliberate:
/// - it never writes to the baseline store, so the baseline stays a stable
///   reference instead of drifting toward the previous run;
/// - strict mode would panic on a machine that has never recorded a baseline,
///   which is hostile on first run — lenient mode simply reports timings
///   until a baseline exists, then diffs against it.
///
/// Recording (or refreshing) the `main` baseline is a deliberate act:
/// `cargo bench -- --save-baseline main`.
fn custom_criterion() -> Criterion {
    // `retain_baseline` (lenient) is the only way to get compare-without-
    // overwrite: `Criterion::default()` alone still sets `Baseline::Save`
    // on directory "base".
    Criterion::default().retain_baseline("main".to_string(), false)
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = db_search
}
criterion_main!(benches);
