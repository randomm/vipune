//! Golden embedding fixture for [`crate::embedding::EmbeddingEngine`].
//!
//! Purpose: pin the **numbers** the ONNX pipeline produces. Every other
//! model-backed test in this crate checks a shape or a band (L2 norm ≈ 1.0,
//! self-match > 0.99), so a change in the runtime that shifts the vectors
//! slightly — an ONNX Runtime upgrade, a different pooling kernel — stays
//! invisible. `EmbeddingClass::Real` is norm-based, so a reindex cannot see
//! drift either. This module compares the element-wise output of the
//! production role path against a committed fixture.
//!
//! ## What is compared
//!
//! The values returned by [`EmbeddingEngine::embed_passage`] (post-pooling,
//! post-L2-normalisation) for each fixture text: 384 f32 per text,
//! element-wise max abs difference <= `MAX_ABS_DIFF` (1e-4).
//!
//! ## Data file
//!
//! `golden_bge_small.txt`, next to this file. One record per line, fields
//! separated by a single space:
//!
//! ```text
//! <text> <384 floats in %.9e>
//! ```
//!
//! `<text>` is the raw golden text, so a space inside it would break the
//! single-space field separator; the generator therefore collapses every
//! whitespace run out of the corpus before embedding, and `GOLDEN_TEXTS`
//! stores the texts in that already-collapsed form (see `whitespace_key`).
//! The file stores the texts too, and `test_golden_fixture_shape` pins file
//! texts against that array so a hand-edit of either side fails loudly.
//!
//! ## Regenerating
//!
//! Only when the model id/revision changes (never to make a drift failure
//! pass — that is the whole point of the test):
//!
//! ```text
//! VIPUNE_REGENERATE_GOLDEN=1 cargo test golden_fixture_regenerate -- --ignored --nocapture
//! ```
//!
//! The env var is the opt-in and it is load-bearing: the test is `#[ignore]`d
//! but a bare `cargo test -- --ignored` (the nightly CI lane) still runs every
//! ignored test, and a generator that always rewrote the data file would both
//! dirty the tree and make [`test_golden_embeddings_match_fixture`] pass
//! vacuously. Only the exact value `1` enables the generator; unset or any
//! other value does nothing.
//!
//! The generator re-embeds `GOLDEN_TEXTS` through `embed_passage` at the
//! pinned revision and rewrites the file. Commit the new file together with
//! the revision bump, and re-embed existing databases
//! (`vipune reindex --force --all-projects`).
//!
//! The generator embeds the whitespace-collapsed form of each text, which is
//! what `GOLDEN_TEXTS` stores — so no escaping is needed in the data file.

use crate::embedding::{EMBED_MODEL_ID, EMBEDDING_DIMS, EmbeddingEngine};

/// Relative path of the committed fixture, resolved against the crate root.
const GOLDEN_FILE: &str = "src/embedding/tests/golden_bge_small.txt";

/// Element-wise tolerance for the golden comparison (issue #265).
const MAX_ABS_DIFF: f32 = 1e-4;

/// The fixed corpus: short, long, unicode and code-like texts, all non-empty
/// and all well under the 512-token limit (the engine short-circuits empty
/// input to a zero vector, which would pin nothing). The last entry carries
/// the collapsed form of a wider-spaced original — whitespace runs cannot
/// survive the fixture format, so it pins the collapse, not the raw spacing.
///
/// Each entry is stored in the canonical `whitespace_key` form (whitespace
/// runs collapsed to one space, ends trimmed) because the data file uses a
/// single space as its field separator.
const GOLDEN_TEXTS: [&str; 10] = [
    "hello world",
    "the cat sat on the mat",
    "Memory stores keep context alive across sessions for AI agents.",
    "API endpoint returns 429 Too Many Requests when the rate limit is exceeded.",
    "Der Speicher hält den Kontext für KI-Agenten über Sitzungen hinweg bereit.",
    "记忆为智能体跨会话保留上下文信息",
    "記憶エージェントの文脈をセッション横断で保持する",
    "SELECT m.id, m.content FROM memories m WHERE m.status = 'active' ORDER BY m.created_at DESC LIMIT 25;",
    "fn main() { let store = MemoryStore::new(&path, model_id, config)?; store.add(\"p\", \"fact\", None)?; }",
    "multiple spaces and tabs",
];

/// The canonical fixture form of `raw`: every whitespace run collapsed to a
/// single space and both ends trimmed — the form the data file can store
/// without escaping, and the form the generator feeds the model.
fn whitespace_key(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Builds the fixture line for one (text, embedding) pair.
///
/// The `{:.9e}` field format is load-bearing, not cosmetic: nine mantissa
/// digits round-trips every `f32` exactly, and a fixed exponent width keeps
/// every record the same byte layout, so the positional split in
/// [`decode_line`] and the file's size stay stable across regenerations.
fn encode_line(text: &str, embedding: &[f32]) -> String {
    let mut line = text.to_string();
    for value in embedding {
        line.push(' ');
        line.push_str(&format!("{value:.9e}"));
    }
    line
}

/// Parses one fixture line into `(text, embedding)`.
///
/// Fields are separated by single spaces and every record holds exactly
/// [`EMBEDDING_DIMS`] floats, so the split is positional: the last
/// `EMBEDDING_DIMS` fields are the floats and everything before them is the
/// text. A content-based split ("leading non-numeric fields") would corrupt
/// any text containing a bare number — `"... returns 429 Too Many ..."` is in
/// the corpus — by eating the number and the floats that follow it as text.
fn decode_line(line: &str) -> Option<(String, Vec<f32>)> {
    let fields: Vec<&str> = line.split(' ').collect();
    let split = fields.len().checked_sub(EMBEDDING_DIMS)?;
    if split == 0 {
        return None;
    }
    let floats = fields[split..]
        .iter()
        .map(|token| token.parse::<f32>().ok())
        .collect::<Option<Vec<f32>>>()?;
    // A non-finite float (`nan`, `inf` — each parses successfully) would
    // poison every downstream comparison: `NaN.abs()` is NaN and `NaN <= tol`
    // is false, so the drift test fails with a message that blames the
    // runtime. Reject it at the parse boundary instead.
    if !floats.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((fields[..split].join(" "), floats))
}

/// The fixture file contents, read from the crate root (a test-only read of
/// a tracked file, so `unwrap` on IO is the accepted style here).
fn read_fixture_lines() -> Vec<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(&manifest_dir).join(GOLDEN_FILE);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.to_string())
        .collect()
}

/// Model-free sanity check on the committed data file: it parses to exactly
/// `GOLDEN_TEXTS.len()` records, each with exactly [`EMBEDDING_DIMS`] finite
/// floats, and each carrying the text from `GOLDEN_TEXTS` in order. Runs on
/// every `cargo test` — no model, no download.
#[test]
fn test_golden_fixture_shape() {
    let lines = read_fixture_lines();
    assert_eq!(
        lines.len(),
        GOLDEN_TEXTS.len(),
        "fixture must hold one record per golden text"
    );

    for (index, line) in lines.iter().enumerate() {
        let (text, embedding) = decode_line(line)
            .unwrap_or_else(|| panic!("fixture line {} does not parse", index + 1));
        assert_eq!(
            text,
            GOLDEN_TEXTS[index],
            "fixture line {} must carry the golden text at that index",
            index + 1
        );
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIMS,
            "fixture line {} must hold exactly {EMBEDDING_DIMS} floats",
            index + 1
        );
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "fixture line {} must hold only finite floats",
            index + 1
        );
        // The fixture pins post-normalisation output, so each vector must be a
        // unit vector: a fixture whose norm is off cannot detect the drift it
        // exists to detect (the production path L2-normalises, so 1.0 is the
        // only defensible expectation).
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() <= 1e-3,
            "fixture line {} must be L2-normalised (norm {norm})",
            index + 1
        );
    }
}

/// Re-embeds every golden text through the production passage path and
/// asserts the element-wise max abs difference against the fixture stays
/// within [`MAX_ABS_DIFF`]. Prints the measured max diff.
///
/// Ignored: downloads/loads the ONNX model (nightly CI lane).
#[ignore]
#[test]
fn test_golden_embeddings_match_fixture() {
    let lines = read_fixture_lines();
    assert_eq!(lines.len(), GOLDEN_TEXTS.len());

    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID)
        .unwrap_or_else(|e| panic!("load model {EMBED_MODEL_ID} for golden comparison: {e}"));
    let mut measured_max_diff = 0.0f32;
    let mut worst_text = String::new();

    for (index, line) in lines.iter().enumerate() {
        let (text, expected) = decode_line(line)
            .unwrap_or_else(|| panic!("fixture line {} does not parse", index + 1));
        assert_eq!(text, GOLDEN_TEXTS[index]);
        // Zip is pairwise-truncating, so a short fixture vector would compare
        // a prefix and report the drift of the record it silently dropped.
        assert_eq!(
            expected.len(),
            EMBEDDING_DIMS,
            "fixture line {} must hold exactly {EMBEDDING_DIMS} floats",
            index + 1
        );

        let actual = engine
            .embed_passage(&text)
            .unwrap_or_else(|e| panic!("embed golden text {index}: {e}"));
        assert_eq!(actual.len(), EMBEDDING_DIMS);

        let diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        println!("golden[{index}] max abs diff = {diff:.3e}  text = {text:?}");
        if diff > measured_max_diff {
            measured_max_diff = diff;
            worst_text = text;
        }
    }

    println!(
        "golden fixture: measured max abs diff over {} texts = {measured_max_diff:.3e}",
        GOLDEN_TEXTS.len()
    );
    assert!(
        measured_max_diff <= MAX_ABS_DIFF,
        "golden embedding drift {measured_max_diff:.3e} exceeds {MAX_ABS_DIFF:e} (worst text {worst_text:?}) — \
         the runtime changed the vectors; re-embed existing databases with \
         `vipune reindex --force --all-projects` and report the measured drift",
    );
}

/// Env var that opts in to rewriting the committed fixture. It must be set to
/// exactly `1`; any other value (or no value) makes [`golden_fixture_regenerate`]
/// return immediately.
const REGENERATE_ENV_VAR: &str = "VIPUNE_REGENERATE_GOLDEN";

/// Regenerates the committed fixture from the current runtime.
///
/// Opt-in via [`REGENERATE_ENV_VAR`]: `#[ignore]`d tests still run under a
/// bare `cargo test -- --ignored`, and an always-on generator would rewrite
/// the committed fixture on that run — dirtying the tree and neutralising
/// [`test_golden_embeddings_match_fixture`]. Only the exact value `1` enables
/// it; any other value (including `true` or `0`) prints a skip line and
/// returns without touching the file or loading the model, so a stray
/// `VIPUNE_REGENERATE_GOLDEN=0` in an environment cannot rewrite the fixture.
///
/// With the opt-in it is intentionally destructive to the data file: run it
/// only when the model id/revision legitimately changes (see the module
/// header). To capture a baseline from scratch, delete the file first — this
/// test rewrites it either way, the shape test is what catches a bad
/// regeneration.
#[ignore]
#[test]
fn golden_fixture_regenerate() {
    if std::env::var(REGENERATE_ENV_VAR).as_deref() != Ok("1") {
        println!(
            "skipped: {REGENERATE_ENV_VAR} is not set, leaving {GOLDEN_FILE} untouched \
             (set {REGENERATE_ENV_VAR}=1 to regenerate)"
        );
        return;
    }

    let mut engine = EmbeddingEngine::new(EMBED_MODEL_ID)
        .unwrap_or_else(|e| panic!("load model {EMBED_MODEL_ID} for golden regeneration: {e}"));

    let mut out = String::new();
    for text in GOLDEN_TEXTS {
        let text = whitespace_key(text);
        let embedding = engine
            .embed_passage(&text)
            .unwrap_or_else(|e| panic!("embed {text:?}: {e}"));
        assert_eq!(embedding.len(), EMBEDDING_DIMS, "embed {text:?}");
        out.push_str(&encode_line(&text, &embedding));
        out.push('\n');
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(&manifest_dir).join(GOLDEN_FILE);
    std::fs::write(&path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} golden records to {}",
        GOLDEN_TEXTS.len(),
        path.display()
    );
}

#[cfg(test)]
mod unit {
    use super::*;

    /// The encode/decode pair must round trip every corpus text exactly —
    /// text preserved, all 384 floats recovered, no precision lost to the
    /// `{:.9e}` field format. Model-free.
    ///
    /// It must be exercised on the real corpus rather than a synthetic string:
    /// the split is positional, so only the entry containing the bare number
    /// `429` catches a parser that splits at the first float (see
    /// `decode_line`).
    #[test]
    fn encode_decode_roundtrips_text_and_floats() {
        let embedding: Vec<f32> = (0..EMBEDDING_DIMS)
            .map(|i| (i as f32 - 192.0) / 1000.0)
            .collect();
        for text in GOLDEN_TEXTS {
            let text = whitespace_key(text);
            assert_eq!(whitespace_key(&text), text, "{text:?} is not canonical");
            let line = encode_line(&text, &embedding);
            let (decoded_text, decoded_embedding) =
                decode_line(&line).expect("generated line parses");
            assert_eq!(decoded_text, text);
            assert_eq!(
                decoded_embedding.len(),
                EMBEDDING_DIMS,
                "{text:?}: a text word was parsed as a float"
            );
            let diff = decoded_embedding
                .iter()
                .zip(embedding.iter())
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff <= MAX_ABS_DIFF,
                "text {text:?} lost precision: {diff:e}"
            );
        }
    }

    #[test]
    fn whitespace_key_collapses_runs_and_trims() {
        assert_eq!(whitespace_key("  a\t\n b   c "), "a b c");
        assert_eq!(whitespace_key("no-change"), "no-change");
    }

    /// The data file relies on every text surviving a space-joined round trip:
    /// some corpus entries are multi-word, the CJK entries are deliberately
    /// single-token, so both shapes must survive the split.
    #[test]
    fn corpus_covers_both_single_word_and_multi_word_texts() {
        let multi = GOLDEN_TEXTS.iter().filter(|t| t.contains(' ')).count();
        let single = GOLDEN_TEXTS.iter().filter(|t| !t.contains(' ')).count();
        assert!(multi >= 2, "corpus must hold multi-word texts");
        assert!(single >= 1, "corpus must hold space-free texts");
    }

    #[test]
    fn decode_line_rejects_a_record_with_too_few_fields() {
        // Fewer fields than EMBEDDING_DIMS + 1: no room for a text and the vector.
        assert!(decode_line("no floats here").is_none());
    }

    #[test]
    fn decode_line_rejects_a_non_numeric_float_field() {
        let line = encode_line("some words", &[0.5f32; EMBEDDING_DIMS - 1]);
        assert!(decode_line(&format!("{line} notanumber")).is_none());
    }

    /// `"nan"`/`"inf"` parse as `f32`, so a corrupted fixture record would
    /// otherwise survive the parse and fail the drift test with a message
    /// blaming the runtime instead of the file (see `decode_line`).
    #[test]
    fn decode_line_rejects_non_finite_floats() {
        for token in ["nan", "inf", "-inf"] {
            let line = encode_line("some words", &[0.5f32; EMBEDDING_DIMS - 1]);
            assert!(
                decode_line(&format!("{line} {token}")).is_none(),
                "{token} must not decode as a fixture float"
            );
        }
    }

    /// The positional split must keep a numeric-looking word inside the text:
    /// a "split at the first float" parser returns `"returns"` plus 386 floats
    /// for this input, so the text silently loses words while the float count
    /// alone still looks plausible. This is the corpus case (`429`) that made
    /// the content-based split wrong.
    #[test]
    fn decode_line_keeps_numeric_looking_words_in_the_text() {
        let text = "returns 429 too many";
        let embedding: Vec<f32> = (0..EMBEDDING_DIMS).map(|i| i as f32).collect();
        let line = encode_line(text, &embedding);
        let (decoded_text, decoded_embedding) = decode_line(&line).expect("line parses");
        assert_eq!(decoded_text, text);
        assert_eq!(decoded_embedding, embedding);
    }
}
