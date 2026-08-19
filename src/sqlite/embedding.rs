//! Embedding BLOB conversion, cosine similarity computation, and embedding classification.

use super::Error;

pub type Result<T> = std::result::Result<T, Error>;

const EMBEDDING_DIMS: usize = 384;
const EMBEDDING_BLOB_SIZE: usize = EMBEDDING_DIMS * 4; // 384 f32 values × 4 bytes each

/// Classification of an embedding vector based on its L2 norm.
///
/// `EmbeddingEngine::embed` applies `l2_normalize` unconditionally to all model
/// output, so any real embedding has norm ≈ 1. Mock vectors (uniform [-1,1] over
/// 384 dims) have norm ≈ 11.3. Anything else is unknown/corrupted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingClass {
    /// L2-normalised vector from the real model (norm ∈ [0.99, 1.01]).
    Real,
    /// Non-normalised mock or hand-crafted vector (norm > 2.0).
    Mock,
    /// Zero vector or other unexpected norm — skipped by reindex.
    Unknown,
}

/// Classify an embedding vector by computing its L2 norm.
///
/// Thresholds:
/// - `norm ∈ [0.99, 1.01]` → `Real` (L2-normalised by `EmbeddingEngine::embed`)
/// - `norm > 2.0` → `Mock` (e.g., uniform [-1,1] over 384 dims ≈ 11.3)
/// - anything else (including `norm == 0`) → `Unknown`/corrupted
pub fn classify_embedding(embedding: &[f32]) -> EmbeddingClass {
    let norm: f64 = embedding
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if (0.99..=1.01).contains(&norm) {
        EmbeddingClass::Real
    } else if norm > 2.0 {
        EmbeddingClass::Mock
    } else {
        EmbeddingClass::Unknown
    }
}

/// Convert a vector of f32 embedding values to a BLOB (little-endian bytes).
///
/// # Errors
///
/// Returns `Error::MismatchedDimensions` if the vector length is not exactly 384.
pub fn vec_to_blob(vec: &[f32]) -> Result<Vec<u8>> {
    if vec.len() != EMBEDDING_DIMS {
        return Err(Error::MismatchedDimensions {
            expected: EMBEDDING_DIMS,
            actual: vec.len(),
        });
    }
    Ok(vec.iter().flat_map(|&x| x.to_le_bytes()).collect())
}

/// Convert a BLOB (little-endian bytes) to a vector of f32 embedding values.
///
/// # Errors
///
/// Returns `Error::InvalidBlobSize` if the blob length is not exactly 1,536 bytes.
pub fn blob_to_vec(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() != EMBEDDING_BLOB_SIZE {
        return Err(Error::InvalidBlobSize {
            expected: EMBEDDING_BLOB_SIZE,
            actual: blob.len(),
        });
    }
    let mut vec = Vec::with_capacity(EMBEDDING_DIMS);
    for chunk in blob.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        vec.push(val);
    }
    Ok(vec)
}

/// Compute cosine similarity between two embedding vectors.
///
/// # Errors
///
/// - Returns `Error::EmptyVector` if either vector is empty.
/// - Returns `Error::MismatchedDimensions` if vectors have different lengths.
/// - Returns `Error::InvalidEmbedding` if any value is NaN or infinite.
// Only called from tests; production search paths use
// `cosine_similarity_with_norm` with a hoisted query norm. The exact-equality
// test locks this function as the reference implementation.
#[cfg_attr(not(test), allow(dead_code))]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64> {
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    cosine_similarity_with_norm(a, norm_a, b)
}

/// Compute cosine similarity between two embedding vectors, taking the L2 norm
/// of `a` as a precomputed parameter so a caller (e.g. `Database::search`) can
/// hoist the query-vector norm out of a per-row loop.
///
/// **Invariant:** `norm_a` MUST be the L2 norm of `a`, computed with the same
/// f64 accumulation expression as here (`(*x as f64).powi(2)` / `.sum::<f64>()`
/// / `.sqrt()`). Passing a norm computed from `b` — or computed in f32, e.g.
/// via `l2_normalize` — silently yields wrong results: the zero-norm guard is
/// bypassed or misfired, and the division is off. The exact-equality test in
/// `tests` is the lock against operand-order or precision regressions.
///
/// # Errors
///
/// - Returns `Error::EmptyVector` if either vector is empty.
/// - Returns `Error::MismatchedDimensions` if vectors have different lengths.
/// - Returns `Error::InvalidEmbedding` if any value is NaN or infinite.
pub(crate) fn cosine_similarity_with_norm(a: &[f32], norm_a: f64, b: &[f32]) -> Result<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::EmptyVector);
    }

    if a.len() != b.len() {
        return Err(Error::MismatchedDimensions {
            expected: a.len(),
            actual: b.len(),
        });
    }

    if a.iter().any(|x| x.is_nan() || x.is_infinite())
        || b.iter().any(|x| x.is_nan() || x.is_infinite())
    {
        return Err(Error::InvalidEmbedding(
            "Vector contains NaN or infinite values".to_string(),
        ));
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_to_blob_correct_dimensions() {
        let vec = vec![0.1f32; 384];
        let blob = vec_to_blob(&vec).unwrap();
        assert_eq!(blob.len(), 1536);
    }

    #[test]
    fn test_vec_to_blob_wrong_dimensions() {
        let vec = vec![0.1f32; 256];
        assert!(matches!(
            vec_to_blob(&vec),
            Err(Error::MismatchedDimensions { .. })
        ));
    }

    #[test]
    fn test_blob_to_vec_correct_size() {
        let vec = vec![0.1f32; 384];
        let blob = vec_to_blob(&vec).unwrap();
        let recovered = blob_to_vec(&blob).unwrap();
        assert_eq!(recovered.len(), 384);
        for (a, b) in vec.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blob_to_vec_wrong_size() {
        let blob = vec![0u8; 1500];
        assert!(matches!(
            blob_to_vec(&blob),
            Err(Error::InvalidBlobSize { .. })
        ));
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec = vec![1.0f32; 384];
        let sim = cosine_similarity(&vec, &vec).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let mut a = vec![0.0f32; 384];
        let mut b = vec![0.0f32; 384];
        a[0] = 1.0;
        b[1] = 1.0;
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty_vector() {
        let empty = vec![];
        let vec = vec![1.0f32; 384];
        assert!(cosine_similarity(&empty, &vec).is_err());
    }

    #[test]
    fn test_cosine_similarity_mismatched_dimensions() {
        let a = vec![1.0f32; 384];
        let b = vec![1.0f32; 256];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_cosine_similarity_nan_values() {
        let mut a = vec![1.0f32; 384];
        a[0] = f32::NAN;
        let b = vec![1.0f32; 384];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_cosine_similarity_infinite_values() {
        let mut a = vec![1.0f32; 384];
        a[0] = f32::INFINITY;
        let b = vec![1.0f32; 384];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        let zero = vec![0.0f32; 384];
        let vec = vec![1.0f32; 384];
        let sim = cosine_similarity(&zero, &vec).unwrap();
        assert_eq!(sim, 0.0);
    }

    fn norm_of(v: &[f32]) -> f64 {
        v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
    }

    /// Fixed-seed pseudorandom vector in [-1, 1] (SplitMix64, seed 42).
    fn fixed_seed_pseudorandom() -> Vec<f32> {
        let mut state: u64 = 42;
        (0..384)
            .map(|_| {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                (z % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// Behaviour-preservation lock: the hoisted-norm variant must be bit-
    /// identical to `cosine_similarity` for the same inputs. `assert_eq!` is
    /// exact — any reordering of the f64 accumulation or an operand swap in
    /// `norm_a` breaks at least one of these fixtures.
    #[test]
    fn test_cosine_similarity_with_norm_exact_equality() {
        let b = vec![0.7f32; 384];
        let fixtures: Vec<Vec<f32>> = vec![
            vec![0.1f32; 384],
            vec![0.5f32; 384],
            vec![1.0f32; 384],
            (0..384)
                .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
                .collect(),
            fixed_seed_pseudorandom(),
            vec![0.0f32; 384],
        ];
        for a in &fixtures {
            let expected = cosine_similarity(a, &b).unwrap();
            let actual = cosine_similarity_with_norm(a, norm_of(a), &b).unwrap();
            assert_eq!(actual, expected, "fixture norm: {}", norm_of(a));
        }
    }

    /// A zero-norm query through the hoisted variant returns Ok(0.0), matching
    /// `test_cosine_similarity_zero_norm`.
    #[test]
    fn test_cosine_similarity_with_norm_zero_norm_query() {
        let zero = vec![0.0f32; 384];
        let vec = vec![1.0f32; 384];
        let sim = cosine_similarity_with_norm(&zero, norm_of(&zero), &vec).unwrap();
        assert_eq!(sim, 0.0);
    }

    /// Both entry points must return the same Error variant for each invalid
    /// input shape: empty, mismatched dimensions, NaN, infinite.
    #[test]
    fn test_cosine_similarity_error_parity() {
        let empty = Vec::new();
        let a384 = vec![1.0f32; 384];
        let b256 = vec![1.0f32; 256];
        let mut nan = vec![1.0f32; 384];
        nan[0] = f32::NAN;
        let mut inf = vec![1.0f32; 384];
        inf[0] = f32::INFINITY;

        let cases: [(&[f32], &[f32]); 4] = [
            (&empty, &a384),
            (&a384, &b256),
            (&nan, &a384),
            (&inf, &a384),
        ];
        for (a, b) in cases {
            let na = norm_of(a);
            match (
                cosine_similarity(a, b),
                cosine_similarity_with_norm(a, na, b),
            ) {
                (Err(e1), Err(e2)) => assert_eq!(format!("{e1:?}"), format!("{e2:?}")),
                other => panic!("expected both to error, got {:?}", other),
            }
        }
    }

    // ---- Embedding classification tests ----

    #[test]
    fn test_classify_embedding_l2_normalised_is_real() {
        // A unit vector (norm = 1.0) — like output from EmbeddingEngine::embed
        let mut vec = vec![0.0f32; 384];
        vec[0] = 1.0;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Real);
    }

    #[test]
    fn test_classify_embedding_mock_vector_is_mock() {
        // Mock vectors are uniform [-1,1] over 384 dims, norm ≈ 11.3
        let mut vec = Vec::with_capacity(384);
        let hash: u64 = 0x123456789abcdef;
        for i in 0..384 {
            let mut dim_hash = hash.wrapping_add(i as u64);
            dim_hash ^= dim_hash >> 33;
            dim_hash = dim_hash.wrapping_mul(0xff51afd7ed558ccd);
            dim_hash ^= dim_hash >> 33;
            dim_hash = dim_hash.wrapping_mul(0xc4ceb9fe1a85ec53);
            let value = ((dim_hash % 2000) as f32 - 1000.0) / 1000.0;
            vec.push(value);
        }
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Mock);
    }

    #[test]
    fn test_classify_embedding_uniform_ones_is_mock() {
        // vec![1.0f32; 384] has norm ≈ sqrt(384) ≈ 19.6 — mock
        let vec = vec![1.0f32; 384];
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Mock);
    }

    #[test]
    fn test_classify_embedding_zero_vector_is_unknown() {
        // norm == 0 — unknown/corrupted
        let vec = vec![0.0f32; 384];
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Unknown);
    }

    #[test]
    fn test_classify_embedding_near_boundary_real_lower() {
        // norm = 1.0 (safe within [0.99, 1.01])
        let mut vec = vec![0.0f32; 384];
        vec[0] = 1.0;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Real);
    }

    #[test]
    fn test_classify_embedding_near_boundary_real_upper() {
        // norm ≈ 0.995 (safely within [0.99, 1.01])
        let mut vec = vec![0.0f32; 384];
        vec[0] = 0.995f32;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Real);
    }

    #[test]
    fn test_classify_embedding_just_below_real_range() {
        // norm = 0.98 — unknown (below real range)
        let mut vec = vec![0.0f32; 384];
        vec[0] = 0.98f32;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Unknown);
    }

    #[test]
    fn test_classify_embedding_just_above_real_range_below_mock() {
        // norm = 1.5 — unknown (between real and mock thresholds)
        let mut vec = vec![0.0f32; 384];
        vec[0] = 1.5f32;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Unknown);
    }

    #[test]
    fn test_classify_embedding_at_mock_threshold() {
        // norm = 2.01 — mock (just above threshold)
        let mut vec = vec![0.0f32; 384];
        vec[0] = 2.01f32;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Mock);
    }

    #[test]
    fn test_classify_embedding_at_mock_boundary_exclusive() {
        // norm = 2.0 — unknown (not strictly greater than 2.0)
        let mut vec = vec![0.0f32; 384];
        vec[0] = 2.0f32;
        assert_eq!(classify_embedding(&vec), EmbeddingClass::Unknown);
    }
}
