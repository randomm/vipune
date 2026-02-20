//! Embedding BLOB conversion and cosine similarity computation.

use super::Error;

pub type Result<T> = std::result::Result<T, Error>;

const EMBEDDING_DIMS: usize = 384;
const EMBEDDING_BLOB_SIZE: usize = EMBEDDING_DIMS * 4; // 384 f32 values × 4 bytes each

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
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64> {
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
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
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
}
