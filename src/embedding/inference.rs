//! ONNX inference for the embedding engine: tokenisation, model run, and
//! mean pooling + L2 normalisation of the hidden state.
//!
//! Lives in its own module so the engine struct file stays under the project
//! 500-line cap; the only coupling is the `EmbeddingEngine` fields the
//! inference path reads (see `mod.rs`).

use ort::inputs;
use ort::value::Tensor;

use super::engine::{EMBEDDING_DIMS, EmbeddingEngine};
use crate::errors::{Error, with_prefix};

/// Tokenise `input`, run the model, and pool + L2-normalize the output.
pub(crate) fn encode_and_infer(
    engine: &mut EmbeddingEngine,
    input: &str,
) -> Result<Vec<f32>, Error> {
    let encoding = engine
        .tokenizer
        .encode(input, true)
        .map_err(|e| with_prefix("inference tokenizer:")(Error::from(e)))?;
    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    if input_ids.is_empty() {
        return Ok(vec![0.0f32; EMBEDDING_DIMS]);
    }

    let seq_len = input_ids.len();

    let input_ids_vec: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let attention_mask_vec: Vec<i64> = attention_mask.iter().map(|&m| m as i64).collect();

    let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids_vec))?;
    let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask_vec))?;

    // Only include token_type_ids if the model requires it
    let outputs = if engine.requires_token_type_ids {
        let token_type_ids_vec: Vec<i64> = vec![0i64; seq_len]; // Single sentence, all zeros
        let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids_vec))?;
        let inputs = inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ];
        engine.session.run(inputs)?
    } else {
        let inputs = inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor
        ];
        engine.session.run(inputs)?
    };

    // rc.15's `try_extract_tensor` returns `(&Shape, &[f32])`: the shape and a
    // flat C-contiguous slice straight from the tensor, so the rc.9
    // `as_slice().unwrap()` is gone — a failed extraction is an `Err`, never a
    // bad slice. `Shape` derefs to `[i64]` (rc.9 indexed `[usize]`), so
    // [`validate_output_shape`] converts the dims with `usize::try_from` — a
    // negative dim is an error, not a wrapped `usize` — and pins the buffer
    // length against the shape before the pooling loop indexes it.
    let (shape, data) = outputs
        .get("last_hidden_state")
        .or_else(|| outputs.get("token_embeddings"))
        .ok_or_else(|| {
            Error::Inference(
                "Output tensor 'last_hidden_state' or 'token_embeddings' not found".to_string(),
            )
        })?
        .try_extract_tensor::<f32>()?;

    validate_output_shape(shape, seq_len, data.len())?;
    // Implied by the shape check above (hidden == EMBEDDING_DIMS and
    // data.len() == batch * seq_len * hidden), so every `chunk` the pooling
    // loop indexes is exactly `hidden_dim` long.
    let hidden_dim = EMBEDDING_DIMS;

    let mut pooled = vec![0.0f32; EMBEDDING_DIMS];

    for (token_idx, chunk) in data.chunks(hidden_dim).take(seq_len).enumerate() {
        let mask_value = attention_mask.get(token_idx).copied().unwrap_or(0) as f32;

        for (dim, pooled_value) in pooled.iter_mut().enumerate() {
            *pooled_value += chunk[dim] * mask_value;
        }
    }

    let mask_sum: f32 = attention_mask
        .iter()
        .take(seq_len)
        .map(|&m| m as f32)
        .sum::<f32>()
        .max(1e-9);

    for value in pooled.iter_mut() {
        *value /= mask_sum;
    }

    let normalized = l2_normalize(&pooled);
    Ok(normalized)
}

/// Validates the `(batch, seq_len, hidden)` shape of a pooled-model output
/// against what the pooling loop below is about to index.
///
/// The model is external input, so every dimension is suspect: a negative
/// dim (an ORT sentinel such as `LONG_MIN`, not a plausible batch size) must
/// never reach `as usize`, which would wrap it into a huge positive number.
/// `data_len` is checked against `batch * seq_len * hidden` so the pooling
/// loop can never index a `chunk` shorter than the hidden dimension —
/// `data.chunks(n)` yields a short tail chunk whenever the buffer is smaller
/// than the shape claims, and `chunk[dim]` would panic on it.
///
/// Pure (no engine, no ORT types) so the failure modes are testable without
/// a model.
fn validate_output_shape(shape: &[i64], seq_len: usize, data_len: usize) -> Result<(), Error> {
    if shape.len() != 3 {
        return Err(Error::Inference(format!(
            "Expected 3D output (batch, seq_len, hidden), got {:?}",
            shape
        )));
    }

    let dims: [usize; 3] = [
        usize::try_from(shape[0]).map_err(|_| negative_dim_error(shape))?,
        usize::try_from(shape[1]).map_err(|_| negative_dim_error(shape))?,
        usize::try_from(shape[2]).map_err(|_| negative_dim_error(shape))?,
    ];
    let [batch_size, output_seq_len, hidden_dim] = dims;

    if batch_size != 1 || hidden_dim != EMBEDDING_DIMS {
        return Err(Error::Inference(format!(
            "Unexpected output shape: {shape:?}, batch=1, hidden=384 expected",
        )));
    }

    if output_seq_len != seq_len {
        return Err(Error::Inference(format!(
            "Unexpected output seq_len {output_seq_len}, {seq_len} expected (shape {shape:?})",
        )));
    }

    // `checked_mul` so a huge (but positive) dim from a corrupt model cannot
    // wrap the expected length around into a small number that `data_len`
    // happens to match.
    let expected_len = batch_size
        .checked_mul(output_seq_len)
        .and_then(|n| n.checked_mul(hidden_dim))
        .ok_or_else(|| Error::Inference(format!("Output shape overflows usize: {shape:?}")))?;
    if data_len != expected_len {
        return Err(Error::Inference(format!(
            "Output tensor holds {data_len} f32 values, shape {shape:?} implies {expected_len}",
        )));
    }

    Ok(())
}

#[cold]
fn negative_dim_error(shape: &[i64]) -> Error {
    Error::Inference(format!("Negative output dimension in shape: {shape:?}"))
}

pub(crate) fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm = norm.max(1e-9);

    vec.iter().map(|&x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid bge output: batch 1, `seq_len` tokens, 384 hidden.
    fn valid_shape(seq_len: usize) -> [i64; 3] {
        [1, seq_len as i64, EMBEDDING_DIMS as i64]
    }

    #[test]
    fn validate_output_shape_accepts_a_valid_output() {
        let shape = valid_shape(4);
        assert!(validate_output_shape(&shape, 4, 4 * EMBEDDING_DIMS).is_ok());
    }

    /// A negative dim must be an error: `as usize` would wrap it into a huge
    /// positive number that then passes no meaningful bound.
    #[test]
    fn validate_output_shape_rejects_a_negative_dim() {
        for dim in [0usize, 1, 2] {
            let mut shape = valid_shape(4);
            shape[dim] = -1;
            let err = validate_output_shape(&shape, 4, 4 * EMBEDDING_DIMS)
                .expect_err("a negative dim must not validate");
            assert!(
                matches!(err, Error::Inference(_)),
                "dim {dim}: expected Inference, got {err:?}"
            );
            assert!(
                err.to_string().contains("-1"),
                "dim {dim}: message must name the offending shape, got {err}"
            );
        }
    }

    #[test]
    fn validate_output_shape_rejects_a_wrong_rank() {
        // 2D (seq_len, hidden) — the shape a non-batched export would carry.
        let shape = [4i64, EMBEDDING_DIMS as i64];
        let err = validate_output_shape(&shape, 4, 4 * EMBEDDING_DIMS)
            .expect_err("a 2D output must not validate");
        assert!(matches!(err, Error::Inference(_)), "got {err:?}");

        let shape = [1i64, 1, 4, EMBEDDING_DIMS as i64];
        let err = validate_output_shape(&shape, 4, 4 * EMBEDDING_DIMS)
            .expect_err("a 4D output must not validate");
        assert!(matches!(err, Error::Inference(_)), "got {err:?}");
    }

    /// The panic this guard exists to prevent: with a buffer shorter than the
    /// shape claims, `data.chunks(hidden_dim)` yields a short tail chunk and
    /// the pooling loop indexes past it.
    #[test]
    fn validate_output_shape_rejects_a_short_data_buffer() {
        let shape = valid_shape(4);
        let err = validate_output_shape(&shape, 4, 4 * EMBEDDING_DIMS - 1)
            .expect_err("a short buffer must not validate");
        assert!(matches!(err, Error::Inference(_)), "got {err:?}");
        assert!(
            err.to_string().contains("implies"),
            "message must name the length mismatch, got {err}"
        );
    }

    #[test]
    fn validate_output_shape_rejects_a_long_data_buffer() {
        let shape = valid_shape(4);
        assert!(validate_output_shape(&shape, 4, 4 * EMBEDDING_DIMS + 1).is_err());
    }

    /// batch != 1 and hidden != 384 keep the pre-existing guards, and a
    /// seq_len that disagrees with the tokenised input must not pool zeros for
    /// the tokens the model did not return.
    #[test]
    fn validate_output_shape_rejects_unexpected_batch_hidden_or_seq_len() {
        let mut shape = valid_shape(4);
        shape[0] = 2;
        assert!(validate_output_shape(&shape, 4, 8 * EMBEDDING_DIMS).is_err());

        let mut shape = valid_shape(4);
        shape[2] = (EMBEDDING_DIMS + 1) as i64;
        assert!(validate_output_shape(&shape, 4, 4 * (EMBEDDING_DIMS + 1)).is_err());

        let shape = valid_shape(3);
        let err = validate_output_shape(&shape, 4, 3 * EMBEDDING_DIMS)
            .expect_err("a short seq_len must not validate");
        assert!(matches!(err, Error::Inference(_)), "got {err:?}");
    }
}
