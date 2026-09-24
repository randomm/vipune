//! ONNX inference for the embedding engine: tokenisation, model run, and
//! mean pooling + L2 normalisation of the hidden state.
//!
//! Lives in its own module so the engine struct file stays under the project
//! 500-line cap; the only coupling is the `EmbeddingEngine` fields the
//! inference path reads (see `mod.rs`).

use ort::inputs;
use ort::value::Tensor;

use super::engine::{EMBEDDING_DIMS, EmbeddingEngine};
use crate::errors::Error;

/// Tokenise `input`, run the model, and pool + L2-normalize the output.
pub(crate) fn encode_and_infer(
    engine: &mut EmbeddingEngine,
    input: &str,
) -> Result<Vec<f32>, Error> {
    let encoding = engine.tokenizer.encode(input, true)?;
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
        engine.session.run(inputs?)?
    } else {
        let inputs = inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor
        ];
        engine.session.run(inputs?)?
    };

    let last_hidden_state = outputs
        .get("last_hidden_state")
        .or_else(|| outputs.get("token_embeddings"))
        .ok_or_else(|| {
            Error::Inference(
                "Output tensor 'last_hidden_state' or 'token_embeddings' not found".to_string(),
            )
        })?
        .try_extract_tensor::<f32>()?;

    let shape = last_hidden_state.shape();
    let data = last_hidden_state.as_slice().unwrap();
    if shape.len() != 3 {
        return Err(Error::Inference(format!(
            "Expected 3D output (batch, seq_len, hidden), got {:?}",
            shape
        )));
    }

    let batch_size = shape[0];
    let hidden_dim = shape[2];

    if batch_size != 1 || hidden_dim != EMBEDDING_DIMS {
        return Err(Error::Inference(format!(
            "Unexpected output shape: {:?}, batch=1, hidden=384 expected",
            shape
        )));
    }

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

pub(crate) fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm = norm.max(1e-9);

    vec.iter().map(|&x| x / norm).collect()
}
