# Embedding Pipeline

This guide explains vipune's text-to-vector embedding pipeline for contributors.

## Pipeline Overview

Text input → Tokenizer → ONNX session → Mean pooling → L2 normalization → 384-dim vector → BLOB storage

### Step-by-Step

```
1. Input text
   ↓
2. HuggingFace tokenizer (max_length=512, truncation enabled)
   ↓
3. ONNX inference (bge-small-en-v1.5 model)
   → Returns [1, seq_len, 384] tensor
   ↓
4. Mean pooling across sequence length
   → Reduces to [384] vector
   ↓
5. L2 normalization
   → Unit vector (norm = 1.0)
   ↓
6. Convert to little-endian bytes
   → 1536 bytes (384 × 4 bytes/f32)
   ↓
7. Store as SQLite BLOB
```

## Model Details

### bge-small-en-v1.5

- **Source**: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Dimensions**: 384
- **Type**: Fine-tuned BERT for semantic embeddings
- **Why chosen**: Size/quality tradeoff — 384 dimensions sufficient for memory retrieval while keeping embeddings small (1,536 bytes each)

### Model Cache

- **Location**: by default `~/.cache/huggingface/hub` (or `$HF_HOME/hub` when `HF_HOME` is set)
- **Download**: First use only via `hf_hub` crate
- **Reuse**: All subsequent operations use cached model
- **Size**: ~66MB (ONNX model + tokenizer)

## ONNX Integration

### Version Pinning

vipune uses ONNX Runtime v2.0.0-rc.15, **pinned with exact version**:

```toml
# In Cargo.toml
ort = { version = "=2.0.0-rc.15", default-features = false, features = ["std", "download-binaries", "tls-rustls", "copy-dylibs", "api-31"] }
ort-sys = "=2.0.0-rc.15"
```

The feature list is explicit (with `default-features = false`) because the
rc.15 defaults include `tls-native`, which pulls `native-tls`/`openssl-sys` in
just to download the prebuilt ONNX Runtime at build time. `tls-rustls` keeps
the rustls + `ring` + webpki-roots stack the rc.9 upgrade used, so the tree has
one TLS stack and no OpenSSL. `ort-sys` needs no TLS feature of its own —
`ort`'s `tls-rustls` enables the matching `ort-sys` feature. `ndarray` is not
enabled: no code in this crate builds or extracts an `ndarray` tensor.

**Why pinned**: the rc series carries the memory-safety / undefined-behaviour
fixes this release needs, and `ort` declares exactly the matching `ort-sys`, so
the two pins always move together (see `.github/dependabot.yml`, which groups
them). Upgrading ONNX requires careful testing, as newer versions may shift the
vectors the stored embeddings were produced with — `src/embedding/tests/golden.rs`
compares the production passage output against a committed fixture to catch that.

### `&mut self` Requirement

The `EmbeddingEngine::embed_passage()` / `embed_query()` methods require `&mut self` because ONNX internally mutates tensor state for inference:

```rust
impl EmbeddingEngine {
    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>, Error> {
        // ONNX tensor allocation mutates internal state
        let outputs = self.session.run(inputs)?;
        // ...
    }
}
```

This affects `MemoryStore` methods that generate embeddings (`add`, `search`, `update`) — they also require `&mut self`.

## Input Constraints

### `MAX_INPUT_LENGTH: 100_000`

- **Where defined**: `src/memory/store.rs`
- **Purpose**: Database and embedding pipeline protection
- **Behavior**: Rejected if exceeded, returns `Error::InputTooLong`
- **Validation**: Checked before tokenization

### Tokenization Truncation

- **Max tokens**: 512 (tokenizer configuration)
- **Behavior**: Silently truncates text exceeding 512 tokens
- **Implementation**: `tokenizer.with_truncation(...)` in `EmbeddingEngine::new()`

**Note**: Input can be up to 100,000 characters, but only first 512 tokens are embedded.

### Empty Input Handling

- **Empty strings**: Return zero vector `[0.0; 384]`
- **Whitespace-only**: Tokenized to empty sequence, returns zero vector
- **Validation**: Trimmed empty input rejected at higher level (`Error::EmptyInput`)

## Output Specifications

### `EMBEDDING_DIMS: 384`

- **Where defined**: `src/embedding/engine.rs`
- **Value**: 384 f32 values (model-specific)
- **Used by**:
  - Database BLOB size validation
  - Cosine similarity computation
  - Search result processing

### BLOB Storage Format

- **Format**: Little-endian f32 bytes
- **Size**: Exactly 1,536 bytes (384 × 4 bytes/f32)
- **Storage**: SQLite BLOB column `memories.embedding`

### Cosine Similarity Computed at Query Time

Embeddings are stored as raw bytes; cosine similarity is computed in Rust during search:

```rust
// Pseudocode (see src/sqlite/embedding.rs for implementation)
fn cosine_similarity(query_embedding: &[f32], stored_bytes: &[u8]) -> f64 {
    let stored_embedding = decode_bytes_to_f32(stored_bytes);  // Little-endian
    // Compute dot product / (norm_query * norm_stored)
}
```

**Why**: SQLite extensions add complexity; Rust is fast enough.

## Changing Embedding Models

### Incompatibility Warning

Changing the embedding model breaks existing databases:

1. **New model dimensions** may differ from 384
2. **Different semantic space**: Old embeddings no longer comparable to new
3. **Migration required**: Regenerate all embeddings

### Migration Process

Re-embed an existing database with the currently configured model:

```bash
vipune reindex --force --all-projects
```

(`--all-projects` is required with `--force`: `handle_reindex` refuses a forced
re-embed that would cover only some of the projects in a shared store.) Library
consumers can call
`vipune::migrate_model(db_path, &config)` in-process; both run the same
crash-safe pre-flight → marker-first → re-embed lifecycle documented in the
README (Switching models on an existing database).

A plain `vipune reindex` re-embeds nothing here: the stored vectors still look
real to `EmbeddingClass::Real`, which checks the norm, not the values. Use
`--force` when a runtime or model change moved the vectors.

### Testing Model Changes

When testing a new model:

1. Use temporary database: `--db-path /tmp/test.db`
2. Run integration tests: `cargo test --test lib_integration`
3. Verify embedding dimensions match `EMBEDDING_DIMS` constant
4. Compare semantic search quality with old model

## Performance Characteristics

### Embedding Generation Speed

- **Single text**: ~10-20ms on modern CPU
- **Batch processing**: Not supported (synchronous only)
- **Cold start**: +30-60s for first model download

### Memory Usage

- **Model loading**: ~100MB RAM (model weights)
- **Per embedding**: 1,536 bytes in database + 1,536 bytes during computation
- **Peak**: ~500MB during embedding generation

### Cosine Similarity Computation

- **Per memory comparison**: ~1 microsecond
- **Search overhead**: Negligible compared to embedding generation

## Implementation Files

- **`src/embedding/engine.rs`**: ONNX environment + session, tokenizer, `requires_token_type_ids`
- **`src/embedding/inference.rs`**: model run, mean pooling, `l2_normalize()`
- **`src/sqlite/embedding.rs`**: BLOB I/O, cosine similarity
- **`src/memory/crud.rs`**: Embedding generation in add/update/search
- **`src/memory/store.rs`**: `MAX_INPUT_LENGTH` constant

## Debugging Embedding Issues

### Model Download Fails

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/
vipune add "Test text"
```

### Wrong Embedding Dimensions

Check `EMBEDDING_DIMS` constant:

```bash
# In src/embedding/engine.rs
pub const EMBEDDING_DIMS: usize = 384;  // Must match model output
```

### Embedding Not L2-Normalized

Check `l2_normalize()` in `src/embedding/inference.rs`:

```rust
fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm = norm.max(1e-9);
    vec.iter().map(|&x| x / norm).collect()
}
```

Test: `cargo test --package vipune --lib embedding::tests::test_l2_normalize`

---

For user-facing search guidance, see the [Search Guide](search.md).

For architecture overview, see [Architecture](architecture.md).