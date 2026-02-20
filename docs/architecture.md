# vipune Architecture

## Overview

vipune is a single Rust binary CLI tool for semantic memory storage and search. It was designed for simplicity and predictability:

- **Fully synchronous**: No async runtime (no tokio), no event loops. All operations block until complete. This eliminates complexity and runtime overhead.
- **No daemon**: CLI tool only — runs, executes, and exits. No long-lived server process.
- **No network at runtime**: All dependencies are bundled. HuggingFace Hub model downloads happen once and are cached locally.
- **SQLite for persistence**: Data stored in `~/.vipune/memories.db` using rusqlite (bundled, no external SQLite installation required).
- **ONNX for embeddings**: bge-small-en-v1.5 model (384 dimensions) for semantic search, with local inference via ONNX Runtime.

## Module Map

| Module | Purpose |
|--------|---------|
| `src/main.rs` | CLI entry point, argument parsing with clap, command dispatch (add, search, get, list, delete, update, version) |
| `src/memory.rs` | High-level orchestration of embedding generation and persistence; conflict detection for similar memories |
| `src/sqlite.rs` | SQLite persistence layer with schema, insert/search/update/delete operations, FTS5 hybrid search support |
| `src/embedding.rs` | ONNX model loading and text-to-vector conversion using bge-small-en-v1.5 and HuggingFace tokenizer |
| `src/project.rs` | Project auto-detection from git remote, environment variable, or working directory |
| `src/config/` | Configuration loading from TOML files, environment variables, and validation |
| `src/errors.rs` | Unified error types wrapping rusqlite, ONNX, tokenizer, and HuggingFace Hub errors |
| `src/output.rs` | JSON response types for CLI output (add, search, get, list responses) |
| `src/temporal.rs` | Recency decay scoring with exponential/linear decay functions for search result weighting |
| `src/rrf.rs` | Reciprocal Rank Fusion (RRF) algorithm for merging semantic and BM25 search rankings |
| `src/memory_types.rs` | Shared type definitions (AddResult, ConflictMemory) |

## Embedding Pipeline

**Model**: bge-small-en-v1.5 from HuggingFace (fine-tuned BERT for semantic embeddings)

**Dimensions**: 384 × f32 values per embedding

**Storage**: Little-endian binary BLOB, exactly 1,536 bytes per embedding (384 × 4 bytes)

**Processing**:
1. Text is tokenized using HuggingFace tokenizers with max_length=512 and truncation
2. Tokens are fed to ONNX model for inference
3. Output embeddings are mean-pooled and L2-normalized
4. Raw f32 array is converted to little-endian bytes for storage
5. Cosine similarity computed in Rust during search (not via SQL extension)

**Caching**: Model files downloaded on first use via `hf_hub`, cached in `~/.vipune/models/`, reused for all subsequent operations.

## Hybrid Search

vipune supports two search modes:

**Semantic search** (default): Cosine similarity between query embedding and stored embeddings
- Fast exact-match similarity
- Works well for paraphrases and conceptual similarity

**Hybrid search**: Combines semantic (embedding cosine) and lexical (BM25) rankings using Reciprocal Rank Fusion (RRF)
- BM25 implemented via SQLite FTS5 full-text search
- RRF merges both rankings without score normalization
- Formula: fused_score = Σ (1 / (k + rank)) per result across both rankings
- Documents appearing in both lists get boosted scores

**Recency weighting**: Optional exponential or linear decay applied to scores based on creation timestamp, with configurable grace period.

## Database Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,      -- 1536 bytes (384 × f32 little-endian)
    metadata TEXT,                -- JSON metadata (optional)
    created_at TEXT NOT NULL,     -- ISO 8601 timestamp
    updated_at TEXT NOT NULL      -- ISO 8601 timestamp
);

CREATE INDEX idx_memories_project ON memories(project_id);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    project_id UNINDEXED,
    tokenize='porter unicode61',
    content_rowid='rowid',
    content='memories'
);

-- Triggers maintain FTS5 index in sync with memories table
CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, project_id)
    VALUES (new.rowid, new.content, new.project_id);
END;

CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, project_id)
    VALUES('delete', old.rowid, old.content, old.project_id);
END;

CREATE TRIGGER memories_fts_update AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, project_id)
    VALUES('delete', old.rowid, old.content, old.project_id);
    INSERT INTO memories_fts(rowid, content, project_id)
    VALUES (new.rowid, new.content, new.project_id);
END;
```

**Project scoping**: Each memory is tagged with a project_id to isolate memories by project. Project auto-detection uses git remote URL, environment variable, or working directory name.

## Dependency Rationale

| Dependency | Reason |
|------------|--------|
| `rusqlite` (bundled) | Embedded SQLite without external system dependencies. Single C library included in binary. |
| `ort` v2 | ONNX Runtime for cross-platform local model inference (CPU-only, no GPU). Auto-downloads platform-specific shared libraries. |
| `tokenizers` | HuggingFace tokenizer library for efficient BPE tokenization matching bge-small-en-v1.5 preprocessing. |
| `hf-hub` (with ureq feature) | Download models and tokenizers from HuggingFace Hub with local caching. Sync API (ureq) matches synchronous design. |
| `clap` | Robust CLI argument parsing with subcommands, defaults, help, and structured error messages. |
| `thiserror` | Ergonomic error type derivation via `#[derive(Error)]` with automatic Display and Error trait impl. |
| `serde`/`serde_json` | Serialization for JSON output mode and metadata storage. Standard Rust serialization. |
| `uuid` v4 | Generate unique memory IDs, replacing incremental counters for distributed safety. |
| `chrono` | ISO 8601 timestamps for created_at/updated_at with parsing and formatting. |
| `toml` | Configuration file parsing (TOML format) for user settings. |
| `dirs` | XDG-compliant home directory paths for `~/.vipune/` cache and database locations. |

**Intentionally excluded**:
- ❌ `tokio`: Async runtime unnecessary for synchronous CLI operation
- ❌ `reqwest`: HTTP client not needed (model downloads via hf-hub with ureq blocking I/O)
- ❌ `pyo3`: Python bindings not required (Rust-only tool)
- ❌ `sqlx`: Async database toolkit incompatible with synchronous design

## Configuration

Configuration can be provided via:
1. TOML file at `~/.config/vipune/config.toml` (XDG base directory)
2. Environment variables: `VIPUNE_*` (e.g., `VIPUNE_SIMILARITY_THRESHOLD`)
3. CLI flags: `--project`, `--db-path`, `--recency`, `--hybrid`, etc.

Priority: CLI flags > environment variables > TOML file > defaults

Configurable parameters include:
- `similarity_threshold`: Minimum score for conflict detection (default: 0.85)
- `recency_weight`: Mix semantic and temporal signals (0.0-1.0)
- `decay_function`: Exponential or linear recency decay
- `decay_lambda`: Decay rate parameter
- `cache_dir`: Override model cache location
- `db_path`: Override database location

## Design Constraints

**Synchronous only**: No async/await, no tokio, no `.await` operators. All I/O is blocking, matching the simplicity requirement for a CLI tool.

**Single crate**: No workspaces, no lib.rs/main.rs split. All code in one binary simplifies distribution (single release artifact).

**No daemon**: Tool exits after operation. State lives only in SQLite; no in-memory caches survive between invocations.

**File size limits**: Source files capped at 500 lines (exceptions justified). Keeps modules focused, testable, and maintainable.

**Zero technical debt**: No TODO/FIXME/HACK comments in src/. Incomplete work tracked in GitHub issues, not left in code.

