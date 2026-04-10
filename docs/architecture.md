# vipune Architecture

## Overview

vipune is a single Rust binary CLI tool for semantic memory storage and search. It was designed for simplicity and predictability:

- **Library core is sync-only**: No async runtime in `src/lib.rs` (no tokio). All operations block until complete. This eliminates complexity and runtime overhead.
- **MCP server uses tokio**: The MCP server module uses an async runtime (tokio) when enabled (default feature).
- **No daemon**: CLI tool only — runs, executes, and exits. No long-lived server process.
- **No network at runtime**: All dependencies are bundled. HuggingFace Hub model downloads happen once and are cached locally.
- **SQLite for persistence**: Data stored in `~/.vipune/memories.db` using rusqlite (bundled, no external SQLite installation required).
- **ONNX for embeddings**: bge-small-en-v1.5 model (384 dimensions) for semantic search, with local inference via ONNX Runtime.

## Module Map

| Module | Purpose |
|--------|---------|
| `src/main.rs` | CLI entry point, argument parsing with clap, command dispatch (add, search, get, list, delete, update, version) |
| `src/commands.rs` | Command dispatch logic, CLI argument handling |
| `src/memory/` | Memory store CRUD operations and search (replaces old `src/memory.rs`) |
| `src/memory/mod.rs` | Module re-exports |
| `src/memory/store.rs` | Core MemoryStore struct, `&mut self` embedding requirements |
| `src/memory/crud.rs` | Add, get, list, update, delete operations with conflict detection |
| `src/memory/search.rs` | Search orchestration, hybrid search coordination |
| `src/memory/tests.rs` | Memory module unit tests |
| `src/sqlite/` | SQLite persistence layer (replaces old `src/sqlite.rs`) |
| `src/sqlite/mod.rs` | Module re-exports |
| `src/sqlite/embedding.rs` | BLOB I/O, cosine similarity computation |
| `src/sqlite/fts.rs` | FTS5 full-text search, BM25, triggers, auto-migration |
| `src/sqlite/search.rs` | Hybrid search coordination, semantic + BM25 |
| `src/embedding.rs` | ONNX model loading and text-to-vector conversion using bge-small-en-v1.5 and HuggingFace tokenizer |
| `src/project.rs` | Project auto-detection from git remote, environment variable, or working directory |
| `src/config/` | Configuration loading from TOML files, environment variables, and validation |
| `src/errors.rs` | Unified error types wrapping rusqlite, ONNX, tokenizer, and HuggingFace Hub errors |
| `src/memory_types.rs` | Shared type definitions (AddResult, ConflictMemory) |
| `src/output.rs` | JSON response formatting for CLI output |
| `src/temporal.rs` | Recency decay scoring with exponential/linear decay functions for search result weighting |
| `src/rrf.rs` | Reciprocal Rank Fusion (RRF) algorithm for merging semantic and BM25 search rankings |

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

**See also**: [Embedding Pipeline Details](embedding-pipeline.md) for contributor-level documentation on ONNX integration, model versioning, and BLOB format.

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

**See also**: [Search Guide](search.md) for user-facing guidance on when to use each search mode and how recency weighting works.

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

**Synchronous core; MCP server adds tokio when enabled (default feature)**: No async/await, no `.await` operators in library code (`src/lib.rs`). All I/O is blocking, matching the simplicity requirement for a CLI tool. The MCP module is the exception — it uses tokio for async MCP protocol handling.

**Lib and bin targets**: `src/lib.rs` (public API exports) and `src/main.rs` (CLI entry point). Single release artifact enables both library use and CLI tool.

**No daemon**: Tool exits after operation. State lives only in SQLite; no in-memory caches survive between invocations.

**File size limits**: Source files capped at 500 lines (exceptions justified). Keeps modules focused, testable, and maintainable.

**Zero technical debt**: No TODO/FIXME/HACK comments in src/. Incomplete work tracked in GitHub issues, not left in code.

## See Also

- **[Embedding Pipeline](embedding-pipeline.md)** — Contributor-level documentation on ONNX integration, model versioning, and BLOB format
- **[Search Guide](search.md)** — User-facing guidance on semantic vs hybrid search, recency weighting, and query strategies
- **[Testing Guide](testing.md)** — How to write tests, test utilities, and coverage expectations

