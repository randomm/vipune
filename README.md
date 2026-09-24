# vipune `/ˈʋi.pu.ne/`

```

              ███                                          
             ░░░                                           
 █████ █████ ████  ████████  █████ ████ ████████    ██████ 
░░███ ░░███ ░░███ ░░███░░███░░███ ░███ ░░███░░███  ███░░███
 ░███  ░███  ░███  ░███ ░███ ░███ ░███  ░███ ░███ ░███████ 
 ░░███ ███   ░███  ░███ ░███ ░███ ░███  ░███ ░███ ░███░░░  
  ░░█████    █████ ░███████  ░░████████ ████ █████░░██████ 
   ░░░░░    ░░░░░  ░███░░░    ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░  
                   ░███                                    
                   █████                                   
                  ░░░░░                                    

```

A minimal memory layer for AI agents.

In Finnish mythology, Antero Vipunen is a giant who sleeps underground, holding all the world's knowledge and ancient songs. vipune is your agent's sleeping giant — a local knowledge store that remembers everything.

Store semantic memories, search by meaning, and detect conflicts. Single binary CLI. No API keys required.

## Features

- **Semantic search** - Find memories by meaning, not keywords (ONNX embeddings, bge-small-en-v1.5)
- **Conflict detection** - Automatically warns when adding duplicate or similar memories
- **Zero configuration** - Works out of the box (auto-detected git projects, sensible defaults)
- **Single binary** - Just one CLI tool, no daemon, no database server
- **No API keys** - Everything runs locally, no network dependencies
- **Project scoped** - Memories isolated by git repository

## Installation

### Platform Support

**Supported:** macOS ARM64, Linux x86_64, Linux ARM64  
**Not supported:** Windows (due to ONNX Runtime compilation complexity)

### Quick install (macOS and Linux)

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/randomm/vipune/releases/latest/download/vipune-installer.sh | sh
```

> The installer detects your platform, downloads the correct binary, and adds it to `~/.cargo/bin/` by default. If you don't use Rust, restart your shell or run `export PATH="$HOME/.cargo/bin:$PATH"` to make vipune available. Use `--prefix=/usr/local` for system-wide installs.

To verify the download before running, download the checksum file and check:
```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-installer.sh.sha256
sha256sum -c vipune-installer.sh.sha256
sh vipune-installer.sh
```

### Pre-built binary

**macOS Apple Silicon (arm64)**

Download and extract:

```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-aarch64-apple-darwin.tar.xz
tar xf vipune-aarch64-apple-darwin.tar.xz --strip-components=1
sudo mkdir -p /usr/local/bin && sudo mv vipune /usr/local/bin/
```

**Linux x86_64**

Download and extract:

```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-x86_64-unknown-linux-gnu.tar.xz
tar xf vipune-x86_64-unknown-linux-gnu.tar.xz --strip-components=1
sudo mkdir -p /usr/local/bin && sudo mv vipune /usr/local/bin/
```

**Linux ARM64**

Download and extract:

```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-aarch64-unknown-linux-gnu.tar.xz
tar xf vipune-aarch64-unknown-linux-gnu.tar.xz --strip-components=1
sudo mkdir -p /usr/local/bin && sudo mv vipune /usr/local/bin/
```

### Build from source

Requires the [Rust toolchain](https://rustup.rs) (1.85+). On Linux, you may also need `libgomp1` and `libc6`.

**Latest release (recommended)**
```bash
cargo install vipune
```

**Or clone and build manually**

```bash
git clone https://github.com/randomm/vipune.git
cd vipune && cargo build --release
```

The binary will be at `./target/release/vipune`. Install it:

```bash
sudo mkdir -p /usr/local/bin && sudo cp target/release/vipune /usr/local/bin/
```

Or add to PATH temporarily:

```bash
export PATH="$(pwd)/target/release:$PATH"
```

### Install Claude/Pi skill (optional)

For agents with skill auto-discovery (Claude Code, Pi), install the vipune skill artifact to enable software-development-tuned memory patterns:

```bash
mkdir -p ~/.claude/skills/vipune && \
  curl -fsSL --connect-timeout 10 --max-time 30 https://raw.githubusercontent.com/randomm/vipune/main/skills/vipune/SKILL.md \
  -o ~/.claude/skills/vipune/SKILL.md
```

_The skill becomes available at this URL once this change is merged to `main`._

This adds domain-specific patterns: issue/PR linkage, failed-approach tracking, pre-flight gotcha checks, dev-loop mapping, and ADR-lite decision capture. Use `<project>/.claude/skills/` for project-scoped installation. See [Agent Integration](docs/agent-integration.md#using-skillmd) for details.

**Note**: Tier-2 (automated install via `vipune skill install`) and Tier-3 (cross-agent AGENTS.md snippets) are future follow-ups.

### Uninstall

```bash
rm ~/.cargo/bin/vipune   # or: sudo rm /usr/local/bin/vipune
```

Or via cargo: `cargo uninstall vipune`

Optionally, clear all data:

```bash
rm -rf ~/.vipune ~/.config/vipune
```

## Embedding Models

vipune embeds memories with a local ONNX model downloaded from HuggingFace Hub on first use, pinned to an exact revision for reproducibility. The model is chosen **per database** via the `embedding_model` setting (see [Configuration](#configuration)) — set it before your first `vipune add`, because a database records which model it was created with.

### Built-in profiles

| Model ID | Revision | Languages | Download size | Prefixes |
|----------|----------|-----------|---------------|----------|
| `BAAI/bge-small-en-v1.5` (default) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | English | ~66 MB | none |
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | ~100 languages (xlm-roberta) | ~470 MB | automatic `query: ` / `passage: ` |

**Choosing a model:** the default `bge-small-en-v1.5` is fast and small but English-focused. If your memories are in other languages (e.g. Finnish, Swedish, German), select the multilingual profile instead — it embeds ~100 languages, at the cost of a much larger one-time download:

```bash
# Either set the environment variable
export VIPUNE_EMBEDDING_MODEL="intfloat/multilingual-e5-small"

# ...or in ~/.config/vipune/config.toml
embedding_model = "intfloat/multilingual-e5-small"
```

### Automatic prefixes (e5 only)

The e5 model requires task prefixes to work well. vipune applies them **automatically**: `query: ` is prepended to search queries and `passage: ` to stored text at embedding time. **Do not add these prefixes yourself** — they are never written into stored content, search output, or the FTS index; they exist only at embed time.

The `bge-small-en-v1.5` profile uses no prefixes.

Note: the prefix consumes tokens of the 512-token limit, so slightly longer texts that embed fine under bge may be rejected (exit code 3) under e5 — over-length input is always rejected, never truncated.

### Switching models on an existing database

A database records the model id + revision it was created with. If the configured model no longer matches the recorded identity, `add`, `update`, and `search` refuse with an error naming both identities — the stored embeddings are in a different vector space and results would be meaningless.

To migrate an existing database to a different model:

```bash
vipune reindex --force
```

This re-embeds **every** stored memory with the newly configured model (a plain `vipune reindex` would re-embed nothing, since existing vectors already look real). The migration is crash-safe and pre-checked:

1. **Pre-flight** — `reindex --force` first token-counts every stored row with the target model's passage prefix. If any row would exceed the 512-token limit once prefixed, it refuses to start, writes nothing, and lists the offending memory ids. The migration marker is only ever written in a state the re-embed pass can complete.
2. **Marker-first** — it records a `migrating to <id>@<revision>` marker before re-embedding.
3. **Re-embed + record** — it re-embeds every row, then in one transaction records the new identity and clears the marker.

If the run is interrupted, the marker stays and operations refuse until you re-run `vipune reindex --force`, which performs a full idempotent pass from the beginning.

## Air-gapped / Offline Usage

vipune downloads the embedding model from HuggingFace Hub on first run. Each model is pinned to a specific revision to ensure reproducibility:

- `BAAI/bge-small-en-v1.5` — revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- `intfloat/multilingual-e5-small` — revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`

For air-gapped environments, pre-fetch the default model before going offline:

```bash
# Install huggingface-cli first if you don't have it
pip install huggingface_hub

# Download the model revision to the cache directory
huggingface-cli download BAAI/bge-small-en-v1.5 \
  --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a \
  --cache-dir ~/.cache/huggingface/hub
```

If you selected the multilingual profile instead, pre-fetch that one as well:

```bash
huggingface-cli download intfloat/multilingual-e5-small \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --cache-dir ~/.cache/huggingface/hub
```

The model will be cached in the HF Hub cache layout at `~/.cache/huggingface/hub/` and vipune will use it without network access. You can verify the cache before going offline:

```bash
ls ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/
ls ~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/
```

**Note**: When upgrading vipune, the pinned revision may change. Check the release notes and re-download the new revision if the SHA has changed.

**Library consumers**: The constants `vipune::EMBED_MODEL_ID` and `vipune::EMBED_MODEL_REVISION` are exported for tracking the pinned model version programmatically.

## Quick Start

Add a memory:

```bash
vipune add "Alice works at Microsoft"
```

Search by semantic meaning:

```bash
vipune search "where does alice work"
```

Add with metadata (optional):

```bash
vipune add "Auth uses JWT tokens" --metadata '{"topic": "authentication"}'
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `vipune add <text>` | Store a memory (with type, status, conflict detection) |
| `vipune search <query>` | Find memories by meaning (filters by type/status) |
| `vipune get <id>` | Retrieve a memory by ID |
| `vipune list` | List memories (default: active only) |
| `vipune delete <id>` | Delete a memory |
| `vipune update <id> [text]` | Update content, metadata, type, or status |
| `vipune validate <text>` | Check if text is within embedding token limits |
| `vipune version` | Show version |

[Complete CLI reference](docs/cli-reference.md) • [Quickstart guide](docs/quickstart.md) • [Search guide](docs/search.md) • [Architecture](docs/architecture.md)

## Library Usage

vipune can also be used as a Rust crate for programmatic integration:

```toml
# Cargo.toml
[dependencies]
vipune = "0.5.0"
```

```rust
use vipune::{Config, MemoryStore, MemoryType, MemoryStatus, detect_project};

// Initialize memory store
let config = Config::default();
let mut store = MemoryStore::new(
    config.database_path.as_path(),
    &config.embedding_model,
    config.clone()
).expect("Failed to initialize store");

// Add a memory
let project_id = "my-project";
let result = store.add_with_conflict(&project_id, "Alice works at Microsoft", None, false, vipune::MemoryType::Fact, vipune::MemoryStatus::Active)
    .expect("Failed to add memory");

// Search memories
let results = store.search(&project_id, "where does alice work", 10, 0.0, vipune::memory::SearchOptions::default())
    .expect("Failed to search");

for memory in results {
    println!("{:.2}: {}", memory.similarity.unwrap_or(0.0), memory.content);
}
```

**v0.4+ features**: `MemoryType`, `MemoryStatus`, `supersedes` flag, and telemetry (retrieval_count) are available for type-aware memory management. See the [CLI reference](docs/cli-reference.md) for details.

**See the crate documentation at [docs.rs](https://docs.rs/vipune) for complete API reference.**

## Configuration

vipune works with zero configuration. All paths use the user's home directory:

**Default paths:**
- Database: `~/.vipune/memories.db`
- Model cache: `~/.cache/huggingface/hub/` (standard HuggingFace Hub layout, created by `hf-hub` on first download)
- Config file: `~/.config/vipune/config.toml`

**Environment variables (override defaults):**
- `VIPUNE_DATABASE_PATH` - SQLite database location
- `VIPUNE_EMBEDDING_MODEL` - embedding model profile: `BAAI/bge-small-en-v1.5` (default) or `intfloat/multilingual-e5-small` — see [Embedding Models](#embedding-models)
- `HF_HOME` - HuggingFace cache home directory (changes the model cache location to `$HF_HOME/hub`)
- `VIPUNE_PROJECT` - Project identifier (overrides auto-detection)
- `VIPUNE_SIMILARITY_THRESHOLD` - Conflict detection threshold, 0.0-1.0 (default: `0.85`)
- `VIPUNE_RECENCY_WEIGHT` - Recency bias in search results, 0.0-1.0 (default: `0.3`)
- `VIPUNE_HYBRID` - Enable hybrid search (semantic + BM25), true/false or 1/0

**Config file (`~/.config/vipune/config.toml`):**
```toml
database_path = "~/.vipune/memories.db"
embedding_model = "BAAI/bge-small-en-v1.5"
similarity_threshold = 0.85
recency_weight = 0.3
```

## MCP Server

vipune can act as an MCP (Model Context Protocol) server, enabling AI agents like Claude Code and Cursor to use it as a native memory provider.

> **Note:** MCP is enabled by default. Library users can use `default-features = false` for sync-only builds.

### Setup (Claude Code)

Add to your Claude Code configuration (`~/.claude/settings.json` or project `.claude.json`):

```json
{
  "mcpServers": {
    "vipune": {
      "command": "vipune",
      "args": ["mcp"]
    }
  }
}
```

### Setup (Cursor)

Add to your Cursor MCP configuration with the same JSON structure.

### Available Tools

- **store_memory**: Store information for later recall
- **search_memories**: Find memories by meaning (supports `hybrid` param for semantic + BM25)
- **list_memories**: List recent memories
- **supersede_memory**: Replace an existing memory with new content
- **get_memory**: Retrieve a memory by ID
- **delete_memory**: Permanently delete a memory
- **update_memory**: Update a memory's content, metadata, type, or status

**See the [Claude Desktop on macOS guide](docs/agent-integration.md#claude-desktop-on-macos-mcp) for a complete Desktop setup with configuration examples and macOS-specific gotchas.**

## Agent Integration

vipune works with any agent that can run shell commands — no plugins, adapters, or API keys required. Configure your agent with a few lines of instructions, grant shell command permissions, and the agent can use `vipune search` and `vipune add` to maintain persistent memory across tasks.

**[→ See Agent Integration Guide](docs/agent-integration.md)** for per-tool setup instructions (Claude Code, Cursor, Windsurf, Cline, Roo Code, GitHub Copilot, Goose, Aider, OpenCode, Zed, and more).

**See also:** [Search Guide](docs/search.md) for agent-friendly query patterns.

**Exit codes for agent workflows:**
- `0` - Success
- `1` - Error (missing file, invalid input, etc.)
- `2` - Conflicts detected (similar memories found)
- `3` - Content too long (exceeds embedding token limit)

## Recency Scoring

Search results can be weighted by recency using the `--recency` flag or `VIPUNE_RECENCY_WEIGHT` config:

```bash
# Increase recency bias (recent memories rank higher)
vipune search "authentication" --recency 0.7

# Pure semantic similarity (no recency bias)
vipune search "authentication" --recency 0.0
```

The final score combines semantic similarity and recency time decay:
- `score = (1 - recency_weight) * similarity + recency_weight * time_score`
- Default balance: 70% semantic, 30% recency

## Benchmarks

The repo ships a [Criterion](https://github.com/bheisler/criterion.rs) benchmark for the `Database::search` path. It benchmarks search over a synthetic corpus of 384-dim vectors at 1k and 10k rows, seeded only through the public API — no ONNX model and no network access required.

Run the benchmark locally:

```bash
cargo bench
```

A plain `cargo bench` compares the run against a stored `main` baseline when one exists (lenient mode: reports differences, never fails on them) and **never overwrites the baseline**.

Record or refresh the `main` baseline deliberately — this is the only run that writes to the baseline store:

```bash
cargo bench -- --save-baseline main
```

After a performance-shaped change, force a strict comparison against it (fails only if the baseline is missing; a regression is reported in red text but does not fail the run):

```bash
cargo bench -- --baseline main
```

Or a lenient comparison (reports the difference, no failure if the baseline is missing):

```bash
cargo bench -- --baseline-lenient main
```

Criterion stores results and baselines under `target/criterion/`. No committed baseline is bundled: baselines are machine-specific, so each machine records its own `main` baseline and later runs diff against that stable reference numerically.

The bench reports timings only — it asserts no timing thresholds, so a slower result is a signal to look at, never a build failure.

## License

Apache-2.0 © [Janni Turunen](https://github.com/randomm/vipune)

## Links

- [GitHub](https://github.com/randomm/vipune)
- [Issues](https://github.com/randomm/vipune/issues)
- [CLI Reference](docs/cli-reference.md)
- [Quickstart](docs/quickstart.md)

