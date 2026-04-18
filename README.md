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

### Uninstall

```bash
rm ~/.cargo/bin/vipune   # or: sudo rm /usr/local/bin/vipune
```

Or via cargo: `cargo uninstall vipune`

Optionally, clear all data:

```bash
rm -rf ~/.vipune ~/.config/vipune
```

## Air-gapped / Offline Usage

vipune downloads the embedding model from HuggingFace Hub on first run. The model is pinned to a specific SHA to ensure reproducibility:

- **Model ID**: `BAAI/bge-small-en-v1.5`
- **Pinned revision SHA**: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`

For air-gapped environments, pre-fetch the model before going offline:

```bash
# Install huggingface-cli first if you don't have it
pip install huggingface_hub

# Download the pinned model to the cache directory
huggingface-cli download BAAI/bge-small-en-v1.5 \
  --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a \
  --cache-dir ~/.cache/huggingface/hub
```

The model will be cached in the HF Hub cache layout at `~/.cache/huggingface/hub/` and vipune will use it without network access. You can verify the cache before going offline:

```bash
ls ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/
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
| `vipune add <text>` | Store a memory |
| `vipune search <query>` | Find memories by meaning |
| `vipune get <id>` | Retrieve a memory by ID |
| `vipune list` | List all memories |
| `vipune delete <id>` | Delete a memory |
| `vipune update <id> <text>` | Update a memory's content |
| `vipune version` | Show version |

[Complete CLI reference](docs/cli-reference.md) • [Quickstart guide](docs/quickstart.md) • [Search guide](docs/search.md) • [Architecture](docs/architecture.md)

## Library Usage

vipune can also be used as a Rust crate for programmatic integration:

```toml
# Cargo.toml
[dependencies]
vipune = "0.2"
```

```rust
use vipune::{Config, MemoryStore, detect_project};

// Initialize memory store
let config = Config::default();
let mut store = MemoryStore::new(
    config.database_path.as_path(),
    &config.embedding_model,
    config.clone()
).expect("Failed to initialize store");

// Add a memory
let project_id = "my-project";
let memory_id = store.add(&project_id, "Alice works at Microsoft", None)
    .expect("Failed to add memory");

// Search memories
let results = store.search(&project_id, "where does alice work", 10, 0.0)
    .expect("Failed to search");

for memory in results {
    println!("{:.2}: {}", memory.similarity.unwrap_or(0.0), memory.content);
}
```

**See the crate documentation at [docs.rs](https://docs.rs/vipune) for complete API reference.**

## Configuration

vipune works with zero configuration. All paths use the user's home directory:

**Default paths:**
- Database: `~/.vipune/memories.db`
- Model cache: `~/.vipune/models/`
- Config file: `~/.config/vipune/config.toml`

**Environment variables (override defaults):**
- `VIPUNE_DATABASE_PATH` - SQLite database location
- `VIPUNE_EMBEDDING_MODEL` - HuggingFace model ID (default: `BAAI/bge-small-en-v1.5`)
- `VIPUNE_MODEL_CACHE` - Model download cache directory
- `VIPUNE_PROJECT` - Project identifier (overrides auto-detection)
- `VIPUNE_SIMILARITY_THRESHOLD` - Conflict detection threshold, 0.0-1.0 (default: `0.85`)
- `VIPUNE_RECENCY_WEIGHT` - Recency bias in search results, 0.0-1.0 (default: `0.3`)

**Config file (`~/.config/vipune/config.toml`):**
```toml
database_path = "~/.vipune/memories.db"
embedding_model = "BAAI/bge-small-en-v1.5"
model_cache = "~/.vipune/models"
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
- **search_memories**: Find memories by meaning
- **list_memories**: List recent memories

## Agent Integration

vipune works with any agent that can run shell commands — no plugins, adapters, or API keys required. Configure your agent with a few lines of instructions, grant shell command permissions, and the agent can use `vipune search` and `vipune add` to maintain persistent memory across tasks.

**[→ See Agent Integration Guide](docs/agent-integration.md)** for per-tool setup instructions (Claude Code, Cursor, Windsurf, Cline, Roo Code, GitHub Copilot, Goose, Aider, OpenCode, Zed, and more).

**See also:** [Search Guide](docs/search.md) for agent-friendly query patterns.

**Exit codes for agent workflows:**
- `0` - Success
- `1` - Error (missing file, invalid input, etc.)
- `2` - Conflicts detected (similar memories found)

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

## License

Apache-2.0 © [Janni Turunen](https://github.com/randomm/vipune)

## Links

- [GitHub](https://github.com/randomm/vipune)
- [Issues](https://github.com/randomm/vipune/issues)
- [CLI Reference](docs/cli-reference.md)
- [Quickstart](docs/quickstart.md)

