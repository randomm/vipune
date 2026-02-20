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

### Prerequisites

For source installation:
- Rust 1.70+ (install via https://rustup.rs)
- System dependencies for ONNX Runtime:
  - Linux: libgomp1, libc6
  - macOS: None required

### Pre-built binary

**macOS Apple Silicon (arm64)**
```bash
# Download
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-aarch64-apple-darwin.tar.gz

# Extract
tar xzf vipune-aarch64-apple-darwin.tar.gz

# Install to system directory (requires sudo)
sudo mv vipune /usr/local/bin/

# Or install to user directory (no sudo)
mkdir -p ~/.local/bin
mv vipune ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
```

**Linux x86_64**
```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-x86_64-unknown-linux-gnu.tar.gz
tar xzf vipune-x86_64-unknown-linux-gnu.tar.gz
sudo mv vipune /usr/local/bin/
```

**Linux ARM64**
```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-aarch64-unknown-linux-gnu.tar.gz
tar xzf vipune-aarch64-unknown-linux-gnu.tar.gz
sudo mv vipune /usr/local/bin/
```

### Build from source

**Latest release (recommended)**
```bash
# Check https://github.com/randomm/vipune/releases for the latest version
cargo install --git https://github.com/randomm/vipune --tag v0.1.1 vipune
```

**Or clone and build manually**
```bash
git clone https://github.com/randomm/vipune.git
cd vipune && cargo build --release

# Binary at ./target/release/vipune

# Add to PATH temporarily
export PATH="$(pwd)/target/release:$PATH"

# Or install permanently
sudo cp target/release/vipune /usr/local/bin/
```

### Uninstall

```bash
# Remove pre-built binary
sudo rm /usr/local/bin/vipune

# Remove from user directory
rm ~/.local/bin/vipune

# Remove via cargo
cargo uninstall vipune

# Clear data (optional)
rm -rf ~/.vipune ~/.config/vipune
```

## Quick Start

```bash
# Add a memory
vipune add "Alice works at Microsoft"

# Search by semantic meaning
vipune search "where does alice work"

# Add with metadata (optional)
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

[Complete CLI reference](docs/cli-reference.md) • [Quickstart guide](docs/quickstart.md)

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

## Agent Integration

vipune works with any agent that can run shell commands — no plugins, adapters, or API keys required. Configure your agent with a few lines of instructions, grant shell command permissions, and the agent can use `vipune search` and `vipune add` to maintain persistent memory across tasks.

**[→ See Agent Integration Guide](docs/agent-integration.md)** for per-tool setup instructions (Claude Code, Cursor, Windsurf, Cline, Roo Code, GitHub Copilot, Goose, Aider, OpenCode, Zed, and more).

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

