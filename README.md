# vipune `/ˈʋi.pu.ne/`

```
██╗   ██╗██╗██████╗ ██╗   ██╗███╗   ██╗███████╗
██║   ██║██║██╔══██╗██║   ██║████╗  ██║██╔════╝
██║   ██║██║██████╔╝██║   ██║██╔██╗ ██║█████╗  
╚██╗ ██╔╝██║██╔═══╝ ██║   ██║██║╚██╗██║██╔══╝  
 ╚████╔╝ ██║██║     ╚██████╔╝██║ ╚████║███████╗
  ╚═══╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚══════╝
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

**macOS Intel (x86_64)**
```bash
curl -sSfLO https://github.com/randomm/vipune/releases/latest/download/vipune-x86_64-apple-darwin.tar.gz
tar xzf vipune-x86_64-apple-darwin.tar.gz
sudo mv vipune /usr/local/bin/
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
cargo install --git https://github.com/randomm/vipune --tag $LATEST_TAG vipune
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
rm -rf ~/.local/share/vipune ~/.cache/vipune ~/.config/vipune
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
| `vipune import <source>` | Import memories from SQLite or JSON |
| `vipune version` | Show version |

[Complete CLI reference](docs/cli-reference.md) • [Quickstart guide](docs/quickstart.md)

## Configuration

vipune works with zero configuration. All paths use platform-standard XDG directories:

**Default paths:**
- Database: `~/.local/share/vipune/memories.db`
- Model cache: `~/.cache/vipune/models/`
- Config file: `~/.config/vipune/config.toml`

**Environment variables (override defaults):**
- `VIPUNE_DATABASE_PATH` - SQLite database location
- `VIPUNE_EMBEDDING_MODEL` - HuggingFace model ID (default: `sentence-transformers/bge-small-en-v1.5`)
- `VIPUNE_MODEL_CACHE` - Model download cache directory
- `VIPUNE_PROJECT` - Project identifier (overrides auto-detection)
- `VIPUNE_SIMILARITY_THRESHOLD` - Conflict detection threshold, 0.0-1.0 (default: `0.85`)
- `VIPUNE_RECENCY_WEIGHT` - Recency bias in search results, 0.0-1.0 (default: `0.3`)

**Config file (`~/.config/vipune/config.toml`):**
```toml
database_path = "/custom/path/memories.db"
embedding_model = "sentence-transformers/bge-small-en-v1.5"
model_cache = "~/.cache/vipune/models"
similarity_threshold = 0.85
recency_weight = 0.3
```

## Agent Integration

vipune is designed for AI agents to maintain persistent memory across tasks.

**Conflict detection workflow:**

```bash
# Agent attempts to add memory
vipune add "Authentication uses OAuth2"

# If similar memory exists, exit code 2 and JSON output:
{
  "status": "conflicts",
  "proposed": "Authentication uses OAuth2",
  "conflicts": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "content": "Auth system uses OAuth2 for login",
      "similarity": 0.94
    }
  ]
}

# Agent can then:
# 1. Skip dupe (exit code 2)
# 2. Force add to bypass conflict detection
vipune add "Authentication uses OAuth2" --force
# 3. Update existing memory
vipune update 123e4567-e89b-12d3-a456-426614174000 "Auth system uses OAuth2 for login"
```

**Exit codes:**
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

## Code Search + Memory Management

Use semantic code search (e.g., [ColGREP](https://github.com/colgrep/colgrep)) to discover relevant code, then store findings:

```bash
# Find authentication implementation
colgrep "authentication middleware"

# Store the finding with context
vipune add "JWT token validation in src/auth/middleware.rs (validate_token function)"
```

## License

Apache-2.0 © [Janni Turunen](https://github.com/randomm/vipune)

## Links

- [GitHub](https://github.com/randomm/vipune)
- [Issues](https://github.com/randomm/vipune/issues)
- [CLI Reference](docs/cli-reference.md)
- [Quickstart](docs/quickstart.md)
