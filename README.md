# vipune

A minimal memory layer for AI agents.

Store semantic memories, search by meaning, and detect conflicts. Single binary CLI. No API keys required.

## Features

- **Semantic search** - Find memories by meaning, not keywords (ONNX embeddings, bge-small-en-v1.5)
- **Conflict detection** - Automatically warns when adding duplicate or similar memories
- **Zero configuration** - Works out of the box (auto-detected git projects, sensible defaults)
- **Single binary** - Just one CLI tool, no daemon, no database server
- **No API keys** - Everything runs locally, no network dependencies
- **Project scoped** - Memories isolated by git repository

## Quick Start

```bash
# Install from source
cargo install --git https://github.com/randomm/vipune vipune

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

**Config file (`~/.config/vipune/config.toml`):**
```toml
database_path = "/custom/path/memories.db"
embedding_model = "sentence-transformers/bge-small-en-v1.5"
model_cache = "~/.cache/vipune/models"
similarity_threshold = 0.85
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
# 2. Force add (not available in v0.1.0, see issue #6)
# vipune add "Authentication uses OAuth2" --force  # Coming soon
# 3. Update existing memory
vipune update 123e4567-e89b-12d3-a456-426614174000 "Auth system uses OAuth2 for login"
```

**Exit codes:**
- `0` - Success
- `1` - Error (missing file, invalid input, etc.)
- `2` - Conflicts detected (similar memories found)

## Code Search + Memory Management

Use semantic code search (e.g., [ColGREP](https://github.com/colgrep/colgrep)) to discover relevant code, then store findings:

```bash
# Find authentication implementation
colgrep "authentication middleware"

# Store the finding with context
vipune add "JWT token validation in src/auth/middleware.rs (validate_token function)"
```

## Migration from remory

Coming soon. See [issue #9](https://github.com/randomm/vipune/issues/9).

## License

Apache-2.0 © [Janni Turunen](https://github.com/randomm/vipune)

## Links

- [GitHub](https://github.com/randomm/vipune)
- [Issues](https://github.com/randomm/vipune/issues)
- [CLI Reference](docs/cli-reference.md)
- [Quickstart](docs/quickstart.md)