# Quickstart Guide

Get started with vipune in 5 minutes.

## Step 1: Install

```bash
# From crates.io (recommended)
cargo install vipune

# Or build from source
git clone https://github.com/randomm/vipune.git
cd vipune
cargo build --release
cargo install --path .
```

**First run:** vipune will automatically download the ONNX model (~400MB) to `~/.cache/vipune/models/`. This happens once and takes 30-60 seconds depending on your connection.

## Step 2: Add Your First Memory

```bash
# Store a simple memory
vipune add "Alice works at Microsoft as a senior engineer"

# Output: Added memory: 123e4567-e89b-12d3-a456-426614174000
```

vipune stores:
- The text content
- Semantic embedding (384 dimensions)
- Creation timestamp
- Project ID (auto-detected from git)

## Step 3: Search Memories

```bash
# Search by meaning, not keywords
vipune search "where does alice work"

# Output:
# 123e4567-e89b-12d3-a456-426614174000 [score: 0.95]
#   Alice works at Microsoft as a senior engineer
```

**Tips for searching:**
- Use natural language: "how do we handle payments", "database schema for users"
- Try variations if results aren't ideal: same meaning, different words
- Check similarity scores: 0.9+ = very similar, 0.7-0.9 = related, below 0.7 = distant

## Step 4: Conflict Detection

vipune warns when you add duplicate or very similar memories.

```bash
# Try adding a similar memory
vipune add "Alice is a senior engineer at Microsoft"

# Output:
# Conflicts detected: 1 similar memory/memories found
# Proposed: Alice is a senior engineer at Microsoft
# Use --force to add anyway
#   123e4567-e89b-12d3-a456-426614174000 (similarity: 0.94)
#     Alice works at Microsoft as a senior engineer
# Exit code: 2

# Options when conflicts occur:
# 1. Skip (don't add)
# 2. Force add (not available in v0.1.0, see issue #6)
# vipune add "Alice is a senior engineer at Microsoft" --force  # Coming soon
# 3. Update existing memory instead
vipune update 123e4567-e89b-12d3-a456-426614174000 "Alice works at Microsoft as a senior engineer (since 2020)"
```

**Why conflict detection matters:**
- Prevents redundant memories
- Maintains knowledge quality
- Forces agents to resolve ambiguity explicitly

## Step 5: Configuration (Optional)

vipune works with zero configuration. Customize only if needed.

### Check Current Database Path

```bash
# See where memories are stored (default: ~/.local/share/vipune/memories.db)
ls ~/.local/share/vipune/memories.db
```

### Override Database Path

```bash
# Use custom database location
vipune add "Test" --db-path /tmp/test.db

# Or set environment variable
export VIPUNE_DATABASE_PATH="/custom/path/memories.db"
vipune add "Test"

# Override project scope via environment
export VIPUNE_PROJECT="my-custom-project"
vipune add "Project-specific memory"
```

### Adjust Conflict Threshold

**Default:** 0.85 (memories with similarity ≥ 0.85 trigger conflicts)

```bash
# More strict (catch near-duplicates)
export VIPUNE_SIMILARITY_THRESHOLD="0.9"

# More permissive (only exact/near-exact matches conflict)
export VIPUNE_SIMILARITY_THRESHOLD="0.95"

# Disable conflict detection (not recommended)
export VIPUNE_SIMILARITY_THRESHOLD="1.0"
```

### Config File Example

Create `~/.config/vipune/config.toml`:

```toml
database_path = "~/.local/share/vipune/memories.db"
embedding_model = "sentence-transformers/bge-small-en-v1.5"
model_cache = "~/.cache/vipune/models"
similarity_threshold = 0.85
```

## Step 6: Migration from remory (Optional)

If you're migrating from remory:

1. Export memories from remory
2. Import into vipune (command coming soon - see [issue #9](https://github.com/randomm/vipune/issues/9))

## Common Workflows

### Storing Code Knowledge

```bash
# After reading code or documentation
vipune add "Authentication middleware validates JWT tokens in src/auth/middleware.rs"

# Tag with metadata for organization
vipune add "Users table schema: id, email, password_hash, created_at, updated_at" \
  --metadata '{"table": "users", "schema": true}'
```

### Fact-Checking Before Adding

```bash
# Search first to avoid duplicates
vipune search "authentication implementation"

# If similar results found, update existing instead of adding new
vipune update <existing-id> "Auth uses JWT with refresh tokens (expires in 24h)"
```

### Listing All Memories

```bash
# See all memories for current project
vipune list

# More results
vipune list --limit 50

# Export to JSON for processing
vipune list --limit 9999 --json > memories.json
```

### Project Isolation

```bash
# Memories are scoped to git repository
cd ~/projects/myapp
vipune add "Myapp uses PostgreSQL"
cd ~/projects/otherproject
vipune add "Otherapp uses SQLite"

# Search respects project scope automatically
vipune search "database"  # Only finds memories for otherproject
```

### JSON for Scripting

```bash
# Add memory and get ID
ID=$(vipune add --json "Fact" | jq -r '.id')

# Search and extract highest similarity
vipune search --json "test" | jq '.results[0]'

# Check for conflicts in script
if vipune add --json "New fact" | jq -e '.conflicts' > /dev/null; then
  echo "Conflict detected!"
fi
```

## Troubleshooting

### Model Download Fails

If the first run fails to download the model:

```bash
# Clear cache and retry
rm -rf ~/.cache/vipune/models/
vipune add "Test"  # Will re-download
```

### Conflicts Too Aggressive

If legitimate memories are flagged as conflicts:

```bash
# Lower threshold temporarily
export VIPUNE_SIMILARITY_THRESHOLD="0.9"
vipune add "Your memory"
```

### Can't Find Memories After Adding

```bash
# Check project scope (--project is a global flag, must come before command)
vipune --project "git@github.com:user/repo.git" list

# Or search without project filter (if using default scope)
vipune --project "default" search "query"
```

### Database Locked

```bash
# Only one vipune process can access the database at a time
# Close other terminals running vipune, then retry
```

## Next Steps

- Read the [complete CLI reference](cli-reference.md)
- Check [issue #9](https://github.com/randomm/vipune/issues/9) for remory migration
- Explore agent integration patterns in the [README](../README.md#agent-integration)