# Quickstart Guide

Get started with vipune in 5 minutes.

## Step 1: Install

```bash
# Install from source
cargo install --git https://github.com/randomm/vipune vipune

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

**Recency scoring:**
Search results can be biased toward recent memories:

```bash
# Default balance (70% semantic, 30% recency)
vipune search "authentication"

# High recency bias (recent memories rank higher)
vipune search "authentication" --recency 0.7

# Pure semantic search (no time bias)
vipune search "authentication" --recency 0.0
```

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
# 2. Force add to bypass conflict detection
vipune add "Alice is a senior engineer at Microsoft" --force
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

### Adjust Recency Bias

**Default:** 0.3 (30% recency, 70% semantic similarity)

```bash
# High recency bias (recent memories always rank higher)
export VIPUNE_RECENCY_WEIGHT="0.8"

# Pure semantic search (no time bias)
export VIPUNE_RECENCY_WEIGHT="0.0"

# Balance semantic and recency equally
export VIPUNE_RECENCY_WEIGHT="0.5"
```

### Config File Example

Create `~/.config/vipune/config.toml`:

```toml
database_path = "~/.local/share/vipune/memories.db"
embedding_model = "BAAI/bge-small-en-v1.5"
model_cache = "~/.cache/vipune/models"
similarity_threshold = 0.85
recency_weight = 0.3
```

## Step 6: Migration from remory (Optional)

If you're migrating from remory, import your existing memories directly:

```bash
# Preview what would be imported
vipune import ~/.local/share/remory/memories.db --dry-run

# Output:
# Dry run: would import from /home/user/.local/share/remory/memories.db
# Total memories: 150
# Imported: 0
# Skipped duplicates: 0
# Skipped corrupted: 0
# Projects: 3
#   - git@github.com:user/repo1.git
#   - git@github.com:user/repo2.git
#   - default

# Perform the actual import
vipune import ~/.local/share/remory/memories.db

# Output:
# Imported from /home/user/.local/share/remory/memories.db
# Total memories: 150
# Imported: 142
# Skipped duplicates: 8
# Skipped corrupted: 0
# Projects: 3
#   - git@github.com:user/repo1.git
#   - git@github.com:user/repo2.git
#   - default
```

Import formats:
- **SQLite (default):** remory database format
- **JSON:** vipune JSON export format

```bash
# Import from remory SQLite
vipune import ~/.local/share/remory/memories.db --format sqlite

# Import from JSON export
vipune import export.json --format json
```

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

### Recency-Weighted Search

```bash
# Find recent changes first (high recency bias)
vipune search "API changes" --recency 0.8

# Find fundamental knowledge (pure semantic)
vipune search "authentication patterns" --recency 0.0

# Balance relevance and freshness
vipune search "database schema" --recency 0.4
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

# Or force add the memory
vipune add "Your memory" --force
```

### Can't Find Memories After Adding

```bash
# Check project scope (--project is a global flag, must come before command)
vipune --project "git@github.com:user/repo.git" list

# Or search without project filter (if using default scope)
vipune --project "default" search "query"
```

### Search Results Don't Rank by Time

If you want recent memories to rank higher:

```bash
# Increase recency bias
export VIPUNE_RECENCY_WEIGHT="0.7"
vipune search "recent changes"

# Or use --recency flag per search
vipune search "recent changes" --recency 0.7
```

### Database Locked

```bash
# Only one vipune process can access the database at a time
# Close other terminals running vipune, then retry
```

## Next Steps

- Read the [complete CLI reference](cli-reference.md)
- Explore agent integration patterns in the [README](../README.md#agent-integration)
- Check project issues for upcoming features