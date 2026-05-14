# CLI Reference

Complete reference for all vipune commands.

## Global Flags

These flags apply to all commands:

| Flag | Short | Description |
|------|-------|-------------|
| `--json` | | Output as JSON (pretty-printed) instead of human-readable text |
| `--project <id>` | `-p` | Project identifier (auto-detected from git if omitted) |
| `--db-path <path>` | | Override database path |

## Commands

### add

Store a memory.

```
vipune add <text> [--metadata <json>] [--force] [--memory-type <type>] [--status <status>] [--supersedes <id>]
```

**Arguments:**
- `text` - Memory text content (required)

**Flags:**
- `-m, --metadata <json>` - Optional JSON metadata (e.g., `{"topic": "auth"}`)
- `--force` - Bypass conflict detection and add regardless
- `--memory-type <type>` - Memory type: `fact` (default), `preference`, `procedure`, `guard`, `observation`
- `--status <status>` - Initial status: `active` (default) or `candidate`
- `--supersedes <id>` - Atomically supersede an existing memory (mutually exclusive with `--force`)

**Behavior:**
- Generates semantic embedding for the text
- Checks for similar existing memories (similarity ≥ threshold)
- If conflicts found: returns exit code 2, lists conflicting memories
- If `--force` used: skips conflict check and adds memory
- If `--supersedes` used: atomically adds new memory and marks target as `superseded`

**Exit codes:**
- `0` - Successfully added
- `1` - Error (invalid input, database error)
- `2` - Conflicts detected (similar memories exist)
- `3` - Content too long (exceeds embedding token limit)

**Human output:**
```
Added memory: 123e4567-e89b-12d3-a456-426614174000
```

**Conflicts output:**
```
Conflicts detected: 1 similar memory/memories found
Proposed: Authentication uses OAuth2
Use --force to add anyway
  123e4567-e89b-12d3-a456-426614174000 (similarity: 0.94)
    Auth system uses OAuth2 for login
```

**JSON output (success):**
```json
{
  "status": "added",
  "id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**JSON output (conflicts):**
```json
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
```

**Supersede example:**
```bash
# Replace an outdated memory atomically
vipune add "Alice now works at Google" --supersedes abc123-old-memory-id
```
This inserts the new memory and marks the old one as `superseded` in a single transaction.

---

### search

Find memories by semantic similarity.

```
vipune search <query> [--limit <n>] [--recency <weight>] [--hybrid] [--memory-type <types>] [--status <statuses>] [--include-candidates]
```

**Arguments:**
- `query` - Search query text (required)

**Flags:**
- `-l, --limit <n>` - Maximum results to return (default: `5`)
- `--recency <weight>` - Recency bias for scoring, 0.0 to 1.0 (default: from config, typically `0.3`)
- `--hybrid` - Enables hybrid search combining semantic similarity with FTS5 full-text search using Reciprocal Rank Fusion (RRF)
- `--no-hybrid` - Disables hybrid search even when enabled in config (useful to force pure semantic search)
- `--memory-type <types>` - Filter by memory type (comma-separated, e.g., `guard,procedure`)
- `--status <statuses>` - Filter by status (comma-separated, e.g., `active,candidate`)
- `--include-candidates` - Shorthand to include both `active` and `candidate` memories

**Behavior:**
- Generates embedding for query
- Finds memories with highest cosine similarity
- Combines semantic similarity with time decay for final score
- Returns results sorted by final score (highest first)
- All memories in current project scope
- **Default filtering:** Only `active` memories are returned by default. Use `--status` or `--include-candidates` to include other statuses.

**Recency scoring:**
The final score combines: `(1 - recency_weight) * similarity + recency_weight * time_score`
- `recency_weight = 0.0`: Pure semantic similarity
- `recency_weight = 1.0`: Pure recency (newest first)
- `recency_weight = 0.3`: Default balance (70% semantic, 30% recency)

**Exit codes:**
- `0` - Success (may return empty results if no matches)

**Human output:**
```
123e4567-e89b-12d3-a456-426614174000 [score: 0.95]
  Alice works at Microsoft as a senior engineer

234e5678-e89b-12d3-a456-426614174001 [score: 0.87]
  Bob is a software engineer at Google
```

**JSON output:**
```json
{
  "results": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "content": "Alice works at Microsoft as a senior engineer",
      "similarity": 0.95,
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "id": "234e5678-e89b-12d3-a456-426614174001",
      "content": "Bob is a software engineer at Google",
      "similarity": 0.87,
      "created_at": "2024-01-16T14:20:00Z"
    }
  ]
}
```

**Recency example:**
```bash
# Default recency balance (0.3)
vipune search "authentication"

# High recency bias (recent memories rank higher)
vipune search "authentication" --recency 0.7

# Pure semantic similarity (no time bias)
vipune search "authentication" --recency 0.0
```

---

### get

Retrieve a memory by ID.

```
vipune get <id>
```

**Arguments:**
- `id` - Memory ID (required)

**Exit codes:**
- `0` - Memory found
- `1` - Memory not found or error

**Human output:**
```
ID: 123e4567-e89b-12d3-a456-426614174000
Content: Alice works at Microsoft as a senior engineer
Project: git@github.com:user/repo.git
Metadata: {"topic": "team"}
Created: 2024-01-15T10:30:00Z
Updated: 2024-01-15T10:30:00Z
```

**JSON output:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "content": "Alice works at Microsoft as a senior engineer",
  "project_id": "git@github.com:user/repo.git",
  "metadata": "{\"topic\": \"team\"}",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

### list

List all memories in the current project.

```
vipune list [--limit <n>] [--memory-type <types>] [--status <statuses>] [--include-candidates]
```

**Flags:**
- `-l, --limit <n>` - Maximum results to return (default: `10`)
- `--memory-type <types>` - Filter by memory type (comma-separated, e.g., `guard,procedure`)
- `--status <statuses>` - Filter by status (comma-separated, e.g., `active,candidate`)
- `--include-candidates` - Shorthand to include both `active` and `candidate` memories

**Behavior:**
- Returns memories ordered by creation time (newest first)
- Limited to current project scope
- **Default filtering:** Only `active` memories are returned by default. Use `--status` or `--include-candidates` to include other statuses.

**Exit codes:**
- `0` - Success (may return empty list)

**Human output:**
```
123e4567-e89b-12d3-a456-426614174000: Alice works at Microsoft
234e5678-e89b-12d3-a456-426614174001: Bob is a software engineer at Google
```

**JSON output:**
```json
{
  "memories": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "content": "Alice works at Microsoft",
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "id": "234e5678-e89b-12d3-a456-426614174001",
      "content": "Bob is a software engineer at Google",
      "created_at": "2024-01-16T14:20:00Z"
    }
  ]
}
```

---

### delete

Delete a memory by ID.

```
vipune delete <id>
```

**Arguments:**
- `id` - Memory ID (required)

**Exit codes:**
- `0` - Memory deleted
- `1` - Memory not found or error

**Human output:**
```
Deleted memory: 123e4567-e89b-12d3-a456-426614174000
```

**JSON output:**
```json
{
  "status": "deleted",
  "id": "123e4567-e89b-12d3-a456-426614174000"
}
```

---

### update

Update a memory's content.

```
vipune update <id> [text] [--metadata <json>]
```

**Arguments:**
- `id` - Memory ID (required)
- `text` - New content (optional — if omitted, only metadata is updated)

**Flags:**
- `-m, --metadata <json>` - Replace metadata (must be valid JSON)

**Behavior:**
- If only `text` provided: generates new embedding, preserves metadata
- If only `--metadata` provided: updates metadata without re-embedding
- If both `text` and `--metadata` provided: updates both
- Metadata must be valid JSON (validated on write)

**Exit codes:**
- `0` - Memory updated
- `1` - Memory not found or error

**Human output:**
```
Updated memory: 123e4567-e89b-12d3-a456-426614174000
```

**JSON output:**
```json
{
  "status": "updated",
  "id": "123e4567-e89b-12d3-a456-426614174000"
}
```

---

### validate

Check if text content is within the embedding model's token limit.

```
vipune validate <text>
```

**Arguments:**
- `text` - Text to validate (required)

**Exit codes:**
- `0` - Within token limit
- `3` - Content exceeds token limit

**Human output:**
```
Token count: 42/512 — within limit
```

**JSON output:**
```json
{
  "token_count": 42,
  "max_tokens": 512,
  "within_limit": true
}
```

---

### version

Display version information.

```
vipune version
```

**Exit codes:**
- `0` - Success

**Human output:**
```
vipune 0.1.1
```

**JSON output:**
```json
{
  "version": "0.1.1",
  "name": "vipune"
}
```

*Note: Output version matches your installed version. Update this example when upgrading vipune.*

---

### mcp

Start vipune as an MCP (Model Context Protocol) server over stdio. This enables AI agents like Claude Code and Cursor to use vipune as a native memory provider.

```
vipune mcp
```

**Behavior:**
- Starts MCP server on stdin/stdout
- Exposes three tools: `store_memory`, `search_memories`, `list_memories`
- Automatically detects project from git repository
- Uses default database path (`~/.vipune/memories.db`)

**Available Tools:**

| Tool | Description |
|------|-------------|
| `store_memory` | Store information for later recall |
| `search_memories` | Find memories by meaning, with type/status filters |
| `list_memories` | List recent memories, with type/status filters |
| `supersede_memory` | Replace an existing memory with new content |
| `get_memory` | Retrieve a specific memory by ID |
| `delete_memory` | Delete a memory permanently by ID |
| `update_memory` | Update an existing memory's content, metadata, type, or status |

**Exit codes:**
- `0` - Success
- `1` - Error (initialization failed)

**New v0.3 tool parameters:**

`store_memory` accepts:
- `text` — the information to remember (required)
- `metadata` — optional structured labels as JSON (e.g., `{"topic": "auth"}`)
- `memory_type` — type of memory: `fact` (default), `preference`, `procedure`, `guard`, `observation`
- `status` — initial status: `active` (default) or `candidate`
- `supersedes` — ID of memory to supersede (atomically replaces old memory)
- `force` — force store even if conflicts detected (default: `false`)

`supersede_memory` accepts:
- `new_text` — the new content that replaces the old memory (required)
- `old_memory_id` — ID of the memory to supersede (required)
- `memory_type` — optional type for the new memory (default: `fact`)
- `metadata` — optional structured labels as JSON

`get_memory` accepts:
- `id` — ID of the memory to retrieve (required)
- `no_touch` — skip updating retrieval telemetry (default: `false`)

`delete_memory` accepts:
- `id` — ID of the memory to delete (required)

`update_memory` accepts:
- `id` — ID of the memory to update (required)
- `text` — optional new content for the memory
- `metadata` — optional new metadata as JSON object
- `memory_type` — optional new memory type
- `status` — optional new status: `active` or `candidate`
  (At least one optional field must be provided)

`search_memories` and `list_memories` accept optional:
- `memory_types` — array of types to filter by (e.g., `["guard", "procedure"]`)
- `statuses` — array of statuses to filter by (e.g., `["active", "candidate"]`)
- `no_touch` — skip updating retrieval telemetry (default: `false`)

`search_memories` also accepts:
- `recency_weight` — recency bias for scoring, 0.0 to 1.0 (default: config value)
- `hybrid` — use hybrid search (semantic + BM25), true/false (default: config value)

**Example MCP configuration (Claude Code):**
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

---

## Upgrading from v0.2

v0.3 includes automatic schema migrations — no manual steps required. On first run after upgrading, vipune will add the new `type`, `status`, and `superseded_by` columns to your existing database. All existing memories default to type `fact` and status `active`.

**Breaking change:** Content that previously was silently truncated when exceeding the embedding token limit now fails with exit code 3. Use `vipune validate <text>` to check content length before adding.

---

## Exit Codes Summary

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (invalid input, database error, not found) |
| `2` | Conflicts detected (similar memories exist) |
| `3` | Content too long (exceeds embedding token limit) |

---

## Project Detection

vipune automatically detects projects from git repositories. When inside a git repo, the project ID is inferred from the remote origin URL.

**To override project scope:**
```bash
vipune add "Memory for specific project" --project "my-custom-project"
```

**Project ID examples:**
- Inside `~/projects/myapp/.git`: `git@github.com:user/myapp.git`
- No git repository: `default` (all memories share default scope)

---

## Error Handling

All commands return exit code `1` on error, with error message to stderr or JSON error response.

**JSON error format:**
```json
{
  "error": "Memory not found"
}
```

**Common errors:**
- Memory not found (`get`, `update`, `delete`)
- Invalid metadata (not valid JSON)
- Database errors (permissions, disk full)
- Missing or invalid configuration

---

## Examples

**Semantic search:**
```bash
vipune search "how do we handle authentication"
```

**Hybrid search (keywords):**
```bash
vipune search "JWT tokens" --hybrid
```

*See [Search Guide](search.md) for when to use hybrid vs semantic search and how recency weighting works.*

**Find by metadata (via search):**
```bash
# High recency bias for time-sensitive queries
vipune search "recent changes" --recency 0.8

# Pure semantic search for knowledge retrieval
vipune search "authentication architecture" --recency 0.0
```

**Force add despite conflicts:**
```bash
vipune add "Duplicate memory" --force
```

**Batch import (loop in shell):**
```bash
for fact in facts.txt; do
  vipune add "$fact" || break
done
```

**Export all memories:**
```bash
vipune list --limit 9999 --json > memories.json
```

**Find and update:**
```bash
# Search for memory
vipune search "auth implementation"
# Get output ID, then update
vipune update 123e4567-e89b-12d3-a456-426614174000 "Auth uses JWT with refresh tokens"
```

**JSON processing with jq:**
```bash
# Add and extract ID
ID=$(vipune add --json "Important fact" | jq -r '.id')
echo "Added: $ID"

# Search and get highest similarity
vipune search --json "test" | jq '.results[0].similarity'

# Check for conflicts in script
if vipune add --json "New fact" | jq -e '.conflicts' > /dev/null; then
  echo "Conflict detected!"
fi
```

**Project-specific operations:**
```bash
# Add memory to specific project
vipune --project "my-company/project" add "Company-specific knowledge"

# Search within project
vipune --project "my-company/project" search "API keys"
```

**See also:**
- [Search Guide](search.md) — When to use hybrid vs semantic search, recency weighting
- [Query Guide](vipune-query-guide.md) — Writing effective semantic queries