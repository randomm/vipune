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
vipune add <text> [--metadata <json>] [--force]
```

**Arguments:**
- `text` - Memory text content (required)

**Flags:**
- `-m, --metadata <json>` - Optional JSON metadata (e.g., `{"topic": "auth"}`)
- `--force` - Bypass conflict detection and add regardless (**not implemented in v0.1.0**, see issue #6)

**Behavior:**
- Generates semantic embedding for the text
- Checks for similar existing memories (similarity ≥ threshold)
- If conflicts found: returns exit code 2, lists conflicting memories
- If `--force` used: would skip conflict check (not implemented in v0.1.0)

**Exit codes:**
- `0` - Successfully added
- `1` - Error (invalid input, database error)
- `2` - Conflicts detected (similar memories exist)

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

---

### search

Find memories by semantic similarity.

```
vipune search <query> [--limit <n>]
```

**Arguments:**
- `query` - Search query text (required)

**Flags:**
- `-l, --limit <n>` - Maximum results to return (default: `5`)

**Behavior:**
- Generates embedding for query
- Finds memories with highest cosine similarity
- Returns results sorted by similarity (highest first)
- All memories in current project scope

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
  "metadata": "{\"topic\": \"team\"}",  // or null if no metadata
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

### list

List all memories in the current project.

```
vipune list [--limit <n>]
```

**Flags:**
- `-l, --limit <n>` - Maximum results to return (default: `10`)

**Behavior:**
- Returns memories ordered by creation time (newest first)
- Limited to current project scope

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
vipune update <id> <text>
```

**Arguments:**
- `id` - Memory ID (required)
- `text` - New content (required)

**Behavior:**
- Generates new embedding for updated content
- Preserves: ID, project ID, creation timestamp
- Updates: content, embedding, updated_at timestamp

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

### version

Display version information.

```
vipune version
```

**Exit codes:**
- `0` - Success

**Human output:**
```
vipune 0.1.0
```

**JSON output:**
```json
{
  "version": "0.1.0",
  "name": "vipune"
}
```

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
  "error": "Memory not found: 123e4567-e89b-12d3-a456-426614174000"
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

**Find by metadata (via search):**
```bash
vipune add "Users table has email, password hash, created_at" --metadata '{"table": "users"}'
vipune search "table schema"  # Will find it by semantic meaning
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
```