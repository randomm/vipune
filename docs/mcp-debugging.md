# MCP Debugging Guide

This guide helps developers debug the vipune MCP server integration with AI agents like Claude Code and Cursor.

Note: unlike all other vipune commands, `vipune mcp` is long-lived by design — it's the single architectural exception to vipune's no-daemon policy.

## Running the MCP Server Locally

### Standalone Mode

The MCP server runs via stdio and expects JSON-RPC messages. Start it directly:

```bash
# Start MCP server (auto-detected project)
vipune mcp

# With explicit project ID
vipune mcp --project my-project
```

### Via Cargo (Development)

When developing locally:

```bash
# Build and run MCP server
cargo run -- mcp

# With explicit project
cargo run -- mcp --project my-project
```

The server will block and wait for JSON-RPC messages on stdin. It runs until the client disconnects or stdin closes.

## Testing JSON-RPC Requests

### Basic Test Script

Create a test script (`test-mcp.sh`) to send requests via stdin:

```bash
#!/bin/bash
# Send a JSON-RPC request to MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | vipune mcp
```

### Example Tool Calls

Each MCP tool has a specific JSON-RPC format:

#### store_memory

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "store_memory",
    "arguments": {
      "text": "Alice works at Microsoft",
      "memory_type": "fact",
      "metadata": {"topic": "people"}
    }
  }
}
```

#### search_memories

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search_memories",
    "arguments": {
      "query": "where does alice work",
      "memory_types": ["fact"],
      "limit": 5
    }
  }
}
```

#### list_memories

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "list_memories",
    "arguments": {
      "limit": 10,
      "memory_types": ["fact"]
    }
  }
}
```

#### supersede_memory

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "supersede_memory",
    "arguments": {
      "old_memory_id": "uuid-here",
      "new_text": "Alice works at Google as a senior engineer",
      "memory_type": "fact"
    }
  }
}
```

### Interactive Testing

Use `rlwrap` for an interactive stdin session:

```bash
# macOS
brew install rlwrap
# Ubuntu/Debian
sudo apt install rlwrap
```

```bash
# Note: responses will interleave with the readline prompt
rlwrap vipune mcp
```

Then paste JSON-RPC payloads directly (press Ctrl+D after each to send).

## Common Issues

### Buffering Problems

**Symptom**: Responses don't appear immediately.

**Cause**: stdio buffering delays output.

**Solution**: Ensure JSON-RPC messages end with newline (`\n`). The MCP protocol expects line-delimited JSON.

```bash
# Bad: no trailing newline (server may not process the request)
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | vipune mcp

# Good: explicit newline using echo (echo always adds \n)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | vipune mcp
```

### Line Ending Issues

**Symptom**: Server hangs or parse errors.

**Cause**: Windows CRLF (`\r\n`) vs Unix LF (`\n`) line endings.

**Solution**: Use POSIX-compliant tools:

```bash
# macOS/Linux - correct line endings
echo '{"jsonrpc":"2.0","id":1}' | vipune mcp

# Windows/convert - sanitize line endings
echo '{"jsonrpc":"2.0","id":1}' | dos2unix | vipune mcp
```

### JSON Format Errors

**Symptom**: `Parse error: invalid JSON`.

**Cause**: Malformed JSON-RPC payload.

**Common mistakes:**
- Missing `jsonrpc: "2.0"` field
- Missing `id` field (required for requests)
- Unclosed braces or quotes
- Invalid escape sequences in `text` field

**Debug with jq**:
```bash
echo 'your-json-here' | jq -c .  # Validate JSON before sending (compact/single-line output)
```

### Tool Registration Issues

**Symptom**: Agent reports "tool not found" or blank tool list.

**Cause**: MCP handler not exposing tools correctly.

**Verification steps:**

1. Check `tools/list` returns tools:
```json
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

Expected response includes `store_memory`, `search_memories`, `list_memories`, `supersede_memory`.

2. Verify tool names match agent expectations (case-sensitive):
- `store_memory` ✅
- `Store_Memory` ❌
- `storeMemory` ❌

3. Check server logs (see Debug Logging below)

### Project Detection Failures

**Symptom**: Missing project_id in results, or wrong project.

**Cause**: Not in a git repository, or git detection fails.

**Debug:**

```bash
# Force explicit project
vipune mcp --project my-project

# Check git detection
git rev-parse --git-dir  # Should return .git path
```

## Debug Logging

### Environment Variables

vipune uses Rust's `env_logger`. Enable logging before starting MCP:

```bash
# Enable all debug output
RUST_LOG=debug vipune mcp

# MCP-specific debugging
RUST_LOG=vipune::mcp=debug vipune mcp

# Store wrapper debugging
RUST_LOG=vipune::mcp::store_wrapper=trace vipune mcp
```

### Log Levels

- `trace` - Extremely verbose, every step
- `debug` - Detailed information flow
- `info` - High-level operations (default)
- `warn` - Warning messages
- `error` - Error conditions only

### Example Session with Logging

```bash
RUST_LOG=debug vipune mcp <<EOF
{"jsonrpc":"2.0","id":1,"method":"tools/list"}
EOF
```

Output shows:
- Tools registered
- Request received
- Response generated
- Errors (if any)

## Verification Checklist

### Server Starts Successfully

- [ ] `vipune mcp` doesn't crash immediately
- [ ] Process blocks waiting for input (expected)
- [ ] No panics or assertion failures

### tools/list Works

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | vipune mcp
```

- [ ] Returns JSON response
- [ ] `result.tools` array has 4 tools
- [ ] Tool names: `store_memory`, `search_memories`, `list_memories`, `supersede_memory`

### Store Memory Works

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"store_memory","arguments":{"text":"test memory"}}}
```

- [ ] Returns success response
- [ ] `toolCallId` or result contains memory ID
- [ ] No errors in response

### Search Memory Works

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_memories","arguments":{"query":"test"}}}
```

- [ ] Returns array of memories
- [ ] Each memory has `content`, `id`, `similarity` (or `score`)

### Agent Integration Tests

After verifying basic JSON-RPC, test with actual agent:

- [ ] Claude Code: Add to `~/.claude/settings.json`, check MCP tools appear
- [ ] Cursor: Add to MCP config, test `store_memory` call
- [ ] Verify project auto-detection works (project_id appears in results)
- [ ] Check search results include metadata when provided

## Getting Help

If issues persist:

1. Check server logs with `RUST_LOG=debug`
2. Verify JSON-RPC payloads are valid (`jq -c .`)
3. Test with minimal examples above
4. Check project isolation (use `--project` to bypass git detection)
5. Review agent's MCP client logs

For bug reports, include:
- VIPune version: `vipune version`
- OS and platform
- JSON-RPC request payload (sanitized)
- Full error message
- `RUST_LOG=debug` output