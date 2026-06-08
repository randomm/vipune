# Agent Integration

vipune is a generic CLI tool for persistent memory that works with any agent capable of running shell commands. Integration requires two steps: (1) tell the agent about vipune by adding a configuration snippet, and (2) grant permission to execute shell commands. Once configured, agents can use `vipune search` and `vipune add` to maintain a knowledge loop across tasks without plugins, API keys, or additional setup.

The knowledge loop is simple: before starting work, search for relevant context; execute the work; then store important discoveries for future recall. This pattern helps agents avoid redundant work, maintain consistency, and build cumulative knowledge over time.

## The knowledge loop

Use vipune to create a feedback cycle in your agent workflow:

*See also: [Search Guide](search.md) for query patterns that work best with agents.*

1. **Search before starting** - `vipune search "relevant topic"` to recall context from past work
2. **Do the work** - Execute your task with the knowledge in mind
3. **Store learnings** - `vipune add "important discovery"` after completing meaningful work

### Memory types

Memories can be categorized by type for routing and filtering:

```bash
# Store a guard (safety constraint)
vipune add "Never deploy on Fridays" --memory-type guard

# Store a procedure
vipune add "Run cargo test before pushing" --memory-type procedure

# Search only guard memories
vipune search "deployment rules" --memory-type guard

# Replace an outdated memory
vipune add "Alice now works at Google" --supersedes <old-memory-id>
```

Available types: `fact` (default), `preference`, `procedure`, `guard`, `observation`

Default search returns only `active` memories. Use `--include-candidates` to also see `candidate` memories.

This pattern is especially useful for:
- Avoiding duplicate implementations or analyses
- Maintaining architectural consistency across multiple tasks
- Building project-specific context that outlives individual agent sessions
- Tracking decisions and their rationale for future reference

**Important:** Never store sensitive information (API keys, credentials, passwords, personal data) in vipune. Memories are persisted locally in plaintext and should contain only project context, decisions, and non-sensitive learnings.

## Tools

### Claude Code

Claude Code is the official Anthropic CLI with native integration for AI coding tasks.

**Config file:** `CLAUDE.md` at project root (takes precedence), OR `.claude/CLAUDE.md`, OR `~/.claude/CLAUDE.md` (global)

Create the file with:

```markdown
# Claude Code Configuration

You have access to the `vipune` CLI for persistent memory management across tasks.

## Using vipune for memory

Before starting significant work, search for relevant context:
- `vipune search "relevant topic"` returns ranked memories as plain text

After completing work, store important findings:
- `vipune add "specific discovery or decision"` stores a memory
- `vipune list` shows recent memories
- `vipune --help` displays all available commands

**MCP tools** are also available (via `vipune mcp`):
- `store_memory(text, metadata?, memory_type?, status?, supersedes?, force?)` — store with optional type/status/supersede
- `search_memories(query, limit?, memory_types?, statuses?, recency_weight?, hybrid?, no_touch?)` — find by meaning
- `list_memories(limit?, memory_types?, statuses?)` — list recent
- `supersede_memory(new_text, old_memory_id, memory_type?, metadata?)` — replace existing memory
- `get_memory(id, no_touch?)` — retrieve a memory by ID
- `delete_memory(id)` — permanently delete a memory
- `update_memory(id, text?, metadata?, memory_type?, status?)` — update content, metadata, type, or status

Keep entries focused: one atomic fact per memory for better retrieval.

This memory persists across sessions and is scoped to your git project.
```

**Allowing vipune to run:** Pre-approve vipune commands with Claude Code's `--allowedTools` flag:
```bash
claude --allowedTools "Bash(vipune search *)" "Bash(vipune add *)"
```
Or allow user approval on first use (Claude will prompt before running vipune commands).

Use `CLAUDE.local.md` for personal configuration (automatically gitignored). Remember: never store API keys, credentials, or secrets in vipune memories.

### Claude Desktop on macOS (MCP)

Claude Desktop is the native macOS application for Claude. It integrates with local vipune via the MCP server (`vipune mcp`) — this is the only viable Desktop integration path since ZIP-uploaded skills run in a hosted sandbox without access to the local vipune binary or `~/.vipune/` directory.

**Config file:** `~/Library/Application Support/Claude/claude_desktop_config.json`

Add this configuration block to enable the vipune MCP server:

```json
{
  "mcpServers": {
    "vipune": {
      "command": "/Users/<you>/.cargo/bin/vipune",
      "args": ["mcp"],
      "env": {
        "VIPUNE_DATABASE_PATH": "/Users/<you>/.vipune/memories.db"
      }
    }
  }
}
```

If you installed the prebuilt binary to `/usr/local/bin/`, use this path instead:
```json
{ "command": "/usr/local/bin/vipune", "args": ["mcp"] }
```

Replace `<you>` with your actual username. Do NOT use `~` in absolute paths — Desktop expands from launch directory, not your home.

**Available MCP tools:**
- `store_memory` — store information for later recall
- `search_memories` — find memories by meaning (supports `hybrid` param for semantic + BM25)
- `list_memories` — list recent memories
- `supersede_memory` — replace an existing memory with new content
- `get_memory` — retrieve a memory by ID
- `delete_memory` — permanently delete a memory
- `update_memory` — update a memory's content, metadata, type, or status

**Environment variables** (set in the `env` block):
- `VIPUNE_DATABASE_PATH` — SQLite database location (default: `~/.vipune/memories.db`)
- `VIPUNE_MODEL_CACHE` — Model download cache directory (default: `~/.vipune/models`)
- `VIPUNE_PROJECT` — Project identifier override (auto-detected from git by default)
- `VIPUNE_EMBEDDING_MODEL` — HuggingFace model ID (default: `BAAI/bge-small-en-v1.5`)
- `VIPUNE_SIMILARITY_THRESHOLD` — Conflict detection threshold, 0.0-1.0 (default: `0.85`)
- `VIPUNE_RECENCY_WEIGHT` — Recency bias in search results, 0.0-1.0 (default: `0.3`)
- `VIPUNE_HYBRID` — Enable hybrid search (semantic + BM25), true/false or 1/0

**macOS gotchas:**

- **Absolute paths required** — Claude Desktop launches from the Dock and does NOT inherit your shell PATH. Use the full binary path (e.g., `/Users/<you>/.cargo/bin/vipune`, not just `vipune`).

- **Env vars not inherited** — Desktop does not load `.zshrc`, `.bashrc`, or shell environment. Put all `VIPUNE_*` variables you need in the `env` block of the config.

- **Gatekeeper quarantine** — Downloaded binaries may be quarantined by macOS. If Desktop fails to launch vipune, run:
  ```bash
  xattr -d com.apple.quarantine /Users/<you>/.cargo/bin/vipune
  ```

- **Executable bit** — Ensure the binary is executable:
  ```bash
  chmod +x /Users/<you>/.cargo/bin/vipune
  ```

**Verification:**

1. Fully quit Claude Desktop (⌘Q, not just close window)
2. Restart Claude Desktop
3. Confirm the tools/hammer icon appears in the UI (MCP indicator)
4. Check logs for MCP server issues: `~/Library/Logs/Claude/mcp*.log`

**Surface split:**

- **Claude Desktop on macOS** → Use MCP (this guide). The Desktop app's MCP integration gives direct access to your local vipune binary and `~/.vipune/` directory.
- **Claude Code (CLI)** → Use the skill at `~/.claude/skills/vipune/SKILL.md` (see [Using SKILL.md](#using-skillmd) below). Desktop ZIP skills are not applicable to vipune — they run in a hosted sandbox without filesystem access.

Remember: never store API keys, credentials, or secrets in vipune memories.

### Cursor

Cursor is a modern IDE with AI-powered agent capabilities.

**Config file:** `.cursor/rules/vipune.mdc`

Create the file with:

```markdown
---
alwaysApply: true
---

# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management.

## Using vipune

Before starting work, search for relevant context:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After completing work, store important findings:
- `vipune add "specific discovery"` - stores a memory with automatic conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep entries atomic: one discovery per memory for optimal retrieval.

Memories are scoped to the git project and persist across sessions.
```

**Allowing vipune to run:** Shell command execution is available in Agent mode (Cmd+I). Chat mode behavior may vary depending on your Cursor version and configuration. In Cursor settings, vipune commands are auto-run if your Cursor environment has "Auto run code" enabled, otherwise you'll approve each command manually.

The deprecated `.cursorrules` file is not used; use the `.cursor/rules/` directory structure instead. Remember: never store API keys, credentials, or secrets in vipune memories.

### Windsurf

Windsurf is an AI IDE with advanced agent capabilities.

**Config file:** `.windsurf/rules/vipune.md`

Create the file with:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory across tasks.

## Using vipune

Before starting work, search relevant memories:
- `vipune search "topic"` - semantic search returning ranked results as plain text

After completing work, store findings:
- `vipune add "specific discovery"` - stores with automatic conflict detection
- `vipune list` - lists recent memories
- `vipune --help` - shows all commands

Keep memories atomic: one discovery per entry for better retrieval.

Memories are project-scoped and persist across sessions.
```

**Allowing vipune to run:** Set the rule to "Always On" via Windsurf Settings (Settings → Agent Rules → vipune.md → Always On). Also add `vipune` to the cascade commands allow list in Windsurf Settings (search "cascade" or "allow" in settings) to enable autonomous execution. Setting names may vary by Windsurf version — if the exact path differs, search for related terms in Windsurf Settings to locate the command allowlist configuration.

Settings menu structure may vary by Windsurf version. Consult current Windsurf documentation if these paths don't match. Note: Windsurf may enforce character limits on rule files — keep instructions concise. Remember: never store API keys, credentials, or secrets in vipune memories.

### Cline

Cline is a popular VS Code extension (58k+ stars) for autonomous coding tasks.

**Config file:** `.clinerules/vipune.md`

Create the file with:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management.

## Using vipune

Before starting work, search for context:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After work, store discoveries:
- `vipune add "specific discovery"` - stores memory with conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep entries atomic: one discovery per memory.

Memories are project-scoped and persistent.
```

**Allowing vipune to run:** In Cline Settings → Auto Approve, enable command execution for safe/read-only commands. vipune search and vipune add are non-destructive and will be treated as safe. Alternatively, approve each command manually when Cline prompts.

Remember: never store API keys, credentials, or secrets in vipune memories.

### Roo Code

Roo Code is a VS Code extension (22k+ stars) and community fork of Cline with extended capabilities.

**Config file:** `.roo/rules/vipune.md`

Create the file with:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management.

## Using vipune

Before starting work, search relevant memories:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After work, store findings:
- `vipune add "specific discovery"` - stores with automatic conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep memories atomic: one discovery per entry.

Memories are project-scoped and persistent across sessions.
```

**Allowing vipune to run:** Add to VS Code `settings.json`:
```json
"roo-cline.allowedCommands": ["vipune search", "vipune add", "vipune list"]
```

Roo Code can read `AGENTS.md` from your workspace root automatically (setting `roo-cline.useAgentRules`, enabled by default). This allows you to place vipune instructions there instead of creating a separate `.roo/rules/` file. If both `.roo/rules/vipune.md` and `AGENTS.md` exist, `.roo/rules/` takes precedence.

Remember: never store API keys, credentials, or secrets in vipune memories.

### GitHub Copilot

GitHub Copilot in VS Code provides code suggestions and chat, but cannot execute shell commands directly.

**Config file:** `.github/copilot-instructions.md`

Create the file with:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management. The user will execute commands you suggest.

## Using vipune

For semantic memory search:
- Suggest: `vipune search "topic"`

For storing discoveries:
- Suggest: `vipune add "specific discovery"`

For listing memories:
- Suggest: `vipune list`

For help:
- Suggest: `vipune --help`

Keep memory entries atomic: one discovery per entry for optimal retrieval.
```

**Important caveat:** GitHub Copilot cannot execute shell commands — it only suggests them. You must run vipune commands manually. This configuration tells Copilot about vipune so it can suggest appropriate commands in its responses.

Remember: never store API keys, credentials, or secrets in vipune memories.

### Goose

Goose (by Block) is an autonomous CLI agent that executes shell commands.

**Config file:** `.goosehints` at project root (local), OR `~/.config/goose/.goosehints` (global)

Create the file with:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory across tasks.

## Using vipune

Before starting work, search for relevant context:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After completing work, store important findings:
- `vipune add "specific discovery"` - stores a memory with automatic conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep entries atomic: one discovery per memory for better retrieval.

Memories are scoped to the git project and persist across sessions.
```

**Allowing vipune to run:** Goose executes shell commands autonomously when running in Autonomous mode — no extra setup needed. If you want to customize which files Goose reads, set the `CONTEXT_FILE_NAMES` environment variable.

Goose also reads `AGENTS.md` automatically if present.

Remember: never store API keys, credentials, or secrets in vipune memories.

### Aider

Aider is a CLI tool for pair programming with LLMs.

**Config file:** `CONVENTIONS.md` (community convention)

For one-time use, load with: `aider --read CONVENTIONS.md`

For persistent configuration (recommended), create `.aider.conf.yml` in your home directory or git repository root with:
```yaml
read: CONVENTIONS.md
```

The config file approach eliminates the need to pass `--read` on every invocation.

Create `CONVENTIONS.md` with:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management.

## Using vipune

Before starting work, search for context:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After work, store discoveries:
- `vipune add "specific discovery"` - stores memory with conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep entries atomic: one discovery per memory.

Memories are project-scoped and persistent.
```

**Allowing vipune to run:** Aider executes commands autonomously via `/run` in chat — no separate permission model. Once configured, you can call vipune commands directly in chat.

Remember: never store API keys, credentials, or secrets in vipune memories.

### OpenCode

OpenCode (by SST) is a web-based IDE and development platform.

**Config file:** `.opencode/agents/vipune-instructions.md` or configure in `opencode.json`

You can configure vipune instructions in two ways:

**Option 1: Auto-discovered files** - Create `.opencode/agents/vipune-instructions.md`:

```markdown
---
alwaysApply: true
---

# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management.

## Using vipune

Before starting work, search for context:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After work, store discoveries:
- `vipune add "specific discovery"` - stores with automatic conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep memories atomic: one discovery per entry.

Memories are project-scoped and persistent.
```

**Option 2: Explicit configuration** - Add to `opencode.json`:
```json
{
  "instructions": [".opencode/agents/vipune-instructions.md"]
}
```

Files in `.opencode/agents/` are auto-discovered by OpenCode. Use `opencode.json` to explicitly point to instruction files in other locations outside the auto-discovery directory.

**Allowing vipune to run:** OpenCode executes shell commands in agent mode — configure which commands are auto-run in your workspace settings.

Remember: never store API keys, credentials, or secrets in vipune memories.

### Zed

Zed is a high-performance code editor with AI capabilities.

**Config file:** `.rules` at project root (or `.cursorrules`, `CLAUDE.md`, `AGENTS.md`)

Zed reads configuration files in this order of precedence: `.rules` → `.cursorrules` → `CLAUDE.md` → `AGENTS.md`. Create whichever file makes sense for your project:

```markdown
# vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory management.

## Using vipune

Before starting work, search for context:
- `vipune search "topic"` - semantic search, returns ranked results as plain text

After work, store discoveries:
- `vipune add "specific discovery"` - stores with automatic conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep memories atomic: one discovery per entry.

Memories are project-scoped and persistent.
```

**Allowing vipune to run:** Zed's AI assistant can suggest shell commands, but execution depends on your Zed configuration and the agent's permission level. Commands may require manual approval.

Zed's AI configuration and precedence order may vary by version. Consult current Zed documentation if behavior doesn't match the configuration above.

Remember: never store API keys, credentials, or secrets in vipune memories.

## Using AGENTS.md

Some tools automatically read `AGENTS.md` from your project root as a default configuration source. This allows you to define agent instructions once and have multiple tools recognize them:

**Tools that auto-read AGENTS.md:**
- Roo Code (enabled by default; configure with `roo-cline.useAgentRules`)
- Goose (reads automatically; customize with `CONTEXT_FILE_NAMES` env var)
- Zed (reads as fallback after checking `.rules`, `.cursorrules`, and `CLAUDE.md`)

**Using AGENTS.md instead of tool-specific files:**

Create `AGENTS.md` at your project root with a section for vipune:

```markdown
# Agent Configuration

## vipune Memory Integration

You have access to the `vipune` CLI tool for persistent memory.

### Using vipune

Before starting work, search for context:
- `vipune search "topic"` - semantic search

After work, store discoveries:
- `vipune add "specific discovery"` - stores with conflict detection
- `vipune list` - shows recent memories
- `vipune --help` - displays all commands

Keep memories atomic.
```

This approach is useful if you're using multiple agents in the same project — a single `AGENTS.md` file becomes the source of truth for tool integration without duplicating instructions across tool-specific config files.

**Important:** If you use `AGENTS.md`, avoid also creating tool-specific config files like `.cursor/rules/vipune.mdc` or `.roo/rules/vipune.md` in the same project — this can lead to duplicate or conflicting instructions being applied. Use one approach per project.

## Using SKILL.md

Some Claude-compatible agent systems (including Claude Code and Pi) support **auto-discoverable skill files** in standardized locations. A SKILL.md file at `~/.claude/skills/<skill-name>/SKILL.md` (global) or `<project>/.claude/skills/<skill-name>/SKILL.md` (project-scoped) is automatically loaded when the agent detects the specified skill name in its configuration or prompt.

**vipune provides a software-development-tuned skill artifact** at `skills/vipune/SKILL.md` in this repository. This skill extends the generic vipune instructions with domain-specific patterns for software development:

- Issue/PR linkage via `--metadata` flag for traceability
- Failed-approach tracking (observation type with experiment metadata)
- Pre-flight quality-gate gotcha checks before running tests/linting
- Dev-loop phase→action mapping (read-issue → implement → test → commit)
- Lightweight ADR capture (storing decision rationale, not just what)

This SKILL.md is designed for **Claude and Pi agent systems** that support the skill auto-discovery convention. It includes YAML frontmatter with the skill name and description for automatic recognition.

### Installing the vipune skill

**Tier-1: Manual copy (current, works across all Claude-compatible agents)**

```bash
mkdir -p ~/.claude/skills/vipune && \
  curl -fsSL --connect-timeout 10 --max-time 30 https://raw.githubusercontent.com/randomm/vipune/main/skills/vipune/SKILL.md \
  -o ~/.claude/skills/vipune/SKILL.md
```

_The skill becomes available at this URL once this change is merged to `main`._

This places the skill in the global skills directory where Claude/Pi agents will auto-discover it. Use `<project>/.claude/skills/` instead of `~/.claude/skills/` for project-scoped installation.

**Tier-2 and Tier-3 are out of scope** (future follow-ups):
- Tier-2: Automated install via `vipune skill install` subcommand
- Tier-3: Cross-agent reach via AGENTS.md snippets compatible with non-Claude agents
- These require additional CLI infrastructure and cross-tool standardization work

**Skill directory convention** (for Claude/Pi agents):
- Global: `~/.claude/skills/<skill-name>/SKILL.md`
- Project-scoped: `<project>/.claude/skills/<skill-name>/SKILL.md`
- Each skill lives in its own directory with the descriptive name
- The skill's frontmatter (`name: vipune`) enables agent auto-discovery

The `skills/vipune/SKILL.md` artifact in this repository is the canonical source — install from there to get the latest enhancements.
