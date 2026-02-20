# Agent Integration

vipune is a generic CLI tool for persistent memory that works with any agent capable of running shell commands. Integration requires two steps: (1) tell the agent about vipune by adding a configuration snippet, and (2) grant permission to execute shell commands. Once configured, agents can use `vipune search` and `vipune add` to maintain a knowledge loop across tasks without plugins, API keys, or additional setup.

The knowledge loop is simple: before starting work, search for relevant context; execute the work; then store important discoveries for future recall. This pattern helps agents avoid redundant work, maintain consistency, and build cumulative knowledge over time.

## The knowledge loop

Use vipune to create a feedback cycle in your agent workflow:

1. **Search before starting** - `vipune search "relevant topic"` to recall context from past work
2. **Do the work** - Execute your task with the knowledge in mind
3. **Store learnings** - `vipune add "important discovery"` after completing meaningful work

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

Keep entries focused: one atomic fact per memory for better retrieval.

This memory persists across sessions and is scoped to your git project.
```

**Allowing vipune to run:** Pre-approve vipune commands with Claude Code's `--allowedTools` flag:
```bash
claude --allowedTools "Bash(vipune search *)" "Bash(vipune add *)"
```
Or allow user approval on first use (Claude will prompt before running vipune commands).

Use `CLAUDE.local.md` for personal configuration (automatically gitignored). Remember: never store API keys, credentials, or secrets in vipune memories.

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
