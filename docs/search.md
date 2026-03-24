# Search Guide

This guide explains vipune's search modes and how to use them effectively.

## Search Modes

vipune supports three search modes, each optimized for different use cases.

### Pure Semantic Search (Default)

**This is the default search mode** — it runs automatically when you call `vipune search` without the `--hybrid` flag.

Semantic search finds memories by meaning, not keywords.

**Use when:**
- Searching for concepts or ideas
- Finding paraphrased content
- Exploratory knowledge retrieval

**Example:**
```bash
# Find memories about authentication concepts
vipune search "how do we handle authentication"

# Semantic search works with natural language
vipune search "database schema for users table"

# Finds content even with different wording
vipune search "user login process"
```

**How it works:**
- Generates embedding for query
- Compares to stored embeddings using cosine similarity
- Returns results ranked by similarity (0.0 to 1.0)

### Hybrid Search

Hybrid search combines semantic similarity and keyword matching for precision.

**Use `--hybrid` when:**
- Searching for keywords, proper nouns, or code identifiers
- Using short queries (1-3 words)
- Looking for exact terms in specific contexts

**Do NOT use `--hybrid` when:**
- Exploring conceptual relationships
- Using long natural language queries
- The meaning matters more than exact words

**Example:**
```bash
# Keyword-heavy query
vipune search "JWT tokens" --hybrid

# Proper noun search
vipune search "PostgreSQL configuration" --hybrid

# Code identifier search
vipune search "MessageHandler" --hybrid

# Short term search
vipune search "auth" --hybrid
```

**How it works:**
- Runs semantic search (cosine similarity)
- Runs keyword search (BM25 via SQLite FTS5)
- Merges results using Reciprocal Rank Fusion (RRF)
- Documents in both lists get boosted scores

### Recency Weighting

Recency weighting favors recent memories without ignoring semantic relevance.

**Use `--recency` (default: 0.3) when:**
- Working on time-sensitive projects
- Searching for recent changes
- Knowledge becomes stale quickly

**Recency weights:**
- `0.0`: Pure semantic similarity (no time bias)
- `0.3`: Default balance (70% semantic, 30% recency)
- `0.5`: Equal weight on meaning and time
- `0.7-1.0`: Strong recency bias (recent first)

**Example:**
```bash
# High recency bias (recent memories rank higher)
vipune search "recent API changes" --recency 0.8

# Default balance (70% semantic, 30% recency)
vipune search "authentication flows"

# Pure semantic search (no time bias)
vipune search "authentication patterns" --recency 0.0

# Equal balance
vipune search "database schema" --recency 0.5
```

**How it works:**
- Final score combines similarity and recency decay
- Formula: `(1 - recency_weight) * similarity + recency_weight * time_score`
- Newer memories get higher time scores, older memories decay

## Score Interpretation

Search results include similarity scores:

### For pure semantic search:

- **0.90+**: Excellent match — content is nearly identical in meaning
- **0.80-0.89**: Very good match — strong semantic relationship
- **0.70-0.79**: Good match — related concepts or paraphrases
- **0.60-0.69**: Fair match — loosely related, may need refinement
- **Below 0.60**: Weak match — consider rephrasing your query

### For hybrid search (with `--hybrid`):

Scores are RRF fused values (no direct semantic meaning). Focus on ranking order, not absolute scores.

### For recency-weighted search:

Scores blend similarity and time decay. Use ranking order to judge relevance.

## Practical Examples

### Scenario 1: Finding Architecture Documentation

```bash
# Pure semantic search best for conceptual searches
vipune search "architecture of message handling system"
```

### Scenario 2: Finding Specific Configuration

```bash
# Hybrid search better for specific terms
vipune search "JWT configuration" --hybrid
```

### Scenario 3: Finding Recent Changes

```bash
# Combine hybrid search with recency bias
vipune search "auth API changes" --hybrid --recency 0.7
```

### Scenario 4: Exploring Related Concepts

```bash
# Pure semantic search finds paraphrased content
vipune search "how users authenticate"
```

### Scenario 5: Finding Code Snippets

```bash
# Hybrid search for code identifiers
vipune search "MessageHandler implementation" --hybrid
```

## Common Failure Modes

### Search Returns No Results

**Possible causes:**
1. Project scope mismatch — you're searching in a different project
2. No memories stored yet
3. Query too unrelated to stored content

**Fixes:**
```bash
# Check project scope
vipune --project "your-project-id" search "query"

# List memories to see what's stored
vipune list

# Try a broader query
vipune search "authentication"  # instead of "JWT token expiration"
```

### Search Returns Irrelevant Results

**Possible causes:**
1. Query too vague
2. Hybrid search needed for keyword-heavy queries
3. Semantic score threshold too high

**Fixes:**
```bash
# Be more specific with technical terms
vipune search "user authentication JWT tokens"

# Try hybrid search for keyword focus
vipune search "auth tokens" --hybrid

# Lower similarity threshold if filtering too aggressively
export VIPUNE_SIMILARITY_THRESHOLD="0.7"
```

### Recent Memories Don't Rank First

**Cause:** Low recency weight

**Fix:**
```bash
# Increase recency weight
vipune search "recent changes" --recency 0.8

# Set globally via config
export VIPUNE_RECENCY_WEIGHT="0.7"
```

### Proper Nouns Not Found

**Cause:** Semantic search may miss exact terms

**Fix:** Use hybrid search for proper nouns and identifiers

```bash
# Semantic search might miss exact term
vipune search "PostgreSQL"

# Hybrid search catches keywords
vipune search "PostgreSQL" --hybrid
```

## Query Tips

### Good Queries

- **Specific technical terms**: "JWT tokens", "MessageHandler"
- **Domain + concept**: "user authentication", "database schema"
- **Action + object**: "handle messages", "validate tokens"
- **Problem + context**: "authentication timeout error"

### Poor Queries

- **Too vague**: "how to do it", "problem"
- **Too long**: unnecessarily verbose descriptions
- **Wrong language**: Rust syntax in Python project
- **Pure keywords without context**: "token", "handler" (ambiguous)

---

For more guidance on writing effective queries, see the [Query Guide](vipune-query-guide.md).

For detailed command-line usage, see the [CLI Reference](cli-reference.md).