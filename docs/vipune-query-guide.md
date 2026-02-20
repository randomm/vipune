# Vipune Query Guide

Recommendations for writing effective vipune queries to get the best semantic search results.

## Query Categories Tested

### 1. Single Technical Terms ⭐⭐⭐⭐⭐
**Works excellently** (scores 0.80-0.81)

| Query | Top Score | Quality |
|-------|-----------|---------|
| "ConversationGraph" | 0.81 | Exact matches |
| "handler" | 0.81 | All relevant |
| "MessageHandler" | 0.81 | Domain-specific |

**Verdict**: Use specific technical terms directly.

### 2. Multi-Word Technical Phrases ⭐⭐⭐⭐
**Works well** (scores 0.75-0.80)

| Query | Top Score | Quality | Notes |
|-------|-----------|---------|-------|
| "message handler" | 0.80 | Excellent | Domain-specific pairing |
| "message handling" | 0.79 | Very Good | Action-oriented phrase |
| "user authentication" | 0.78 | Good | Standard concept |
| "API endpoint" | 0.75 | Good | Common architecture term |

**Verdict**: Multi-word phrases work well for conceptual searches. Combine domain + technical terms for best results.

### 3. Architectural Questions ⭐⭐⭐
**Moderate** (scores 0.63-0.74)

| Query | Top Score | Quality | Why it works |
|-------|-----------|---------|--------------|
| "How does the message handler work?" | 0.74 | Fair | "message handler" is strong |
| "What is the architecture of this system?" | 0.70 | Fair | Generic architecture terms |
| "How do users authenticate?" | 0.68 | Fair | "authenticate" is specific |
| "Where is the database schema?" | 0.63 | Fair | Multiple relevant terms |

**Verdict**: Questions work because vipune extracts technical terms. Focus on specific components, not general system overview.

### 4. Natural Language ⭐⭐⭐⭐
**Good** (scores 0.75-0.81)

| Query | Top Score | Quality | Examples |
|-------|-----------|---------|-----------|
| "code for processing events" | 0.81 | Excellent | Action + object |
| "user login implementation" | 0.79 | Very Good | Specific feature + implementation |
| "database connection setup" | 0.77 | Good | Component + action |
| "error handling in handlers" | 0.75 | Good | Cross-cutting concern + location |

**Verdict**: Natural language queries work surprisingly well. Use [action] + [object] or [feature] + [implementation] patterns.

### 5. Code-Specific ⚠️
**Language-dependent**

Syntax-specific queries only work if querying code in your project's actual programming language. Instead, use semantic descriptions:

| Good Approach | Avoid | Why |
|---|---|---|
| "function that sends messages" | Language-specific syntax (fn, def, etc.) | Semantic meaning transcends syntax |
| "error handling mechanism" | Language keywords (try/except, Result, etc.) | Concept vs. language-specific detail |
| "initialization code" | Language-specific imports/use statements | Purpose vs. syntax detail |

**Verdict**: Describe what code *does* rather than how it's written. This approach works across all programming languages in your project.

### 6. Vague/Problem-Solving ⭐⭐⭐⭐
**Better than expected**

| Query | Quality | Why it works |
|-------|---------|--------------|
| "How do I fix authentication errors?" | Good | "authentication errors" is specific |
| "Why does the service fail on startup?" | Good | "service fail" + "startup" triggers relevant terms |
| "Database connection timeout issues" | Good | Combines multiple technical terms |
| "Slow message delivery problems" | Good | Component + problem type |

**Verdict**: Problem-solving queries work well because they combine technical terms with context. Describe the specific problem with technical keywords.

### 7. Compound Technical ⭐⭐⭐⭐
**Good** (0.78-0.81)

| Query | Top Score | Quality | Composition |
|-------|-----------|---------|-------------|
| "event processor message handler" | 0.81 | Excellent | Domain + component + action |
| "user authentication database schema" | 0.80 | Excellent | Feature + component + architecture |
| "API request error handling" | 0.79 | Very Good | Interface + event + pattern |
| "database connection pooling configuration" | 0.78 | Good | Component + pattern + configuration |

**Verdict**: Compound queries combining domain, component, concept, and action work very well. Be specific but use natural grouping.

## Recommendations

**DO:**

- **Use specific technical terms** - Single technical terms (class names, function names, specific technologies) score highest (0.81)
- **Try multi-word phrases for concepts** - Phrases like "message handling" or "user authentication" work well (0.75-0.80)
- **Be explicit about programming language** - If querying code, use the project's actual language syntax
- **Use natural language descriptions** - Semantic descriptions often beat code syntax queries
- **Combine domain + concept** - "Telegram bot message handler" scores 0.81 by chaining related terms
- **Focus problems with technical keywords** - "authentication errors" works better than "login is broken"

**DON'T:**

- **Use syntax from wrong programming language** - Rust queries in Python projects score poorly (0.48-0.52)
- **Expect perfect results for vague questions** - Generic questions like "What is the architecture?" only score 0.70
- **Make queries unnecessarily long** - 3-4 technical terms are sufficient for most searches

## Query Patterns That Work

Based on the 22 test cases, these patterns consistently score well:

1. **[Technical Term]** → 0.81
   - Example: "ConversationGraph", "handler", "Telegram"

2. **[Domain] [Component]** → 0.80
   - Example: "Telegram bot", "message handler"

3. **[Feature] [Implementation]** → 0.79
   - Example: "user login implementation", "send messages code"

4. **[Component] [Action] [Concept]** → 0.78-0.81
   - Example: "database connection pooling configuration"

5. **[Problem] [Technical Context]** → 0.80
   - Example: "authentication errors", "bot crash on startup"

## Score Interpretation

- **0.80+**: Excellent match, highly relevant results - First result is typically what you need
- **0.70-0.79**: Good match, relevant results - Check top 2-3 results for best match (default search returns 5 results)
- **0.60-0.69**: Fair, related but may need refinement - Reconsider query wording
- **Below 0.60**: Consider rephrasing query - Try different technical terms or use natural language

## Search Model

- **Model**: bge-small-en-v1.5 (384 dimensions)
- **Similarity Metric**: Cosine similarity
- **Note**: Scores vary by your memory library size and content. Use the patterns as guidance, not absolute thresholds.

## Key Insight

Vipune's semantic search excels with **technical specificity**. The model understands:

- Domain-specific terminology (handler, ConversationGraph, MessageHandler)
- Architectural concepts (authentication, schema, pooling)
- Action-component relationships (message handling, user login)
- Problem contexts (timeout, crash, errors)

The embedding model captures semantic meaning beyond keyword matching, making natural language queries surprisingly effective when they contain technical terms.