# Contributing to vipune

Thank you for your interest in contributing to vipune! This document outlines how to get involved.

## Reporting Bugs

Found a bug? Please report it on GitHub Issues:
https://github.com/randomm/vipune/issues

Include:
- What you were trying to do
- What happened (vs. what you expected)
- Steps to reproduce
- Rust version (`rustc --version`)
- Output of `cargo build` or relevant error messages

## Proposing Features

Have an idea for vipune? Start with a GitHub Issue:
https://github.com/randomm/vipune/issues

**Before coding**: Open an issue and discuss your approach. This ensures alignment and prevents wasted effort on features that may not fit the project's scope. Don't worry about perfect formatting — we'll help you refine the idea!

## Development Setup

### Prerequisites
- Rust stable (install from https://rustup.rs/)
- Git

### Getting Started
```bash
git clone git@github.com:randomm/vipune.git
# (If SSH isn't configured, use: git clone https://github.com/randomm/vipune.git)
cd vipune
cargo build
cargo test
```

### First Contribution

Looking for a good first issue? Check the [GitHub issues](https://github.com/randomm/vipune/issues) and look for labels like `good first issue` or `help wanted`.

**Tips for first-time contributors:**
- Start small: Documentation improvements, test additions, or minor bug fixes
- Ask questions: Comments on issues are welcome — clarifying before coding saves time
- Focus on one thing: Each PR should address a single issue or feature
- Existing patterns: Follow the code style and structure of existing modules

**Suggested first PR size**: 100-300 lines of well-tested code or documentation changes.

### Testing Guidance

For writing tests, see the [Testing Guide](docs/testing.md) for:
- How to run tests (`cargo test`, `cargo test -- --nocapture`)
- Test organization (unit vs integration tests)
- Test utilities (`ENV_MUTEX`, temporary directories)
- Coverage expectations (80%+ for new code)
- What to test vs. what NOT to test

**See also:** [Architecture Documentation](docs/architecture.md) for module structure and design constraints.

## Quality Gates

All code changes must pass these checks before PR submission:

```bash
cargo fmt --check     # Code formatting
cargo clippy -- -D warnings  # Linting (all warnings are errors)
cargo test            # All tests pass
```

Run all checks together:
```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

**These checks must pass locally before pushing.** CI verifies, but does not discover — fix issues locally first.

## Commit Style

vipune uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Tests
- `perf`: Performance improvement
- `chore`: Maintenance

Format:
```
feat(#123): brief description

Optional body with details.
```

Example:
```
fix(#45): correct embedding BLOB size validation

Previously accepted 1600-byte blobs. Now correctly validates 1536 bytes (384 × 4).
```

## Branch Naming

Use this convention for feature branches:
```
feature/issue-{NUMBER}-short-description
```

Example: `feature/issue-12-add-json-output`

## Release Process

Releases are automated via release-plz. When your PR is merged to `main`, the release process runs automatically:

1. **Release-plz scans commits**: Uses conventional commits to determine version bump
2. **Creates Git tag**: Tags the commit with new version number
3. **Generates CHANGELOG.md**: Auto-updates from commit messages
4. **Creates GitHub release**: Draft release with changelog

**Hard rules — do not violate:**

- **NEVER leave a release PR open across subsequent merges to main.** release-plz may not recompute its version or changelog, so the PR can silently go stale. Close and regenerate instead (see issue [#155](https://github.com/randomm/vipune/issues/155)).
- **A release PR is armed — merging it publishes to crates.io irreversibly.** It must never be merged to "close" it. If a release PR needs to be abandoned, close it via the GitHub UI.

**Important:**
- Do NOT manually edit `CHANGELOG.md` — it's auto-generated
- Do NOT manually create Git tags — release-plz handles this
- Ensure your commit messages follow conventional commits format
- Include issue numbers in commits: `feat(#123): description`

If you need to modify an in-progress release:
- Contact maintainers to coordinate manual intervention

## Debugging Tips

### Running with Verbose Output

When investigating issues, verbose output can help:

```bash
# Set RUST_BACKTRACE for error traces
RUST_BACKTRACE=1 vipune add "Test"

# Use --json for structured output
vipune search "query" --json | jq .
```

### Inspecting the SQLite Database Directly

You can inspect vipune's database with sqlite3:

```bash
# Open database (default path)
sqlite3 ~/.vipune/memories.db

# List tables
.tables

# View schema
.schema

# Query memories
SELECT id, content, created_at FROM memories LIMIT 5;

# Count memories per project
SELECT project_id, COUNT(*) FROM memories GROUP BY project_id;

# Search FTS5 index (keyword search)
SELECT rowid, content FROM memories_fts WHERE content MATCH 'authentication';

# Check FTS5 table exists
SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts';
```

### Common Setup Issues

**Model download fails:**
```bash
# Clear cache and retry
rm -rf ~/.vipune/models/
vipune add "Test"
```

**Database locked:**
```bash
# Ensure no other vipune processes are running
ps aux | grep vipune
# Close other terminals running vipune
```

**Configuration not applied:**
```bash
# Check config file syntax
cat ~/.config/vipune/config.toml

# Check environment variables
env | grep VIPUNE_
```

**Project detection wrong:**
```bash
# Explicit project override
vipune --project "my-project" add "Test"

# Current project detection
vipune add "Test"  # Use --project if wrong project detected
```

## Pull Request Process

1. Create a feature branch from `main` (see naming above)
2. Make your changes and commit with conventional commits
3. Push to your fork
4. Open a PR against `randomm/vipune`
5. Ensure CI passes (all checks green)
6. Address any code review comments
7. PR is squash-merged to main

## Code of Conduct

Be respectful of others. Harassment, discrimination, and hostile behavior are not tolerated.

---

Questions? Open an issue on GitHub:
https://github.com/randomm/vipune/issues
