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
