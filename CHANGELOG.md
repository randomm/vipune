# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.1] - 2026-02-19

### Features

- Semantic memory storage with vector embeddings (BAAI/bge-small-en-v1.5)
- SQLite backend with schema, CRUD, and cosine similarity search
- Conflict detection: flags similar memories on add (configurable threshold)
- Project auto-detection from git repository context
- Config system with TOML file and environment variable overrides
- Recency scoring: time-weighted search result ranking
- Hybrid search with BM25 full-text and semantic re-ranking
- JSON output for all commands (machine-readable for agent integration)
- CLI commands: add, search, get, list, delete, update, version

### Platform Support

- macOS ARM64
- Linux x86_64
- Linux ARM64

[0.1.1]: https://github.com/randomm/vipune/releases/tag/v0.1.1
