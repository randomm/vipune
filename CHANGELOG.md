# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.9.0] - 2026-08-07

### Bug Fixes

- *(#149)* Honour loaded config in MCP server sessions ([#175](https://github.com/randomm/vipune/pull/175))

  `run_mcp` rebuilt its `Config` from only `embedding_model` and `database_path` with `..Config::default()`, so `similarity_threshold`, `recency_weight` and `hybrid` silently reverted to defaults for every MCP session. MCP now honours the same file- and environment-loaded configuration a CLI invocation does.

  Note: `model_cache` is carried through the mapping but remains inert — nothing outside `config/` reads it. Making it functional is tracked in #148.

### ⚠ Breaking Changes

- `vipune::mcp::server::run_mcp` now takes `(config: Config, project_id: &str)` instead of `(embedding_model: String, project_id: &str, db_path: PathBuf)`. Library consumers calling `run_mcp` directly must pass a fully-loaded `Config` (e.g. from `Config::load()`). CLI and MCP users are unaffected.


## [0.8.0] - 2026-08-05

### Features

- Add project_id scoping to update() SQL query for defense-in-depth ([#172](https://github.com/randomm/vipune/pull/172))

### Miscellaneous

- Realign version to 0.7.1 after accidental revert ([#155](https://github.com/randomm/vipune/pull/155)) ([#170](https://github.com/randomm/vipune/pull/170))


## [0.7.1] - 2026-08-04

### Features

- *(#158)* project merge command to repair fragmented project_ids ([#165](https://github.com/randomm/vipune/pull/165))
- *(#158)* doctor --projects to detect fragmented project_ids ([#166](https://github.com/randomm/vipune/pull/166))

### Bug Fixes

- *(#158)* stop new project_id splits ([#162](https://github.com/randomm/vipune/pull/162))
- 4 #[ignore]d tests in src/embedding.rs fail ([#159](https://github.com/randomm/vipune/pull/159))

## [0.7.0] - 2026-08-02

### Features

- *(#147)* Remove mock-embedding fallback and add doctor/reindex repair ([#151](https://github.com/randomm/vipune/pull/151))
- *(#139)* Bundle software-dev-tuned vipune skill artifact ([#140](https://github.com/randomm/vipune/pull/140))

### Bug Fixes

- Stop MCP tests loading a real ONNX model ([#154](https://github.com/randomm/vipune/pull/154))
- *(#148)* Correct air-gapped model revision and cache path in docs ([#151](https://github.com/randomm/vipune/pull/151))

### Documentation

- *(#142)* Add Claude Desktop (macOS) MCP setup guide ([#143](https://github.com/randomm/vipune/pull/143))

### Upgrade notes

**Run `vipune reindex` after upgrading.**

Before this release, `vipune add` stored a hash-derived placeholder vector whenever the embedding model was not loaded, while `vipune search` queried with a real one. Cosine similarity between those two spaces is noise, so semantic search returned effectively random results for CLI users. This has been the case since v0.2.1.

This release stops new writes being corrupted, but rows already in your database keep their placeholder vectors. To check and repair:

```bash
vipune doctor --embeddings   # report real / mock / unknown rows per project
vipune reindex               # re-embed the mock rows
```

`reindex` is idempotent and leaves already-correct rows byte-identical. Rows it cannot classify are reported and skipped rather than overwritten.

**Expect more conflict detections.** With real vectors, paraphrases can exceed the default `similarity_threshold` of 0.85, whereas placeholder vectors only ever matched exact duplicates. Seeing more conflicts on ingest is expected behaviour, not a regression. Lower `similarity_threshold` if it is too aggressive for your corpus.

**Air-gapped installs**: the documented pre-download procedure named the wrong model revision, so vipune missed the cache and attempted a network download. That is corrected in this release — re-run the `huggingface-cli download` command from the README, which now specifies the pinned revision.

## [0.6.0] - 2026-05-15


## [0.5.0] - 2026-05-01

### Added

- Add retrieval_count and last_retrieved_at telemetry (#118)
- Add public MemoryStore::supersede method (#113)

### Fixed

- Update embedding model revision to main branch (#117)

### Changed

- Refactor oversized files to meet 500-line limit (#120)
- Use MemoryType/MemoryStatus enums at API boundaries (#119)

### Documentation

- Fix documentation drift from v0.3.0 changes (#114)

### Dependencies

- Bump actions/upload-artifact from 6 to 7
- Bump actions/download-artifact from 7 to 8
- Bump toml from 0.8.23 to 1.0.6+spec-1.1.0
- Upgrade cargo-dist to v0.31.0 (#123)


## [0.4.0] - 2026-04-28

### Fixed

- Fix lifecycle bugs, search consistency, and hybrid config (#105)


## [0.3.0] - 2026-04-27

### Added

- Add memory type, lifecycle status, and atomic supersede (#97)
- Extend update command to support metadata changes (#95)
- Include metadata and project_id in MCP search results (#93)

### Fixed

- Fail-fast on over-length content instead of silent truncation (#92)

### Changed

- Add schema migration framework for SQLite (#91)

### Documentation

- Update documentation for v0.3 API changes (#98)


## [0.2.6] - 2026-04-18

### Added

- Pin embedding model revision and expose public constants (#84)


## [0.2.5] - 2026-04-12

### Fixed

- Keep MCP server alive until client disconnects (#80)


## [0.2.4] - 2026-04-11

### Changed

- Make MCP a default feature (#76)


## [0.2.3] - 2026-04-10

### Added

- Add vipune mcp subcommand with MCP server (#74)


## [0.2.2] - 2026-04-09

### Added

- Add shell installer and fix macOS tar extraction bug (#70)


## [0.2.1] - 2026-03-27

### Added

- Add batch ingest API with per-item outcomes (#68)
- Add ergonomic ingest API to MemoryStore (#65)
- Add list_since and get_many read helpers (#66)

### Documentation

- Audit and evolve project documentation (#59)

## [0.2.0] - 2026-03-24

### Added

- Expose Memory.embedding field and add MemoryStore::list() (#54)

### Changed

- Downgrade ort from 2.0.0-rc.11 to 2.0.0-rc.9 (#56)


## [0.1.9] - 2026-03-09

### Fixed

- Add download-binaries feature to ort dependency (#46)


## [0.1.8] - 2026-03-02

### Changed

- Upgrade to Rust edition 2024 with MSRV 1.85 (#40)


## [0.1.7] - 2026-03-02

### Fixed

- MacOS install instructions and code block copy-paste UX


## [0.1.6] - 2026-03-01

### Fixed

- Correct install instructions to use .tar.xz instead of .tar.gz


## [0.1.5] - 2026-02-28

### Fixed

- Lazy embedding init to prevent flaky tests on HF 404
- Keep all tests active, cache HF models in CI instead of #[ignore]

### Documentation

- Replace hardcoded version tag with cargo install from crates.io


## [0.1.4] - 2026-02-27

### Fixed

- Upgrade ort from rc.10 to rc.11 and fix inputs() API (#18)
- Pin macOS runner to macos-14 for ort rc.11 Xcode 15.4 compatibility (#25)
- Switch CI to Ubicloud Ubuntu 24.04 runners (#22)


## [0.1.3] - 2026-02-26

### Fixed

- Upgrade rusqlite from 0.32 to 0.38 (#17)


## [0.1.2] - 2026-02-21

### Added

- Add lib.rs public API, security hardening, and integration tests (#11)

### Fixed

- Add release-pr job to release-plz workflow (#14)


## [0.1.1] - 2026-02-19

### Added

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
