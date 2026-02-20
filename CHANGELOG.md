# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

## [0.1.1] - 2026-02-19

### Bug Fixes

- *(ci)* Update release workflow to build all 4 platforms
- *(ci)* Fix release workflow for 3 platforms
- *(ci)* Use macos-12 runner for Intel builds
- *(ci)* Make binary validation regex more flexible
- *(ci)* Use POSIX glob patterns instead of bash regex
- *(ci)* Correct --prerelease flag expansion in release workflow
- *(ci)* Add checkout step to release job for --generate-notes
- *(ci)* Correct release-plz.toml package table syntax
- *(ci)* Use [workspace] section in release-plz.toml for single-crate project

### CI/CD

- Switch to Ubicloud runners for faster builds
- Drop Intel Mac support, use Ubicloud for ARM64+Linux
- Use mixed runners - GitHub for macOS, Ubicloud for Linux

### Documentation

- Remove migration from remory section from README ([#37](https://github.com/randomm/vipune/pull/37))
- Add temporary remory-to-vipune migration guide ([#38](https://github.com/randomm/vipune/pull/38))

### Miscellaneous

- Initialize dev repo with configuration files
- Add remory as reference submodule
- Add .worktrees/ to .gitignore
- Sync Cargo.toml version to v0.1.1 (already released) ([#49](https://github.com/randomm/vipune/pull/49))


### Features
- *(#9)* Implement data migration from remory databases
- *(#6)* Implement conflict detection for add operation
- *(#5)* Implement CLI interface with clap commands and JSON output
- *(#8)* Implement config system with env vars and TOML file
- *(#7)* Implement project auto-detection from git
- *(#4)* Implement core memory operations wiring embedding and SQLite
- *(#3)* Implement synchronous ONNX embedding engine
- *(#2)* Implement SQLite backend with schema, CRUD, and vector search
- *(#1)* Project scaffolding with Cargo.toml, CI, and minimal CLI

### Documentation
- *(#12)* Create AGENTS.md for vipune project governance
- *(#10)* Add README, CLI reference, and quickstart guide

### Miscellaneous
- Add .worktrees/ to .gitignore
- Add release automation configuration with git-cliff and release-plz