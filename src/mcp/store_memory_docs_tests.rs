//! Drift tests for the documented `store_memory` signature (issue #256).
//!
//! `docs/agent-integration.md` spells the tool's parameters out as a signature
//! line and `docs/cli-reference.md` lists them per tool. Both live outside
//! `src/`, so nothing compiles them: when `StoreMemoryParams` gained
//! `importance`, an agent reading the docs had no way to learn the parameter
//! existed and every MCP-written memory stayed at `medium` — invisible to
//! `prune` protection. Same drift class as
//! `test_air_gapped_docs_and_ci_cache_match_embed_revision` in
//! `src/embedding/tests/model_free.rs`, which pins the README's pinned
//! revision to the code constant.
//!
//! Model-free: reading two markdown files needs no embedding session.

use crate::memory::lifecycle::MemoryImportance;

/// The documented parameter names of the `store_memory` tool in
/// `docs/agent-integration.md`, e.g. `text` / `importance?` from
/// `store_memory(text, metadata?, importance?, …)`.
fn documented_store_memory_params(doc: &str) -> Vec<String> {
    let line = doc
        .lines()
        .find(|line| line.contains("`store_memory("))
        .expect("agent-integration.md documents the store_memory signature");
    let params = line
        .split('(')
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("the store_memory signature has a parenthesised parameter list");
    params
        .split(',')
        .map(|param| param.trim().trim_end_matches('?').to_string())
        .collect()
}

/// Every field of `StoreMemoryParams` that the docs are supposed to name.
///
/// Kept as a literal list rather than derived from the struct: the struct is
/// the thing the schema is generated from, so the list is the independent
/// statement the docs are checked against.
const EXPECTED_PARAMS: [&str; 7] = [
    "text",
    "metadata",
    "memory_type",
    "status",
    "importance",
    "supersedes",
    "force",
];

/// The four levels the tool accepts, listed here as the docs are checked
/// against them: `MemoryImportance::from_str` is the one gate both the MCP and
/// the CLI surfaces go through, so this is the independent statement the
/// documentation has to agree with (and the enum has no `VARIANTS` list to
/// borrow — see `test_importance_vocabularies_agree_with_the_read_side` for the
/// sibling hand-maintained roster).
const LEVELS: [&str; 4] = ["low", "medium", "high", "critical"];

/// The agent-facing signature must name every parameter the handler accepts.
///
/// `importance` is the field #256 added; a signature that lost it would leave
/// agents storing every memory at `medium` because the docs never told them the
/// level could be asked for.
#[test]
fn agent_integration_signature_names_every_store_memory_param() {
    let doc = std::fs::read_to_string(repo_file("docs/agent-integration.md"))
        .expect("read docs/agent-integration.md");
    let documented = documented_store_memory_params(&doc);

    for param in EXPECTED_PARAMS {
        assert!(
            documented.iter().any(|d| d == param),
            "docs/agent-integration.md's store_memory signature omits `{param}` — \
             it lists {documented:?}, and an agent that cannot see a parameter will not send it"
        );
    }

    // The line must also say what the level is for, the way it already says
    // what `status` does: a bare name does not tell an agent when to use it.
    let line = doc
        .lines()
        .find(|line| line.contains("`store_memory("))
        .expect("store_memory signature line");
    assert!(
        line.contains("importance") && line.contains("prune"),
        "the signature line must document why importance matters (prune exemption): {line}"
    );
}

/// The per-tool parameter list in the CLI reference must list `importance`
/// under `store_memory`, with the same vocabulary the parser accepts.
#[test]
fn cli_reference_lists_store_memory_importance_with_the_canonical_levels() {
    let doc =
        std::fs::read_to_string(repo_file("docs/cli-reference.md")).expect("read cli-reference.md");

    let section = doc
        .split("`store_memory` accepts:")
        .nth(1)
        .and_then(|rest| rest.split("supersede_memory` accepts:").next())
        .expect("cli-reference.md has a 'store_memory accepts:' section");

    let levels: Vec<&str> = LEVELS
        .iter()
        .filter(|level| section.contains(&format!("`{level}`")))
        .copied()
        .collect();
    assert!(
        section.contains("- `importance`"),
        "cli-reference.md must list `importance` as a store_memory parameter"
    );
    assert_eq!(
        levels.len(),
        LEVELS.len(),
        "the documented level vocabulary must be exactly the levels the parser accepts \
         ({LEVELS:?}), got {levels:?}"
    );
}

/// The tool schema and the CLI's own flag must advertise the same four levels,
/// so an agent and an operator cannot be taught two vocabularies.
///
/// `MemoryImportance::from_str` is the one gate both surfaces go through, so
/// this pins the doc-facing list to that gate rather than to a copy of it.
#[test]
fn documented_importance_levels_are_the_ones_the_parser_accepts() {
    for level in LEVELS {
        let parsed = MemoryImportance::from_str(level)
            .unwrap_or_else(|e| panic!("the documented level {level:?} is rejected: {e}"));
        assert_eq!(parsed.as_str(), level);
        // Case-insensitive on the way in, canonical on the way out — the
        // column must read `high`, never `HIGH` (mirrors the import path's
        // test_import_uppercase_importance_is_stored_canonically).
        assert_eq!(
            MemoryImportance::from_str(&level.to_uppercase())
                .expect("uppercase level")
                .as_str(),
            level,
        );
    }
}

/// A repo-root-relative file, resolved through the manifest dir the way the
/// existing docs-drift tests do.
fn repo_file(rel: &str) -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    std::path::Path::new(&manifest_dir).join(rel)
}
