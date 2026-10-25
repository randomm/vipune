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
//! These assert *presence* and *exclusivity* of the vocabulary, not the shape
//! of the prose: rewording a bullet must not fail the build, documenting a
//! level the parser rejects must.
//!
//! Model-free: reading two markdown files needs no embedding session.

use crate::memory::lifecycle::MemoryImportance;

/// The four levels the tool accepts, listed here as the docs are checked
/// against them: `MemoryImportance::from_str` is the one gate both the MCP and
/// the CLI surfaces go through, so this is the independent statement the
/// documentation has to agree with (and the enum has no `VARIANTS` list to
/// borrow — see `test_importance_vocabularies_agree_with_the_read_side` for the
/// sibling hand-maintained roster).
const LEVELS: [&str; 4] = ["low", "medium", "high", "critical"];

/// Every documented level must be a word the parser actually accepts.
///
/// The docs tests below compare markdown against `LEVELS`; this compares
/// `LEVELS` against the one gate, so the roster cannot drift from the parser
/// while both doc tests stay green.
#[test]
fn every_documented_level_is_one_the_parser_accepts() {
    for level in LEVELS {
        let parsed = MemoryImportance::from_str(level)
            .unwrap_or_else(|e| panic!("the documented level {level:?} is rejected: {e}"));
        assert_eq!(parsed.as_str(), level);
    }
}

/// A repo-root-relative file, resolved through the manifest dir the way the
/// existing docs-drift tests do.
fn repo_file(rel: &str) -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    std::path::Path::new(&manifest_dir).join(rel)
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

/// The one signature line for `store_memory` in `docs/agent-integration.md`.
///
/// Anchored on the list-item form rather than a bare substring so a prose
/// mention anywhere earlier in the file ("call `store_memory(text)` first")
/// cannot become the assertion target, and required to be unique so an
/// ambiguous file fails loudly instead of picking the first hit.
fn signature_line(doc: &str) -> String {
    let matches: Vec<&str> = doc
        .lines()
        .filter(|line| line.starts_with("- `store_memory("))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "agent-integration.md must contain exactly one '- `store_memory(' signature line, found {}",
        matches.len()
    );
    matches[0].to_string()
}

/// The parameter names in a `store_memory(…)` signature line, `?` stripped.
fn params_of(line: &str) -> Vec<String> {
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

/// The agent-facing signature must list every parameter the handler accepts,
/// in the order the schema declares them.
///
/// `importance` is the field #256 added; a signature that lost it would leave
/// agents storing every memory at `medium` because the docs never told them the
/// level could be asked for.
#[test]
fn agent_integration_signature_names_every_store_memory_param() {
    let doc = std::fs::read_to_string(repo_file("docs/agent-integration.md"))
        .expect("read docs/agent-integration.md");
    let line = signature_line(&doc);
    let documented = params_of(&line);

    assert_eq!(
        documented, EXPECTED_PARAMS,
        "docs/agent-integration.md's store_memory signature must list the parameters \
         the handler accepts, in order"
    );

    // The line must also say what the level is for, the way it already says
    // what `status` does: a bare name does not tell an agent when to use it.
    assert!(
        line.contains("importance") && line.contains("prune"),
        "the signature line must document why importance matters (prune exemption): {line}"
    );
}

/// The per-tool parameter list in the CLI reference must list `importance`
/// under `store_memory`, with exactly the vocabulary the parser accepts.
#[test]
fn cli_reference_lists_store_memory_importance_with_the_canonical_levels() {
    let doc =
        std::fs::read_to_string(repo_file("docs/cli-reference.md")).expect("read cli-reference.md");

    let section = doc
        .split("`store_memory` accepts:")
        .nth(1)
        .and_then(|rest| rest.split("supersede_memory` accepts:").next())
        .expect("cli-reference.md has a 'store_memory accepts:' section");

    assert!(
        section.contains("- `importance`"),
        "cli-reference.md must list `importance` as a store_memory parameter"
    );

    let documented: Vec<&str> = documented_levels(section);
    assert_eq!(
        documented, LEVELS,
        "the documented level vocabulary must be exactly the levels the parser accepts"
    );
}

/// The level words in the `store_memory` importance bullet — the enumeration
/// that opens the bullet, `low`, `medium` (default), `high`, or `critical`.
/// Extracting the set (rather than filtering `LEVELS` down to what the section
/// mentions) is what makes an extra level fail the test instead of passing
/// silently.
///
/// Only that leading enumeration is read: the bullet goes on to name other
/// backticked things (`supersedes`, `prune`) and to re-mention levels in prose,
/// none of which belong to the vocabulary list. The enumeration is taken to its
/// first word outside the parser's set, so a level the parser rejects —
/// `normal`, `permanent`, anything — is where the read stops, and the equality
/// asserts below are what report it.
fn documented_levels(section: &str) -> Vec<&str> {
    let bullet = section
        .lines()
        .find(|line| line.starts_with("- `importance`"))
        .expect("cli-reference.md lists an `importance` parameter under store_memory");
    let enumeration = bullet
        .split(':')
        .nth(1)
        .expect("the importance bullet enumerates its levels before qualifying them");
    let words = backticked_words(enumeration);
    let end = words
        .iter()
        .position(|word| !LEVELS.contains(word))
        .unwrap_or(words.len());
    words[..end].to_vec()
}

/// Every backticked span in `line`.
fn backticked_words(line: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        match after.find('`') {
            Some(close) => {
                words.push(&after[..close]);
                rest = &after[close + 1..];
            }
            None => break,
        }
    }
    words
}

/// The tool schema and the CLI reference must advertise one importance
/// vocabulary — the two places an agent and an operator read the levels.
///
/// Both sides are separately worded copies of the same list (`params.rs` and
/// `docs/cli-reference.md`), so nothing but this assert keeps them in step. The
/// enum↔parser half of the agreement (`from_str` round-trip, case
/// insensitivity) is owned by `src/memory/lifecycle.rs` and is not re-derived
/// here.
#[test]
fn schema_and_cli_reference_advertise_the_same_importance_levels() {
    let doc =
        std::fs::read_to_string(repo_file("docs/cli-reference.md")).expect("read cli-reference.md");
    let section = doc
        .split("`store_memory` accepts:")
        .nth(1)
        .and_then(|rest| rest.split("supersede_memory` accepts:").next())
        .expect("cli-reference.md has a 'store_memory accepts:' section");

    let schema_json = serde_json::to_value(rmcp::schemars::schema_for!(
        crate::mcp::params::StoreMemoryParams
    ))
    .expect("StoreMemoryParams schema is serialisable");
    let description = schema_json
        .get("properties")
        .and_then(|v| v.get("importance"))
        .and_then(|v| v.get("description"))
        .and_then(|v| v.as_str())
        .expect("the schema documents the importance parameter");

    let in_schema: Vec<&str> = LEVELS
        .iter()
        .copied()
        .filter(|level| description.contains(level))
        .collect();
    assert_eq!(
        in_schema, LEVELS,
        "the schema description must name every level the parser accepts: {description}"
    );

    assert_eq!(
        documented_levels(section),
        in_schema,
        "the CLI reference and the tool schema must advertise the same levels"
    );
}
