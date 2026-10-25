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
//! Two things are pinned: that the docs name every parameter the tool declares
//! and every level the parser accepts, and that they name **no** level the
//! parser would reject (an agent that follows a documented-but-invented level
//! gets a rejection it was never warned about). The vocabulary comes from
//! `MemoryImportance::STRINGS`, the roster the enum itself maintains, so a fifth
//! variant fails these asserts instead of drifting silently.
//!
//! The parsers find a bullet by its `- \`param\`` prefix and read backticked
//! words anywhere inside it, so rewording or re-indenting a bullet is a copy
//! edit and not a build failure. What is *not* tolerated is a vocabulary change:
//! `documented_levels` compares the set of level words, and separately rejects
//! any other backticked span in the bullet, so a level appended after `critical`
//! fails as surely as one inserted in the middle (a truncate-at-first-unknown
//! rule would have dropped it silently).
//!
//! Model-free: reading two markdown files needs no embedding session.

use crate::memory::lifecycle::MemoryImportance;

/// The levels the tool accepts, taken from the enum that gates them.
///
/// `MemoryImportance::from_str` is the one gate both the MCP and the CLI
/// surfaces go through, and this roster is the enum's own
/// (`test_all_and_strings_match_the_variants` pins it to the variants), so a new
/// level cannot leave this list stale.
pub(crate) const LEVELS: [&str; 4] = MemoryImportance::STRINGS;

/// A repo-root-relative file, resolved through the manifest dir the way the
/// existing docs-drift tests do (`src/embedding/tests/model_free.rs`).
pub(crate) fn repo_file(rel: &str) -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    std::path::Path::new(&manifest_dir).join(rel)
}

/// Read a repo-root-relative file, with the path in the panic message.
pub(crate) fn read_repo_file(rel: &str) -> String {
    std::fs::read_to_string(repo_file(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// The `store_memory` parameter list in `docs/cli-reference.md`, up to the
/// `supersede_memory` list that follows it. Parsed by one helper so the two
/// tests that read it cannot disagree about where the section is.
fn store_memory_section() -> String {
    let doc = read_repo_file("docs/cli-reference.md");
    doc.split("`store_memory` accepts:")
        .nth(1)
        .and_then(|rest| rest.split("supersede_memory` accepts:").next())
        .expect("cli-reference.md has a 'store_memory accepts:' section")
        .to_string()
}

/// The line of a doc section documenting a parameter, i.e. `- \`name\` …`.
/// Leading whitespace is ignored so a re-indented list still resolves.
fn bullet<'a>(section: &'a str, param: &str) -> Option<&'a str> {
    let prefix = format!("- `{param}`");
    section
        .lines()
        .find(|line| line.trim_start().starts_with(&prefix))
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

/// Words an `importance` bullet may legitimately backtick besides the levels:
/// the parameter and the command it points at, and nothing else. The whitelist is
/// deliberately narrow — a bullet may not invent a fifth level and hide it among
/// qualifying prose.
fn allowed_non_level_span(word: &str) -> bool {
    matches!(
        word,
        "importance" | "supersedes" | "prune" | "status" | "force" | "active" | "candidate"
    )
}

/// The level vocabulary an `importance` bullet advertises: its backticked level
/// words, in `LEVELS` order.
///
/// Two halves. **Coverage** is by set membership, not by position, so the pin is
/// about the vocabulary rather than about where in the bullet the list sits —
/// qualifying the bullet before the enumeration is a copy edit. **Exclusivity**
/// is the assert inside: every other backticked span of the whole bullet must be
/// a known word, so an extra level appended after `critical` fails instead of
/// being truncated away the way a read-that-stops-at-the-first-unknown-word
/// would drop it.
fn documented_levels(section: &str) -> Vec<&str> {
    let bullet = bullet(section, "importance")
        .expect("cli-reference.md lists an `importance` parameter under store_memory");
    let spans = backticked_words(bullet);
    for word in &spans {
        if !LEVELS.contains(word) {
            assert!(
                allowed_non_level_span(word),
                "the importance bullet backticks {word:?}, which the parser does not accept \
                 as a level: {bullet}"
            );
        }
    }
    let named: Vec<&str> = LEVELS
        .iter()
        .copied()
        .filter(|l| spans.contains(l))
        .collect();
    assert_eq!(
        spans.iter().filter(|w| LEVELS.contains(w)).count(),
        named.len(),
        "the importance bullet must name each level exactly once: {bullet}"
    );
    named
}

/// Every word in `text` that names an importance level, canonicalised.
///
/// Matching is on whole words, so `lowish` in a schema description is not
/// counted as `low` — the failure mode of the `description.contains(level)`
/// check this replaces. Case is ignored because the parser is case-insensitive:
/// a description advertising `CRITICAL` is advertising `critical`.
pub(crate) fn level_words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter_map(|token| {
            let lowered = token.to_lowercase();
            LEVELS
                .iter()
                .copied()
                .find(|level| *level == lowered)
                .map(str::to_string)
        })
        .collect()
}

/// The `properties` object of the generated `StoreMemoryParams` schema — the
/// artefact an MCP client actually receives.
pub(crate) fn store_memory_properties() -> serde_json::Value {
    let schema_json = serde_json::to_value(rmcp::schemars::schema_for!(
        crate::mcp::params::StoreMemoryParams
    ))
    .expect("StoreMemoryParams schema is serialisable");
    schema_json
        .get("properties")
        .expect("the schema has a properties object")
        .clone()
}

/// The schema description of one `store_memory` parameter.
pub(crate) fn schema_description(param: &str) -> String {
    store_memory_properties()
        .get(param)
        .and_then(|v| v.get("description"))
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("the schema documents the {param} parameter"))
        .to_string()
}

/// The agent-facing signature must list every parameter the tool declares.
///
/// `importance` is the field #256 added; a signature that lost it would leave
/// agents storing every memory at `medium` because the docs never told them the
/// level could be asked for.
#[test]
fn agent_integration_signature_names_every_store_memory_param() {
    let doc = read_repo_file("docs/agent-integration.md");
    let matches: Vec<&str> = doc
        .lines()
        .filter(|line| line.trim_start().starts_with("- `store_memory("))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "agent-integration.md must contain exactly one '- `store_memory(' signature line, \
         found {}",
        matches.len()
    );
    let line = matches[0];

    let documented: Vec<String> = line
        .split('(')
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("the store_memory signature has a parenthesised parameter list")
        .split(',')
        .map(|param| param.trim().trim_end_matches('?').to_string())
        .collect();

    // Compared against the schema's own property list rather than a hand-typed
    // roster: a field added to `StoreMemoryParams` and left out of the docs fails
    // here, which is the #256 defect one level up. A JSON Schema `properties`
    // object is unordered, so the comparison is as sets; the signature's own
    // order is pinned separately below, because a reader is supposed to meet the
    // parameters in the order the handler documents them.
    let declared: Vec<String> = store_memory_properties()
        .as_object()
        .expect("schema properties")
        .keys()
        .cloned()
        .collect();
    let mut documented_sorted = documented.clone();
    let mut declared_sorted = declared.clone();
    documented_sorted.sort();
    declared_sorted.sort();
    assert_eq!(
        documented_sorted, declared_sorted,
        "docs/agent-integration.md's store_memory signature must list exactly the parameters \
         the tool schema declares (documented {documented:?}, schema {declared:?})"
    );
    assert_eq!(
        documented,
        [
            "text",
            "metadata",
            "memory_type",
            "status",
            "importance",
            "supersedes",
            "force"
        ],
        "the store_memory signature must keep listing the parameters in this order"
    );

    // The line must also say what the level is *for*, the way it already says what
    // `status` does: naming the parameter is not enough to tell an agent when to
    // use it. `importance` itself is already guaranteed by the equality above, so
    // what is asserted here is the rationale — the exemption and the two levels
    // that grant it.
    assert!(
        line.contains("prune") && line.contains("high") && line.contains("critical"),
        "the signature line must document why importance matters (the high/critical prune \
         exemption): {line}"
    );
}

/// The per-tool parameter list in the CLI reference must list `importance` under
/// `store_memory`, with exactly the vocabulary the parser accepts.
#[test]
fn cli_reference_lists_store_memory_importance_with_the_canonical_levels() {
    let section = store_memory_section();
    assert!(
        bullet(&section, "importance").is_some(),
        "cli-reference.md must list `importance` as a store_memory parameter"
    );

    assert_eq!(
        documented_levels(&section),
        LEVELS,
        "the documented level vocabulary must be exactly the levels the parser accepts"
    );
}

/// The tool schema and the CLI reference must advertise one importance
/// vocabulary — the two places an agent and an operator read the levels.
///
/// Both sides are separately worded copies of the same list (`params.rs` and
/// `docs/cli-reference.md`), so nothing but this assert keeps them in step. The
/// enum↔parser half of the agreement (`from_str` round-trip, case insensitivity)
/// is owned by `src/memory/lifecycle.rs` and is not re-derived here.
#[test]
fn schema_and_cli_reference_advertise_the_same_importance_levels() {
    let section = store_memory_section();
    let documented = documented_levels(&section);
    let description = schema_description("importance");

    let named = level_words(&description);
    for level in LEVELS {
        assert!(
            named.iter().any(|word| word == level),
            "the schema description must name the level {level:?}: {description}"
        );
    }
    // The description names the two exempt levels twice (once in the list, once in
    // the exemption clause), so exclusivity is asserted over the set of words it
    // uses rather than over their count: no word outside the parser's vocabulary may
    // read as a level.
    assert!(
        named.iter().all(|word| LEVELS.contains(&word.as_str())),
        "the schema description must advertise no level outside the parser's set: {description}"
    );

    assert_eq!(
        documented, LEVELS,
        "the CLI reference and the tool schema must advertise the same levels"
    );
}
