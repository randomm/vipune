//! Model-free drift lock for the cargo-deny gate (issue #269).
//!
//! The gate is one command string hand-copied into `.github/workflows/ci.yml`,
//! `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md` and `docs/testing.md`, and
//! cargo-deny itself is pinned only in `ci.yml`. A copy that drops `--locked`
//! or `bans`, or a doc that quotes a tool version the workflow no longer
//! installs, is a silently weaker gate — nothing else reads those strings.
//!
//! Every check here is a substring or line scan over files read with `std::fs`
//! relative to `CARGO_MANIFEST_DIR`: no config parsers (a parser that accepts
//! the shape a mistyped row does not have would pass on the row it is meant to
//! catch) and no cargo-deny invocation. Same drift class as
//! `test_air_gapped_docs_and_ci_cache_match_embed_revision` and `msrv_drift.rs`.

use std::path::Path;

const GATE: &str = "cargo deny --locked check advisories bans";

/// Docs that restate the gate command. `SECURITY.md` describes the gate in prose
/// without printing the command, so it is not a copy and not listed. This file
/// is out of scope for the scan by construction: the scan reads only these paths.
const GATE_DOCS: [&str; 4] = [
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "docs/testing.md",
];

fn read(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: unreadable: {e}"))
}

/// The `cargo deny …` **command** issued by a line, with markdown/table wrappers
/// gone, or `None` if the line does not issue one. Requiring the flag plus `check`
/// right after the binary is what separates a command from a prose mention
/// ("`cargo deny` needs `cargo install cargo-deny` once") or a bare tool name.
fn deny_command(line: &str) -> Option<String> {
    let start = line.find("cargo deny --")?;
    let mut cmd = &line[start..];
    // Truncate at the nearest wrapper or trailing comment, not at the first
    // occurrence of each: cutting at a later `  #` than the closing backtick would
    // let a `run: cargo deny … # note` keep trailing text and hide dropped args.
    let end = ["`", " #", " |"]
        .iter()
        .filter_map(|stop| cmd.find(stop))
        .min();
    cmd = cmd.get(..end.unwrap_or(cmd.len())).unwrap_or(cmd);
    Some(cmd.trim().to_string())
}

/// The value of `key:` in a workflow step's `with:` map, e.g. `tool`.
fn workflow_with_value(ci: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}: ");
    ci.lines()
        .map(str::trim)
        .find(|l| l.starts_with(&prefix))
        .map(|l| l[prefix.len()..].trim().to_string())
}

/// `major.minor` of the version in the `name = { version = "…"` line.
fn declared_version(manifest: &str, dep: &str) -> (u32, u32) {
    let line = manifest
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("{dep} = ")))
        .unwrap_or_else(|| panic!("Cargo.toml has no `{dep}` dependency"));
    let after = line
        .split("version = \"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .unwrap_or_else(|| panic!("Cargo.toml: no quoted version on `{dep}`"));
    let mut parts = after.split('.');
    let major = parts
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("Cargo.toml: {dep} version `{after}` has no numeric major"));
    let minor = parts
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("Cargo.toml: {dep} version `{after}` has no numeric minor"));
    (major, minor)
}

#[test]
fn test_every_gate_copy_is_the_canonical_command() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ci = read(root, ".github/workflows/ci.yml");

    // In ci.yml the command must be a run step, not just text inside a comment.
    let run = format!("run: {GATE}");
    assert!(
        ci.lines().any(|l| l.trim() == run),
        "ci.yml has no step running the canonical gate; expected a line `{run}`"
    );

    for doc in GATE_DOCS {
        let text = read(root, doc);
        assert!(
            text.contains(GATE),
            "{doc} does not state the canonical gate command `{GATE}`"
        );

        for (n, line) in text.lines().enumerate() {
            if let Some(cmd) = deny_command(line) {
                assert_eq!(
                    cmd,
                    GATE,
                    "{doc}:{} quotes `{cmd}` instead of the canonical `{GATE}`",
                    n + 1
                );
            }
        }
    }
}

#[test]
fn test_ci_deny_pin_matches_the_docs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ci = read(root, ".github/workflows/ci.yml");

    let tool = workflow_with_value(&ci, "tool").expect("ci.yml: no `tool:` install step");
    let mut parts = tool.split('@');
    assert_eq!(parts.next(), Some("cargo-deny"), "ci.yml: {tool}");
    let pin = parts
        .next()
        .unwrap_or_else(|| panic!("ci.yml: `{tool}` has no @version"));
    assert!(
        parts.next().is_none(),
        "ci.yml: `{tool}` is not name@version"
    );

    let docs = read(root, "docs/testing.md");
    assert!(
        docs.contains(&format!(
            "[cargo-deny](https://github.com/EmbarkStudios/cargo-deny) **{pin}**"
        )),
        "docs/testing.md does not state the ci.yml cargo-deny pin {pin} plainly"
    );
}

#[test]
fn test_deny_config_keeps_its_load_bearing_rows() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let deny = read(root, "deny.toml");
    let manifest = read(root, "Cargo.toml");

    assert!(
        deny.contains("multiple-versions = \"warn\""),
        "deny.toml: duplicate crate versions are not reported as warnings"
    );
    // A mistyped crate or feature in this row passes cargo-deny silently, so the
    // literals are pinned here.
    assert!(
        deny.contains("base64@0.23") && deny.contains("deny = [\"simd-unsafe\"]"),
        "deny.toml: the base64@0.23 simd-unsafe deny row is missing or mistyped"
    );
    assert!(
        manifest.contains("base64 = { version = \"0.23\", default-features = false"),
        "Cargo.toml: base64 must keep default-features = false, or simd-unsafe can be unified back in"
    );
}

#[test]
fn test_rmcp_floor_is_at_or_above_the_patched_release() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = read(root, "Cargo.toml");

    // Numeric compare, not a literal match, so a routine rmcp bump stays green.
    let (major, minor) = declared_version(&manifest, "rmcp");
    assert!(
        (major, minor) >= (1, 8),
        "Cargo.toml: rmcp {major}.{minor} permits versions before the 1.4.0 fix for RUSTSEC-2026-0189"
    );
}
