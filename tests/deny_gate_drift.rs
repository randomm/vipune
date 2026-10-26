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
    // Note the `--` requirement above: a doc that prints the command without a
    // flag (`cargo deny check advisories bans`) yields `None` here and escapes the
    // per-line scan; only the `text.contains(GATE)` assertion can catch that.
    let end = ["`", " #", " |"]
        .iter()
        .filter_map(|stop| cmd.find(stop))
        .min();
    cmd = cmd.get(..end.unwrap_or(cmd.len())).unwrap_or(cmd);
    Some(cmd.trim().to_string())
}

/// Lines after the `with:` map of the step starting at line `install` (its
/// `uses: …` line): the contiguous block of `with:` and `tool`/`tools` keys.
/// Anything else — the next `- name:`/`- run:` step, a job key — ends the map, so
/// a later step's `tool:` cannot be attributed to this one. Blank lines and `#`
/// comments do not (neither is YAML content), and a block scalar (`|`/`>`) body
/// never matches a key prefix.
fn step_with_map<'a>(ci: &'a str, install: usize) -> impl Iterator<Item = &'a str> + 'a {
    ci.lines().skip(install + 1).map(str::trim).take_while(|l| {
        l.is_empty()
            || l.starts_with('#')
            || l.starts_with("with:")
            || (l.starts_with("tool") && l.contains(": "))
    })
}

/// The cargo-deny pin: the `tool: cargo-deny@<version>` line of the
/// `taiki-e/install-action` step that installs cargo-deny, e.g. `0.20.2`.
///
/// There can be several install steps in one workflow (`taiki-e/install-action`
/// is the usual way to add `cargo-machete`, `cargo-audit`, …), so the scan is
/// scoped to the step that actually names cargo-deny and to its own `with:` map:
/// a bare first-`tool:` scan reads an unrelated step above the deny job and fails
/// with "expected cargo-deny". Comment lines are handled in the reader: the scan
/// for the install step is line-based, so a `# uses:` cannot name a step, and the
/// pin lookup drops `#` lines so a commented-out `# tool:` cannot supply the pin.
fn ci_deny_pin(ci: &str) -> Option<String> {
    const INSTALL: &str = "uses: taiki-e/install-action@";
    const KEY: &str = "tool: cargo-deny@";
    ci.lines()
        .map(str::trim)
        .enumerate()
        // Only a step whose own `with:` map names cargo-deny qualifies.
        .find(|(n, line)| {
            line.starts_with(INSTALL) && step_with_map(ci, *n).any(|t| t.starts_with(KEY))
        })
        .and_then(|(n, _)| {
            step_with_map(ci, n)
                .filter(|l| !l.starts_with('#'))
                .find_map(|l| l.strip_prefix(KEY))
        })
        .map(|v| v.trim().to_string())
}

/// Each row of the `[advisories]` `ignore = [ … ]` array, comment lines gone —
/// single- or multi-line, both shapes deny.toml uses.
///
/// Hand-rolled, not parsed: a TOML parser accepts the `{ crate, version }` table
/// form this must reject, so it would pass on the very row that drops `version`
/// and mutes every release of a crate (deny.toml; docs/testing.md, yank triage).
fn deny_ignore_rows(deny: &str) -> Vec<String> {
    let mut rows = Vec::new();
    // `open` tracks the array across lines: an `array` still empty after the
    // opening line (a row-per-line array) must keep collecting, not read as closed.
    let mut open = false;
    let mut array = String::new();
    for line in uncommented(deny) {
        // Only the line carrying the key opens the array, so a `db-urls = ["…"]`
        // elsewhere in `[advisories]` is not mistaken for it.
        if let Some(rest) = line.strip_prefix("ignore = [") {
            open = true;
            array = rest.to_string();
        } else if open {
            array.push(' ');
            array.push_str(&line);
        } else {
            continue;
        }
        // The array closes at the first `]`; `ignore = []` closes on its own
        // opening line and yields no rows. A `{ … }` row would close early on the
        // `]` of its own inline table, but any such fragment is rejected anyway —
        // no valid row contains `]`, so the early close cannot let one through.
        if let Some(end) = array.find(']') {
            rows.extend(
                array[..end]
                    .split(',')
                    .map(str::trim)
                    .filter(|r| !r.is_empty())
                    .map(str::to_string),
            );
            open = false;
            array.clear();
        }
    }
    rows
}

/// Non-comment lines, trimmed.
fn uncommented(text: &str) -> impl Iterator<Item = String> + '_ {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .map(str::to_string)
}

/// `Some(reason)` for an `ignore` row that is not a bare quoted spec string
/// `"name@x.y.z"` with an exact dotted version, or an advisory id (`RUSTSEC-…`,
/// the other shape cargo-deny accepts — docs/testing.md, yank triage).
fn ignore_row_violation(row: &str) -> Option<String> {
    // Brace check first: a `{ crate, version }` row split on its internal commas
    // would otherwise report as a fragment missing its `@`.
    if row.contains('{') {
        return Some("a `{ crate, version }` table row".to_string());
    }
    let Some(spec) = row.strip_prefix('"').and_then(|r| r.strip_suffix('"')) else {
        return Some("not a bare quoted spec string".to_string());
    };
    if spec.starts_with("RUSTSEC-") {
        return None;
    }
    let Some((name, version)) = spec.rsplit_once('@') else {
        return Some("no `@` — a bare crate name ignores every release".to_string());
    };
    let exact = !version.is_empty()
        && !version.contains(['*', '?', '<', '>', '=', ' '])
        && version
            .split('.')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
    match (name.is_empty(), exact) {
        (true, _) => Some("no crate name before `@`".to_string()),
        (false, false) => Some(format!(
            "version `{version}` is not an exact dotted version"
        )),
        (false, true) => None,
    }
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
    // A commented-out `# run: …` is not a step and must not satisfy the lock.
    let run = format!("run: {GATE}");
    assert!(
        ci.lines()
            .map(str::trim)
            .any(|l| !l.starts_with('#') && l == run),
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

    let pin = ci_deny_pin(&ci)
        .expect("ci.yml: no `tool: cargo-deny@<version>` under a taiki-e/install-action step");
    assert!(
        !pin.is_empty()
            && pin
                .split('.')
                .all(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())),
        "ci.yml: cargo-deny pin `{pin}` is not a dotted numeric version"
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

    // Comment lines repeat these literals as prose, so they cannot satisfy a pin —
    // only a live row can.
    let rows = uncommented(&deny).collect::<Vec<_>>().join("\n");

    for (literal, why) in [
        (
            "multiple-versions = \"warn\"",
            "duplicate crate versions are not reported",
        ),
        (
            "yanked = \"deny\"",
            "a yanked release a build still compiles passes the gate",
        ),
        (
            "unused-ignored-advisory = \"deny\"",
            "an ignore row that stops matching rots at the 0.20.2 default of warn",
        ),
        // Contiguous, not two substrings: a row keeping the crate but dropping the
        // denied feature (or the reverse) satisfies a split pin.
        (
            "{ deny = [\"simd-unsafe\"], crate = \"base64@0.23\"",
            "the base64@0.23 simd-unsafe deny row is missing or mistyped",
        ),
    ] {
        assert!(
            rows.contains(literal),
            "deny.toml: {literal} missing — {why}"
        );
    }
    for row in deny_ignore_rows(&rows) {
        assert!(
            ignore_row_violation(&row).is_none(),
            "deny.toml: ignore row `{row}` is not a bare \"name@x.y.z\" spec — the \
             table form and any non-exact version mute every release of the crate"
        );
    }
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

mod helper_probes {
    use super::*;

    /// The finding this lock exists for: a `with:`-less `taiki-e/install-action`
    /// step added above the deny job. A naive "find the install action, then take
    /// the next `tool:` anywhere below it" scan reads that step's tool (or finds
    /// none) and the lock dies with a confusing "expected cargo-deny".
    const FOREIGN_STEP: &str = "      - name: Install cargo-machete\n        uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-machete@1\n";
    const DENY_STEP: &str = "      - name: Install cargo-deny\n        uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.20.2\n";

    #[test]
    fn ci_deny_pin_skips_a_foreign_tool_line() {
        let ci = format!("{FOREIGN_STEP}{DENY_STEP}");
        assert_eq!(ci_deny_pin(&ci).as_deref(), Some("0.20.2"));
    }

    #[test]
    fn ci_deny_pin_is_none_when_deny_is_installed_elsewhere() {
        // No install action at all: fail the lock rather than borrow a pin.
        let ci =
            "      - name: Install cargo-deny\n        run: cargo install cargo-deny --locked\n";
        assert_eq!(ci_deny_pin(ci), None);
        // An install action whose only tool is not cargo-deny: also None, so a
        // workflow that drops the deny install step cannot pass on a stale doc.
        assert_eq!(ci_deny_pin(FOREIGN_STEP), None);
    }

    #[test]
    fn ci_deny_pin_ignores_a_commented_tool_line() {
        let ci = "uses: taiki-e/install-action@v2\n        # tool: cargo-deny@0.1\n        tool: cargo-deny@0.20.2\n";
        assert_eq!(ci_deny_pin(ci).as_deref(), Some("0.20.2"));
    }

    #[test]
    fn ci_deny_pin_stops_at_the_next_step() {
        // A `tool:` belonging to a step after the deny install must not be taken.
        let ci = "uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.20.2\n      - name: Next\n        uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.1.0\n";
        assert_eq!(ci_deny_pin(ci).as_deref(), Some("0.20.2"));
    }

    #[test]
    fn commented_ci_lines_do_not_count_as_gate_or_install() {
        // Both comment-skips in one probe. A commented-out `run:` is not a step, so
        // it must leave the canonical-gate assertion unsatisfied — and a live line
        // at a different indent must still satisfy it, so the first case is not
        // passing by rejecting everything.
        let run = format!("run: {GATE}");
        let satisfied = |text: &str| uncommented(text).any(|l| l == run);
        assert!(
            !satisfied("      # run: cargo deny --locked check advisories bans\n"),
            "a commented-out run line must not count"
        );
        assert!(
            satisfied("run: cargo deny --locked check advisories bans\n"),
            "a live run line must count"
        );
        // Same skip on the install side: a commented-out `# uses:` must not make a
        // tool line look like it belongs to an install step.
        let ci =
            "# uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.1\n";
        assert_eq!(
            ci_deny_pin(ci),
            None,
            "a commented-out install line must not count"
        );
    }

    #[test]
    fn ci_deny_pin_requires_the_install_action() {
        let ci =
            "jobs:\n  deny:\n    steps:\n      - run: cargo deny --locked check advisories bans\n";
        assert_eq!(ci_deny_pin(ci), None);
    }

    #[test]
    fn ci_deny_pin_reads_the_real_workflow() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let ci = read(root, ".github/workflows/ci.yml");
        assert!(ci_deny_pin(&ci).is_some(), "real ci.yml must yield a pin");
    }

    #[test]
    fn deny_command_strips_wrappers() {
        assert_eq!(
            deny_command("        run: cargo deny --locked check advisories bans").as_deref(),
            Some(GATE)
        );
        assert_eq!(
            deny_command("run: cargo deny --locked check advisories bans # note").as_deref(),
            Some(GATE)
        );
        assert_eq!(
            deny_command("| `cargo deny --locked check advisories bans` |").as_deref(),
            Some(GATE)
        );
        // A weakened copy is returned, not swallowed, so the assertion bites.
        assert_eq!(
            deny_command("run: cargo deny --locked check advisories").as_deref(),
            Some("cargo deny --locked check advisories")
        );
        // A prose mention without flags yields None (see the note on `deny_command`).
        assert_eq!(
            deny_command("`cargo deny` needs cargo install").as_deref(),
            None
        );
    }

    #[test]
    fn deny_ignore_rows_reads_both_array_shapes() {
        // Both shapes deny.toml uses, with comment lines skipped: the commented
        // `stale@9.9.9` row must not appear, and `ignore = []` must not stay open
        // and swallow the `[bans]` key below it.
        let single = "[advisories]\nignore = [\"a@1.2.3\", \"b@0.1.0\"]\n\n[bans]\n";
        assert_eq!(
            deny_ignore_rows(single),
            ["\"a@1.2.3\"", "\"b@0.1.0\""],
            "single-line array"
        );
        let multi = "[advisories]\n# ignore = [\"stale@9.9.9\"]\nignore = [\n    \"c@2.0.0\",\n]\n";
        assert_eq!(deny_ignore_rows(multi), ["\"c@2.0.0\""], "multi-line");
        assert!(
            deny_ignore_rows("[advisories]\nignore = []\n\n[bans]\nyanked = \"deny\"\n").is_empty(),
            "empty array"
        );
        assert!(
            deny_ignore_rows("[bans]\nmultiple-versions = \"warn\"\n").is_empty(),
            "no ignore key"
        );
    }

    #[test]
    fn ignore_row_violation_accepts_and_rejects_the_right_shapes() {
        // Accept: an exact spec string, and the advisory-id shape cargo-deny also
        // accepts. Reject: each form that lets a row match more than the one
        // triaged release — the table form drops `version` for the yanked
        // allowlist, an operator or wildcard widens the range, and a bare name
        // matches every release of the crate.
        for row in ["\"a@1.2.3\"", "\"base64@0.23.1\"", "\"RUSTSEC-2026-0189\""] {
            assert_eq!(ignore_row_violation(row), None, "must accept {row}");
        }
        for row in [
            "{ crate = \"a\", version = \"=1.2.3\" }",
            "\"a@>=1.2\"",
            "\"a@1.2.*\"",
            "\"a\"",
            "a",
        ] {
            assert!(ignore_row_violation(row).is_some(), "must reject {row}");
        }
    }

    #[test]
    fn declared_version_reads_major_minor() {
        let manifest = "rmcp = { version = \"1.8\", features = [\"server\"] }\n";
        assert_eq!(declared_version(manifest, "rmcp"), (1, 8));
        let bumped = "rmcp = { version = \"2.11\", optional = true }\n";
        assert_eq!(declared_version(bumped, "rmcp"), (2, 11));
    }

    #[test]
    #[should_panic(expected = "no numeric major")]
    fn declared_version_panics_on_a_non_numeric_major() {
        declared_version("rmcp = { version = \"^1.8\" }\n", "rmcp");
    }
}
