//! Model-free drift lock for the cargo-deny gate (issue #269): one command string
//! hand-copied into ci.yml and the docs, with cargo-deny pinned only in ci.yml, so a
//! copy dropping `--locked` or `bans` — or quoting a version the workflow no longer
//! installs — is a silently weaker gate. Every check is a substring or line scan under
//! `CARGO_MANIFEST_DIR`: no config parsers (a parser accepting the shape a mistyped row
//! lacks passes on the row it must catch) and no cargo-deny run. Same drift class as
//! the embed-revision and msrv locks.

use std::path::Path;

const GATE: &str = "cargo deny --locked check advisories bans";

/// Docs restating the gate command; `SECURITY.md` only describes it in prose.
const GATE_DOCS: [&str; 4] = [
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "docs/testing.md",
];

fn read(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: unreadable: {e}"))
}

/// The `cargo deny …` **command** issued by a line, wrappers gone, or `None` if none.
/// The flag-plus-`check` requirement separates a command from a prose mention.
fn deny_command(line: &str) -> Option<String> {
    let start = line.find("cargo deny --")?;
    let mut cmd = &line[start..];
    // Truncate at the nearest wrapper or trailing comment, not the first occurrence
    // of each: cutting at a later `  #` than the closing backtick lets `… # note`
    // keep trailing text and hide dropped args. Without the `--` requirement above a
    // flagless copy yields `None` here and only `text.contains(GATE)` can catch it.
    let end = ["`", " #", " |"]
        .iter()
        .filter_map(|stop| cmd.find(stop))
        .min();
    cmd = cmd.get(..end.unwrap_or(cmd.len())).unwrap_or(cmd);
    Some(cmd.trim().to_string())
}

/// Lines after the `uses: …` line of the step at line `install`: taken through blank
/// lines and `#` (real ci.yml has a blank above the deny step's `uses:`), stopping at
/// the first non-key line — next step, job key, or near-miss `tool-versions:`. Keys
/// compare exactly before the first `:`: a prefix test on "tool" keeps the map open for
/// `tool-versions: cargo-deny@9.9.9`.
fn step_with_map<'a>(ci: &'a str, install: usize) -> impl Iterator<Item = &'a str> + 'a {
    ci.lines()
        .skip(install + 1)
        .map(str::trim)
        .take_while(|l| l.is_empty() || l.starts_with('#') || is_with_key(l))
}

/// A `with:`/`tool:`/`tools:` key line of a step's map.
fn is_with_key(line: &str) -> bool {
    matches!(
        line.split_once(':').map(|(k, _)| k),
        Some("with" | "tool" | "tools")
    )
}

/// The cargo-deny pin: the `tool: cargo-deny@<version>` line of the install step for
/// cargo-deny, e.g. `0.20.2`. A workflow can have several install steps, so the scan is
/// scoped to the step naming cargo-deny and its own `with:` map; a `# uses:` cannot name
/// a step, a `# tool:` cannot be the pin.
fn ci_deny_pin(ci: &str) -> Option<String> {
    const INSTALL: &str = "uses: taiki-e/install-action@";
    const KEY: &str = "tool: cargo-deny@";
    numbered(ci)
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .find(|(n, line)| {
            line.starts_with(INSTALL) && step_with_map(ci, *n).any(|t| t.starts_with(KEY))
        })
        .and_then(|(n, _)| step_with_map(ci, n).find_map(|l| l.strip_prefix(KEY)))
        .map(|v| v.trim().to_string())
}

/// The `[advisories]` `ignore` key line: split once on `=`, LHS `ignore`, RHS starting
/// with `[` — `ignore=[` and `ignore  =  [` are one key to reader and caller.
fn is_ignore_key_line(line: &str) -> bool {
    line.split_once('=')
        .is_some_and(|(k, v)| k.trim() == "ignore" && v.trim_start().starts_with('['))
}

/// Each row of the `[advisories]` `ignore = [ … ]` array, comments gone — single- or
/// multi-line, both deny.toml shapes. Hand-rolled, not parsed: a TOML parser takes the
/// `{ crate, version }` table this must reject, so it passes on the row that drops
/// `version` (docs/testing.md, yank triage).
fn deny_ignore_rows(deny: &str) -> Vec<String> {
    let mut rows = Vec::new();
    // `open` tracks the array across lines: an `array` still empty after the opening
    // line (row-per-line) must keep collecting, not read as closed. Only the key line
    // opens it, and the key match is whitespace-insensitive: that is what makes the
    // scan total rather than vacuous on `ignore=[`.
    let mut open = false;
    let mut array = String::new();
    for (_, line) in numbered(deny).filter(|(_, l)| !l.is_empty()) {
        if let Some((_, rhs)) = line.split_once('=').filter(|_| is_ignore_key_line(&line)) {
            open = true;
            // The bracket is not a row: drop it, or `ignore = []` yields a `[` row.
            array = rhs.trim_start()[1..].to_string();
        } else if open {
            array.push(' ');
            array.push_str(&line);
        } else {
            continue;
        }
        // The array closes at the first `]`; `ignore = []` closes on its own line. A
        // `{ … }` row closes early on the `]` of its own inline table, but no valid row
        // contains `]`, so that cannot let one through.
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

/// Trimmed non-comment lines with 0-based line numbers; blanks kept so a reader can
/// tell a map's own keys from the next step's.
fn numbered(text: &str) -> impl Iterator<Item = (usize, String)> + '_ {
    text.lines()
        .enumerate()
        .map(|(n, l)| (n, l.trim().to_string()))
        .filter(|(_, l)| !l.starts_with('#'))
}

/// `Some(reason)` for a row that is neither a `"name@x.y.z"` spec with an exact dotted
/// version nor a bare advisory id (below).
fn is_digits(s: Option<&str>, len: usize) -> bool {
    s.is_some_and(|v| v.len() == len && v.bytes().all(|b| b.is_ascii_digit()))
}

/// The bare advisory id cargo-deny also accepts: exactly `RUSTSEC-<year>-<4 or 5
/// digits>` — a third hyphen or an `@` lands in `id` and fails the digit tests, so
/// `RUSTSEC-2026-0189@1.2.3` and `RUSTSEC-x` are not ids.
fn is_advisory_id(spec: &str) -> bool {
    let Some((year, id)) = spec
        .strip_prefix("RUSTSEC-")
        .and_then(|r| r.split_once('-'))
    else {
        return false;
    };
    is_digits(Some(year), 4) && (is_digits(Some(id), 4) || is_digits(Some(id), 5))
}

fn ignore_row_violation(row: &str) -> Option<String> {
    // Brace first: a `{ crate, version }` row split on its own commas would otherwise
    // report as a fragment missing its `@`.
    if row.contains('{') {
        return Some("a `{ crate, version }` table row".to_string());
    }
    let Some(spec) = row.strip_prefix('"').and_then(|r| r.strip_suffix('"')) else {
        return Some("not a bare quoted spec string".to_string());
    };
    if spec.starts_with("RUSTSEC-") {
        return (!is_advisory_id(spec))
            .then(|| "advisory id is not RUSTSEC-<year>-<4 or 5 digits>".to_string());
    }
    let Some((name, version)) = spec.rsplit_once('@') else {
        return Some("no `@` — a bare crate name ignores every release".to_string());
    };
    let exact = !version.is_empty()
        && !version.contains(['*', '?', '<', '>', '=', ' '])
        && version.split('.').all(|p| is_digits(Some(p), p.len()));
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
        .and_then(|r| r.split('"').next())
        .unwrap_or_else(|| panic!("Cargo.toml: no quoted version on `{dep}`"));
    let mut parts = after.split('.');
    let num = |p: Option<&str>| p.and_then(|p| p.parse().ok());
    match (num(parts.next()), num(parts.next())) {
        (Some(major), Some(minor)) => (major, minor),
        _ => panic!("Cargo.toml: {dep} version `{after}` has no numeric major.minor"),
    }
}

#[test]
fn test_every_gate_copy_is_the_canonical_command() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ci = read(root, ".github/workflows/ci.yml");

    // In ci.yml it must be a run step: a commented-out `# run:` must not satisfy.
    let run = format!("run: {GATE}");
    assert!(
        numbered(&ci).any(|(_, l)| l == run),
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
        !pin.is_empty() && pin.split('.').all(|s| is_digits(Some(s), s.len())),
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

    // Comment lines repeat these literals as prose, so only a live row pins them.
    let uncommented_lines = numbered(&deny)
        .map(|(_, l)| l)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>();
    let rows = uncommented_lines.join("\n");

    // The reader matches the `ignore` key whitespace-insensitively, so a mistyped key
    // (`ignores = [`) opens nothing and the scan below passes on zero rows.
    let key_lines = uncommented_lines
        .iter()
        .filter(|l| is_ignore_key_line(l))
        .count();
    assert_eq!(
        key_lines, 1,
        "deny.toml: expected exactly one live `ignore = [` key line, found {key_lines} — a \
         mistyped key leaves the ignore rows unread and the scan below vacuous"
    );

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
        // denied feature satisfies a split pin.
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

    /// The lock's reason for being: a `with:`-less install step above the deny job —
    /// "first install action, then next `tool:`" reads its tool.
    const FOREIGN_STEP: &str = "      - name: Install cargo-machete\n        uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-machete@1\n";
    const DENY_STEP: &str = "      - name: Install cargo-deny\n        uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.20.2\n";

    #[test]
    fn ci_deny_pin_skips_a_foreign_tool_line() {
        assert_eq!(
            ci_deny_pin(&format!("{FOREIGN_STEP}{DENY_STEP}")).as_deref(),
            Some("0.20.2")
        );
    }

    #[test]
    fn ci_deny_pin_is_none_when_deny_is_installed_elsewhere() {
        // No install action at all, and an install action whose only tool is not
        // cargo-deny: fail the lock rather than borrow a pin.
        let ci =
            "      - name: Install cargo-deny\n        run: cargo install cargo-deny --locked\n";
        assert_eq!(ci_deny_pin(ci), None);
        assert_eq!(ci_deny_pin(FOREIGN_STEP), None);
    }

    #[test]
    fn ci_deny_pin_ignores_commented_or_later_tool_lines() {
        let ci = "uses: taiki-e/install-action@v2\n        # tool: cargo-deny@0.1\n        tool: cargo-deny@0.20.2\n";
        assert_eq!(ci_deny_pin(ci).as_deref(), Some("0.20.2"));
        // A `tool:` of a later step must not be taken.
        let ci = "uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.20.2\n      - name: Next\n        uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.1.0\n";
        assert_eq!(ci_deny_pin(ci).as_deref(), Some("0.20.2"));
    }

    #[test]
    fn commented_ci_lines_do_not_count_as_gate_or_install() {
        // Both comment-skips in one probe: a commented-out `run:` is not a step and
        // leaves the canonical-gate assertion unsatisfied, while a live line at another
        // indent still satisfies it.
        let run = format!("run: {GATE}");
        let satisfied = |text: &str| numbered(text).any(|(_, l)| l == run);
        assert!(
            !satisfied("      # run: cargo deny --locked check advisories bans\n"),
            "a commented-out run line must not count"
        );
        assert!(
            satisfied("run: cargo deny --locked check advisories bans\n"),
            "a live run line must count"
        );
        // Same skip on the install side: `# uses:` must not own the tool line below.
        let ci =
            "# uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.1\n";
        assert_eq!(
            ci_deny_pin(ci),
            None,
            "a commented-out install line must not count"
        );
    }

    #[test]
    fn ci_deny_pin_ignores_a_tool_versions_key() {
        // `tool-versions:` is no key of this step's map: a prefix test on "tool" stops
        // the map there, so the real `tool:` below it (the only 9.9.9) is outside it.
        let ci = "uses: taiki-e/install-action@v2\n        with:\n          tool: cargo-deny@0.20.2\n          tool-versions: x\n          tool: cargo-deny@9.9.9\n      - name: Later\n        run: cargo deny --locked check advisories bans\n";
        assert_eq!(ci_deny_pin(ci).as_deref(), Some("0.20.2"));
    }

    #[test]
    fn ci_deny_pin_requires_the_install_action() {
        let ci =
            "jobs:\n  deny:\n    steps:\n      - run: cargo deny --locked check advisories bans\n";
        assert_eq!(ci_deny_pin(ci), None);
        // And the real workflow must still yield one.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(ci_deny_pin(&read(root, ".github/workflows/ci.yml")).is_some());
    }

    #[test]
    fn deny_command_strips_wrappers() {
        assert_eq!(
            deny_command("run: cargo deny --locked check advisories bans").as_deref(),
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
        // A prose mention without flags yields None (see `deny_command`).
        assert_eq!(
            deny_command("`cargo deny` needs cargo install").as_deref(),
            None
        );
    }

    #[test]
    fn deny_ignore_rows_reads_both_array_shapes() {
        // Both shapes, comments skipped: the commented row must not show and `[]` must
        // not swallow the `[bans]` key below it.
        let single = "[advisories]\nignore = [\"a@1.2.3\", \"b@0.1.0\"]\n\n[bans]\n";
        assert_eq!(deny_ignore_rows(single), ["\"a@1.2.3\"", "\"b@0.1.0\""]);
        let multi = "[advisories]\n# ignore = [\"stale@9.9.9\"]\nignore = [\n    \"c@2.0.0\",\n]\n";
        assert_eq!(deny_ignore_rows(multi), ["\"c@2.0.0\""], "multi-line");
        assert!(
            deny_ignore_rows("[advisories]\nignore = []\n\n[bans]\nyanked = \"deny\"\n").is_empty(),
            "empty array"
        );
        assert!(deny_ignore_rows("[bans]\nmultiple-versions = \"warn\"\n").is_empty());
        // Whitespace-insensitive key match: keyed on the literal `ignore = [` the reader
        // yields zero rows for valid `ignore=[` and passes vacuously.
        assert_eq!(
            deny_ignore_rows("[advisories]\nignore=[\"h@1.0.0\",\"i@2.0.0\"]\n[bans]\n"),
            ["\"h@1.0.0\"", "\"i@2.0.0\""],
            "spacing variants of the key still open the array"
        );
        // A non-key line whose value begins with `[` must not open the array.
        assert!(
            deny_ignore_rows("[advisories]\ndb-urls = [\"https://example.com\"]\nignore = []\n")
                .is_empty(),
            "a non-ignore array key must not be read as the ignore list"
        );
    }

    #[test]
    fn ignore_row_violation_accepts_and_rejects_the_right_shapes() {
        // Accept the exact spec string and the advisory-id shape; reject each form
        // letting a row match more than the triaged release: the table form drops
        // `version`, an operator or wildcard widens the range, and a bare name or an id
        // carrying a version matches more than itself (a prefix test on "RUSTSEC-"
        // accepts both of the last two).
        for row in ["\"a@1.2.3\"", "\"base64@0.23.1\"", "\"RUSTSEC-2026-0189\""] {
            assert_eq!(ignore_row_violation(row), None, "must accept {row}");
        }
        for row in [
            "{ crate = \"a\", version = \"=1.2.3\" }",
            "\"a@>=1.2\"",
            "\"a@1.2.*\"",
            "\"a\"",
            "a",
            "\"RUSTSEC-2026-0189@1.2.3\"",
            "\"RUSTSEC-x\"",
        ] {
            assert!(ignore_row_violation(row).is_some(), "must reject {row}");
        }
    }

    #[test]
    fn declared_version_reads_major_minor() {
        let manifest = "rmcp = { version = \"1.8\", features = [\"server\"] }\n";
        assert_eq!(declared_version(manifest, "rmcp"), (1, 8));
        assert_eq!(
            declared_version("rmcp = { version = \"2.11\", optional = true }\n", "rmcp"),
            (2, 11)
        );
    }

    #[test]
    #[should_panic(expected = "no numeric major")]
    fn declared_version_panics_on_a_non_numeric_major() {
        declared_version("rmcp = { version = \"^1.8\" }\n", "rmcp");
    }
}
