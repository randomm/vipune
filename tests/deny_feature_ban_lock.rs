//! Every `[bans] features` row of `deny.toml` must name a crate@version that
//! Cargo.lock actually resolves (issue #317).
//!
//! cargo-deny 0.20.2 silently ignores a feature-ban row whose `crate =
//! "name@version"` spec matches nothing in the lockfile: rows pinned to
//! `ort@2.0.0-rc.9` and to `orts@2.0.0-rc.15` both printed `bans ok`. A typo,
//! or a row left behind by an ort bump, is therefore a green no-op — the ban
//! looks configured while nothing enforces it. `tests/deny_gate_drift.rs` pins
//! the rows its scan covers as literals; only this scan can tell a pinned row
//! from a dead one, and it pins the ort/ort-sys literals too.
//!
//! std-only line scanners, like the other drift locks: no TOML parser (one
//! accepting the shape a dead row has passes on the row it must catch) and no
//! cargo-deny run.

use std::path::Path;

fn read(rel: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: unreadable: {e}"))
}

/// The `crate = "…"` spec of a `[bans] features` row, or `None` for a line that
/// is not a live row. Brace before key: `{ deny = … }` is a row, a bare `crate
/// = "…"` outside one is not, and a comment line is neither (`numbered`). The
/// value is required to be quoted: an unquoted `crate = a@1.2.3` is invalid
/// TOML, and reading it anyway would let a typo look like a live row.
fn banned_crate_spec(line: &str) -> Option<String> {
    if !line.trim_start().starts_with('{') {
        return None;
    }
    let after_crate = line.split("crate =").nth(1)?;
    let spec = after_crate.split('"').nth(1)?;
    (!spec.is_empty()).then(|| spec.to_string())
}

/// Trimmed non-comment, non-blank lines.
fn numbered(text: &str) -> impl Iterator<Item = String> + '_ {
    text.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

/// `[[package]]` blocks of Cargo.lock as (name, version) pairs. Each block runs
/// from its header to the next, so the name/version are the block's own top-level
/// keys: a `dependencies` entry sits inside a `[dependencies]`-style indented
/// list and never decodes (they are quoted bare strings, not `name = "…"`).
fn lockfile_packages(lock: &str) -> Vec<(String, String)> {
    const HEADER: &str = "[[package]]";
    let mut packages = Vec::new();
    for block in lock.split(HEADER).skip(1) {
        let mut name: Option<String> = None;
        let mut version: Option<String> = None;
        for line in block.lines() {
            let line = line.trim();
            if let Some(value) = line.strip_prefix("name = ")
                && name.is_none()
            {
                name = unquote(value);
            } else if let Some(value) = line.strip_prefix("version = ")
                && version.is_none()
            {
                version = unquote(value);
            }
        }
        if let (Some(name), Some(version)) = (name, version) {
            packages.push((name, version));
        }
    }
    packages
}

/// The quoted value of a `key = "value"` fragment, or `None` if unquoted.
fn unquote(value: &str) -> Option<String> {
    let rest = value.trim_start().strip_prefix('"')?;
    let value = rest.split('"').next()?;
    (!value.is_empty()).then(|| value.to_string())
}

/// `Some(reason)` for a spec cargo-deny could read as matching more than the one
/// release the row intends: no `@` at all (a bare crate name bans the crate at
/// every version), no crate name, or a version carrying an operator or wildcard.
/// A *version string* is deliberately not validated beyond that — cargo-deny
/// accepts both an exact version (`ort@2.0.0-rc.15`, pre-release included) and a
/// `major.minor` prefix (`base64@0.23`, which does bite: verified with 0.20.2 on
/// a probe crate that enables ort's `tls-native`), and requiring `x.y.z` here
/// would reject rows that work.
fn spec_violation(spec: &str) -> Option<String> {
    let Some((name, version)) = spec.rsplit_once('@') else {
        return Some("no `@` — a bare crate name matches every release".to_string());
    };
    if name.is_empty() {
        return Some("no crate name before `@`".to_string());
    }
    if version.is_empty() || version.contains(['*', '?', '<', '>', '=', ' ']) {
        return Some(format!(
            "version `{version}` is not a pinned version or version prefix"
        ));
    }
    None
}

/// Whether `spec` names a package Cargo.lock resolves: the part before the last
/// `@` is the crate name (a pre-release keeps its own hyphens, so this is not a
/// plain `split_once`), and the rest is either an exact lock version or a
/// cargo-deny `major.minor` prefix — which must break on a `.` boundary, so
/// `base64@0.2` does not match the resolved `0.23.1`.
fn resolves_in(lock: &[(String, String)], spec: &str) -> bool {
    let Some((name, version)) = spec.rsplit_once('@') else {
        return false;
    };
    lock.iter().any(|(n, v)| {
        n == name
            && (v == version
                || (v.starts_with(version) && v.get(version.len()..version.len() + 1) == Some(".")))
    })
}

#[test]
fn test_every_banned_feature_row_names_a_crate_resolved_in_the_lockfile() {
    let deny = read("deny.toml");
    let lock = read("Cargo.lock");
    let resolved: Vec<(String, String)> = lockfile_packages(&lock);

    let rows: Vec<String> = numbered(&deny)
        .filter_map(|l| banned_crate_spec(&l))
        .collect();
    assert!(
        rows.len() >= 3,
        "deny.toml: read only {} `[bans] features` row(s) ({rows:?}) — the scan is vacuous, so \
         a dead row would pass unnoticed",
        rows.len()
    );

    for spec in &rows {
        // Malformed first: a spec cargo-deny would widen past one release, or
        // read as a bare crate name, cannot be compared to one lock entry.
        if let Some(reason) = spec_violation(spec) {
            panic!("deny.toml: feature-ban row crate = \"{spec}\" — {reason}");
        }
        assert!(
            resolves_in(&resolved, spec),
            "deny.toml: feature-ban row crate = \"{spec}\" matches nothing in Cargo.lock — \
             cargo-deny ignores such a row silently (verified: a row naming a crate the lock \
             does not resolve prints `bans ok`), so the ban would not be enforced. Bump the \
             row together with the pin in Cargo.toml, or fix the crate name."
        );
    }
}

#[test]
fn test_the_ort_tls_ban_rows_are_the_ones_the_gate_exists_to_enforce() {
    // The whole point of the two rows is the OpenSSL TLS stack, so pin what each
    // bans as well as its crate: a row narrowed to one feature, or to a
    // non-TLS feature, would leave tls-native-vendored (vendored OpenSSL) open.
    let deny = read("deny.toml");
    let rows: Vec<String> = numbered(&deny)
        .filter_map(|l| banned_crate_spec(&l))
        .collect();

    for crate_name in ["ort", "ort-sys"] {
        let spec = format!("{crate_name}@2.0.0-rc.15");
        // Scope the scan to the row itself: the row's `reason` string also names
        // both features, so a `deny` list narrowed to one of them must not be
        // certified by a substring test over the whole line.
        let row = deny
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with('{') && l.contains(&format!("crate = \"{spec}\"")))
            .unwrap_or_else(|| panic!("deny.toml: no live row for crate = \"{spec}\""));
        let denied = row
            .split("deny =")
            .nth(1)
            .map(|rest| rest.split(']').next().unwrap_or(rest))
            .expect("row has a deny list");
        for feature in ["tls-native", "tls-native-vendored"] {
            assert!(
                denied.contains(&format!("\"{feature}\"")),
                "deny.toml: the {spec} row must deny \"{feature}\" in its `deny` list, or \
                 OpenSSL can be unified back in: {row}"
            );
        }
        assert!(
            rows.contains(&spec),
            "the {spec} ban row is not read as a `[bans] features` row"
        );
    }
}

/// Contiguous, not two substrings: a row that keeps the crate but drops one
/// denied feature satisfies a split pin — the same rule
/// `tests/deny_gate_drift.rs` applies to the rows its own scan covers. Living
/// here rather than there is a file-size matter, not a difference in what is
/// pinned: deleting either literal from deny.toml fails this test.
#[test]
fn test_the_ort_tls_deny_rows_are_pinned_as_contiguous_literals() {
    let deny = read("deny.toml");
    let rows = numbered(&deny).collect::<Vec<_>>().join("\n");

    for crate_name in ["ort", "ort-sys"] {
        let literal = format!(
            "{{ deny = [\"tls-native\", \"tls-native-vendored\"], crate = \"{crate_name}@2.0.0-rc.15\""
        );
        assert!(
            rows.contains(&literal),
            "deny.toml: {literal} missing — the {crate_name} OpenSSL TLS deny row is absent \
             from the live (uncommented) rows, or its deny list / crate spec is mistyped"
        );
    }
}

mod helper_probes {
    use super::*;

    #[test]
    fn banned_crate_spec_reads_only_live_rows() {
        assert_eq!(
            banned_crate_spec("    { deny = [\"x\"], crate = \"a@1.2.3\", reason = \"r\" },"),
            Some("a@1.2.3".to_string())
        );
        // A comment repeating the row is not a row.
        assert_eq!(
            banned_crate_spec("# { deny = [\"x\"], crate = \"a@1.2.3\" }"),
            None
        );
        // Nor is a spec written outside a row, or a row whose spec is malformed.
        assert_eq!(banned_crate_spec("crate = \"a@1.2.3\""), None);
        assert_eq!(
            banned_crate_spec("{ deny = [\"x\"], crate = a@1.2.3 }"),
            None
        );
        assert_eq!(banned_crate_spec("{ deny = [\"x\"], crate = \"\" }"), None);
        // `crate =` with only one space before `=` is not read: this scan takes
        // the whitespace form deny.toml actually uses (rustfmt keeps it there).
        assert_eq!(banned_crate_spec("{ crate= \"a@1.2.3\" }"), None);
    }

    #[test]
    fn lockfile_packages_reads_headers_not_dependency_lines() {
        let lock = "[[package]]\nname = \"a\"\nversion = \"1.2.3\"\nsource = \"x\"\ndependencies = [\n \"b 0.9.9\",\n]\n\n[[package]]\nname = \"b\"\nversion = \"0.9.9\"\n";
        assert_eq!(
            lockfile_packages(lock),
            [
                ("a".to_string(), "1.2.3".to_string()),
                ("b".to_string(), "0.9.9".to_string())
            ]
        );
        // A block with no version contributes nothing rather than borrowing one.
        assert!(lockfile_packages("[[package]]\nname = \"a\"\n").is_empty());
    }

    #[test]
    fn spec_violation_rejects_the_specs_that_match_more_than_one_release() {
        // A malformed call yields Some too: the caller must not read `None` as
        // "valid" for a spec with no `@` at all. `base64@0.23` is accepted — a
        // major.minor prefix spec bites in cargo-deny, so rejecting it here
        // would fail a row that works.
        for good in ["a@1.2.3", "ort@2.0.0-rc.15", "base64@0.23", "a@1"] {
            assert_eq!(spec_violation(good), None, "must accept {good}");
        }
        for bad in ["a@1.2.*", "a@>=1.2", "a@", "@1.2.3", "a"] {
            assert!(spec_violation(bad).is_some(), "must reject {bad}");
        }
        // Non-numeric version text is not rejected here — cargo-deny's own
        // matcher decides it — but it resolves to no package, and
        // `resolves_in_matches_exact_and_prefix_specs` pins that it is caught
        // against the lock rather than by this shape check.
        assert!(spec_violation("a@1.2.x").is_none());
        assert!(!resolves_in(
            &[("a".to_string(), "1.2.3".to_string())],
            "a@1.2.x"
        ));
    }

    #[test]
    fn resolves_in_matches_exact_and_prefix_specs() {
        let lock = vec![
            ("base64".to_string(), "0.23.1".to_string()),
            ("ort".to_string(), "2.0.0-rc.15".to_string()),
        ];
        assert!(resolves_in(&lock, "base64@0.23"), "major.minor prefix");
        assert!(resolves_in(&lock, "ort@2.0.0-rc.15"), "exact");
        // The two ways a row goes dead: a stale version and a mistyped name.
        assert!(!resolves_in(&lock, "ort@2.0.0-rc.9"), "stale rc pin");
        assert!(!resolves_in(&lock, "orts@2.0.0-rc.15"), "mistyped name");
        // A prefix that is not a version boundary must not match ("0.2" of 0.23).
        assert!(!resolves_in(&lock, "base64@0.2"), "partial component");
        assert!(!resolves_in(&lock, "a"), "no @");
    }

    #[test]
    fn the_committed_files_still_satisfy_the_scans() {
        // A thin read means the scan above is vacuous, so assert on both files
        // directly rather than trusting the test that uses them.
        assert!(lockfile_packages(&read("Cargo.lock")).len() > 50);
        assert!(numbered(&read("deny.toml")).count() > 10);
        assert!(
            numbered(&read("deny.toml"))
                .any(|l| banned_crate_spec(&l) == Some("base64@0.23".to_string()))
        );
    }
}
