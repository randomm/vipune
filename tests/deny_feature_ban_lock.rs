//! Every `[bans] features` row of `deny.toml` must name a crate@version that
//! Cargo.lock actually resolves (issue #317).
//!
//! cargo-deny 0.20.2 silently ignores a feature-ban row whose `crate =
//! "name@version"` spec matches nothing in the lockfile: rows pinned to
//! `ort@2.0.0-rc.9` and to `orts@2.0.0-rc.15` both printed `bans ok`. A typo,
//! or a row left behind by an ort bump, is therefore a green no-op — the ban
//! looks configured while nothing enforces it. `tests/deny_gate_drift.rs` pins
//! the rows its scan covers as literals; only this scan can tell a pinned row
//! from a dead one, and it pins the ort/ort-sys literals too. Coverage is per
//! release: cargo-deny matches a row against every package of that name, so a
//! second copy the row does not name keeps its banned features.
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

/// Whether a version matches a spec's version part: exactly, or as a cargo-deny
/// `major.minor` prefix that breaks on a `.` boundary (so `0.2` does not match
/// the resolved `0.23.1`).
fn version_matches(version: &str, spec_version: &str) -> bool {
    version == spec_version
        || (version.starts_with(spec_version)
            && version.get(spec_version.len()..spec_version.len() + 1) == Some("."))
}

/// The crate name of a `name@version` spec: the part before the last `@` (a
/// pre-release keeps its own hyphens, so this is not a plain `split_once`).
fn spec_crate(spec: &str) -> &str {
    spec.rsplit_once('@').map_or(spec, |(name, _)| name)
}

/// The versions Cargo.lock resolves for crate `name`, in lock order.
fn copies_of<'a>(lock: &'a [(String, String)], name: &str) -> Vec<&'a str> {
    lock.iter()
        .filter(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
        .collect()
}

/// Whether `spec` resolves at all: its crate has at least one copy in the lock
/// (otherwise cargo-deny ignores the row silently) and its version matches one of
/// them with `version_matches`. Pure "any copy" — the row test asserts the
/// stricter `covers_every_copy`, so this predicate stays probeable on its own.
fn resolves_in(lock: &[(String, String)], spec: &str) -> bool {
    let Some((name, version)) = spec.rsplit_once('@') else {
        return false;
    };
    copies_of(lock, name)
        .iter()
        .any(|v| version_matches(v, version))
}

/// Whether `spec` bans **every** copy of its crate that Cargo.lock resolves: at
/// least one copy exists and each resolved version matches with
/// `version_matches`.
///
/// Per release, not "any copy matches": cargo-deny matches a `name@version` row
/// against every package of that name, so a second copy the row does not name
/// keeps its banned features while the gate prints `bans ok`. `multiple-versions
/// = "warn"` makes several copies of one crate normal in this lock, so an extra
/// unbanned `ort` must fail here rather than pass. A crate whose duplicates cannot
/// be covered by any one row needs a `MULTI_COPY_EXCEPTIONS` entry, audited
/// separately by `test_multi_copy_exceptions_are_declared_where_the_row_states_them`;
/// the exemption lives in that assertion, so this predicate keeps meaning "all
/// copies match" and its probe below stays meaningful.
fn covers_every_copy(lock: &[(String, String)], spec: &str) -> bool {
    let Some((name, version)) = spec.rsplit_once('@') else {
        return false;
    };
    let copies = copies_of(lock, name);
    !copies.is_empty() && copies.iter().all(|v| version_matches(v, version))
}

/// The rows of a `[bans] features` list whose crate has more than one release in
/// Cargo.lock, as `(crate name, versions)` — the rows `resolves_in` cannot
/// certify, because a row is keyed by `name@version` while the lock keys a crate
/// by name. The crate name is reported (not the whole spec): the exception roster
/// is keyed on the crate, so a row for the same crate at a different version is
/// caught rather than silently re-labelled.
fn multi_copy_crates<'a>(
    lock: &'a [(String, String)],
    rows: &[String],
) -> Vec<(String, Vec<&'a str>)> {
    rows.iter()
        .filter_map(|spec| {
            let name = spec_crate(spec);
            let copies = copies_of(lock, name);
            (copies.len() > 1).then(|| (name.to_string(), copies))
        })
        .collect()
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
            covers_every_copy(&resolved, spec) || MULTI_COPY_EXCEPTIONS.contains(&spec_crate(spec)),
            "deny.toml: feature-ban row crate = \"{spec}\" does not cover every copy of its \
             crate that Cargo.lock resolves — cargo-deny ignores a row matching nothing \
             silently (verified: a row naming a crate the lock does not resolve prints `bans \
             ok`), and an extra copy the row does not name keeps its banned features. Bump \
             the row together with the pin in Cargo.toml, fix the crate name, or resolve the \
             duplicate. A crate whose duplicates genuinely cannot be covered by one row needs \
             an entry in MULTI_COPY_EXCEPTIONS, which \
             test_multi_copy_exceptions_are_declared_where_the_row_states_them then audits."
        );
    }
}

/// The only crates a ban row may name while Cargo.lock resolves several releases
/// of them. A row is keyed by `name@version`, so a crate with several releases
/// cannot be covered by any single row, however it is written — such a row is a
/// coarse statement about one release, so it has to be on this roster (audited by
/// `test_multi_copy_exceptions_are_declared_where_the_row_states_them`) rather
/// than pass unnoticed. base64 is that case: the lock resolves 0.13.1 and 0.22.1
/// as well, through independent upstream constraints, and `simd-unsafe` exists
/// only in 0.23, so `base64@0.23` is the whole ban.
const MULTI_COPY_EXCEPTIONS: [&str; 1] = ["base64"];

#[test]
fn test_multi_copy_exceptions_are_declared_where_the_row_states_them() {
    let deny = read("deny.toml");
    let lock = read("Cargo.lock");
    let resolved: Vec<(String, String)> = lockfile_packages(&lock);
    let rows: Vec<String> = numbered(&deny)
        .filter_map(|l| banned_crate_spec(&l))
        .collect();

    let multi_copy = multi_copy_crates(&resolved, &rows);
    let actual: Vec<&str> = multi_copy.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        actual, MULTI_COPY_EXCEPTIONS,
        "a ban row names a crate with several releases in Cargo.lock: cargo-deny matches \
         rows on version, so one row cannot cover every copy and the row is decorative for \
         the copies it does not name. Either resolve the duplicate, or add a row per copy and \
         list the crate in MULTI_COPY_EXCEPTIONS with the reason pinned by the loop below."
    );

    for (name, copies) in &multi_copy {
        // The exception must be documented where a reader of the row sees it.
        let explains = deny.lines().any(|l| {
            l.trim_start().starts_with('{')
                && l.contains(&format!("crate = \"{name}@"))
                && l.contains("duplicate")
        });
        assert!(
            explains,
            "deny.toml: the {name} row(s) name a crate the lock resolves as {copies:?} — the \
             row must carry a `reason` naming the duplicate releases, so nobody mistakes a \
             one-release ban for a whole-crate one"
        );
    }
}

/// Contiguous, not two substrings: a row that keeps the crate but drops one
/// denied feature satisfies a split pin — the same rule
/// `tests/deny_gate_drift.rs` applies to the rows its own scan covers. Living
/// here rather than there is a file-size matter, not a difference in what is
/// pinned: deleting either literal from deny.toml fails this test.
///
/// Per live line, not over a joined blob: `numbered` strips and joins, so the
/// literal can only be required to *start* a line. A blob search would also
/// match it split across the end of one row and the start of the next, which is
/// how the `deny` list of the row above could certify the row below it.
///
/// This subsumes pinning what each ort row bans: the literal carries both TLS
/// features and the crate spec, so a row narrowed to one feature, or to a
/// non-TLS feature, fails here.
#[test]
fn test_the_ort_tls_deny_rows_are_pinned_as_contiguous_literals() {
    let deny = read("deny.toml");

    for crate_name in ["ort", "ort-sys"] {
        let literal = format!(
            "{{ deny = [\"tls-native\", \"tls-native-vendored\"], crate = \"{crate_name}@2.0.0-rc.15\""
        );
        assert!(
            numbered(&deny).any(|l| l.starts_with(&literal)),
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
        // No copy of the crate at all — the row is a green no-op.
        assert!(!resolves_in(&lock, "absent-crate@1.0.0"), "absent crate");
    }

    #[test]
    fn covers_every_copy_demands_all_copies_resolves_in_only_one() {
        let lock = vec![
            ("ort".to_string(), "2.0.0-rc.15".to_string()),
            ("ort".to_string(), "2.0.0-rc.9".to_string()),
        ];
        // The reason for the stricter predicate: one matching copy is enough for
        // cargo-deny to accept the row, and not enough to ban the tree.
        assert!(resolves_in(&lock, "ort@2.0.0-rc.15"), "one copy matches");
        assert!(
            !covers_every_copy(&lock, "ort@2.0.0-rc.15"),
            "the other copy stays unbanned"
        );
        // One prefix on a `.` boundary covers both releases, so it bans both.
        assert!(
            covers_every_copy(&lock, "ort@2.0.0-rc"),
            "prefix covers both"
        );
        // With a single copy the two predicates agree.
        let one = vec![("ort".to_string(), "2.0.0-rc.15".to_string())];
        assert!(resolves_in(&one, "ort@2.0.0-rc.15"));
        assert!(covers_every_copy(&one, "ort@2.0.0-rc.15"));
        // No copies: neither predicate rescues a row naming an absent crate.
        assert!(!resolves_in(&lock, "nope@1.0.0"));
        assert!(!covers_every_copy(&lock, "nope@1.0.0"));
    }

    #[test]
    fn multi_copy_crates_lists_only_rows_whose_crate_has_duplicates() {
        let lock = vec![
            ("base64".to_string(), "0.13.1".to_string()),
            ("base64".to_string(), "0.23.1".to_string()),
            ("ort".to_string(), "2.0.0-rc.15".to_string()),
        ];
        let rows = ["base64@0.23".to_string(), "ort@2.0.0-rc.15".to_string()];
        let multi = multi_copy_crates(&lock, &rows);
        assert_eq!(multi.len(), 1, "only the base64 row names a duplicate");
        // Reported by crate name, so the exception roster is keyed where a row
        // for a second release of the same crate would also land.
        assert_eq!(multi[0].0, "base64");
        assert_eq!(
            multi[0].1,
            ["0.13.1", "0.23.1"],
            "the reported copies are every release the lock resolves"
        );
        // A row whose crate is absent is not a multi-copy row (resolves_in catches it).
        let rows = ["nope@1.0.0".to_string()];
        assert!(multi_copy_crates(&lock, &rows).is_empty());
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
