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

/// The `crate = "…"` spec of a `[bans] features` row, else `None`.
///
/// Requiring the leading brace keeps a bare `crate = "…"` outside a row from
/// reading as one, and requiring quotes keeps an unquoted (so invalid-TOML)
/// `crate = a@1.2.3` typo from looking live. Comments are already gone (`numbered`).
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

/// `[[package]]` blocks of Cargo.lock as (name, version) pairs.
///
/// Each block runs to the next header, so name/version are the block's own
/// top-level keys: an entry in a `dependencies` list is a quoted bare string and
/// never decodes as `name = "…"`.
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
/// release the row intends: no `@` at all, no crate name, or a version carrying
/// an operator or wildcard.
///
/// A *version string* is deliberately not validated beyond that: cargo-deny
/// accepts both an exact version (`ort@2.0.0-rc.15`, pre-release included) and a
/// `major.minor` prefix (`base64@0.23`, verified to bite with 0.20.2 on a probe
/// crate enabling ort's `tls-native`), so requiring `x.y.z` here would reject
/// rows that work.
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

/// The rows of `lock` for crate `name` that `spec_version` matches — the copies
/// cargo-deny would apply the ban to.
///
/// Exposed as a name list rather than only as `covers_every_copy`'s all/none
/// test because a `base64@0.13.1` row bites exactly one copy while a `base64@0.99`
/// row bites none, and the row assertion must reject both for different reasons.
fn matching_copies<'a>(
    lock: &'a [(String, String)],
    name: &str,
    spec_version: &str,
) -> Vec<&'a str> {
    copies_of(lock, name)
        .into_iter()
        .filter(|v| version_matches(v, spec_version))
        .collect()
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

/// The two halves the row assertion demands, spelled out because an `assert!` on
/// their conjunction cannot say which one a row failed: `resolves_in` and
/// `covers_every_copy`.
///
/// `covers` reports the **ungarnished** coverage — no roster exemption — because
/// a roster row is the common case and saying `ok` for it would hide which copies
/// the ban does not reach. The caller ORs the exemption in, so pass is
/// `resolves && (covers || exempt)`, never this `covers` alone.
fn row_verdict(lock: &[(String, String)], spec: &str) -> (&'static str, bool, bool) {
    let resolves = resolves_in(lock, spec);
    let covers = covers_every_copy(lock, spec);
    let why = match (resolves, covers) {
        (true, true) => "ok",
        (false, false) => "resolves in no copy of its crate and leaves an extra copy unbanned",
        (false, true) => "resolves in no copy of its crate that Cargo.lock resolves",
        (true, false) => "does not cover every copy of its crate that Cargo.lock resolves",
    };
    (why, resolves, covers)
}

/// Whether the row assertion passes `spec`: both halves of `row_verdict`, with the
/// declared roster exemption ORed into the coverage half only.
fn row_passes(lock: &[(String, String)], spec: &str) -> bool {
    let (_, resolves, covers) = row_verdict(lock, spec);
    resolves && (covers || MULTI_COPY_EXCEPTIONS.contains(&spec_crate(spec)))
}

/// Whether `spec` resolves at all: its crate has at least one copy in the lock
/// (otherwise cargo-deny ignores the row silently) and its version matches one of
/// them with `version_matches`.
///
/// Pure "any copy": the row test asserts the stricter `covers_every_copy`, so this
/// predicate stays probeable on its own.
fn resolves_in(lock: &[(String, String)], spec: &str) -> bool {
    let Some((name, version)) = spec.rsplit_once('@') else {
        return false;
    };
    copies_of(lock, name)
        .iter()
        .any(|v| version_matches(v, version))
}

/// Whether `spec` bans **every** copy of its crate that Cargo.lock resolves: at
/// least one copy exists and each matches with `version_matches`.
///
/// Per release, not "any copy matches": cargo-deny matches a `name@version` row
/// against every package of that name, so a second copy the row does not name
/// keeps its banned features while the gate prints `bans ok` — and
/// `multiple-versions = "warn"` makes several copies of one crate normal in this
/// lock, so an extra unbanned `ort` must fail here rather than pass.
///
/// A crate whose duplicates no one row can cover needs a `MULTI_COPY_EXCEPTIONS`
/// entry, audited by `multi_copy_exceptions_are_declared_where_the_row_states_them`;
/// the exemption lives in that assertion, so
/// this predicate keeps meaning "all copies match" and its probe stays meaningful.
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
/// by name.
///
/// The crate name is reported, not the whole spec: the exception roster is keyed
/// on the crate, so a row for the same crate at a different version is caught
/// rather than silently re-labelled.
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
        // Both halves, not one: `MULTI_COPY_EXCEPTIONS` exempts a row from *coverage*
        // of every copy, and it must not exempt it from resolving at all — otherwise a
        // roster crate could be banned at a version the lock does not resolve
        // (`base64@0.99`), cargo-deny reports "unknown feature" for the named copy and
        // prints `bans ok`, and the row is a green no-op. A roster row that resolves at
        // some other release (`base64@0.13.1`) is not caught by this assertion either —
        // it is a legitimate one-release ban — and is instead answered by the roster
        // audit below, which requires the row to name every release the lock resolves.
        let (why, resolves, covers) = row_verdict(&resolved, spec);
        let exempt = MULTI_COPY_EXCEPTIONS.contains(&spec_crate(spec));
        assert!(
            resolves && (covers || exempt),
            "deny.toml: feature-ban row crate = \"{spec}\" — {why}{}: cargo-deny ignores a \
             row matching nothing silently (verified: a row naming a crate the lock does not \
             resolve prints `bans ok`), and an extra copy the row does not name keeps its \
             banned features. Bump the row together with the pin in Cargo.toml, fix the crate \
             name, or resolve the duplicate. A crate whose duplicates genuinely cannot be \
             covered by one row needs an entry in MULTI_COPY_EXCEPTIONS — which exempts the \
             row from covering every copy, never from resolving in the lock — and \
             test_multi_copy_exceptions_are_declared_where_the_row_states_them then audits \
             that roster.",
            if exempt && !covers {
                " (its crate is on MULTI_COPY_EXCEPTIONS, so coverage is exempt but \
                 resolving is not)"
            } else {
                ""
            }
        );
    }
}

/// The only crates a ban row may name while Cargo.lock resolves several releases
/// of them.
///
/// A row is keyed by `name@version`, so a crate with several releases cannot be
/// covered by any single row, however it is written — such a row is a coarse
/// statement about one release, so it has to be on this roster (audited by
/// `multi_copy_exceptions_are_declared_where_the_row_states_them`, which also
/// requires the row's `reason` to name **every** one of those releases) rather
/// than pass unnoticed. base64 is that case: the lock resolves 0.13.1 and 0.22.1
/// as well, through independent upstream constraints, and `simd-unsafe` exists
/// only in 0.23, so `base64@0.23` is the whole ban. Naming every release is what
/// stops a row that excuses itself against two of three copies from reading as a
/// complete account of the duplicates.
///
/// Being on the roster exempts a row from covering **every** copy only; the row
/// must still resolve in the lock. Without that, a roster crate could be banned at
/// a version nobody resolves (`base64@0.99`) and cargo-deny would print `bans ok`
/// for a row that enforces nothing. Pinned by `row_verdict_separates_the_two_halves`.
const MULTI_COPY_EXCEPTIONS: [&str; 1] = ["base64"];

/// Whether `line` names every version in `copies` as a **token**, not as a
/// substring.
///
/// Two substring traps this avoids: `contains("0.23.1")` is satisfied by a reason
/// that says only `0.23` while `0.23.1` appears elsewhere on the line (a real
/// mutation passed the substring form), and `contains("0.23")` is satisfied by
/// `0.23.1`. A token is a maximal run of digits, `.`, `-` and letters — `-`
/// because a pre-release such as `2.0.0-rc.15` is itself a resolved version, so a
/// hyphen must not end the token — so `0.23.1,` and `(0.23.1)` count while
/// `release-0.23.1` and a bare `0.23` do not count for `0.23.1`.
fn names_all_versions(line: &str, copies: &[&str]) -> bool {
    let tokens: Vec<&str> = line
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c.is_ascii_alphabetic()))
        .filter(|t| !t.is_empty())
        .collect();
    copies.iter().all(|version| tokens.contains(version))
}

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
        // The exception must be documented where a reader of the row sees it, and it
        // must state every release the lock resolves. A bare `contains("duplicate")`
        // certifies the shape of the excuse, not its accuracy: a row whose reason names
        // two of three copies reads as a complete account while the third is invisible
        // to anyone auditing the ban.
        let explains = deny.lines().any(|l| {
            l.trim_start().starts_with('{')
                && l.contains(&format!("crate = \"{name}@"))
                && names_all_versions(l, copies)
        });
        assert!(
            explains,
            "deny.toml: the {name} row(s) name a crate the lock resolves as {copies:?} — the \
             row must carry a `reason` that names **every** one of those releases (each of \
             {copies:?} must appear on the row line), so nobody mistakes a one-release ban \
             for a whole-crate one"
        );
    }
}

/// Contiguous, not two substrings: a row that keeps the crate but drops one denied
/// feature satisfies a split pin — the same rule `tests/deny_gate_drift.rs` applies
/// to the rows its own scan covers. Living here rather than there is a file-size
/// matter, not a difference in what is pinned: deleting either literal from
/// deny.toml fails this test.
///
/// Per live line, not over a joined blob: `numbered` strips and joins, so the
/// literal can only be required to *start* a line — a blob search would also match
/// it split across the end of one row and the start of the next, which is how the
/// `deny` list of the row above could certify the row below it.
///
/// This subsumes pinning what each ort row bans: the literal carries both TLS
/// features and the crate spec, so a row narrowed to one feature, or to a non-TLS
/// feature, fails here.
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

/// Unit probes of the scanners above, kept in a sibling file so each stays under
/// the per-file cap while still reaching the helpers as `super::` items. A
/// subdirectory keeps Cargo from auto-discovering it as a second test target.
#[path = "deny_feature_ban_lock/helper_probes.rs"]
mod helper_probes;
