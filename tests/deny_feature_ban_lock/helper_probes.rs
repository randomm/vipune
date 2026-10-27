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
fn names_all_versions_needs_the_version_as_a_whole_token() {
    let copies = ["0.13.1", "0.22.1", "0.23.1"];
    // The committed row line names all three.
    let good = "{ deny = [\"simd-unsafe\"], crate = \"base64@0.23\", reason = \"the lock resolves duplicate base64 releases 0.13.1, 0.22.1 and 0.23.1\" },";
    assert!(names_all_versions(good, &copies));
    // The mutation that made this check token-based: dropping 0.23.1 from the
    // reason must fail even though `0.23` still appears twice on the line — once
    // in the crate spec and once in `pins the 0.23 copy`.
    let dropped = "{ deny = [\"simd-unsafe\"], crate = \"base64@0.23\", reason = \"base64 0.23 decodes untrusted import files; simd-unsafe is a 0.23 default feature and must stay off; the lock resolves duplicate base64 releases 0.13.1 and 0.22.1, so this row pins the 0.23 copy that has the feature\" },";
    assert!(
        !names_all_versions(dropped, &copies),
        "a row that never names 0.23.1 must not certify the 0.23.1 release"
    );
    // A version glued to a letter is part of a larger token, not a version token.
    assert!(
        !names_all_versions("base64x0.23.1 0.13.1 0.22.1", &copies),
        "a substring inside a larger token does not count"
    );
    assert!(
        !names_all_versions("release-0.23.1 0.13.1 0.22.1", &copies),
        "a hyphen does not end a version token, since a pre-release keeps its own"
    );
    // Punctuation around a version is a separator, not part of the token.
    assert!(names_all_versions("v (0.13.1) [0.22.1]; 0.23.1", &copies));
    // Empty roster entry is vacuously satisfied — the caller only reaches this
    // for a crate the lock resolves more than once.
    assert!(names_all_versions("anything", &[]));
}

#[test]
fn row_verdict_separates_the_two_halves() {
    // Every base64 release the committed lock resolves, so a spec's verdict here is
    // the verdict the row test would reach on the real tree.
    let lock = lockfile_packages(&read("Cargo.lock"));
    let base64_copies = copies_of(&lock, "base64");
    assert!(
        base64_copies.len() > 1,
        "probe needs a multi-copy base64: {base64_copies:?}"
    );
    // The committed roster row: resolves, and the copies it does not name are
    // exempted by declaration — so it passes, while the verdict still says plainly
    // that it does not cover every copy.
    let (why, resolves, covers) = row_verdict(&lock, "base64@0.23");
    assert!(resolves, "the roster row must resolve: {why}");
    assert!(
        !covers,
        "the roster row covers one copy of {base64_copies:?}: {why}"
    );
    assert!(why.contains("does not cover every copy"), "{why}");
    assert!(
        row_passes(&lock, "base64@0.23"),
        "the roster exempts the coverage half"
    );
    // The hole the roster used to open. Both specs below are exempt from
    // *coverage* by the roster and both used to pass the row assertion:
    // `base64@0.99` names no release at all (cargo-deny prints `bans ok` for a row
    // matching nothing), and `base64@0.13.1` bans a copy that has no `simd-unsafe`
    // to ban. The first is caught by `resolves_in`; the second resolves, so it is
    // caught only because the lock resolves *other* base64 copies the row does not
    // name — which is why the assertions below are driven off `matching_copies`
    // rather than off a hardcoded version list that would rot with the lock.
    for dead in ["base64@0.99", "base64@0.13", "base64@0.13.1"] {
        let spec_version = dead.rsplit_once('@').expect("spec has an @").1;
        let bites = matching_copies(&lock, "base64", spec_version);
        let (why, resolves, covers) = row_verdict(&lock, dead);
        assert!(!covers, "{dead} must not cover every copy: {why}");
        assert_eq!(
            resolves,
            !bites.is_empty(),
            "{dead}: {why} — a row resolves exactly when it bites a copy"
        );
        if bites.is_empty() {
            assert!(
                why.contains("resolves in no copy"),
                "{dead} must name the resolving half: {why}"
            );
        } else {
            assert!(
                bites.len() < base64_copies.len(),
                "{dead} bites every copy ({bites:?}) — the not-covering branch needs a \
                 spec that leaves one out of {base64_copies:?}"
            );
            assert!(
                why.contains("does not cover every copy"),
                "{dead} must name the coverage half: {why}"
            );
        }
    }
    // A spec that bites every copy covers the crate without the roster: this is
    // what an `ort` row must do, and what a base64 row can only do if upstream
    // constraints ever collapse to one release.
    let single = vec![("ort".to_string(), "2.0.0-rc.15".to_string())];
    let (why, resolves, covers) = row_verdict(&single, "ort@2.0.0-rc.15");
    assert!((resolves, covers) == (true, true), "{why}");
    assert_eq!(why, "ok", "no roster exemption is in play");
    assert!(row_passes(&single, "ort@2.0.0-rc.15"));
    // Non-roster crate with an extra copy: resolves but leaves one unbanned.
    let two = vec![
        ("ort".to_string(), "2.0.0-rc.15".to_string()),
        ("ort".to_string(), "2.0.0-rc.9".to_string()),
    ];
    let (why, resolves, covers) = row_verdict(&two, "ort@2.0.0-rc.15");
    assert!(resolves && !covers, "{why}");
    assert!(why.contains("does not cover every copy"), "{why}");
    // A crate absent from the lock fails both halves.
    let (why, resolves, covers) = row_verdict(&two, "nope@1.0.0");
    assert!(!resolves && !covers, "{why}");
    assert!(why.contains("resolves in no copy"), "{why}");
    // The roster never rescues a row whose crate is absent from the lock.
    assert!(!row_passes(&lock, "nope@1.0.0"));
    // And it never rescues a row that resolves in no copy: the two specs above are
    // roster crates, so coverage is exempt and `resolves_in` alone decides.
    assert!(!row_passes(&lock, "base64@0.99"), "the reported hole");
    assert!(
        row_passes(&lock, "base64@0.13.1"),
        "a roster row that resolves passes: it is a legitimate one-release ban, not \
         the hole — the hole is a roster row that resolves in nothing"
    );
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
