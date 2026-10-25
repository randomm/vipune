//! Model-free drift lock for the Rust version pins (issue #257).
//!
//! The MSRV is stated in five places, and only a test can keep them together:
//! `rust-version` in `Cargo.toml` (the promise cargo enforces for users), the
//! toolchain pin of the `msrv` job in `.github/workflows/ci.yml` (the gate
//! that is supposed to verify that promise), the two `cargo check` steps that
//! job runs, the minimum in README.md, and the prerequisites floor in
//! CONTRIBUTING.md. Every one of the last four is a hand-copied string or step,
//! so any of them can silently drift from the declaration.
//!
//! Deliberately independent of `model_free.rs`: that file is compiled under
//! `--no-default-features` too, and this lock reads only
//! `Cargo.toml` + `ci.yml` + docs — all always present regardless of features
//! — so it guards the MSRV on both feature sets without dragging the engine
//! imports into the tokio-free build. Same drift class as
//! `test_air_gapped_docs_and_ci_cache_match_embed_revision`.

use std::path::Path;

/// `major.minor` of a `MAJOR.MINOR[.PATCH]` version string, `None` if the
/// first two components aren't both numeric.
fn major_minor(version: &str) -> Option<String> {
    let mut parts = version.split('.');
    let major = parts.next()?.trim();
    let minor = parts.next()?.trim();
    if major.is_empty()
        || minor.is_empty()
        || !major.bytes().all(|b| b.is_ascii_digit())
        || !minor.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some(format!("{major}.{minor}"))
}

/// The value after `key` on a line, trimmed of the quotes/commas/whitespace
/// TOML and YAML put around it. `key` is matched as a whole word (the
/// character before and after it must not be alphanumeric or `_`), so
/// `rust-version` cannot accidentally be found inside another key.
fn value_after_key(line: &str, key: &str) -> String {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(offset) = line[from..].find(key) {
        let start = from + offset;
        let end = start + key.len();
        let boundary_before = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let boundary_after = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if boundary_before && boundary_after {
            return line[end..]
                .trim()
                .trim_matches(|c| c == '=' || c == ':' || c == '"' || c == ',' || c == ' ')
                .to_string();
        }
        from = end;
    }
    String::new()
}

/// First line containing `needle`, so an assertion can name the offending line
/// instead of only reporting "the file doesn't contain X".
fn line_containing(text: &str, needle: &str) -> Option<String> {
    text.lines()
        .find(|line| line.contains(needle))
        .map(str::to_string)
}

#[test]
fn test_msrv_version_pins_agree() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let repo_root = Path::new(&manifest_dir);

    let cargo_toml =
        std::fs::read_to_string(repo_root.join("Cargo.toml")).expect("read Cargo.toml");
    let ci = std::fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))
        .expect("read CI workflow");
    let readme = std::fs::read_to_string(repo_root.join("README.md")).expect("read README");
    let contributing =
        std::fs::read_to_string(repo_root.join("CONTRIBUTING.md")).expect("read CONTRIBUTING.md");

    // 1. Cargo.toml `rust-version` is the authoritative declaration.
    let declared_line =
        line_containing(&cargo_toml, "rust-version").expect("Cargo.toml must declare rust-version");
    let declared = value_after_key(&declared_line, "rust-version");
    let declared_mm = major_minor(&declared)
        .unwrap_or_else(|| panic!("unparsable rust-version: {declared_line}"));

    // 2. The msrv job must install exactly that toolchain. The pin is read from
    //    the msrv job body, not the whole file: the other jobs pin
    //    `@stable`/`@master`, and a stray `toolchain:` elsewhere must neither
    //    satisfy this assertion nor be mistaken for the MSRV pin.
    let msrv_job = ci
        .split("\n  msrv:")
        .nth(1)
        .unwrap_or_else(|| panic!("ci.yml must contain an `msrv:` job"));
    let toolchain_line = line_containing(msrv_job, "toolchain:")
        .expect("msrv job must pin a toolchain (dtolnay/rust-toolchain `with: toolchain:`)");
    let toolchain = value_after_key(&toolchain_line, "toolchain:");
    let toolchain_mm =
        major_minor(&toolchain).unwrap_or_else(|| panic!("unparsable pin: {toolchain_line}"));
    assert_eq!(
        toolchain_mm, declared_mm,
        "CI msrv job pins toolchain {toolchain} but Cargo.toml declares rust-version {declared} — \
         the gate would certify a different Rust than the one we promise users"
    );

    // 3. README must state the same minimum to users.
    let readme_floor = format!("Requires the [Rust toolchain](https://rustup.rs) ({declared}+)");
    assert!(
        readme.contains(&readme_floor),
        "README.md must state the MSRV as `{readme_floor}` — drift between the declared \
         rust-version ({declared}) and what users are told"
    );

    // 4. CONTRIBUTING.md prerequisites must state the same floor for contributors.
    let contributing_floor = format!("Rust {declared}+");
    assert!(
        contributing.contains(&contributing_floor),
        "CONTRIBUTING.md prerequisites must state `Rust {declared}+` (the declared rust-version) — \
         contributors would otherwise install a toolchain the MSRV gate rejects"
    );

    // 5. The gate must check BOTH feature sets, or the `--no-default-features`
    //    tree can rot below the declared floor unnoticed (`mcp` is a default
    //    feature, so the two builds have different dependency graphs).
    assert!(
        msrv_job.contains("cargo check --locked --all-targets")
            && msrv_job.contains("cargo check --locked --no-default-features --all-targets"),
        "the msrv job must check both the default and --no-default-features sets at {declared}"
    );
}
