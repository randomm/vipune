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
//! An integration test rather than a module under `src/embedding/tests/`: this
//! lock reads only `Cargo.toml` + `ci.yml` + docs, so it needs nothing from the
//! crate and, as its own crate, it compiles from `std` alone — no `use super::*`
//! pulling the engine imports of a sibling test file into a module that uses
//! none of them (which is what forced a lint suppression when it lived there),
//! and it runs on both feature sets since integration targets are feature-set
//! agnostic. Same drift class as
//! `test_air_gapped_docs_and_ci_cache_match_embed_revision`.

use std::path::Path;

/// `major.minor` of a `MAJOR.MINOR[.PATCH]` version string, `None` if the
/// first two components aren't both purely numeric. A non-numeric component
/// (a placeholder, a template expression) yields `None` rather than a value
/// that could be compared against the declared floor.
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

/// Drop a trailing `#` YAML comment from a logical line.
///
/// Deliberately simpler than a YAML scanner: vipune's `ci.yml` never puts a
/// `#` inside a quoted scalar, so an unquoted `#` always opens a comment and
/// a whole-line comment is the empty string after this. If a value ever does
/// need a literal `#`, this returns a truncated value and the pin assertion
/// fails loudly rather than passing on a comment fragment.
fn strip_yaml_comment(line: &str) -> &str {
    match line.find('#') {
        Some(index) => line[..index].trim_end(),
        None => line,
    }
}

/// Drop the bodies of `<!-- ... -->` HTML comments, including multi-line ones,
/// while keeping every byte outside them (line structure preserved). A
/// commented-out floor is not quoted at users, so it must not satisfy the docs
/// assertions either.
fn strip_html_comments(text: &str) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<!--") {
        kept.push_str(&rest[..start]);
        let after_open = &rest[start + "<!--".len()..];
        rest = match after_open.find("-->") {
            Some(end) => {
                // Keep the newlines the comment spanned so surrounding lines
                // stay whole lines for the lookups below.
                kept.extend(after_open[..end].chars().filter(|ch| *ch == '\n'));
                &after_open[end + "-->".len()..]
            }
            // Unterminated: everything past the opener is comment.
            None => "",
        };
    }
    kept.push_str(rest);
    kept
}

/// The body of the `msrv:` job: the lines after its `  msrv:` header, up to
/// the next job — the next line whose indent is exactly two spaces followed by
/// a key. Without the end boundary a later job's `toolchain:` or `run:` step
/// would satisfy the assertions below, and a duplicated key would too.
fn msrv_job_body(ci: &str) -> Option<&str> {
    let start = ci.find("\n  msrv:")? + "\n  msrv:".len();
    let after_header = &ci[start..];
    let end = after_header
        .lines()
        .skip(1)
        .take_while(|line| !is_next_job(line))
        .map(|line| line.len() + 1)
        .sum::<usize>();
    Some(after_header[..end].trim_end_matches('\n'))
}

/// A line that closes the current job: exactly two spaces of indent followed
/// by a key (so `  msrv:` opens one, `    name: MSRV` and `  # comment` do
/// not, and neither does a blank line).
fn is_next_job(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("  ") else {
        return false;
    };
    !rest.starts_with(' ') && !rest.is_empty() && !rest.starts_with('#') && rest.contains(':')
}

/// The `toolchain:` pin inside `job`: the value of the first non-comment line
/// whose trimmed form starts with `toolchain:`. A commented-out pin is dead
/// configuration — it must not certify the job.
fn toolchain_pin(job: &str) -> Option<String> {
    for line in job.lines() {
        let code = strip_yaml_comment(line).trim();
        if let Some(rest) = code.strip_prefix("toolchain:") {
            let value = rest
                .trim()
                .trim_matches(|c| c == '"' || c == '\'' || c == ',')
                .to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

/// Whether `job` contains a non-comment step line whose trimmed form is
/// exactly `run: <cmd>` (an optional leading `- ` is allowed). Matching the
/// whole line instead of a substring means the command cannot be satisfied by
/// a comment, a `name:` label, or a cache key that mentions it.
fn has_run_step(job: &str, cmd: &str) -> bool {
    let wanted = format!("run: {cmd}");
    job.lines().any(|line| {
        let code = strip_yaml_comment(line).trim();
        let code = code.strip_prefix("- ").unwrap_or(code).trim();
        code == wanted
    })
}

/// The Cargo.toml `rust-version` value: the first non-comment assignment line
/// with inline comments stripped, minus the quotes TOML puts around it.
fn declared_rust_version(manifest: &str) -> Option<String> {
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("rust-version") else {
            continue;
        };
        let rest = strip_yaml_comment(rest)
            .trim()
            .trim_start_matches('=')
            .trim();
        let value = rest
            .trim_matches(|c| c == '"' || c == '\'' || c == ',')
            .trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

#[test]
fn test_msrv_version_pins_agree() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let repo_root = Path::new(&manifest_dir);

    let cargo_toml =
        std::fs::read_to_string(repo_root.join("Cargo.toml")).expect("read Cargo.toml");
    let ci = std::fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))
        .expect("read CI workflow");
    // The docs floors are read with HTML comments already removed: a floor
    // inside `<!-- ... -->` is not quoted at anyone, so it must not satisfy the
    // assertions below (see assertion 3 and 4).
    let readme = strip_html_comments(
        &std::fs::read_to_string(repo_root.join("README.md")).expect("read README"),
    );
    let contributing = strip_html_comments(
        &std::fs::read_to_string(repo_root.join("CONTRIBUTING.md")).expect("read CONTRIBUTING.md"),
    );

    // 1. Cargo.toml `rust-version` is the authoritative declaration. A commented-out
    //    or non-numeric declaration is no declaration: the test must fail, not
    //    fall back to whatever else mentions the key.
    let declared = declared_rust_version(&cargo_toml)
        .expect("Cargo.toml must declare rust-version on a non-comment line");
    let declared_mm =
        major_minor(&declared).unwrap_or_else(|| panic!("unparsable rust-version: {declared}"));

    // 2. The msrv job must install exactly that toolchain. The pin is read from
    //    the msrv job body alone — the other jobs pin `@stable`/`@master`, so a
    //    stray `toolchain:` in a later job must neither satisfy this assertion nor
    //    be mistaken for the MSRV pin.
    let msrv_job = msrv_job_body(&ci)
        .expect("ci.yml must contain an `msrv:` job at two-space indent under `jobs:`");
    let toolchain = toolchain_pin(msrv_job).expect(
        "msrv job must pin a toolchain on a non-comment `toolchain:` line \
         (dtolnay/rust-toolchain `with: toolchain:`)",
    );
    let toolchain_mm =
        major_minor(&toolchain).unwrap_or_else(|| panic!("unparsable pin: {toolchain}"));
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
    const DEFAULT_CHECK: &str = "cargo check --locked --all-targets";
    const NO_DEFAULT_CHECK: &str = "cargo check --locked --no-default-features --all-targets";
    assert!(
        has_run_step(msrv_job, DEFAULT_CHECK) && has_run_step(msrv_job, NO_DEFAULT_CHECK),
        "the msrv job must run `{DEFAULT_CHECK}` and `{NO_DEFAULT_CHECK}` as their own \
         `run:` steps to check both feature sets at {declared}"
    );
}
