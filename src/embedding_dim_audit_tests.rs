//! One-source-of-truth audit for embedding dimensions (issue #217, task-b).
//!
//! `crate::embedding::EMBEDDING_DIMS` is the single source of truth for the
//! 384-dim embedding dimension and `EMBEDDING_DIMS * 4` for the BLOB size.
//! This test fails if a duplicate survives in production `src/` code:
//!
//! - a `1536` literal (the byte size) in a non-test file — byte-size
//!   arithmetic must go through `EMBEDDING_DIMS * 4`, never a hand-written
//!   1536;
//! - a `const EMBEDDING_DIMS` re-declaration outside `src/embedding.rs`
//!   (private shadowing was the specific pattern that hid in
//!   `src/sqlite/embedding.rs` and `src/commands/{export,import}.rs`).
//!
//! Test files and comment lines are out of scope by the issue's acceptance
//! criterion (the audit "scans src/ non-test files only; test files and
//! comments are out of scope"), so a `1536` inside a `#[cfg(test)]` module
//! or on a comment line is permitted.

#![cfg(test)]

use std::path::{Path, PathBuf};

/// Recursively collect all `.rs` files under a directory.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir);
    let Ok(entries) = entries else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// The single file that may declare `EMBEDDING_DIMS` as a `const`.
fn is_embedding_rs(path: &Path) -> bool {
    path.file_name().and_then(|n| n.to_str()) == Some("embedding.rs")
        && path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            == Some("src")
}

/// Whether `line` is a comment line (a `//` comment, including the `//!` and
/// `///` doc forms).
fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// Whether `line` belongs to this audit module's own file (its search
/// patterns and message strings contain the exact substrings the audit looks
/// for, and must not be flagged as violations).
fn is_audit_file(rel: &str) -> bool {
    rel.ends_with("embedding_dim_audit_tests.rs")
}

#[test]
fn no_private_embedding_dims_or_1536_literals_survive_in_src() {
    let manifest_dir_str =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set in tests");
    let manifest_dir = Path::new(&manifest_dir_str);
    let src_dir = manifest_dir.join("src");
    let mut files: Vec<PathBuf> = Vec::new();
    collect_rs_files(&src_dir, &mut files);

    let mut violations: Vec<String> = Vec::new();

    for file in &files {
        let rel = file
            .strip_prefix(manifest_dir)
            .unwrap_or(file.as_path())
            .display()
            .to_string();
        // This audit file is out of scope: its pattern strings contain the
        // exact substrings it hunts for.
        if is_audit_file(&rel) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };

        // Whole-file test files (a `cfg(test)` attribute in the first ~10
        // lines — either `#![cfg(test)]` as in `export_tests.rs` / `import_tests.rs`
        // or a `#[cfg(test)] mod …` wrapper as in `crud_tests.rs`) are test
        // files by the acceptance criterion — out of scope for the audit.
        let all_lines: Vec<&str> = text.lines().collect();
        let mut file_is_test = false;
        for pre in all_lines.iter().take(10) {
            let t = pre.trim();
            if t == "#![cfg(test)]" || t == "#[cfg(test)]" {
                file_is_test = true;
                break;
            }
        }
        let mut in_test_mod = file_is_test;

        for (idx, line) in all_lines.iter().enumerate() {
            let line_no = idx + 1;
            let trimmed = line.trim_start();

            if in_test_mod {
                continue;
            }

            // Track test-module boundaries. `#[cfg(test)]` marks a test
            // module (every test module in this repo carries it); the flag
            // clears at the next `mod` item after it.
            if trimmed == "#[cfg(test)]" {
                in_test_mod = true;
                continue;
            }
            if trimmed.starts_with("mod ") || trimmed.starts_with("pub mod ") {
                in_test_mod = false;
                continue;
            }

            // Comment lines are out of scope (comments excluded by the
            // acceptance criterion).
            if is_comment_line(line) {
                continue;
            }

            // 1) No private re-declaration of EMBEDDING_DIMS outside
            //    src/embedding.rs.
            if trimmed.contains("const EMBEDDING_DIMS") && !is_embedding_rs(file) {
                violations.push(format!(
                    "{rel}:{line_no}: private `const EMBEDDING_DIMS` re-declaration: {trimmed}"
                ));
            }

            // 2) No literal 1536 in non-test, non-comment code.
            //    `EMBEDDING_DIMS * 4` is the sanctioned form.
            if trimmed.contains("1536") {
                violations.push(format!(
                    "{rel}:{line_no}: literal 1536 (use `EMBEDDING_DIMS * 4`): {trimmed}"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "one-source-of-truth violations (EMBEDDING_DIMS must come from crate::embedding):\n{}",
        violations.join("\n")
    );
}
