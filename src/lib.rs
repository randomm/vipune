//! vipune - A minimal memory layer for AI agents.
//!
//! This crate provides a local, semantic memory store with conflict detection.
//! All operations are synchronous (no async/await required).
//!
//! # Example
//!
//! ```no_run
//! use vipune::{Config, MemoryStore, MemoryType, MemoryStatus, detect_project};
//!
//! // Initialize memory store
//! let config = Config::default();
//! let mut store = MemoryStore::new(
//!     config.database_path.as_path(),
//!     &config.embedding_model,
//!     config.clone()
//! ).expect("Failed to initialize store");
//!
//! // Detect project ID
//! let project_id = detect_project(None);
//!
//! // Add a memory with conflict detection
//! let result = store.add_with_conflict(&project_id, "Alice works at Microsoft", None, false, MemoryType::Fact, MemoryStatus::Active);
//! match result {
//!     Ok(vipune::AddResult::Added { id }) => println!("Added memory: {}", id),
//!     Ok(vipune::AddResult::Conflicts { .. }) => println!("Conflict detected"),
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//!
//! // Search memories
//! let results = store.search(&project_id, "where does alice work", 10, 0.0, vipune::memory::SearchOptions::default());
//! for memory in results.unwrap() {
//!     println!("{:.2}: {}", memory.similarity.unwrap_or(0.0), memory.content);
//! }
//! ```
//!
//! # Mutability Requirements
//!
//! Methods that generate embeddings (`add`, `search`, `update`) require `&mut self`
//! because the embedding engine internally mutates state for ONNX tensor allocations.

pub mod config;
pub mod embedding;
pub mod errors;
pub mod memory;
pub mod memory_types; // Library-only: batch ingest API (not used in CLI)
pub mod project;
mod rrf;
mod sqlite;
mod temporal;

#[cfg(feature = "mcp")]
pub mod mcp;

// Re-export public API
pub use config::Config;
pub use embedding::{EMBED_MODEL_ID, EMBED_MODEL_REVISION, EMBEDDING_DIMS, EmbeddingEngine};
pub use errors::Error;
pub use memory::lifecycle::{MemoryStatus, MemoryType};
pub use memory::store::{MAX_INPUT_LENGTH, MAX_SEARCH_LIMIT};
pub use memory::{MemoryStore, UpdateParams};
pub use memory_types::{
    AddResult, BatchIngestItemResult, BatchIngestResult, ConflictMemory, IngestPolicy,
}; // Library-only: conflict detection and batch ingest types
pub use project::{detect_project, detect_project_at};
pub use sqlite::Database;
pub use sqlite::Memory;
pub use sqlite::embedding::EmbeddingClass;
pub use sqlite::embedding::classify_embedding;

#[cfg(test)]
mod integration_tests {
    /// Recursively collect all `.rs` files under a directory.
    fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// Check if a `*_tests.rs` file is wired as a module by scanning the candidate
    /// parent files for `#[path = "filename"]` or `mod full_test_name;` declarations.
    ///
    /// We match on individual lines to avoid false positives from doc comments,
    /// string literals, or production module names (e.g. `mod temporal;` does NOT
    /// wire `temporal_tests.rs`).
    fn is_test_module_wired(test_filename: &str, candidates: &[std::path::PathBuf]) -> bool {
        let test_mod_name = test_filename.trim_end_matches(".rs");
        let path_pat = format!("path = \"{}\"", test_filename);
        let mod_pat = format!("mod {}", test_mod_name);

        for candidate in candidates {
            let Ok(content) = std::fs::read_to_string(candidate) else {
                continue;
            };
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.contains(&path_pat) {
                    return true;
                }
                if trimmed.starts_with(&mod_pat) {
                    let after = trimmed[mod_pat.len()..].trim_start();
                    if after.is_empty() || after.starts_with(';') || after.starts_with('{') {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Guard against orphaned `*_tests.rs` files that exist on disk but are not
    /// declared as Rust modules — the same bug that silently disabled tests since
    /// PR #120 (issue #173).
    ///
    /// Every `src/*_tests.rs` file must be referenced via `#[path]` in its parent
    /// module or declared as `mod test_name;` in a sibling file.
    /// This test fails CI if any are missing.
    #[test]
    fn all_test_modules_are_wired() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let src_dir = std::path::Path::new(&manifest_dir).join("src");

        let mut all_files = Vec::new();
        collect_rs_files(&src_dir, &mut all_files);

        let mut orphaned = Vec::new();

        for path in &all_files {
            let Some(filename) = path.file_name().and_then(|f| f.to_str()) else {
                continue;
            };
            if !filename.ends_with("_tests.rs") {
                continue;
            }

            let parent_dir = path.parent().unwrap();
            let lib_rs = src_dir.join("lib.rs");

            let mut candidates: Vec<std::path::PathBuf> = Vec::new();
            if lib_rs.exists() {
                candidates.push(lib_rs);
            }
            // Check all .rs siblings (parent file, mod.rs, tests.rs, etc.)
            if let Ok(entries) = std::fs::read_dir(parent_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.ends_with(".rs") && name != filename {
                            if !candidates.contains(&p) {
                                candidates.push(p);
                            }
                        }
                    }
                }
            }

            if !is_test_module_wired(filename, &candidates) {
                orphaned.push(
                    path.strip_prefix(&src_dir)
                        .unwrap_or(path)
                        .display()
                        .to_string(),
                );
            }
        }

        assert!(
            orphaned.is_empty(),
            "Orphaned test module(s) found — these files exist but are not declared as modules:\n  {}\n\nFix: add `#[cfg(test)] #[path = \"filename\"]] mod test_name;` to the parent module,\nor declare as `mod test_name;` in the containing module's mod.rs.",
            orphaned.join("\n  "),
        );
    }

    /// Unit tests for the orphan-detection logic using synthetic data.
    /// These prove the guard can actually detect an orphaned module.
    mod detection_tests {
        use super::*;

        #[test]
        fn detects_path_wired_module() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(
                dir.path().join("parent.rs"),
                r#"#[cfg(test)]
#[path = "parent_tests.rs"]
mod parent_tests;"#,
            )
            .unwrap();
            std::fs::write(dir.path().join("parent_tests.rs"), "// test file").unwrap();

            assert!(is_test_module_wired(
                "parent_tests.rs",
                &[dir.path().join("parent.rs")]
            ));
        }

        #[test]
        fn detects_mod_declared_wired_module() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(
                dir.path().join("mod.rs"),
                r#"#[cfg(test)]
mod merge_tests;"#,
            )
            .unwrap();
            std::fs::write(dir.path().join("merge_tests.rs"), "// test file").unwrap();

            assert!(is_test_module_wired(
                "merge_tests.rs",
                &[dir.path().join("mod.rs")]
            ));
        }

        #[test]
        fn detects_orphaned_module() {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("parent.rs"), "fn production() {}").unwrap();
            std::fs::write(dir.path().join("parent_tests.rs"), "// test file").unwrap();

            assert!(!is_test_module_wired(
                "parent_tests.rs",
                &[dir.path().join("parent.rs")]
            ));
        }

        #[test]
        fn does_not_match_production_module_name() {
            let dir = tempfile::tempdir().unwrap();
            // `mod temporal;` is the PRODUCTION module — must NOT wire `temporal_tests.rs`.
            std::fs::write(dir.path().join("lib.rs"), "mod temporal;\nmod project;\n").unwrap();
            std::fs::write(dir.path().join("temporal_tests.rs"), "// test").unwrap();

            assert!(!is_test_module_wired(
                "temporal_tests.rs",
                &[dir.path().join("lib.rs")]
            ));
        }

        #[test]
        fn does_not_match_doc_comment_mention() {
            let dir = tempfile::tempdir().unwrap();
            // Doc comment mentions the test filename — must NOT count as wiring.
            std::fs::write(
                dir.path().join("lib.rs"),
                "/// References project_tests.rs in documentation.\nmod project;\n",
            )
            .unwrap();
            std::fs::write(dir.path().join("project_tests.rs"), "// test").unwrap();

            assert!(!is_test_module_wired(
                "project_tests.rs",
                &[dir.path().join("lib.rs")]
            ));
        }

        #[test]
        fn detects_wired_in_sibling_tests_file() {
            let dir = tempfile::tempdir().unwrap();
            // tests.rs wires external test files (the sqlite/ pattern).
            std::fs::write(
                dir.path().join("tests.rs"),
                r#"#[cfg(test)]
#[path = "fts_tests.rs"]
mod fts;"#,
            )
            .unwrap();
            std::fs::write(dir.path().join("fts_tests.rs"), "// test file").unwrap();

            assert!(is_test_module_wired(
                "fts_tests.rs",
                &[dir.path().join("tests.rs")]
            ));
        }

        #[test]
        fn detects_recursive_orphan_in_subdirectory() {
            let dir = tempfile::tempdir().unwrap();
            let sub = dir.path().join("subdir");
            std::fs::create_dir_all(&sub).unwrap();
            std::fs::write(sub.join("mod.rs"), "mod production;").unwrap();
            std::fs::write(sub.join("subdir_tests.rs"), "// orphaned test").unwrap();

            // collect_rs_files finds it recursively.
            let mut files = Vec::new();
            collect_rs_files(dir.path(), &mut files);
            assert!(
                files
                    .iter()
                    .any(|f| f.file_name().unwrap() == "subdir_tests.rs")
            );

            assert!(!is_test_module_wired(
                "subdir_tests.rs",
                &[sub.join("mod.rs")]
            ));
        }
    }
}
