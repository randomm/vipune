//! The TempDir-leak guard for the two store builders in this module (issue #278).
//!
//! `MemoryStore::test_store` and `MemoryStore::test_db_path` used to
//! `std::mem::forget` their `TempDir`, so every `cargo test` run left one
//! `.tmpXXXXXX/` directory (holding a SQLite DB) behind in `TMPDIR`. On a host
//! with a quota on the temp filesystem that is not cosmetic: tens of thousands
//! of leaked dirs filled the operator's quota and made unrelated runs fail with
//! SQLite `disk I/O error` and `EDQUOT`.
//!
//! The predicate is "the directory the helper made is gone once the helper's
//! return value is dropped". Counting `.tmp*` entries in `std::env::temp_dir()`
//! cannot express that — `cargo test` runs these on many threads, so every
//! other test's live temp dirs appear and disappear in the same window and
//! would be miscounted as leaks either way. Each test therefore identifies its
//! *own* directory by name, and pins the one fact a `mem::forget` would break:
//! it is still there while owned and gone after `drop`.

use super::MemoryStore;

#[test]
fn test_store_drops_its_temp_dir_with_the_store() {
    // Bound with `mut` so the writes below reach the store through `DerefMut`
    // rather than the shared deref: 26 call sites bind
    // `let mut store = MemoryStore::test_store();`, so both deref impls are part
    // of what this file pins — a broken `DerefMut` would not compile here.
    let mut store = MemoryStore::test_store();
    // A `&mut self` method on `MemoryStore`, driven through the `TestStore`
    // binding: this is the arm `DerefMut` exists for.
    store.set_test_embedder(Box::new(crate::memory::crud::test_fake_embedder));
    let dir_path = store.dir_path().to_path_buf();
    assert!(
        dir_path.is_dir(),
        "the helper's directory must exist while the store is alive"
    );

    // Do something real with the store first: the point is that a *used*
    // store's directory is gone once the store itself drops, so the connection
    // has to be open (and able to write) while the dir is alive.
    //
    // Enum-typed lifecycle values rather than bare `&str`: `Database::insert`
    // takes both columns as `&str` and the schema has no CHECK constraint behind
    // them (validation lives on the `memory/` validating edges, not here), so a
    // typo like `"atcive"` would store a row no read path recognises — and this
    // test's assertion (the directory is gone) would still pass on it. Same
    // pattern `mcp/tests.rs` uses.
    use crate::memory::lifecycle::{MemoryStatus, MemoryType};
    store
        .db
        .insert(
            "leak-guard",
            "a row written before the dir is dropped",
            &vec![0.1f32; 384],
            None,
            MemoryType::Fact.as_str(),
            MemoryStatus::Active.as_str(),
        )
        .expect("insert through the test store");

    // And one through the store's own `&mut self` surface, so the directory is
    // proven to outlive a write that goes via `DerefMut` rather than the `db`
    // field (`insert` above only needs `&self`).
    let ingested = store
        .batch_ingest(
            "leak-guard",
            vec![("a second row, written mutably", None)],
            crate::memory_types::IngestPolicy::Force,
        )
        .expect("batch_ingest through the test store");
    assert_eq!(
        ingested.results.len(),
        1,
        "the mutable write must actually have gone through to the store"
    );

    drop(store);

    assert!(
        !dir_path.exists(),
        "MemoryStore::test_store left {} behind after drop — the TempDir is \
         being forgotten again (issue #278)",
        dir_path.display()
    );
}

#[test]
fn test_db_path_returns_an_owned_temp_dir() {
    let (dir, path) = MemoryStore::test_db_path();
    let dir_path = dir.path().to_path_buf();
    assert!(
        path.starts_with(&dir_path),
        "the returned path must live inside the returned dir"
    );
    assert!(dir_path.is_dir(), "the dir exists while owned");

    // The dir is usable while owned — this is the pre-populate flow the helper
    // exists for.
    use crate::memory::lifecycle::{MemoryStatus, MemoryType};
    let db = crate::sqlite::Database::open(&path).expect("open on the returned path");
    db.insert(
        "leak-guard",
        "pre-populated row",
        &vec![0.2f32; 384],
        None,
        MemoryType::Fact.as_str(),
        MemoryStatus::Active.as_str(),
    )
    .expect("insert through the returned path");
    drop(db);

    drop(dir);

    assert!(
        !dir_path.exists(),
        "MemoryStore::test_db_path left {} behind after drop — the TempDir is \
         being forgotten again (issue #278)",
        dir_path.display()
    );
}

#[test]
fn no_helper_in_this_module_forgets_a_temp_dir() {
    // The compile-time half of the guard. Every leak site fixed for #278 was a
    // `std::mem::forget(dir)` in a *test helper*, which no runtime check inside
    // that helper could catch — and a re-introduced one fails the two tests
    // above only for whichever helper it lands in. This asserts the source text
    // itself, so the whole file set stays covered.
    //
    // A doc comment that *mentions* the call (the ones at the top of this file
    // explain what the helpers used to do) is not a call, so it must not fail
    // the guard: every line is first stripped of `//`-comments, then matched for
    // the `(`-terminated call. The `(` is what keeps prose prose.
    //
    // The file set is read from the filesystem rather than hand-listed: a
    // hand-maintained `include_str!` roster drifts the moment a leak site is
    // added or a file is renamed, and the only thing that would notice is the
    // `include_str!` path itself failing to compile. Same walk the module-wiring
    // guard in `src/lib.rs` uses, for the same reason.
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    assert!(
        files.len() > 50,
        "the walk found only {} source files under {} — the sweep is not \
         covering the tree, so it proves nothing",
        files.len(),
        src_dir.display(),
    );

    // Needle built by concatenation so the literal never appears in this file's
    // own text — this file is swept like any other, and a guard that trips on its
    // own source would force an exclusion list, which is the drift this rewrite
    // exists to remove.
    let needle = "mem::".to_owned() + "forget(";

    for path in files {
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let relative = path
            .strip_prefix(src_dir.parent().unwrap_or(&src_dir))
            .unwrap_or(path.as_path())
            .display();
        // Per line, so the failure names a line.
        //
        // Two normalisations before the match, both load-bearing:
        //   * `std::` is stripped, so a call written `mem::forget` (after
        //     `use std::mem;`) is the same call as far as this guard is concerned;
        //   * everything after a `//` is dropped, so the doc comments in this file
        //     and in `store.rs`/`mcp/tests.rs` that describe what the helpers *used
        //     to do* do not read as the call they describe.
        //
        // The one shape a text scan cannot see is `use std::mem::forget;` plus a
        // bare `forget(dir)` — unqualified, it is not textually distinguishable
        // from any other function named `forget`. The runtime tests above own that
        // property (they fail on the directory that survives); this half is here
        // for the obvious regression in a file none of them constructs.
        for (index, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").replace("std::", "");
            if code.contains(needle.as_str()) {
                panic!(
                    "{relative}:{} calls mem::forget — a leaked TempDir is the bug \
                     issue #278 removed from the test helpers",
                    index + 1,
                );
            }
        }
    }
}

/// Every `.rs` file under `dir`, recursively. Shared shape with the
/// `collect_rs_files` walk in `src/lib.rs`'s module-wiring guard.
fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
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
