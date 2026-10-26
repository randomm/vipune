//! The TempDir-leak guard for the two store builders in this module (issue #278).
//!
//! `MemoryStore::test_store` and `MemoryStore::test_db_path` used to leak their
//! `TempDir`, so every `cargo test` run left one `.tmpXXXXXX/` directory
//! (holding a SQLite DB) behind in `TMPDIR`. On a host with a quota on the temp
//! filesystem that is not cosmetic: tens of thousands of leaked dirs filled the
//! operator's quota and made unrelated runs fail with SQLite `disk I/O error`
//! and `EDQUOT`.
//!
//! The predicate is "the directory the helper made is gone once the caller's
//! binding is dropped". Counting `.tmp*` entries in `std::env::temp_dir()`
//! cannot express that — `cargo test` runs these on many threads, so every
//! other test's live temp dirs appear and disappear in the same window and
//! would be miscounted as leaks either way. Each test therefore identifies its
//! *own* directory by name, and pins the fact a leak would break: it is still
//! there while owned and gone after `drop`.
//!
//! Two halves: the runtime tests below own the property for these two builders
//! (they fail on the directory that survives), and
//! [`no_helper_forgets_a_temp_dir_anywhere_in_the_tree`] is a best-effort lint
//! for the same call shape in a file no runtime test here constructs.

use super::MemoryStore;

#[test]
fn test_store_drops_its_temp_dir_with_the_store() {
    // Both halves are bound (and `dir` is kept alive to the end): the helper
    // returns them as one pair so a test cannot drop the directory under an open
    // SQLite handle, which is the shape that produced the original `disk I/O
    // error` failures. `mut` on the store is the store's own `&mut self` API, not
    // a deref impl.
    let (dir, mut store) = MemoryStore::test_store();
    // A `&mut self` method, to prove the store is writable before the drop.
    store.set_test_embedder(Box::new(crate::memory::crud::test_fake_embedder));
    let dir_path = dir.path().to_path_buf();
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
    // test's assertion (the directory is gone) would still pass on it.
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

    // A second write through a `&mut self` store method, asserted on the
    // discriminated outcome rather than the container length: an
    // `Error { message }` item also yields `results.len() == 1`, so counting
    // would not prove the write landed.
    let ingested = store
        .batch_ingest(
            "leak-guard",
            vec![("a second row, written mutably", None)],
            crate::memory_types::IngestPolicy::Force,
        )
        .expect("batch_ingest through the test store");
    assert!(
        matches!(
            ingested.results[0],
            crate::memory_types::BatchIngestItemResult::Added { .. }
        ),
        "the mutable write must actually have landed as an Added item, got {:?}",
        ingested.results[0]
    );

    // Dropping the store alone must not remove the dir (the store does not own
    // it), and dropping the dir must remove it — together they are the pair the
    // helper hands back, and the pair is what the caller has to keep alive.
    drop(store);
    assert!(
        dir_path.is_dir(),
        "the directory is owned by the caller's binding, not by the store: it \
         must still be there while {} is alive",
        dir.path().display()
    );

    drop(dir);

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
    // exists for: open a `Database` on the path, write, then build the store.
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

/// The line shapes the text guard below must tell apart: a call (must match)
/// and a comment describing the call (must not). The four cases are built from
/// parts in code, so the needle and the two normalisations have to stay in step
/// with each other — an edit that breaks one of the three fails here rather than
/// turning the guard into a silent no-op (or into one that panics on its own
/// file).
#[test]
fn leak_guard_needle_matches_a_call_and_not_a_comment() {
    let needle = leak_guard_needle();
    // The same normalisation the sweep applies, spelled once so the two cannot
    // drift apart.
    let normalise = |line: &str| line.split("//").next().unwrap_or("").replace("std::", "");

    // Pieces rather than literals: a whole call written out here would be seen
    // by the sweep itself.
    let call = ["    ", "std::", "mem::", "forget(", "dir);"].concat();
    let use_aliased = ["    ", "mem::", "forget(", "dir);"].concat();
    let doc_comment = [
        "    /// used to ",
        "std::",
        "mem::",
        "forget(",
        "dir)",
        " it",
    ]
    .concat();
    let comment_after_a_url = [
        "    // see https://ex.am ple ",
        "std::",
        "mem::",
        "forget(",
        "dir)",
    ]
    .concat();

    assert!(
        normalise(&call).contains(needle.as_str()),
        "the guard must match a plain call, normalised line was {call:?}"
    );
    assert!(
        normalise(&use_aliased).contains(needle.as_str()),
        "the guard must match the `use std::mem;` spelling too, normalised line was {use_aliased:?}"
    );
    assert!(
        !normalise(&doc_comment).contains(needle.as_str()),
        "the `//`-strip must keep a doc comment that names the call from failing the guard"
    );
    assert!(
        !normalise(&comment_after_a_url).contains(needle.as_str()),
        "comment prose must not read as a call, even one whose text contains `//`"
    );
    // The pre-filter must be no wider than the needle: any line it skips could
    // not have matched anyway, so a needle that outgrew it would silently drop
    // coverage. Wider than the needle is allowed — the sweep's needle test is
    // what decides — but a pre-filter that excludes part of the needle turns
    // the guard into a no-op that still reports green. Built from the same parts
    // so this control is not itself a copy of the needle.
    let pre_filter = ["for", "get", ""].concat();
    assert!(
        needle.contains(pre_filter.as_str()),
        "the sweep's `{pre_filter}` pre-filter must stay a substring of the needle, got {needle:?}"
    );
}

/// The needle the text guard matches on, shared with its positive control
/// above so the two cannot drift apart.
fn leak_guard_needle() -> String {
    // Built by concatenation so the literal never appears in any swept file's
    // own text — this file is swept like any other, and a guard that trips on
    // its own source would force an exclusion list.
    "mem::".to_owned() + "forget("
}

#[test]
fn no_helper_forgets_a_temp_dir_anywhere_in_the_tree() {
    // The best-effort static half of the #278 guard — a lint, not a control.
    // It sweeps every `.rs` file the `cargo test` invocation compiles (`src/`,
    // `tests/`, `benches/`) for the call shape the fix removed. The two runtime
    // tests above own the property for the store builders (they fail on a
    // directory that survives drop); this half exists for the gap they cannot
    // reach: a helper in a file none of them constructs, whose leaked dir would
    // be invisible until the operator's temp quota filled.
    //
    // Its blind spots are the shapes a text scan cannot see, and the runtime
    // tests are the backstop for each: `use std::mem::forget;` plus a bare
    // `forget(dir)` (unqualified, indistinguishable from any other `forget`),
    // and a call on a line whose `//`-prefixed portion is inside a string
    // literal — the `//` strip is not quote-aware, so such a line is not
    // matched. A guard that reports clean
    // on one of those is why the guarantee lives on the runtime side.
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for root in ["src", "tests", "benches"] {
        collect_rs_files(&manifest_dir.join(root), &mut files);
    }
    assert!(
        files.len() > 100,
        "the walk found only {} source files under {} — the sweep is not covering \
         the tree, so it proves nothing",
        files.len(),
        manifest_dir.display(),
    );

    let needle = leak_guard_needle();

    for path in files {
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let relative = path
            .strip_prefix(manifest_dir)
            .unwrap_or(path.as_path())
            .display();
        // Per line, so the failure names a line.
        //
        // Two normalisations before the match, both load-bearing (and both
        // pinned by `leak_guard_needle_matches_a_call_and_not_a_comment`):
        //   * `std::` is stripped, so a call written `mem::forget` (after
        //     `use std::mem;`) is the same call as far as this guard is concerned;
        //   * everything after a `//` is dropped, so the comments in this file and
        //     in `store.rs`/`mcp/tests.rs` that describe what the helpers *used to
        //     do* do not read as the call they describe.
        for (index, line) in src.lines().enumerate() {
            // Only lines that could carry the call are normalised: the rest of the
            // tree costs one substring scan, not two allocations.
            if !line.contains("forget") {
                continue;
            }
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

/// Every `.rs` file under `dir`, recursively. A directory that cannot be read is
/// a failure, not a skipped subtree: silently shrinking the file set would let
/// the guard report green over an un-swept `src/sqlite/`. (The shape is shared
/// with the module-wiring guard's walk in `src/lib.rs`, which is why it is
/// recursive rather than a flat `read_dir`.)
fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
