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
    let store = MemoryStore::test_store();
    let dir_path = store.dir_path().to_path_buf();
    assert!(
        dir_path.is_dir(),
        "the helper's directory must exist while the store is alive"
    );

    // Do something real with the store first: the point is that a *used*
    // store's directory is gone once the store itself drops, so the connection
    // has to be open (and able to write) while the dir is alive.
    store
        .db
        .insert(
            "leak-guard",
            "a row written before the dir is dropped",
            &vec![0.1f32; 384],
            None,
            "fact",
            "active",
        )
        .expect("insert through the test store");

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
    let db = crate::sqlite::Database::open(&path).expect("open on the returned path");
    db.insert(
        "leak-guard",
        "pre-populated row",
        &vec![0.2f32; 384],
        None,
        "fact",
        "active",
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
    // A doc comment that *mentions* `std::mem::forget` (the ones above explain
    // what the helpers used to do) is not a call, hence the `(` requirement.
    //
    // Deliberately scoped to the leak-fix surface rather than all of `src/`:
    // `src/commands/` has its own `std::mem::forget` use on a non-TempDir value,
    // and asserting over that would make this test fail for something #278 did
    // not set out to change.
    let sources = [
        ("src/memory/store.rs", include_str!("store.rs")),
        ("src/memory/tests/crud.rs", include_str!("tests/crud.rs")),
        (
            "src/memory/tests/ingest_boundary.rs",
            include_str!("tests/ingest_boundary.rs"),
        ),
        (
            "src/memory/tests/search.rs",
            include_str!("tests/search.rs"),
        ),
        (
            "src/sqlite/fts_tests.rs",
            include_str!("../sqlite/fts_tests.rs"),
        ),
        (
            "src/sqlite/fts_migration_tests.rs",
            include_str!("../sqlite/fts_migration_tests.rs"),
        ),
        (
            "src/sqlite/crud_tests.rs",
            include_str!("../sqlite/crud_tests.rs"),
        ),
        (
            "src/sqlite/migration_tests.rs",
            include_str!("../sqlite/migration_tests.rs"),
        ),
        ("src/sqlite/search.rs", include_str!("../sqlite/search.rs")),
        (
            "src/sqlite/query_mod.rs",
            include_str!("../sqlite/query_mod.rs"),
        ),
        ("src/mcp/tests.rs", include_str!("../mcp/tests.rs")),
    ];
    for (path, src) in sources {
        assert!(
            !src.contains("std::mem::forget("),
            "{path} calls std::mem::forget again — a leaked TempDir is the \
             bug issue #278 removed from these files"
        );
    }
}
