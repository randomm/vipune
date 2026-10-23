//! The `vipune add` handler's importance plumbing (issue #238).
//!
//! The store-level tests (`memory/tests/crud.rs`, `commands/prune_tests.rs`)
//! call `ingest_with_importance` / `supersede_with_importance` directly, so
//! they pass whether or not `handle_add` ever hands the parsed level to them —
//! exactly how #238 shipped (`let _ = importance_val;` one line above the call,
//! with the row created by the level-less `ingest_with_type_status`). These
//! tests drive the handler itself, the layer where the defect lived, and read
//! the column back.

use super::handlers::handle_add;
use crate::memory::MemoryStore;
use std::process::ExitCode;

/// Run `vipune add` through the handler.
///
/// `--force` is implied for a plain add (the similarity gate is another
/// handler's concern and would otherwise reject probes whose fake embeddings
/// happen to be close); it must stay off for `--supersedes`, which the handler
/// refuses to combine with it.
fn add(
    store: &mut MemoryStore,
    project: &str,
    text: &str,
    importance: &str,
    supersedes: Option<&str>,
) -> ExitCode {
    handle_add(
        store,
        project,
        text,
        None,
        supersedes.is_none(),
        "fact",
        "active",
        importance,
        supersedes,
        false,
    )
    .expect("handle_add ok")
}

/// The stored row `content` names, read back through the store's own read path.
fn stored(store: &MemoryStore, project: &str, content: &str) -> crate::sqlite::Memory {
    store
        .list(project, 10, None, None)
        .expect("list")
        .into_iter()
        .find(|m| m.content == content)
        .unwrap_or_else(|| panic!("no active row with content {content:?} was stored"))
}

#[test]
fn handle_add_writes_the_importance_the_operator_named() {
    let mut store = MemoryStore::test_store();
    // `medium` is the CLI's own default (`#[arg(long, default_value = "medium")]`),
    // so the omitted-flag case is this same call — the level must reach the
    // column either way, never be dropped on the way down.
    for (i, level) in ["critical", "high", "low", "medium"].iter().enumerate() {
        let text = format!("importance probe {i}");
        assert_eq!(
            add(&mut store, "imp", &text, level, None),
            ExitCode::SUCCESS
        );
        assert_eq!(
            stored(&store, "imp", &text).importance,
            *level,
            "the level named on --importance must reach the column, not its DEFAULT"
        );
    }
}

#[test]
fn handle_add_superseding_writes_the_importance_on_the_new_row() {
    // `--supersedes` is a second, independent add path: it creates its row
    // inside the store's supersede transaction, so that INSERT needs the level
    // bound too.
    let mut store = MemoryStore::test_store();
    assert_eq!(
        add(&mut store, "sup", "original fact", "medium", None),
        ExitCode::SUCCESS
    );
    let old = stored(&store, "sup", "original fact");

    assert_eq!(
        add(
            &mut store,
            "sup",
            "replacement fact",
            "critical",
            Some(&old.id),
        ),
        ExitCode::SUCCESS
    );

    let fresh = stored(&store, "sup", "replacement fact");
    assert_eq!(
        fresh.importance, "critical",
        "the superseding row must carry the level the operator asked for"
    );
    // The row it replaced keeps its own level and its superseded lifecycle —
    // the new row's INSERT must not touch it.
    let replaced = store
        .get(&old.id, "sup")
        .expect("get")
        .expect("superseded row still readable");
    assert_eq!(replaced.status, "superseded");
    assert_eq!(replaced.importance, "medium");
    assert_eq!(replaced.superseded_by.as_deref(), Some(fresh.id.as_str()));
}
