//! The `vipune add` handler's importance and status plumbing (issues #238, #258).
//!
//! The store-level tests (`memory/tests/crud.rs`, `commands/prune_tests.rs`)
//! call `ingest_with_importance` / `supersede_with_lifecycle` directly, so
//! they pass whether or not `handle_add` ever hands the parsed level to them —
//! exactly how #238 shipped (`let _ = importance_val;` one line above the call,
//! with the row created by the level-less `ingest_with_type_status`). The same
//! shape shipped for `--status` in #258: the supersedes branch validated the
//! status and then never passed it on. These tests drive the handler itself,
//! the layer where the defects lived, and read the columns back.

use super::handlers::handle_add;
use crate::memory::MemoryStore;
use std::process::ExitCode;

/// Run `vipune add` through the handler with the CLI's own `--status` default.
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
    add_with_status(store, project, text, "active", importance, supersedes)
}

/// Run `vipune add` through the handler with an explicit `--status`, the flag
/// the supersede path used to discard.
fn add_with_status(
    store: &mut MemoryStore,
    project: &str,
    text: &str,
    status: &str,
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
        status,
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

/// The `candidate`/`active` replacement `--supersedes <id> --status <status>`
/// created, plus the `superseded` row it replaced.
///
/// The link is followed in the direction the supersede transaction writes it:
/// the OLD row carries `superseded_by` pointing at the new id (see
/// `Database::supersede_with_importance`, src/sqlite/supersede.rs), so the old
/// row is located first — it is also the only one of the two that default
/// `list` hides, which is precisely why the read goes through `get`.
fn supersede_pair(
    store: &MemoryStore,
    project: &str,
    old_id: &str,
) -> (crate::sqlite::Memory, crate::sqlite::Memory) {
    let replaced = store
        .get(old_id, project)
        .expect("get")
        .expect("superseded row readable");
    let replacement_id = replaced
        .superseded_by
        .as_deref()
        .unwrap_or_else(|| panic!("row {old_id} carries no superseded_by"));
    let replacement = store
        .get(replacement_id, project)
        .expect("get")
        .expect("replacement row readable");
    (replacement, replaced)
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
    let (replacement, replaced) = supersede_pair(&store, "sup", &old.id);
    assert_eq!(replacement.id, fresh.id);
    assert_eq!(replaced.content, "original fact");
    assert_eq!(replaced.status, "superseded");
    assert_eq!(replaced.importance, "medium");
    assert_eq!(replaced.superseded_by.as_deref(), Some(fresh.id.as_str()));
}

#[test]
fn handle_add_superseding_writes_the_status_the_operator_named() {
    // Issue #258, the twin of the #238 importance defect on the same path:
    // handle_add validated --status and then handed the supersedes branch to a
    // store call that hardcoded `active`, so `--status candidate` was accepted
    // and silently stored as `active`. The replacement row is read back through
    // `get` because `list`/`search` would filter a candidate out entirely.
    let mut store = MemoryStore::test_store();
    assert_eq!(
        add(&mut store, "st", "original fact", "medium", None),
        ExitCode::SUCCESS
    );
    let old = stored(&store, "st", "original fact");

    assert_eq!(
        add_with_status(
            &mut store,
            "st",
            "replacement fact",
            "candidate",
            "medium",
            Some(&old.id),
        ),
        ExitCode::SUCCESS
    );

    let (replacement, replaced) = supersede_pair(&store, "st", &old.id);
    assert_eq!(
        replacement.status, "candidate",
        "the status named on --status must reach the superseding row, not 'active'"
    );
    assert_eq!(replacement.content, "replacement fact");
    // The old-row lifecycle is the supersede transaction's own business and the
    // new row's status must not leak into it.
    assert_eq!(replaced.status, "superseded");
    assert_eq!(
        replaced.superseded_by.as_deref(),
        Some(replacement.id.as_str())
    );

    // A candidate replacement is exactly what default list/search hide: the row
    // is present in the database and invisible to the active-only read path.
    assert!(
        store
            .list("st", 20, None, None)
            .expect("list")
            .iter()
            .all(|m| m.content != "replacement fact"),
        "a candidate replacement must stay out of the active-only default list"
    );
}

#[test]
fn handle_add_superseding_defaults_the_new_row_to_active() {
    // The omitted-flag case: clap's `default_value = "active"` makes this the
    // same call the CLI makes for a plain `--supersedes`, so the superseding row
    // must still land `active` — and, unlike the candidate case above, stay
    // visible to the default read path.
    let mut store = MemoryStore::test_store();
    assert_eq!(
        add(&mut store, "sup-active", "original fact", "medium", None),
        ExitCode::SUCCESS
    );
    let old = stored(&store, "sup-active", "original fact");

    assert_eq!(
        add(
            &mut store,
            "sup-active",
            "replacement fact",
            "medium",
            Some(&old.id),
        ),
        ExitCode::SUCCESS
    );

    let (replacement, replaced) = supersede_pair(&store, "sup-active", &old.id);
    assert_eq!(replacement.status, "active");
    assert_eq!(replaced.status, "superseded");
    // The active row is what the default list returns, so the visibility claim
    // is pinned here rather than by a second read of the same row.
    assert!(
        store
            .list("sup-active", 20, None, None)
            .expect("list")
            .iter()
            .any(|m| m.id == replacement.id),
        "the explicit-active case must remain visible to the default list"
    );
}

#[test]
fn handle_add_superseding_rejects_a_status_that_is_not_insertable() {
    // `superseded`/`deprecated` never reach the supersede path: the insert
    // precheck rejects them first, with or without --supersedes. The old row's
    // `superseded` is set by the supersede transaction, never asked for.
    let mut store = MemoryStore::test_store();
    assert_eq!(
        add(&mut store, "st-bad", "original fact", "medium", None),
        ExitCode::SUCCESS
    );
    let old = stored(&store, "st-bad", "original fact");

    for status in ["superseded", "deprecated"] {
        let err = handle_add(
            &mut store,
            "st-bad",
            "replacement fact",
            None,
            false,
            "fact",
            status,
            "medium",
            Some(&old.id),
            false,
        )
        .expect_err("{status} is not a valid status for a new row");
        assert!(
            err.to_string()
                .contains("not valid for new memory insertion"),
            "unexpected error for --status {status}: {err}"
        );
    }
    // A string outside the vocabulary at all is rejected by the same precheck
    // one step earlier (MemoryStatus::from_str), still before the supersedes
    // branch is entered.
    let err = handle_add(
        &mut store,
        "st-bad",
        "replacement fact",
        None,
        false,
        "fact",
        "nonsense",
        "medium",
        Some(&old.id),
        false,
    )
    .expect_err("an unknown status must be rejected");
    assert!(
        err.to_string().contains("Invalid memory status"),
        "unexpected error for --status nonsense: {err}"
    );

    // Nothing was replaced by the rejected calls: the original row is still the
    // live one.
    assert_eq!(stored(&store, "st-bad", "original fact").status, "active");
}
