//! The supersede path: what the status- and level-carrying entry points write,
//! and what they refuse.
//!
//! Split out of `memory/tests/crud.rs` for issue #271 so both files stay under
//! the project's 500-line cap; the add-path helpers the tests below drive are
//! shared from there (`super::crud::add`), the read helpers included.
//!
//! The re-supersede guard — refusing to retire a row that is already retired —
//! is covered in [`resupersede_guard_tests`], beside the storage body it guards.

use super::tests::crud::{add, added_importance, added_status, supersede_with_importance};
use crate::memory::NewMemory;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::store::MemoryStore;
use crate::memory_types::IngestPolicy;

/// Add a row through `vipune add --supersedes`'s own entry point — the
/// status- and level-carrying supersede — so a test can pin what that path
/// writes. `vipune add` reaches the store through `handle_add`, which is
/// covered by `commands/handlers_add_tests.rs`.
fn supersede_with_status(
    store: &mut MemoryStore,
    project: &str,
    content: &str,
    old_id: &str,
    importance: MemoryImportance,
    status: MemoryStatus,
) -> String {
    store
        .supersede_with(&crate::memory::NewSupersedingMemory {
            row: NewMemory {
                project_id: project,
                content,
                metadata: None,
                memory_type: MemoryType::Fact,
                status,
            },
            old_id,
            importance,
        })
        .expect("supersede must succeed")
}

/// Every row in `project`, read across all four statuses so a refusal cannot
/// satisfy a "nothing was written" assertion by having written a row the default
/// active-only list would have hidden.
pub(super) fn all_statuses(store: &MemoryStore, project: &str) -> Vec<crate::sqlite::Memory> {
    store
        .list(
            project,
            20,
            None,
            Some(["active", "candidate", "superseded", "deprecated"].as_slice()),
        )
        .expect("list")
}

#[test]
fn add_persists_the_importance_it_was_given() {
    let (_dir, mut store) = MemoryStore::test_store();
    // `medium` twice on purpose: once through the defaulting entry point, once
    // through the level-carrying one, which is how the CLI's omitted-flag
    // default reaches the column.
    let cases = [
        (None, "medium"),
        (Some(MemoryImportance::Medium), "medium"),
        (Some(MemoryImportance::Low), "low"),
        (Some(MemoryImportance::High), "high"),
        (Some(MemoryImportance::Critical), "critical"),
    ];
    for (i, (level, expected)) in cases.iter().enumerate() {
        let id = add(
            &mut store,
            "imp-proj",
            &format!("importance probe {i}"),
            *level,
        );
        assert_eq!(
            added_importance(&store, &id, "imp-proj"),
            *expected,
            "the level the operator named must reach the column, not its DEFAULT"
        );
    }
}

#[test]
fn add_superseding_persists_the_status_on_the_new_row() {
    // Issue #258: the supersede body hardcoded `status: "active"` into the row
    // it built, so a caller that named `candidate` got `active` and no warning
    // — the same class of drop #238 fixed for `importance`.
    let (_dir, mut store) = MemoryStore::test_store();
    let old_id = add(&mut store, "status-proj", "original fact", None);

    let candidate_id = supersede_with_status(
        &mut store,
        "status-proj",
        "candidate replacement",
        &old_id,
        MemoryImportance::High,
        MemoryStatus::Candidate,
    );
    assert_eq!(
        added_status(&store, &candidate_id, "status-proj"),
        MemoryStatus::Candidate.as_str(),
        "the status the caller named must reach the superseding row"
    );
    assert_eq!(
        added_importance(&store, &candidate_id, "status-proj"),
        "high",
        "the level still travels with it"
    );
    // The old row is the supersede transaction's own business: it is marked
    // `superseded` regardless of what the new row became, and nothing in the
    // new row's INSERT touched it.
    let replaced = store
        .get(&old_id, "status-proj")
        .expect("get")
        .expect("superseded row readable");
    assert_eq!(replaced.status, "superseded");
    assert_eq!(
        replaced.superseded_by.as_deref(),
        Some(candidate_id.as_str())
    );
}

#[test]
fn add_superseding_wrappers_bind_the_lifecycle_defaults_by_name() {
    // The one level-less public wrapper must keep pinning both defaults by
    // name rather than inheriting the column DEFAULT or the new variant's value
    // (the #238/#258 class of silent drop).
    let (_dir, mut store) = MemoryStore::test_store();
    let old_id = add(&mut store, "sup-defaults", "original fact", None);
    let plain_id = store
        .supersede(
            "sup-defaults",
            "active replacement",
            None,
            MemoryType::Fact,
            &old_id,
        )
        .unwrap();
    assert_eq!(added_status(&store, &plain_id, "sup-defaults"), "active");
    assert_eq!(
        added_importance(&store, &plain_id, "sup-defaults"),
        "medium",
        "the level-less wrapper binds the default level by name"
    );
}

#[test]
fn superseding_candidate_stays_out_of_the_default_reads() {
    // A candidate replacement is invisible to the active-only default reads, so
    // the row must be found by id or an explicit candidate filter — never by
    // accident of the default list.
    let (_dir, mut store) = MemoryStore::test_store();
    let old_id = add(&mut store, "status-visible", "original fact", None);
    let control_id = add(&mut store, "status-visible", "unrelated active fact", None);
    let candidate_id = supersede_with_status(
        &mut store,
        "status-visible",
        "candidate replacement",
        &old_id,
        MemoryImportance::Medium,
        MemoryStatus::Candidate,
    );

    let default_list = store.list("status-visible", 20, None, None).expect("list");
    assert!(
        default_list.iter().all(|m| m.id != candidate_id),
        "default list stays active-only"
    );
    // A third, untouched active row is the control: the superseded original is
    // absent from the default list too, so the guard above would also pass
    // against a read path that returned nothing.
    assert!(
        default_list.iter().any(|m| m.id == control_id),
        "the read path itself works: an ordinary active row is listed"
    );
    assert!(
        store
            .list("status-visible", 20, None, Some(["candidate"].as_slice()))
            .expect("list")
            .iter()
            .any(|m| m.id == candidate_id),
        "an explicit candidate filter finds the hidden replacement"
    );
}

#[test]
fn add_superseding_refuses_a_status_that_is_not_insertable() {
    // The store is the last line of defence: a library caller that skips the
    // CLI/MCP precheck must not get a `superseded` row inserted as the
    // replacement (that lifecycle belongs to the old row's UPDATE only).
    let (_dir, mut store) = MemoryStore::test_store();
    let old_id = add(&mut store, "status-bad", "original fact", None);
    for status in [MemoryStatus::Superseded, MemoryStatus::Deprecated] {
        let err = store
            .supersede_with(&crate::memory::NewSupersedingMemory {
                row: NewMemory {
                    project_id: "status-bad",
                    content: "bad replacement",
                    metadata: None,
                    memory_type: MemoryType::Fact,
                    status,
                },
                old_id: &old_id,
                importance: MemoryImportance::Medium,
            })
            .expect_err("only active/candidate are insertable");
        assert!(
            err.to_string()
                .contains("is not valid for new memory insertion"),
            "unexpected error for {status:?}: {err}"
        );
    }
    // The refused supersede left the original live, and inserted nothing. The
    // loop above attempts the refusal twice, so one inserted replacement would
    // already break this count.
    assert_eq!(added_status(&store, &old_id, "status-bad"), "active");
    assert_eq!(
        all_statuses(&store, "status-bad").len(),
        1,
        "a refused supersede must insert no replacement row"
    );
}

#[test]
fn add_superseding_persists_the_importance_on_the_new_row() {
    // `--supersedes` is a second, independent add path: it inserts the new row
    // inside the supersede transaction, so that INSERT needs the binding too.
    let (_dir, mut store) = MemoryStore::test_store();
    let old_id = add(&mut store, "sup-proj", "original fact", None);
    let new_id = supersede_with_importance(
        &mut store,
        "sup-proj",
        "replacement fact",
        &old_id,
        MemoryImportance::Critical,
    );
    assert_eq!(
        added_importance(&store, &new_id, "sup-proj"),
        "critical",
        "the superseding row must carry the level the operator asked for"
    );
    assert_eq!(
        added_status(&store, &new_id, "sup-proj"),
        MemoryStatus::Active.as_str(),
        "the released wrapper's own semantics: the new row is active"
    );
    // The old row was replaced by this call: the released signature keeps the
    // supersede transaction's other half intact.
    assert_eq!(added_status(&store, &old_id, "sup-proj"), "superseded");
    // The superseded row keeps its own level, and the defaulting wrapper lands
    // on medium rather than sharing the variant's value by accident.
    assert_eq!(added_importance(&store, &old_id, "sup-proj"), "medium");
    let plain_id = store
        .supersede("sup-proj", "third fact", None, MemoryType::Fact, &new_id)
        .unwrap();
    assert_eq!(added_importance(&store, &plain_id, "sup-proj"), "medium");
}

/// Every entry point that creates a row must refuse a status the INSERT may not
/// name, no matter which frame a caller enters at — the CLI and MCP prechecks
/// can be skipped by a library caller, and #258 was exactly a status that was
/// validated somewhere above and dropped below. `MemoryStatus::
/// ensure_valid_for_insert` owns the rule; these pin that each write body calls
/// it rather than trusting its caller.
#[test]
fn every_row_creating_entry_point_refuses_a_non_insertable_status() {
    let (_dir, mut store) = MemoryStore::test_store();
    let old_id = add(&mut store, "guard-proj", "original fact", None);

    for status in [MemoryStatus::Superseded, MemoryStatus::Deprecated] {
        // Supersede: the params-based entry point is the only body, and the
        // defaulting wrapper above it binds `active` by name.
        let err = store
            .supersede_with(&crate::memory::NewSupersedingMemory {
                row: NewMemory {
                    project_id: "guard-proj",
                    content: "guard replacement",
                    metadata: None,
                    memory_type: MemoryType::Fact,
                    status,
                },
                old_id: &old_id,
                importance: MemoryImportance::Medium,
            })
            .expect_err("a superseding row cannot be born superseded/deprecated");
        assert!(
            err.to_string()
                .contains("is not valid for new memory insertion"),
            "supersede_with accepted {status:?}: {err}"
        );

        // The params-based add (the one `vipune add` drives).
        let err = store
            .ingest_with_importance(
                &NewMemory {
                    project_id: "guard-proj",
                    content: "guard add",
                    metadata: None,
                    memory_type: MemoryType::Fact,
                    status,
                },
                IngestPolicy::Force,
                MemoryImportance::Medium,
            )
            .expect_err("an added row cannot be born superseded/deprecated");
        assert!(
            err.to_string()
                .contains("is not valid for new memory insertion"),
            "ingest_with_importance accepted {status:?}: {err}"
        );

        // The defaulting add family, which takes the status as an argument.
        let err = store
            .add_with_conflict(
                "guard-proj",
                "guard add",
                None,
                true,
                MemoryType::Fact,
                status,
            )
            .expect_err("an added row cannot be born superseded/deprecated");
        assert!(
            err.to_string()
                .contains("is not valid for new memory insertion"),
            "add_with_conflict accepted {status:?}: {err}"
        );
    }

    // None of the refusals wrote anything: the project still holds only the
    // original row, and that row is still live.
    assert_eq!(added_status(&store, &old_id, "guard-proj"), "active");
    assert_eq!(
        all_statuses(&store, "guard-proj").len(),
        1,
        "a refused row creation must insert nothing"
    );
}
