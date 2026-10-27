//! The re-supersede guard: the one supersede INSERT must refuse a target that a
//! previous supersede, or `prune`, already retired.
//!
//! Split out of [`supersede_tests`] for issue #271 so both files stay under the
//! project's 500-line cap; the read helper they share is
//! [`super::supersede_tests::all_statuses`].
//!
//! # What this file deliberately does *not* test
//!
//! The guard's *atomicity* is not covered by a thread race. That test was
//! written and rejected: two `MemoryStore`s on one file, each parked on a
//! `Barrier` and released together to supersede the same `active` row, stayed
//! green (one `Ok`, one `DatabaseLocked`, one live row) with the guard deleted
//! outright, and equally green with the pre-change topology restored — the read
//! moved onto the plain connection, the transaction opened only before the
//! INSERT, and a 600 ms window between them. SQLite's own busy handling
//! serialises the two writers regardless of where the check sits, so such a test
//! asserts a property it cannot fail, which is worse than no test at all: it
//! reads as coverage of the exact fork the guard exists to prevent.
//!
//! What is pinned here instead is the guard's *behaviour*, which deleting the
//! guard does break — all three of
//! `superseding_a_superseded_row_is_refused_and_writes_nothing`,
//! `superseding_a_deprecated_row_is_refused_and_writes_nothing` and
//! `superseding_a_row_whose_retired_status_is_spelled_oddly_is_refused` fail,
//! each also asserting nothing was written (no `superseded_by`, no extra row).
//! The transaction placement those tests cannot see is documented at
//! [`crate::sqlite::Database::supersede_with_importance`], at the code that has
//! to keep issuing its read through the transaction.

use super::supersede_tests::all_statuses;
use super::tests::crud::add;
use crate::memory::NewMemory;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::store::MemoryStore;

/// The re-supersede guard: a row the supersede transaction already retired is
/// not a supersede target. Re-marking a `superseded` row would overwrite its
/// `superseded_by` and fork the chain into a replacement nothing points at, and
/// a `deprecated` one was retired by `prune`. `deprecated` is reachable as a
/// target today (only inserts are blocked, and `update --status deprecated`
/// writes it), so the guard names both.
///
/// The retired row is written directly through the public `Database::update`,
/// because the retired status has to be in the column first and no store entry
/// point can put it there — that is what `ensure_valid_for_insert` exists for.
#[test]
fn superseding_a_superseded_row_is_refused_and_writes_nothing() {
    let (_dir, mut store) = MemoryStore::test_store();
    let project = "retired-superseded";
    let old_id = add(&mut store, project, "original fact", None);
    let first_replacement = add(&mut store, project, "first replacement", None);
    seed_retired(
        &mut store,
        &old_id,
        project,
        MemoryStatus::Superseded,
        Some(&first_replacement),
    );

    let msg = refuse_retired_supersede(&mut store, project, &old_id, "second replacement");
    assert!(
        msg.contains("Invalid input") && msg.contains("'superseded'"),
        "the refusal must be InvalidInput and name the current status: {msg}"
    );
    assert!(
        msg.contains(&first_replacement),
        "the refusal must name the existing superseded_by: {msg}"
    );

    let row = retired_row(&store, project, &old_id);
    assert_eq!(row.status, "superseded", "the refusal rewrote the status");
    assert_eq!(
        row.superseded_by.as_deref(),
        Some(first_replacement.as_str()),
        "the refusal must not fork the supersede chain"
    );
    assert_eq!(
        all_statuses(&store, project).len(),
        2,
        "a refused supersede must insert no replacement row"
    );
}

/// The `deprecated` twin: `prune` demotes candidates through this status, so a
/// supersede that resurrected one would undo an operator decision.
#[test]
fn superseding_a_deprecated_row_is_refused_and_writes_nothing() {
    let (_dir, mut store) = MemoryStore::test_store();
    let project = "retired-deprecated";
    let old_id = add(&mut store, project, "pruned fact", None);
    seed_retired(&mut store, &old_id, project, MemoryStatus::Deprecated, None);

    let msg = refuse_retired_supersede(&mut store, project, &old_id, "second replacement");
    assert!(
        msg.contains("Invalid input") && msg.contains("'deprecated'"),
        "the refusal must be InvalidInput and name the current status: {msg}"
    );

    let row = retired_row(&store, project, &old_id);
    assert_eq!(row.status, "deprecated", "the refusal rewrote the status");
    assert_eq!(row.superseded_by, None, "the refusal wrote a link");
    assert_eq!(
        all_statuses(&store, project).len(),
        1,
        "a refused supersede must insert no replacement row"
    );
}

/// Move `id` into a retired lifecycle status through the storage tier, in the
/// exact spelling `status` carries (so a test can put a *non-canonical* one in
/// the column). The store refuses to *insert* either status
/// (`ensure_valid_for_insert`) and `update` refuses `superseded`, so the retired
/// row a guard test starts from can only be seeded below the store — which is
/// also how such a row arises in practice: the storage tier takes the status
/// column as an unvalidated `&str`.
fn seed_retired_spelled(
    store: &mut MemoryStore,
    id: &str,
    project: &str,
    status: &str,
    superseded_by: Option<&str>,
) {
    store
        .db
        .update(
            id,
            project,
            crate::sqlite::UpdateOptions {
                content: None,
                embedding: None,
                metadata: None,
                memory_type: None,
                status: Some(status),
                importance: None,
            },
        )
        .expect("seed the retired status");
    if let Some(by) = superseded_by {
        store
            .db
            .conn()
            .execute(
                "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
                rusqlite::params![by, id],
            )
            .expect("seed the superseded_by link");
    }
}

fn seed_retired(
    store: &mut MemoryStore,
    id: &str,
    project: &str,
    status: MemoryStatus,
    superseded_by: Option<&str>,
) {
    seed_retired_spelled(store, id, project, status.as_str(), superseded_by);
}

/// The guard reads the column through the enum, so a retired status in a
/// spelling the write-side parser also accepts — `MemoryStatus::from_str` matches
/// case-insensitively and trims, and the storage tier takes the column as an
/// unvalidated `&str` — is refused exactly like the canonical one. A raw
/// `== "superseded"` comparison admitted these and forked the chain (#271). The
/// refusal names the *canonical* spelling, never whatever the column held.
#[test]
fn superseding_a_row_whose_retired_status_is_spelled_oddly_is_refused() {
    let (_dir, mut store) = MemoryStore::test_store();
    let project = "retired-spelling";
    // (column spelling, the canonical spelling the refusal must quote)
    for (spelling, canonical) in [
        ("Superseded", "superseded"),
        ("DEPRECATED", "deprecated"),
        (" Superseded ", "superseded"),
    ] {
        let old_id = add(&mut store, project, &format!("fact {spelling}"), None);
        seed_retired_spelled(&mut store, &old_id, project, spelling, None);

        let msg =
            refuse_retired_supersede(&mut store, project, &old_id, &format!("{spelling} twin"));
        assert!(
            msg.contains("Invalid input") && msg.contains(&format!("'{canonical}'")),
            "a tolerant spelling must be refused by its canonical status name: {msg}"
        );
        assert!(
            !msg.contains(spelling),
            "the refusal must not echo the raw column spelling {spelling:?}: {msg}"
        );
        assert_eq!(
            retired_row(&store, project, &old_id).superseded_by,
            None,
            "the refusal must not fork the supersede chain"
        );
        // The replacement row the guard refused would have been `active`, so its
        // absence is the "writes nothing" half of the assertion. Counted per
        // spelling rather than over the project: `list` matches the status column
        // exactly, so a row still carrying `Superseded` is invisible to it.
        assert_eq!(
            all_statuses(&store, project)
                .iter()
                .filter(|row| row.content == format!("{spelling} twin"))
                .count(),
            0,
            "a refused supersede must insert no replacement row"
        );
    }
}

/// Attempt to supersede `old_id` with a fresh row and return the rendered error.
/// `content` names the would-be replacement so a caller can tell its rows apart.
fn refuse_retired_supersede(
    store: &mut MemoryStore,
    project: &str,
    old_id: &str,
    content: &str,
) -> String {
    match supersede(store, project, content, old_id, MemoryStatus::Active, None) {
        Err(err) => err.to_string(),
        Ok(new_id) => panic!("superseding a retired row was accepted: {new_id}"),
    }
}

/// The one `supersede_with` call these tests make, with the row fields that do
/// not vary pinned here so each test names only what it is about.
fn supersede(
    store: &mut MemoryStore,
    project: &str,
    content: &str,
    old_id: &str,
    status: MemoryStatus,
    metadata: Option<&str>,
) -> Result<String, crate::errors::Error> {
    store.supersede_with(&crate::memory::NewSupersedingMemory {
        row: NewMemory {
            project_id: project,
            content,
            metadata,
            memory_type: MemoryType::Fact,
            status,
        },
        old_id,
        importance: MemoryImportance::Medium,
    })
}

/// A target that does not exist *in this project* is `NotFound` — the scoped
/// `WHERE` cannot tell an absent id from one owned by another project, so both
/// answer the same way — and the refusal writes no replacement row. Deleting the
/// `ok_or_else` arm of the guard's read turns this into "no such row" being
/// accepted and a replacement inserted for a target that was never there.
#[test]
fn superseding_an_unknown_id_is_not_found_and_writes_nothing() {
    let (_dir, mut store) = MemoryStore::test_store();
    let project = "not-found-target";
    let live = add(&mut store, project, "live fact", None);
    // A row that exists, but under a different project: the same scoped read that
    // misses an absent id misses this one, so the two must not diverge.
    let other = add(&mut store, "not-found-other-project", "foreign fact", None);

    for (label, missing) in [
        ("absent id", "no-such-memory-id"),
        ("other project", &other),
    ] {
        let err = supersede(
            &mut store,
            project,
            "replacement",
            missing,
            MemoryStatus::Active,
            None,
        )
        .expect_err(&format!("{label} must be refused"));
        assert!(
            matches!(err, crate::errors::Error::NotFound(_)),
            "{label} must be NotFound, got {err}"
        );
        assert!(
            err.to_string().contains("not found"),
            "{label} must render as a not-found: {err}"
        );
    }

    // Nothing was written for either refusal: the project still holds only the
    // row this test added to it, and that row is untouched.
    let rows = all_statuses(&store, project);
    assert_eq!(
        rows.len(),
        1,
        "a not-found supersede must insert no replacement row"
    );
    assert_eq!(rows[0].id, live, "the only row is the one this test added");
    assert_eq!(
        rows[0].superseded_by, None,
        "a not-found supersede wrote a link"
    );
    assert_eq!(
        retired_row(&store, "not-found-other-project", &other).status,
        "active",
        "a not-found supersede must not touch the foreign row"
    );
}

/// The metadata rule runs on the supersede path too, and runs *before* the
/// write: `supersede_with` validates the payload ahead of the embedding call and
/// the transaction, so `Some("{not json")` is `InvalidInput` and the target is
/// still live with no replacement beside it.
#[test]
fn supersede_with_rejects_invalid_metadata_and_writes_nothing() {
    let (_dir, mut store) = MemoryStore::test_store();
    let project = "bad-metadata";
    let old_id = add(&mut store, project, "original fact", None);

    let err = supersede(
        &mut store,
        project,
        "replacement with bad metadata",
        &old_id,
        MemoryStatus::Active,
        Some("{not json"),
    )
    .expect_err("invalid metadata JSON must be refused");
    assert!(
        err.to_string().contains("metadata"),
        "the refusal must name metadata: {err}"
    );

    assert_eq!(
        all_statuses(&store, project).len(),
        1,
        "a refused supersede must insert no replacement row"
    );
    let row = retired_row(&store, project, &old_id);
    assert_eq!(row.status, "active", "the refusal retired the target");
    assert_eq!(row.superseded_by, None, "the refusal wrote a link");
}

fn retired_row(store: &MemoryStore, project: &str, id: &str) -> crate::sqlite::Memory {
    store
        .get(id, project)
        .expect("get")
        .expect("retired row still readable")
}
