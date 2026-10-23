//! Tests for basic CRUD operations.

use crate::sqlite::{Database, UpdateOptions};

#[test]
fn test_memory_store_new() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let _path = dir.path().join("test.db");

    // Note: This test is ignored because it requires downloading the model
    // Use `cargo test -- --ignored` to run it
}

#[test]
fn test_add_and_get() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];
    let id = db
        .insert(
            "test-project",
            "test content",
            &embedding,
            Some("metadata"),
            "fact",
            "active",
        )
        .unwrap();

    let memory = db.get(&id, "test-project").unwrap().unwrap();
    assert_eq!(memory.id, id);
    assert_eq!(memory.content, "test content");
    assert_eq!(memory.project_id, "test-project");
    assert_eq!(memory.metadata, Some("metadata".to_string()));
}

#[test]
fn test_update_memory() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding_old = vec![0.3f32; 384];
    let embedding_new = vec![0.8f32; 384];

    let id = db
        .insert(
            "test-project",
            "original",
            &embedding_old,
            None,
            "fact",
            "active",
        )
        .unwrap();

    db.update(
        &id,
        "test-project",
        UpdateOptions {
            content: Some("updated"),
            embedding: Some(&embedding_new),
            metadata: None,
            memory_type: None,
            status: None,
            importance: None,
        },
    )
    .unwrap();

    let memory = db.get(&id, "test-project").unwrap().unwrap();
    assert_eq!(memory.content, "updated");
    assert_ne!(memory.created_at, memory.updated_at);
}

#[test]
fn test_update_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    let result = db.update(
        "does-not-exist",
        "test-project",
        UpdateOptions {
            content: Some("content"),
            embedding: Some(&embedding),
            metadata: None,
            memory_type: None,
            status: None,
            importance: None,
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_update_cross_project_isolation() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    let id = db
        .insert(
            "proj-a",
            "content in proj-a",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();

    // Attempt to update with wrong project_id — should fail (0 rows affected)
    let result = db.update(
        &id,
        "proj-b",
        UpdateOptions {
            content: Some("malicious update"),
            embedding: Some(&embedding),
            metadata: None,
            memory_type: None,
            status: None,
            importance: None,
        },
    );
    assert!(result.is_err(), "update with wrong project_id must fail");

    // Verify content was NOT modified
    let memory = db.get(&id, "proj-a").unwrap().unwrap();
    assert_eq!(memory.content, "content in proj-a");
}

#[test]
fn test_delete_memory() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    let id = db
        .insert(
            "test-project",
            "to delete",
            &embedding,
            None,
            "fact",
            "active",
        )
        .unwrap();
    assert!(db.delete(&id, "test-project").unwrap());

    let memory = db.get(&id, "test-project").unwrap();
    assert!(memory.is_none());
}

#[test]
fn test_delete_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let deleted = db.delete("does-not-exist", "test-project").unwrap();
    assert!(!deleted);
}

#[test]
fn test_get_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let memory = db.get("does-not-exist", "test-project").unwrap();
    assert!(memory.is_none());
}

#[test]
fn test_list_by_project() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let embedding = vec![0.5f32; 384];

    db.insert("project-a", "content a", &embedding, None, "fact", "active")
        .unwrap();
    db.insert("project-b", "content b", &embedding, None, "fact", "active")
        .unwrap();
    db.insert(
        "project-a",
        "content a2",
        &embedding,
        None,
        "fact",
        "active",
    )
    .unwrap();

    let memories_a = db.list("project-a", 10, None, None).unwrap();
    assert_eq!(memories_a.len(), 2);

    let memories_b = db.list("project-b", 10, None, None).unwrap();
    assert_eq!(memories_b.len(), 1);
}

use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::store::MemoryStore;
use crate::memory_types::{AddResult, IngestPolicy};

/// Issue #238: the flag was parsed, validated, and then discarded
/// (`let _ = importance_val;`), so every added row took the column's
/// `NOT NULL DEFAULT 'medium'` and lost the `high`/`critical` prune exemption
/// that reads the same column. These drive the store's add path (the one
/// `handle_add` calls) and read the level back through `MemoryStore::get`.
fn added_importance(store: &MemoryStore, id: &str, project: &str) -> String {
    store
        .get(id, project)
        .expect("get")
        .unwrap_or_else(|| panic!("row {id} was not stored"))
        .importance
}

/// Add past the similarity gate and return the new id. `None` exercises the
/// entry point that expresses no level; the rest pass a parsed one.
fn add(
    store: &mut MemoryStore,
    project: &str,
    content: &str,
    importance: Option<MemoryImportance>,
) -> String {
    let result = match importance {
        Some(level) => store.ingest_with_importance(
            project,
            content,
            None,
            IngestPolicy::Force,
            MemoryType::Fact,
            MemoryStatus::Active,
            level,
        ),
        None => store.ingest_with_type_status(
            project,
            content,
            None,
            IngestPolicy::Force,
            MemoryType::Fact,
            MemoryStatus::Active,
        ),
    }
    .expect("forced add must succeed");
    match result {
        AddResult::Added { id } => id,
        other => panic!("a forced add stores the row, got {other:?}"),
    }
}

#[test]
fn add_persists_the_importance_it_was_given() {
    let mut store = MemoryStore::test_store();
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
fn add_superseding_persists_the_importance_on_the_new_row() {
    // `--supersedes` is a second, independent add path: it inserts the new row
    // inside the supersede transaction, so that INSERT needs the binding too.
    let mut store = MemoryStore::test_store();
    let old_id = add(&mut store, "sup-proj", "original fact", None);
    let new_id = store
        .supersede_with_importance(
            "sup-proj",
            "replacement fact",
            None,
            MemoryType::Fact,
            &old_id,
            MemoryImportance::Critical,
        )
        .unwrap();
    assert_eq!(
        added_importance(&store, &new_id, "sup-proj"),
        "critical",
        "the superseding row must carry the level the operator asked for"
    );
    // The superseded row keeps its own level, and the defaulting wrapper lands
    // on medium rather than sharing the variant's value by accident.
    assert_eq!(added_importance(&store, &old_id, "sup-proj"), "medium");
    let plain_id = store
        .supersede("sup-proj", "third fact", None, MemoryType::Fact, &new_id)
        .unwrap();
    assert_eq!(added_importance(&store, &plain_id, "sup-proj"), "medium");
}
