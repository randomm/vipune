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
    let (_dir, path) = MemoryStore::test_db_path();

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
    let (_dir, path) = MemoryStore::test_db_path();

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
    let (_dir, path) = MemoryStore::test_db_path();

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
    let (_dir, path) = MemoryStore::test_db_path();

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
    let (_dir, path) = MemoryStore::test_db_path();

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
    let (_dir, path) = MemoryStore::test_db_path();

    let db = Database::open(&path).unwrap();
    let deleted = db.delete("does-not-exist", "test-project").unwrap();
    assert!(!deleted);
}

#[test]
fn test_get_nonexistent() {
    let (_dir, path) = MemoryStore::test_db_path();

    let db = Database::open(&path).unwrap();
    let memory = db.get("does-not-exist", "test-project").unwrap();
    assert!(memory.is_none());
}

#[test]
fn test_list_by_project() {
    let (_dir, path) = MemoryStore::test_db_path();

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

use crate::memory::NewMemory;
use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
use crate::memory::store::MemoryStore;
use crate::memory_types::{AddResult, IngestPolicy};

/// Issue #238: the flag was parsed, validated, and then discarded
/// (`let _ = importance_val;`), so every added row took the column's
/// `NOT NULL DEFAULT 'medium'` and lost the `high`/`critical` prune exemption
/// that reads the same column. These drive the store's add path (the one
/// `handle_add` calls) and read the level back through `MemoryStore::get`.
pub(in crate::memory) fn added_importance(store: &MemoryStore, id: &str, project: &str) -> String {
    store
        .get(id, project)
        .expect("get")
        .unwrap_or_else(|| panic!("row {id} was not stored"))
        .importance
}

/// As [`added_importance`], for the lifecycle status column — also read via
/// `get`, which applies no status filter (unlike `list`/`search`, which default
/// to `status = 'active'` and would hide a `candidate` row rather than report it).
pub(in crate::memory) fn added_status(store: &MemoryStore, id: &str, project: &str) -> String {
    store
        .get(id, project)
        .expect("get")
        .unwrap_or_else(|| panic!("row {id} was not stored"))
        .status
}

/// Add past the similarity gate and return the new id. `None` exercises the
/// entry point that expresses no level; the rest pass a parsed one.
pub(in crate::memory) fn add(
    store: &mut MemoryStore,
    project: &str,
    content: &str,
    importance: Option<MemoryImportance>,
) -> String {
    let result = match importance {
        Some(level) => store.ingest_with_importance(
            &NewMemory {
                project_id: project,
                content,
                metadata: None,
                memory_type: MemoryType::Fact,
                status: MemoryStatus::Active,
            },
            IngestPolicy::Force,
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

/// shipped in v0.15.4), so its behaviour is pinned here rather than assumed.
pub(in crate::memory) fn supersede_with_importance(
    store: &mut MemoryStore,
    project: &str,
    content: &str,
    old_id: &str,
    importance: MemoryImportance,
) -> String {
    store
        .supersede_with_importance(project, content, None, MemoryType::Fact, old_id, importance)
        .expect("supersede_with_importance must succeed")
}
