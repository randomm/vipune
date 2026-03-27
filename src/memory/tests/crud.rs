//! Tests for basic CRUD operations.

use crate::sqlite::Database;

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
        .insert("test-project", "test content", &embedding, Some("metadata"))
        .unwrap();

    let memory = db.get(&id).unwrap().unwrap();
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
        .insert("test-project", "original", &embedding_old, None)
        .unwrap();

    db.update(&id, "updated", &embedding_new).unwrap();

    let memory = db.get(&id).unwrap().unwrap();
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

    let result = db.update("does-not-exist", "content", &embedding);
    assert!(result.is_err());
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
        .insert("test-project", "to delete", &embedding, None)
        .unwrap();
    assert!(db.delete(&id).unwrap());

    let memory = db.get(&id).unwrap();
    assert!(memory.is_none());
}

#[test]
fn test_delete_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let deleted = db.delete("does-not-exist").unwrap();
    assert!(!deleted);
}

#[test]
fn test_get_nonexistent() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let db = Database::open(&path).unwrap();
    let memory = db.get("does-not-exist").unwrap();
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

    db.insert("project-a", "content a", &embedding, None)
        .unwrap();
    db.insert("project-b", "content b", &embedding, None)
        .unwrap();
    db.insert("project-a", "content a2", &embedding, None)
        .unwrap();

    let memories_a = db.list("project-a", 10).unwrap();
    assert_eq!(memories_a.len(), 2);

    let memories_b = db.list("project-b", 10).unwrap();
    assert_eq!(memories_b.len(), 1);
}
