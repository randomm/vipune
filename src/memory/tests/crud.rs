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

/// Issue #238: `vipune add --importance <level>` must persist what it
/// validated.
///
/// The flag was parsed, checked against the vocabulary, and then discarded
/// (`let _ = importance_val;`), so every added row took the column's
/// `NOT NULL DEFAULT 'medium'`. The consequence was not cosmetic: prune's
/// hard exclusion keys off `high`/`critical`, so the operator paid for
/// protection they never got. These tests drive the store's add path — the one
/// the CLI handler calls — and read the column back raw, so a future refactor
/// that drops the binding fails here rather than in someone's prune.
fn stored_importance(db: &Database, id: &str) -> String {
    db.conn()
        .query_row(
            "SELECT importance FROM memories WHERE id = ?",
            [id],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|e| panic!("no row for {id}: {e}"))
}

#[test]
fn add_with_importance_persists_each_level_verbatim() {
    use crate::memory::lifecycle::{MemoryImportance, MemoryStatus, MemoryType};
    use crate::memory::store::MemoryStore;
    use crate::memory_types::IngestPolicy;

    let levels = [
        MemoryImportance::Low,
        MemoryImportance::Medium,
        MemoryImportance::High,
        MemoryImportance::Critical,
    ];

    for (i, level) in levels.iter().enumerate() {
        let mut store = MemoryStore::test_store();
        let content = format!("importance probe {i}: {}", level.as_str());
        let result = store
            .ingest_with_importance(
                "imp-proj",
                &content,
                None,
                // Force skips the similarity gate so the row is always written;
                // the write path is the same one the non-forced branch takes.
                IngestPolicy::Force,
                MemoryType::Fact,
                MemoryStatus::Active,
                *level,
            )
            .unwrap();
        let id = match result {
            crate::memory_types::AddResult::Added { id } => id,
            other => panic!("forced add must store the row, got {other:?}"),
        };

        assert_eq!(
            stored_importance(&store.db, &id),
            level.as_str(),
            "--importance {} must reach the column, not the DEFAULT",
            level.as_str(),
        );
    }
}

#[test]
fn add_without_an_explicit_level_stores_medium() {
    // The default the operator gets when they say nothing. Pinning it here
    // means the `medium` literal exists in exactly one place per layer, so a
    // later change to the vocabulary cannot be half-applied.
    use crate::memory::lifecycle::{MemoryStatus, MemoryType};
    use crate::memory::store::MemoryStore;
    use crate::memory_types::IngestPolicy;

    let mut store = MemoryStore::test_store();
    let id = match store
        .ingest_with_type_status(
            "imp-proj",
            "no opinion expressed",
            None,
            IngestPolicy::Force,
            MemoryType::Fact,
            MemoryStatus::Active,
        )
        .unwrap()
    {
        crate::memory_types::AddResult::Added { id } => id,
        other => panic!("forced add must store the row, got {other:?}"),
    };

    assert_eq!(stored_importance(&store.db, &id), "medium");
}

#[test]
fn add_superseding_persists_the_importance_on_the_new_row() {
    // `--supersedes` is a second, independent add path: it inserts the new row
    // inside Database::supersede's transaction, so it needs the binding too.
    use crate::memory::lifecycle::{MemoryImportance, MemoryType};
    use crate::memory::store::MemoryStore;

    let mut store = MemoryStore::test_store();
    let old_id = match store
        .add_with_conflict(
            "sup-proj",
            "original fact",
            None,
            true,
            MemoryType::Fact,
            crate::memory::lifecycle::MemoryStatus::Active,
        )
        .unwrap()
    {
        crate::memory_types::AddResult::Added { id } => id,
        other => panic!("forced add must store the row, got {other:?}"),
    };

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
        stored_importance(&store.db, &new_id),
        "critical",
        "the superseding row must carry the level the operator asked for"
    );
    // The superseded row keeps its own level — supersede updates status and
    // superseded_by, never importance.
    assert_eq!(stored_importance(&store.db, &old_id), "medium");

    // And the defaulting wrapper still lands on medium rather than sharing an
    // accidental value with the variant above.
    let plain_id = store
        .supersede("sup-proj", "third fact", None, MemoryType::Fact, &new_id)
        .unwrap();
    assert_eq!(stored_importance(&store.db, &plain_id), "medium");
}
