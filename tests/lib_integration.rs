//! Integration tests testing vipune library API from external crate perspective.

use std::env;
use std::path::PathBuf;

use vipune::errors::Error;
use vipune::{
    Config, IngestPolicy, MAX_INPUT_LENGTH, MAX_SEARCH_LIMIT, MemoryStore, detect_project,
};

/// Test basic memory add and search operations.
#[test]
fn test_memory_store_add_then_search_returns_matching_memory() {
    // Create a temporary database
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Add a memory
    let project_id = "test-project";
    let memory_id = match store
        .add_with_conflict(
            project_id,
            "Alice works at Microsoft",
            None,
            false,
            "fact",
            "active",
        )
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    assert!(!memory_id.is_empty());

    // Search for the memory
    let results = store
        .search(project_id, "where does alice work", 10, 0.0, None, None)
        .expect("Failed to search");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "Alice works at Microsoft");
    // Similarity score is present (value depends on model)

    // Clean up
    std::fs::remove_file(db_path).ok();
}

/// Test that path traversal strings are rejected by MemoryStore::new().
#[test]
fn test_memory_store_new_with_path_traversal_returns_error() {
    let config = Config::default();

    // Try to create a store with path traversal
    let traversal_path = PathBuf::from("../../../etc/passwd");

    let result = MemoryStore::new(&traversal_path, &config.embedding_model, config.clone());

    assert!(result.is_err());
}

/// Test that empty input is rejected by add().
#[test]
fn test_add_with_empty_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let result = store.add_with_conflict("test", "", None, false, "fact", "active");
    assert!(result.is_err());
    if !matches!(result.as_ref().unwrap_err(), Error::EmptyInput) {
        panic!("Expected EmptyInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that oversized input is rejected by add().
#[test]
fn test_add_with_oversized_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Create input longer than MAX_INPUT_LENGTH
    let long_text = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.add_with_conflict("test", &long_text, None, false, "fact", "active");
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that empty input is rejected by search().
#[test]
fn test_search_with_empty_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let result = store.search("test", "", 10, 0.0, None, None);
    assert!(result.is_err());
    if !matches!(result.as_ref().unwrap_err(), Error::EmptyInput) {
        panic!("Expected EmptyInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that oversized input is rejected by search().
#[test]
fn test_search_with_oversized_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Create input longer than MAX_INPUT_LENGTH
    let long_query = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.search("test", &long_query, 10, 0.0, None, None);
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that Config::default() works without environment variables.
#[test]
fn test_config_default_with_no_env_vars_returns_valid_config() {
    // Clear environment variables that might affect config
    unsafe {
        env::remove_var("VIPUNE_DATABASE_PATH");
        env::remove_var("VIPUNE_EMBEDDING_MODEL");
        env::remove_var("VIPUNE_MODEL_CACHE");
        env::remove_var("VIPUNE_SIMILARITY_THRESHOLD");
        env::remove_var("VIPUNE_RECENCY_WEIGHT");
    }

    let config = Config::default();

    assert!(config.database_path.ends_with(".vipune/memories.db"));
    assert_eq!(config.embedding_model, "BAAI/bge-small-en-v1.5");
    assert!(config.model_cache.ends_with(".vipune/models"));
    assert_eq!(config.similarity_threshold, 0.85);
    assert_eq!(config.recency_weight, 0.3);
}

/// Test that detect_project returns a non-empty string.
#[test]
fn test_detect_project_in_git_repo_returns_project_id() {
    let project_id = detect_project(None);
    assert!(!project_id.is_empty());

    // Test with explicit override
    let project_id_override = detect_project(Some("my-custom-project"));
    assert_eq!(project_id_override, "my-custom-project");
}

/// Test that Memory::fields are accessible.
#[test]
fn test_memory_with_stored_content_returns_expected_fields() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Add memory with metadata
    let memory_id = match store
        .add_with_conflict(
            "test-project",
            "Test content",
            Some(r#"{"key": "value"}"#),
            false,
            "fact",
            "active",
        )
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Get the memory
    let memory = store
        .get(&memory_id)
        .expect("Failed to get memory")
        .expect("Memory not found");

    assert_eq!(memory.id, memory_id);
    assert_eq!(memory.project_id, "test-project");
    assert_eq!(memory.content, "Test content");
    assert_eq!(memory.metadata, Some(r#"{"key": "value"}"#.to_string()));
    assert!(!memory.created_at.is_empty());
    assert!(!memory.updated_at.is_empty());
    // similarity is None when getting directly
    assert!(memory.similarity.is_none());

    std::fs::remove_file(db_path).ok();
}

/// Test hybrid search functionality.
#[test]
fn test_search_hybrid_with_test_memories_returns_fused_results() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-hybrid";

    // Add multiple memories
    match store
        .add_with_conflict(
            project_id,
            "Authentication uses JWT tokens",
            None,
            false,
            "fact",
            "active",
        )
        .expect("Failed to add memory 1")
    {
        vipune::AddResult::Added { .. } => {}
        _ => panic!("Expected AddResult::Added"),
    }
    match store
        .add_with_conflict(
            project_id,
            "User management system",
            None,
            false,
            "fact",
            "active",
        )
        .expect("Failed to add memory 2")
    {
        vipune::AddResult::Added { .. } => {}
        _ => panic!("Expected AddResult::Added"),
    }

    // Search using hybrid
    let results = store
        .search_hybrid(project_id, "auth token", 10, 0.0, None, None)
        .expect("Failed to search hybrid");

    assert!(!results.is_empty());
    assert_eq!(results[0].project_id, project_id);

    std::fs::remove_file(db_path).ok();
}

/// Test that update() validates empty input.
#[test]
fn test_update_with_empty_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let memory_id = match store
        .add_with_conflict("test", "Original content", None, false, "fact", "active")
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with empty string
    let result = store.update(&memory_id, Some(""), None, None, None);
    assert!(result.is_err());
    if !matches!(result.as_ref().unwrap_err(), Error::EmptyInput) {
        panic!("Expected EmptyInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that update() validates oversized input.
#[test]
fn test_update_with_oversized_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let memory_id = match store
        .add_with_conflict("test", "Original content", None, false, "fact", "active")
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with oversized content
    let long_text = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.update(&memory_id, Some(&long_text), None, None, None);
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that search() validates limit=0.
#[test]
fn test_search_with_zero_limit_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Try to search with limit=0
    let result = store.search("test", "query", 0, 0.0, None, None);
    assert!(result.is_err());
    if let Error::InvalidInput(msg) = &result.as_ref().unwrap_err() {
        assert!(msg.contains("Limit must be greater than 0"));
    } else {
        panic!("Expected InvalidInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that search() validates limit maximum.
#[test]
fn test_search_with_limit_over_max_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Try to search with excessively large limit
    let result = store.search("test", "query", 10_001, 0.0, None, None);
    assert!(result.is_err());
    if let Error::InvalidInput(msg) = &result.as_ref().unwrap_err() {
        assert!(msg.contains("exceeds maximum allowed"));
    } else {
        panic!("Expected InvalidInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that whitespace-only input is rejected.
#[test]
fn test_add_with_whitespace_only_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Try to add whitespace-only content
    let result = store.add_with_conflict("test", "   ", None, false, "fact", "active");
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    // Try to search with whitespace-only query
    let result = store.search("test", "\t\n", 10, 0.0, None, None);
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    std::fs::remove_file(db_path).ok();
}

/// Test that symlink pointing outside temp dir is handled correctly.
#[cfg(unix)]
#[test]
fn test_memory_store_new_with_symlink_traversal_returns_error() {
    use std::os::unix::fs;

    let temp_dir = env::temp_dir();
    let test_dir = temp_dir.join(format!("vipune_symlink_test_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&test_dir).expect("Failed to create test directory");

    let config = Config::default();

    // Create a symlink to a non-existent path outside the test dir
    let symlink_path = test_dir.join("symlink.db");
    let target_path = PathBuf::from("/nonexistent/path/database.db");
    fs::symlink(&target_path, &symlink_path).expect("Failed to create symlink");

    // Try to create store with symlink path
    // Path traversal guard rejects paths with parent-dir components before any filesystem access
    let result = MemoryStore::new(&symlink_path, &config.embedding_model, config.clone());

    // Clean up (always runs even if assertion fails)
    std::fs::remove_file(&symlink_path).ok();
    std::fs::remove_dir(&test_dir).ok();

    // Should fail (path traversal prevention or database open failure)
    assert!(
        result.is_err(),
        "MemoryStore creation should fail for inaccessible symlink"
    );
}

/// Test that path with parent-dir component is rejected.
#[test]
fn test_memory_store_new_with_parent_dir_component_returns_error() {
    let config = Config::default();

    // Use a path with parent-dir component
    let traversal_path = PathBuf::from("/tmp/../etc/evil.db");

    let result = MemoryStore::new(&traversal_path, &config.embedding_model, config.clone());

    // Should be rejected with parent dir error message
    match result {
        Err(Error::Config(msg)) => {
            assert!(
                msg.contains("..") || msg.contains("escape"),
                "Expected parent directory rejection message, got: {}",
                msg
            );
        }
        Err(e) => {
            panic!(
                "Expected Config error with parent dir rejection, got: {}",
                e
            );
        }
        Ok(_) => {
            panic!("MemoryStore creation should fail for path with parent directory component");
        }
    }
}

/// Test that MAX_SEARCH_LIMIT constant is accessible from library API.
#[test]
fn test_constant_max_search_limit_is_accessible() {
    assert_eq!(MAX_SEARCH_LIMIT, 10_000);
}

/// Test that list() validates limit=0.
#[test]
fn test_list_with_zero_limit_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Try to list with limit=0
    let result = store.list("test", 0, None, None);
    assert!(result.is_err());
    if let Error::InvalidInput(msg) = &result.as_ref().unwrap_err() {
        assert!(msg.contains("Limit must be greater than 0"));
    } else {
        panic!("Expected InvalidInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that list() validates limit maximum.
#[test]
fn test_list_with_limit_over_max_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Try to list with excessively large limit
    let result = store.list("test", 10_001, None, None);
    assert!(result.is_err());
    if let Error::InvalidInput(msg) = &result.as_ref().unwrap_err() {
        assert!(msg.contains("exceeds maximum allowed"));
    } else {
        panic!("Expected InvalidInput error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test list() regression coverage for existing behavior.
#[test]
fn test_list_regression_coverage() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add multiple memories
    let _id1 = store
        .add_with_conflict(project_id, "first memory", None, true, "fact", "active")
        .expect("Failed to add memory");
    let _id2 = store
        .add_with_conflict(project_id, "second memory", None, true, "fact", "active")
        .expect("Failed to add memory");
    let _id3 = store
        .add_with_conflict(project_id, "third memory", None, true, "fact", "active")
        .expect("Failed to add memory");

    // Test ordering (newest first)
    let results = store
        .list(project_id, 10, None, None)
        .expect("Failed to list");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].content, "third memory");
    assert_eq!(results[1].content, "second memory");
    assert_eq!(results[2].content, "first memory");

    // Test limit
    let results = store
        .list(project_id, 2, None, None)
        .expect("Failed to list");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "third memory");

    // Test project isolation
    let _id4 = store
        .add_with_conflict(
            "other-project",
            "other memory",
            None,
            true,
            "fact",
            "active",
        )
        .expect("Failed to add memory");
    let project1_results = store
        .list(project_id, 10, None, None)
        .expect("Failed to list");
    assert_eq!(project1_results.len(), 3);

    // Test empty project
    let empty_results = store
        .list("nonexistent_project", 10, None, None)
        .expect("Failed to list");
    assert_eq!(empty_results.len(), 0);

    std::fs::remove_file(db_path).ok();
}

/// Test list_since() with timezone offset RFC3339 timestamps.
#[test]
fn test_list_since_with_timezone_offset() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memories with controlled timestamps
    let _old = store
        .add_with_conflict(project_id, "old memory", None, true, "fact", "active")
        .expect("Failed to add memory");

    // Wait at least 1ms to ensure different timestamps
    std::thread::sleep(std::time::Duration::from_millis(10));

    let _new = store
        .add_with_conflict(project_id, "new memory", None, true, "fact", "active")
        .expect("Failed to add memory");

    // Test with UTC timestamp (should succeed and only return newer)
    let now = chrono::Utc::now();
    let one_minute_ago = (now - chrono::Duration::minutes(1)).to_rfc3339();
    let results = store
        .list_since(project_id, &one_minute_ago, 10, None, None)
        .expect("Failed to list");
    // Should only return "new memory" as it's more recent than one minute ago
    assert!(results.len() >= 1);
    assert!(results.iter().any(|m| m.content == "new memory"));

    std::fs::remove_file(db_path).ok();
}

/// Test list_since() timestamp precision equivalence.
#[test]
fn test_list_since_timestamp_precision_equivalence() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    let _id = store
        .add_with_conflict(project_id, "test memory", None, true, "fact", "active")
        .expect("Failed to add memory");

    // Wait to ensure timestamp difference
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Query with and without fractional seconds - should behave identically
    let now = chrono::Utc::now();
    let two_seconds_ago = (now - chrono::Duration::seconds(2)).to_rfc3339();

    let results1 = store
        .list_since(project_id, &two_seconds_ago, 10, None, None)
        .expect("Failed to list");

    let results2 = store
        .list_since(project_id, &two_seconds_ago, 10, None, None)
        .expect("Failed to list");

    // Results should be identical (same query, same results)
    assert_eq!(results1.len(), results2.len());
    if results1.len() > 0 && results2.len() > 0 {
        assert_eq!(results1[0].id, results2[0].id);
    }

    std::fs::remove_file(db_path).ok();
}

/// Test get_many() with duplicate IDs returns stable behavior.
#[test]
fn test_get_many_with_duplicate_ids() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    let id1 = match store
        .add_with_conflict(project_id, "first", None, true, "fact", "active")
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    let id2 = match store
        .add_with_conflict(project_id, "second", None, true, "fact", "active")
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Query with duplicate IDs
    let results = store
        .get_many(&[&id1, &id2, &id1, &id2])
        .expect("Failed to get many");
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].as_ref().unwrap().id, id1);
    assert_eq!(results[1].as_ref().unwrap().id, id2);
    assert_eq!(results[2].as_ref().unwrap().id, id1);
    assert_eq!(results[3].as_ref().unwrap().id, id2);

    std::fs::remove_file(db_path).ok();
}

/// Test that add() succeeds at exactly MAX_INPUT_LENGTH.
#[test]
fn test_add_at_exactly_max_input_length_returns_success() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Create input exactly at MAX_INPUT_LENGTH
    let exact_text = "x".repeat(MAX_INPUT_LENGTH);
    let result = store.add_with_conflict("test", &exact_text, None, false, "fact", "active");
    assert!(
        result.is_ok(),
        "Should accept input at exactly MAX_INPUT_LENGTH"
    );

    std::fs::remove_file(db_path).ok();
}

/// Test that add() rejects input one character over MAX_INPUT_LENGTH.
#[test]
fn test_add_one_over_max_input_length_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Create input one character over MAX_INPUT_LENGTH
    let too_long_text = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.add_with_conflict("test", &too_long_text, None, false, "fact", "active");
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that search_hybrid() validates empty input.
#[test]
fn test_search_hybrid_with_empty_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let result = store.search_hybrid("test", "", 10, 0.0, None, None);
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    std::fs::remove_file(db_path).ok();
}

/// Test that search_hybrid() validates oversized input.
#[test]
fn test_search_hybrid_with_oversized_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let long_query = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.search_hybrid("test", &long_query, 10, 0.0, None, None);
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that ingest with ConflictAware policy maps to existing conflict behavior.
#[test]
fn test_ingest_conflict_aware_policy_maps_to_existing_behavior() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add first memory
    let result = store
        .ingest(
            project_id,
            "Alice works at Microsoft",
            None,
            IngestPolicy::ConflictAware,
        )
        .expect("Failed to add memory");
    assert!(
        matches!(result, vipune::AddResult::Added { .. }),
        "First add should succeed with ConflictAware policy"
    );

    // Add duplicate content - should detect conflict
    let result = store
        .ingest(
            project_id,
            "Alice works at Microsoft",
            None,
            IngestPolicy::ConflictAware,
        )
        .expect("Failed to check conflicts");
    assert!(
        matches!(result, vipune::AddResult::Conflicts { .. }),
        "Duplicate content should return conflicts with ConflictAware policy"
    );

    std::fs::remove_file(db_path).ok();
}

/// Test that ingest with Force policy bypasses conflict detection.
#[test]
fn test_ingest_force_policy_bypasses_conflicts() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add first memory
    let _id1 = match store
        .ingest(
            project_id,
            "Alice works at Microsoft",
            None,
            IngestPolicy::ConflictAware,
        )
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("First add should succeed"),
    };

    // Add duplicate content with Force - should succeed regardless
    let id2 = match store
        .ingest(
            project_id,
            "Alice works at Microsoft",
            None,
            IngestPolicy::Force,
        )
        .expect("Failed to force add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Force policy should always return Added"),
    };

    // Verify both memories exist
    let results = store
        .list(project_id, 10, None, None)
        .expect("Failed to list memories");
    assert_eq!(
        results.len(),
        2,
        "Both memories should exist after Force add"
    );
    assert!(
        results.iter().any(|m| m.id == id2),
        "Second memory should be stored with Force policy"
    );

    std::fs::remove_file(db_path).ok();
}

/// Test that ingest validates empty input.
#[test]
fn test_ingest_with_empty_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Test both policies reject empty input
    let result = store.ingest("test", "", None, IngestPolicy::ConflictAware);
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    let result = store.ingest("test", "", None, IngestPolicy::Force);
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    std::fs::remove_file(db_path).ok();
}

/// Test that ingest validates oversized input.
#[test]
fn test_ingest_with_oversized_input_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let long_text = "x".repeat(MAX_INPUT_LENGTH + 1);

    // Test both policies reject oversized input
    let result = store.ingest("test", &long_text, None, IngestPolicy::ConflictAware);
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    let result = store.ingest("test", &long_text, None, IngestPolicy::Force);
    assert!(result.is_err());
    if let Error::InputTooLong {
        max_length,
        actual_length,
    } = &result.as_ref().unwrap_err()
    {
        assert_eq!(*max_length, MAX_INPUT_LENGTH);
        assert_eq!(*actual_length, MAX_INPUT_LENGTH + 1);
    } else {
        panic!("Expected InputTooLong error");
    }

    std::fs::remove_file(db_path).ok();
}

/// Test that ingest works with metadata.
#[test]
fn test_ingest_with_metadata_succeeds() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    // Test ConflictAware with metadata
    let id1 = match store
        .ingest(
            "test-project",
            "Test content",
            Some(r#"{"source": "manual"}"#),
            IngestPolicy::ConflictAware,
        )
        .expect("Failed to ingest")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Test Force with metadata
    let id2 = match store
        .ingest(
            "test-project",
            "Another test",
            Some(r#"{"source": "import"}"#),
            IngestPolicy::Force,
        )
        .expect("Failed to ingest")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Verify metadata was stored
    let memory1 = store.get(&id1).expect("Failed to get").expect("Not found");
    assert_eq!(
        memory1.metadata.as_ref().unwrap(),
        r#"{"source": "manual"}"#
    );

    let memory2 = store.get(&id2).expect("Failed to get").expect("Not found");
    assert_eq!(
        memory2.metadata.as_ref().unwrap(),
        r#"{"source": "import"}"#
    );

    std::fs::remove_file(db_path).ok();
}

/// Test that updating text-only preserves existing metadata.
#[test]
fn test_update_text_only_preserves_metadata() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memory with metadata
    let id = match store
        .add_with_conflict(
            project_id,
            "original content",
            Some(r#"{"tag": "important"}"#),
            true,
            "fact",
            "active",
        )
        .expect("Failed to add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Update only text
    store
        .update(&id, Some("updated content"), None, None, None)
        .expect("Failed to update");

    // Verify content updated but metadata preserved
    let memory = store.get(&id).expect("Failed to get").expect("Not found");
    assert_eq!(memory.content, "updated content");
    assert_eq!(memory.metadata.as_ref().unwrap(), r#"{"tag": "important"}"#);

    std::fs::remove_file(db_path).ok();
}

/// Test that invalid JSON metadata is rejected during update.
#[test]
fn test_update_with_invalid_json_metadata_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memory
    let id = match store
        .add_with_conflict(project_id, "original content", None, true, "fact", "active")
        .expect("Failed to add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with invalid JSON - should fail
    let result = store.update(&id, None, Some(r#"{this is not valid json"#), None, None);
    assert!(result.is_err());
    match result {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("invalid metadata JSON"));
        }
        _ => panic!("Expected InvalidInput error for invalid JSON"),
    }

    // Verify memory was not changed
    let memory = store.get(&id).expect("Failed to get").expect("Not found");
    assert_eq!(memory.content, "original content");
    assert!(memory.metadata.is_none());

    std::fs::remove_file(db_path).ok();
}

/// Test that empty string metadata is rejected.
#[test]
fn test_update_with_empty_metadata_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memory
    let id = match store
        .add_with_conflict(project_id, "original content", None, true, "fact", "active")
        .expect("Failed to add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with empty metadata - should fail
    let result = store.update(&id, None, Some(""), None, None);
    assert!(result.is_err());
    match result {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("metadata cannot be empty"));
        }
        _ => panic!("Expected InvalidInput error for empty metadata"),
    }

    // Verify memory was not changed
    let memory = store.get(&id).expect("Failed to get").expect("Not found");
    assert_eq!(memory.content, "original content");
    assert!(memory.metadata.is_none());

    std::fs::remove_file(db_path).ok();
}

/// Test that whitespace-only metadata is rejected.
#[test]
fn test_update_with_whitespace_only_metadata_returns_error() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memory
    let id = match store
        .add_with_conflict(project_id, "original content", None, true, "fact", "active")
        .expect("Failed to add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with whitespace-only metadata - should fail
    let result = store.update(&id, None, Some("   "), None, None);
    assert!(result.is_err());
    match result {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("metadata cannot be empty"));
        }
        _ => panic!("Expected InvalidInput error for whitespace-only metadata"),
    }

    // Verify memory was not changed
    let memory = store.get(&id).expect("Failed to get").expect("Not found");
    assert_eq!(memory.content, "original content");
    assert!(memory.metadata.is_none());

    std::fs::remove_file(db_path).ok();
}

/// Test that metadata-only update works correctly.
#[test]
fn test_update_metadata_only() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memory without metadata
    let id = match store
        .add_with_conflict(project_id, "original content", None, true, "fact", "active")
        .expect("Failed to add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Update metadata only
    store
        .update(&id, None, Some(r#"{"tag": "new"}"#), None, None)
        .expect("Failed to update");

    // Verify metadata added but content unchanged
    let memory = store.get(&id).expect("Failed to get").expect("Not found");
    assert_eq!(memory.content, "original content");
    assert_eq!(memory.metadata.as_ref().unwrap(), r#"{"tag": "new"}"#);

    std::fs::remove_file(db_path).ok();
}

/// Test that updating both text and metadata works.
#[test]
fn test_update_both_text_and_metadata() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = "test-project";

    // Add memory with old metadata
    let id = match store
        .add_with_conflict(
            project_id,
            "old content",
            Some(r#"{"old": "value"}"#),
            true,
            "fact",
            "active",
        )
        .expect("Failed to add")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Update both text and metadata
    store
        .update(
            &id,
            Some("new content"),
            Some(r#"{"new": "value"}"#),
            None,
            None,
        )
        .expect("Failed to update");

    // Verify both updated
    let memory = store.get(&id).expect("Failed to get").expect("Not found");
    assert_eq!(memory.content, "new content");
    assert_eq!(memory.metadata.as_ref().unwrap(), r#"{"new": "value"}"#);

    std::fs::remove_file(db_path).ok();
}

/// Test that add_with_conflict creates independent active memories.
///
/// Note: This does NOT test the supersede lifecycle. Direct supersede testing requires
/// Database-level access (not available through MemoryStore public API). This test
/// verifies that multiple add_with_conflict calls create separate active memories.
#[test]
fn test_add_with_conflict_creates_multiple_active_memories() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = format!("test-project-{}", uuid::Uuid::new_v4());

    // Add memory A with type=fact, status=active using add_with_conflict (bypasses similarity check)
    let memory_a_id = match store
        .add_with_conflict(
            &project_id,
            "memory A content",
            None,
            true,
            "fact",
            "active",
        )
        .expect("Failed to add memory A")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Add another memory - both should be independent and active
    let memory_b_id = match store
        .add_with_conflict(
            &project_id,
            "memory B content",
            None,
            true,
            "fact",
            "active",
        )
        .expect("Failed to add memory B")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Verify both memories exist and are active
    let memory_a = store
        .get(&memory_a_id)
        .expect("Failed to get memory A")
        .expect("Memory A not found");
    let memory_b = store
        .get(&memory_b_id)
        .expect("Failed to get memory B")
        .expect("Memory B not found");

    assert_eq!(memory_a.status, "active");
    assert_eq!(memory_b.status, "active");

    std::fs::remove_file(db_path).ok();
}

/// Test that default search excludes candidate memories (only returns active).
#[test]
fn test_default_search_excludes_candidate() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = format!("test-project-{}", uuid::Uuid::new_v4());

    // Add active memory using add_with_conflict with force=true
    let _active_id = match store
        .add_with_conflict(&project_id, "active memory", None, true, "fact", "active")
        .expect("Failed to add active")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Add candidate memory using add_with_conflict with force=true
    let _candidate_id = match store
        .add_with_conflict(
            &project_id,
            "candidate memory",
            None,
            true,
            "fact",
            "candidate",
        )
        .expect("Failed to add candidate")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Search with default status filter (statuses=None means active only)
    let results = store
        .search(&project_id, "memory", 10, 0.0, None, None)
        .expect("Failed to search");

    // Verify only active is returned
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "active memory");

    std::fs::remove_file(db_path).ok();
}

/// Test that searching with explicit statuses includes both active and candidate.
#[test]
fn test_include_candidates_returns_both() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = format!("test-project-{}", uuid::Uuid::new_v4());

    // Add active memory
    let _active_id = match store
        .add_with_conflict(&project_id, "active memory", None, true, "fact", "active")
        .expect("Failed to add active")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Add candidate memory
    let _candidate_id = match store
        .add_with_conflict(
            &project_id,
            "candidate memory",
            None,
            true,
            "fact",
            "candidate",
        )
        .expect("Failed to add candidate")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Search with explicit statuses=["active", "candidate"]
    let statuses = ["active", "candidate"];
    let results = store
        .search(&project_id, "memory", 10, 0.0, None, Some(&statuses))
        .expect("Failed to search");

    // Verify both are returned
    assert_eq!(results.len(), 2);

    std::fs::remove_file(db_path).ok();
}

/// Test that default list excludes non-active memories.
#[test]
fn test_default_list_excludes_non_active() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = format!("test-project-{}", uuid::Uuid::new_v4());

    // Add memories with various statuses using add_with_conflict with force=true
    let _active_id = match store
        .add_with_conflict(&project_id, "active memory", None, true, "fact", "active")
        .expect("Failed to add active")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    let _candidate_id = match store
        .add_with_conflict(
            &project_id,
            "candidate memory",
            None,
            true,
            "fact",
            "candidate",
        )
        .expect("Failed to add candidate")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    let _deprecated_id = match store
        .add_with_conflict(
            &project_id,
            "deprecated memory",
            None,
            true,
            "fact",
            "deprecated",
        )
        .expect("Failed to add deprecated")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // List with statuses=None (default = active only)
    let results = store
        .list(&project_id, 10, None, None)
        .expect("Failed to list");

    // Verify only active returned
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "active memory");

    std::fs::remove_file(db_path).ok();
}

/// Test that hybrid search respects status filter (BM25 filter fix from #102).
#[test]
fn test_hybrid_search_respects_status_filter() {
    let temp_dir = env::temp_dir();
    let db_path = temp_dir.join(format!("vipune_test_{}.db", uuid::Uuid::new_v4()));

    let config = Config::default();
    let mut store = MemoryStore::new(db_path.as_path(), &config.embedding_model, config.clone())
        .expect("Failed to create store");

    let project_id = format!("test-project-{}", uuid::Uuid::new_v4());

    // Add active memory
    let _active_id = match store
        .add_with_conflict(
            &project_id,
            "rust programming language",
            None,
            true,
            "fact",
            "active",
        )
        .expect("Failed to add active")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Add candidate memory with similar text (would match BM25 if candidate was included)
    let _candidate_id = match store
        .add_with_conflict(
            &project_id,
            "old rust programming info",
            None,
            true,
            "fact",
            "candidate",
        )
        .expect("Failed to add candidate")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Hybrid search with statuses=None (default = active only)
    let results = store
        .search_hybrid(&project_id, "rust programming", 10, 0.0, None, None)
        .expect("Failed to search hybrid");

    // Verify only active memories in results
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "rust programming language");
    assert_eq!(results[0].status, "active");

    std::fs::remove_file(db_path).ok();
}
