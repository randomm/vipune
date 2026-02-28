//! Integration tests testing vipune library API from external crate perspective.

use std::env;
use std::path::PathBuf;

use vipune::errors::Error;
use vipune::{detect_project, Config, MemoryStore, MAX_INPUT_LENGTH, MAX_SEARCH_LIMIT};

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
        .add_with_conflict(project_id, "Alice works at Microsoft", None, false)
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    assert!(!memory_id.is_empty());

    // Search for the memory
    let results = store
        .search(project_id, "where does alice work", 10, 0.0)
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

    let result = store.add_with_conflict("test", "", None, false);
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
    let result = store.add_with_conflict("test", &long_text, None, false);
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

    let result = store.search("test", "", 10, 0.0);
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
    let result = store.search("test", &long_query, 10, 0.0);
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
    env::remove_var("VIPUNE_DATABASE_PATH");
    env::remove_var("VIPUNE_EMBEDDING_MODEL");
    env::remove_var("VIPUNE_MODEL_CACHE");
    env::remove_var("VIPUNE_SIMILARITY_THRESHOLD");
    env::remove_var("VIPUNE_RECENCY_WEIGHT");

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
        .add_with_conflict(project_id, "Authentication uses JWT tokens", None, false)
        .expect("Failed to add memory 1")
    {
        vipune::AddResult::Added { .. } => {}
        _ => panic!("Expected AddResult::Added"),
    }
    match store
        .add_with_conflict(project_id, "User management system", None, false)
        .expect("Failed to add memory 2")
    {
        vipune::AddResult::Added { .. } => {}
        _ => panic!("Expected AddResult::Added"),
    }

    // Search using hybrid
    let results = store
        .search_hybrid(project_id, "auth token", 10, 0.0)
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
        .add_with_conflict("test", "Original content", None, false)
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with empty string
    let result = store.update(&memory_id, "");
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
        .add_with_conflict("test", "Original content", None, false)
        .expect("Failed to add memory")
    {
        vipune::AddResult::Added { id } => id,
        _ => panic!("Expected AddResult::Added"),
    };

    // Try to update with oversized content
    let long_text = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.update(&memory_id, &long_text);
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
    let result = store.search("test", "query", 0, 0.0);
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
    let result = store.search("test", "query", 10_001, 0.0);
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
    let result = store.add_with_conflict("test", "   ", None, false);
    assert!(result.is_err());
    assert!(matches!(result.as_ref().unwrap_err(), Error::EmptyInput));

    // Try to search with whitespace-only query
    let result = store.search("test", "\t\n", 10, 0.0);
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
    let result = store.list("test", 0);
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
    let result = store.list("test", 10_001);
    assert!(result.is_err());
    if let Error::InvalidInput(msg) = &result.as_ref().unwrap_err() {
        assert!(msg.contains("exceeds maximum allowed"));
    } else {
        panic!("Expected InvalidInput error");
    }

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
    let result = store.add_with_conflict("test", &exact_text, None, false);
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
    let result = store.add_with_conflict("test", &too_long_text, None, false);
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

    let result = store.search_hybrid("test", "", 10, 0.0);
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
    let result = store.search_hybrid("test", &long_query, 10, 0.0);
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
