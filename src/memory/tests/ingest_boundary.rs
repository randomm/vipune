//! Boundary and edge case tests for ingest operations.
//!
//! These tests require the embedding model, so they run with:
//! `cargo test -- --ignored`

use crate::config::Config;
use crate::memory::MemoryStore;
use crate::memory::store::MAX_INPUT_LENGTH;
use crate::memory_types::IngestPolicy;
use tempfile::TempDir;

/// Test that ingest accepts input at exactly MAX_INPUT_LENGTH - 1 characters.
#[ignore]
#[test]
fn test_ingest_at_max_input_length_minus_one_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    // Create input at MAX_INPUT_LENGTH - 1
    let text = "x".repeat(MAX_INPUT_LENGTH - 1);
    let result = store.ingest("test-project", &text, None, IngestPolicy::Force);

    assert!(
        result.is_ok(),
        "Should accept input at MAX_INPUT_LENGTH - 1"
    );
}

/// Test that ingest accepts input at exactly MAX_INPUT_LENGTH.
#[ignore]
#[test]
fn test_ingest_at_max_input_length_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    // Create input exactly at MAX_INPUT_LENGTH
    let text = "x".repeat(MAX_INPUT_LENGTH);
    let result = store.ingest("test-project", &text, None, IngestPolicy::Force);

    assert!(
        result.is_ok(),
        "Should accept input at exactly MAX_INPUT_LENGTH"
    );
}

/// Test that ingest rejects input at MAX_INPUT_LENGTH + 1.
#[test]
fn test_ingest_at_max_input_length_plus_one_fails() {
    // Validation happens before embedding, so no model needed
    let mut store = MemoryStore::test_store();

    // Create input one character over MAX_INPUT_LENGTH
    let text = "x".repeat(MAX_INPUT_LENGTH + 1);
    let result = store.ingest("test-project", &text, None, IngestPolicy::Force);

    assert!(
        result.is_err(),
        "Should reject input at MAX_INPUT_LENGTH + 1"
    );
}

/// Test that ingest rejects whitespace-only input with clear error message.
///
/// Note: While the embedding layer can process whitespace-only inputs,
/// the ingest API (MemoryStore level) explicitly rejects them as EmptyInput
/// to maintain data quality and avoid storing meaningless memories.
#[test]
fn test_ingest_whitespace_only_rejected_with_explicit_error() {
    // Validation happens before embedding, so no model needed
    let mut store = MemoryStore::test_store();

    let whitespace_inputs = vec!["   ", "\t\n", " \t \n "];

    for input in whitespace_inputs {
        let result = store.ingest("test-project", input, None, IngestPolicy::Force);
        assert!(
            result.is_err(),
            "Should reject whitespace-only input: {:?}",
            input
        );

        if let Err(e) = result {
            assert!(
                matches!(e, crate::errors::Error::EmptyInput),
                "Should return EmptyInput error, got: {:?}",
                e
            );
        }
    }
}

/// Test that ingest handles metadata with quoted and special characters correctly.
#[ignore]
#[test]
fn test_ingest_metadata_with_special_json_characters_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    // Test metadata with various JSON special characters
    let metadata_cases = vec![
        r#"{"key": "value with \"quotes\""}"#,
        r#"{"key": "value with 'apostrophes'"}"#,
        r#"{"key": "value with \t tabs \n and \r\n newlines"}"#,
        r#"{"key": "value with \\ backslashes"}"#,
        r#"{"nested": {"deeply": {"key": "value"}}}"#,
        r#"{"array": ["item1", "item2"]}"#,
    ];

    for metadata in metadata_cases {
        let result = store.ingest(
            "test-project",
            "test content",
            Some(metadata),
            IngestPolicy::Force,
        );

        assert!(
            result.is_ok(),
            "Should accept valid JSON metadata with special characters: {}",
            metadata
        );

        // Verify metadata was stored correctly
        if let Ok(crate::memory_types::AddResult::Added { id }) = result {
            let memory = store.get(&id, "test-project").unwrap().unwrap();
            assert_eq!(memory.metadata, Some(metadata.to_string()));
        }
    }
}

/// Test that ingest accepts metadata with Unicode characters.
#[ignore]
#[test]
fn test_ingest_metadata_with_unicode_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    let metadata_cases = vec![
        r#"{"emoji": "🚀✨🎯"}"#,
        r#"{"chinese": "你好世界"}"#,
        r#"{"japanese": "こんにちは"}"#,
        r#"{"arabic": "مرحبا"}"#,
        r#"{"emoji_text": "Test with 🌟 emojis and 中文 characters"}"#,
    ];

    for metadata in metadata_cases {
        let result = store.ingest(
            "test-project",
            "test content",
            Some(metadata),
            IngestPolicy::Force,
        );

        assert!(
            result.is_ok(),
            "Should accept valid JSON metadata with Unicode: {}",
            metadata
        );

        // Verify metadata was stored correctly
        if let Ok(crate::memory_types::AddResult::Added { id }) = result {
            let memory = store.get(&id, "test-project").unwrap().unwrap();
            assert_eq!(memory.metadata, Some(metadata.to_string()));
        }
    }
}

/// Test that ingest accepts content with Unicode combining characters.
///
/// MemoryStore stores content as-is (UTF-8 string), so it handles all valid Unicode.
#[ignore]
#[test]
fn test_ingest_content_with_combining_characters_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    // Test content with combining characters
    let content_cases = vec![
        "café",                // Normalized form
        "c\u{0065}\u{0301}fe", // Decomposed form with combining mark
        "e\u{0301}",           // Just an e with combining acute accent
        "日本\u{0301}語",      // Japanese with combining accent
    ];

    for content in content_cases {
        let result = store.ingest("test-project", content, None, IngestPolicy::Force);

        assert!(
            result.is_ok(),
            "Should accept content with combining characters: {}",
            content
        );

        // Verify content was stored as-input (deterministic behavior)
        if let Ok(crate::memory_types::AddResult::Added { id }) = result {
            let memory = store.get(&id, "test-project").unwrap().unwrap();
            assert_eq!(
                memory.content, content,
                "Content should be stored exactly as provided"
            );
        }
    }
}

/// Test that ingest accepts content with emoji.
#[ignore]
#[test]
fn test_ingest_content_with_emoji_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    let content_cases = vec![
        "Test with 🚀 emoji",
        "Emojis: 🎉🎊👏",
        "Unicode: ✨🌟💫",
        "Face emojis: 😀😂🥳",
    ];

    for content in content_cases {
        let result = store.ingest("test-project", content, None, IngestPolicy::Force);

        assert!(
            result.is_ok(),
            "Should accept content with emoji: {}",
            content
        );

        // Verify content was stored correctly
        if let Ok(crate::memory_types::AddResult::Added { id }) = result {
            let memory = store.get(&id, "test-project").unwrap().unwrap();
            assert_eq!(memory.content, content);
        }
    }
}

/// Test that ingest accepts content with BOM (Byte Order Mark).
///
/// MemoryStore stores content as-is, accepting BOM which is a valid UTF-8 sequence.
/// Embedding layer may produce unexpected embeddings for BOM-prefixed content,
/// but this is acceptable behavior for the ingest API layer.
#[ignore]
#[test]
fn test_ingest_content_with_bom_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    // UTF-8 BOM (zero-width no-break space)
    let bom = "\u{feff}";
    let content_with_bom = format!("{}Normal content", bom);

    let result = store.ingest("test-project", &content_with_bom, None, IngestPolicy::Force);

    assert!(result.is_ok(), "Should accept content with BOM");

    // Verify content was stored with BOM intact (deterministic behavior)
    if let Ok(crate::memory_types::AddResult::Added { id }) = result {
        let memory = store.get(&id, "test-project").unwrap().unwrap();
        assert_eq!(
            memory.content, content_with_bom,
            "BOM should be preserved in storage"
        );
    }
}

/// Test that ingest accepts any project_id as a namespace identifier.
///
/// Note: project_id is NOT validated by the ingest API. It serves as a simple
/// namespace string to segregate memories. Any string (including those that
/// would be trimmed to empty) is accepted as a valid project identifier.
/// Validation is the caller's responsibility for higher-level constraints if needed.
#[ignore]
#[test]
fn test_ingest_accepts_any_project_id_string() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    std::mem::forget(dir);

    let mut store = MemoryStore::new(&path, "BAAI/bge-small-en-v1.5", Config::default()).unwrap();

    let project_id_cases = vec![
        "simple-project",
        "https://github.com/user/repo.git",
        "project_with_123",
        "Project-With.dots",
        "项目-ID", // Unicode project ID
        "user/repo/branch",
    ];

    for project_id in project_id_cases {
        let result = store.ingest(project_id, "test content", None, IngestPolicy::Force);

        assert!(
            result.is_ok(),
            "Should accept any string as project_id: {}",
            project_id
        );

        // Verify memory is associated with the project_id
        if let Ok(crate::memory_types::AddResult::Added { id }) = result {
            let memory = store.get(&id, project_id).unwrap().unwrap();
            assert_eq!(memory.project_id, project_id);
        }
    }
}
