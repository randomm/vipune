//! Tests for the `From<Error> for rmcp::ErrorData` conversion in
//! `store_wrapper` — the machine-readable `data.type` discriminators an MCP
//! client branches on.
//!
//! Wired from `store_wrapper.rs` via `#[path = "store_wrapper_error_tests.rs"]`
//! to keep that file under the project 500-line cap.

use super::*;

/// The `data.type` field is the discriminator an MCP client must read: rmcp's
/// `ErrorCode` set has no "retry later", so a retryable lock and a server
/// defect share `INTERNAL_ERROR`. If the fields stop being emitted the client
/// can only string-match the message — which is what issue #255 set out to
/// remove.
#[test]
fn database_errors_carry_a_machine_readable_type_discriminator() {
    let cases: Vec<(Error, &str)> = vec![
        (Error::DatabaseLocked, "database_locked"),
        (
            Error::CorruptEmbedding {
                id: "mem-1".to_string(),
                reason: "not enough bytes".to_string(),
            },
            "corrupt_embedding",
        ),
        (
            Error::InvalidBlobSize {
                expected: crate::embedding::EMBEDDING_DIMS * 4,
                actual: 4,
            },
            "invalid_blob_size",
        ),
        (
            Error::MismatchedDimensions {
                expected: 384,
                actual: 768,
            },
            "mismatched_dimensions",
        ),
    ];
    for (err, expected_type) in cases {
        let data = rmcp::ErrorData::from(err)
            .data
            .unwrap_or_else(|| panic!("{expected_type} must carry a data payload"));
        assert_eq!(
            data["type"].as_str(),
            Some(expected_type),
            "expected the {expected_type} discriminator"
        );
    }
}

#[test]
fn the_corrupt_embedding_discriminator_carries_the_row_id() {
    // The #186 payload a client needs to locate the corrupt row must survive
    // the MCP conversion, not just the Display text.
    let err = Error::CorruptEmbedding {
        id: "mem-42".to_string(),
        reason: "invalid BLOB size".to_string(),
    };
    let data = rmcp::ErrorData::from(err).data.expect("data payload");
    assert_eq!(data["id"].as_str(), Some("mem-42"));
    assert_eq!(data["expected"], serde_json::Value::Null);
}
