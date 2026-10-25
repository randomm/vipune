//! Cross-surface parity tests for the error → wire mapping (issue #255 review).
//!
//! Wired from `mcp/mod.rs` rather than `tests/` or `errors.rs`: the claim being
//! checked spans both crates' halves of one type, and only the library's `mcp`
//! module sees them together. `main.rs` compiles its *own* copy of `errors.rs`,
//! so in the binary target `crate::errors::Error` and the `Error` that
//! `store_wrapper`'s `From<Error> for rmcp::ErrorData` is implemented for are
//! distinct nominal types — a parity test wired from `errors.rs` would compile
//! against the wrong pair there and pass without describing either surface.
//!
//! Model-free by construction: every fixture is built from a `rusqlite` result
//! code or a message string, so nothing here downloads or loads the ONNX model.

use crate::errors::Error;

/// Builds one fixture error. Each case is a producer rather than a value: the
/// public enum is not `Clone`, and deriving it to satisfy a test would widen the
/// public surface for a test-only reason.
type Fixture = Box<dyn Fn() -> Error>;

/// `docs/cli-reference.md` claims the rendered database-error message "is the
/// same string on every surface": the human path writes `Error: {}` to stderr,
/// `--json` puts `error: error.to_string()` in the `error` key on stdout, and
/// the MCP server returns it verbatim in its `ErrorData` message.
///
/// Two of those three are literally `Display` and the MCP database arm is
/// `McpError::internal_error(&e.to_string())`, so the claim is a claim about
/// `Display` — and this makes it a checked one instead of prose: one underlying
/// storage error, rendered on both surfaces, byte for byte. If either side ever
/// stops calling `to_string()` (a hand-written message on the MCP side, or a
/// variant that starts rendering differently), the two diverge and this fails.
#[test]
fn a_database_error_renders_the_same_string_on_the_cli_and_the_mcp_surface() {
    let busy_code = rusqlite::ffi::SQLITE_BUSY;
    let busy_text = "database is locked".to_string();
    let cases: Vec<(Fixture, &str)> = vec![
        (
            Box::new(|| {
                Error::from(crate::sqlite::Error::Sqlite(
                    "no such table: memories".to_string(),
                ))
            }),
            "a plain database failure",
        ),
        (
            Box::new(|| {
                Error::from(crate::sqlite::Error::Sqlite(
                    "attempt to write a readonly database".to_string(),
                ))
            }),
            "a permissions-shaped failure",
        ),
        (
            Box::new(move || {
                Error::from(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(busy_code),
                    Some(busy_text.clone()),
                ))
            }),
            "a lock classified from the raw result code",
        ),
        (
            Box::new(|| Error::from(crate::sqlite::Error::Locked)),
            "the storage layer's own lock verdict",
        ),
    ];

    for (produce, what) in cases {
        // The MCP server: `store_wrapper`'s `From<Error> for rmcp::ErrorData`,
        // which consumes the error.
        let mcp_text = rmcp::ErrorData::from(produce()).message;
        // The CLI: `main.rs` renders `Error: {}` for humans and
        // `ErrorResponse { error: error.to_string() }` for `--json`, so this is
        // the text both of those surfaces emit — from the *same* underlying
        // error, produced again rather than cloned.
        let cli_text = produce().to_string();

        assert_eq!(
            cli_text, mcp_text,
            "{what} must render identically on the CLI and MCP surfaces, \
             got CLI {cli_text:?} vs MCP {mcp_text:?}"
        );
        // Guard against the vacuous pass: two empty strings would compare
        // equal, so a refactor that dropped the message on both sides would
        // satisfy the assertion above while breaking the documented contract.
        assert!(
            !cli_text.is_empty(),
            "{what} must render a message, got an empty string"
        );
    }
}
