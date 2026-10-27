//! MCP (Model Context Protocol) server for vipune.
//!
//! This module provides an MCP server over stdio that exposes vipune's memory
//! operations as tools. The module is a default feature — MCP is enabled by
//! default. Use `default-features = false` in your Cargo.toml for sync-only builds.
//!
//! # Architecture
//!
//! Two-layer pattern: async MCP wrappers → sync MemoryStore operations
//! - MCP layer (server.rs, tools.rs, params.rs): async #[tool] wrappers using rmcp
//! - Business layer: calls MemoryStore methods synchronously via Arc<Mutex<MemoryStore>>
//!
//! # Entry Point
//!
//! Use `run_mcp()` to start the server:
//! ```no_run
//! use vipune::config::Config;
//! use vipune::mcp::server::run_mcp;
//!
//! let config = Config::default();
//! run_mcp(config, "default").expect("Failed to start MCP server");
//! ```

mod helpers;
mod params;
pub mod server;
mod store_lock;
pub mod store_wrapper;
#[cfg(test)]
mod test_transport;
pub mod tools;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "store_memory_importance_tests.rs"]
mod store_memory_importance_tests;

/// Supersede-path tests for the MCP surface, split out of `tests.rs` for #271
/// so that file stays inside the per-file cap. The `feature = "mcp"` half of the
/// gate is load-bearing: the tool handler these tests drive does not exist in
/// the `--no-default-features` lane CI runs, and without it the file compiles to
/// zero tests while the orphan guard in `lib.rs` still counts it as wired.
#[cfg(all(test, feature = "mcp"))]
#[path = "supersede_tests.rs"]
mod supersede_tests;

/// `store_memory` / `update_memory` metadata-shape cases (#309) — a new test
/// module, not a split from `tests.rs`. Same `feature = "mcp"` reasoning as
/// `supersede_tests` above.
#[cfg(all(test, feature = "mcp"))]
#[path = "store_metadata_tests.rs"]
mod store_metadata_tests;

/// The get/update MCP surfaces, split out of `tests.rs` for #271 so that file
/// stays inside the per-file cap. Same `feature = "mcp"` reasoning as
/// `supersede_tests` above.
#[cfg(all(test, feature = "mcp"))]
#[path = "update_tests.rs"]
mod update_tests;

#[cfg(all(test, feature = "mcp"))]
#[path = "update_memory_importance_tests.rs"]
mod update_memory_importance_tests;

/// Cross-surface error-rendering parity (issue #255 review). Kept beside the
/// mapping it exercises rather than in `errors.rs`: `main.rs` compiles its own
/// copy of `errors.rs`, so only here are the CLI's `Error` and the one
/// `From<Error> for rmcp::ErrorData` is implemented for the same nominal type.
#[cfg(all(test, feature = "mcp"))]
#[path = "error_surface_parity_tests.rs"]
mod error_surface_parity_tests;
