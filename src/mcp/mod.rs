//! MCP (Model Context Protocol) server for vipune.
//!
//! This module provides an MCP server over stdio that exposes vipune's memory
//! operations as tools. The module is feature-gated behind the `mcp` feature to
//! keep the library fully synchronous (no tokio/async in lib.rs).
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
//! run_mcp(config.embedding_model, "default", config.database_path).expect("Failed to start MCP server");
//! ```

mod params;
pub mod server;
pub mod tools;

#[cfg(test)]
mod tests;
