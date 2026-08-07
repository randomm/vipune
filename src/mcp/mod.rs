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
pub mod store_wrapper;
pub mod tools;

#[cfg(test)]
mod tests;
