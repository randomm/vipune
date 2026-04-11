//! MCP server entry point.

use crate::Config;
use crate::errors::Error;
use crate::mcp::tools::ToolHandler;
use crate::memory::MemoryStore;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Run the MCP server over stdio.
///
/// This function:
/// 1. Creates a MemoryStore instance
/// 2. Creates a tokio runtime
/// 3. Serves the MCP protocol over stdio
///
/// # Errors
///
/// Returns error if:
/// - MemoryStore initialization fails
/// - Project detection fails
pub fn run_mcp(embedding_model: String, project_id: &str, db_path: PathBuf) -> Result<(), Error> {
    // Use provided embedding_model and db_path directly
    let config = Config {
        embedding_model,
        database_path: db_path.clone(),
        ..Config::default()
    };
    let store = MemoryStore::new(
        &config.database_path,
        &config.embedding_model,
        config.clone(),
    )?;
    let store = Arc::new(Mutex::new(store));

    // Create tool handler
    let handler = ToolHandler::new(store, project_id.to_string());

    // Create tokio runtime (single-threaded since MCP stdio handles requests sequentially)
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Config(format!("Failed to create tokio runtime: {}", e)))?;

    // Run MCP server over stdio
    runtime.block_on(async {
        let (stdin, stdout) = rmcp::transport::stdio();
        let _service = rmcp::serve_server(handler, (stdin, stdout))
            .await
            .map_err(|e| Error::Config(format!("MCP server error: {}", e)))?;
        Ok(())
    })
}
