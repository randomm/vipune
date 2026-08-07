//! MCP server entry point.

use crate::Config;
use crate::embedding::EmbeddingEngine;
use crate::errors::Error;
use crate::mcp::tools::ToolHandler;
use crate::memory::MemoryStore;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Run the MCP server over stdio.
///
/// This function blocks until the client disconnects. It:
/// 1. Creates a MemoryStore instance
/// 2. Creates a tokio runtime
/// 3. Serves the MCP protocol over stdio
///
/// The caller is responsible for producing a fully-loaded `Config` (defaults
/// merged with config-file values and `VIPUNE_*` environment overrides, e.g.
/// via `Config::load()`) so that MCP sessions honour the same configuration
/// a CLI invocation would.
///
/// # Errors
///
/// Returns error if:
/// - MemoryStore initialization fails
/// - Project detection fails
pub fn run_mcp(config: Config, project_id: &str) -> Result<(), Error> {
    let mut store = MemoryStore::new(
        &config.database_path,
        &config.embedding_model,
        config.clone(),
    )?;

    // Pre-initialise the embedder before accepting connections.
    // If this fails, the server exits before accepting any MCP client,
    // avoiding a first `store_memory` that holds the mutex through a
    // 66MB download and trips the client protocol timeout.
    //
    // Wrapped in a timeout: the model download (~66MB) runs in a spawned
    // thread so we can detect slow networks or unresponsive HuggingFace Hub
    // rather than hanging indefinitely.
    let model_id = config.embedding_model.clone();
    let model_id_for_thread = model_id.clone();
    let (tx, rx) = mpsc::channel();
    let init_thread = std::thread::spawn(move || {
        // Discarding send error is correct: a failed send means the receiver
        // already hit the 120s timeout and went away, so there is no one left
        // to report to and nothing actionable to do.
        let _ = tx.send(EmbeddingEngine::new(&model_id_for_thread));
    });

    let engine = match rx.recv_timeout(Duration::from_secs(120)) {
        Ok(Ok(engine)) => engine,
        Ok(Err(e)) => {
            let _ = init_thread.join();
            return Err(Error::EmbedderUnavailable {
                reason: format!("Failed to load embedding model '{}': {}", model_id, e),
            });
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // On timeout we deliberately orphan the download thread rather than block
            // on it. There is no cancellation point in the hf-hub sync API, and
            // `join()` would wait for the full remaining download (~66MB) after we've
            // already waited 120 s — re-introducing the unbounded hang the timeout
            // exists to prevent. A bounded startup failure is worth more than a
            // reclaimed thread at process exit.
            drop(init_thread); // drop the JoinHandle so the thread runs unobserved
            return Err(Error::EmbedderUnavailable {
                reason: format!(
                    "Embedding model '{}' download timed out after 120s. Check network connectivity to HuggingFace Hub.",
                    model_id
                ),
            });
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // Thread panicked — join to clean up before returning.
            let _ = init_thread.join();
            return Err(Error::EmbedderUnavailable {
                reason: format!(
                    "Embedding model '{}' initialization thread panicked. Check disk space and permissions for model cache.",
                    model_id
                ),
            });
        }
    };

    // NOTE: `engine` is created from `config.embedding_model` which is validated
    // during Config construction. The thread that created it has been joined (Ok
    // and Disconnected branches) or is orphaned and running unobserved (Timeout
    // branch returns early above).
    store.set_preinitialized_embedder(engine);

    let store = Arc::new(Mutex::new(store));

    // Create tool handler
    let handler = ToolHandler::new(store, project_id.to_string(), config);

    // Create tokio runtime (single-threaded since MCP stdio handles requests sequentially)
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Config(format!("Failed to create tokio runtime: {}", e)))?;

    // Run MCP server over stdio
    runtime.block_on(async {
        let (stdin, stdout) = rmcp::transport::stdio();
        let service = rmcp::serve_server(handler, (stdin, stdout))
            .await
            .map_err(|e| Error::Config(format!("MCP server error: {}", e)))?;
        service
            .waiting()
            .await
            .map_err(|e| Error::Config(format!("MCP server task error: {}", e)))?;
        Ok(())
    })
}
