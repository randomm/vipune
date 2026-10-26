//! The only acquisitions of the MCP server's shared store mutex (issue #275).
//!
//! The MCP server is a long-lived stdio process (`server.rs`), so a panic in
//! one request must not decide the fate of every later one. `unwrap()` on the
//! shared `Mutex<MemoryStore>` does: the first panic poisons the mutex and
//! every acquisition afterwards panics again inside the tool handler — a
//! permanent outage caused by one bad request. Both helpers here *answer* a
//! poisoned mutex with an internal error, so the request fails and the server
//! keeps serving.
//!
//! Neither helper recovers the guard with `into_inner()`/`clear_poison`: the
//! alternative to answering is serving on with state a panic may have left
//! half-written, and the poison is permanent for the process lifetime. The
//! opposite policy is right for a *test* mutex — `config::tests_utils`'s
//! `lock_env_mutex` recovers precisely so one panicking test does not cascade
//! `PoisonError` to its siblings. That reasoning does not transfer to a server
//! whose data a panic may have corrupted mid-transaction.
//!
//! Two helpers because the wrapper's own signatures differ: the tool-facing
//! methods return `rmcp::ErrorData` and the search/list methods return the
//! library `Error`, which `tools.rs` converts. Choose by the caller's return
//! type; they are not interchangeable — routing an `rmcp::ErrorData` method
//! through `lock_store` would answer with `Error::Database`'s `Database error:`
//! frame instead of the plain poison message.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::errors::Error;
use crate::memory::MemoryStore;

use super::store_wrapper::McpError;

/// Acquire the shared store for the methods that answer in `rmcp::ErrorData`.
///
/// `op` names the acquiring wrapper method: std's `PoisonError` says nothing
/// about which acquisition failed, and in a stdio server this message is the
/// only channel an operator reads.
pub(super) fn lock_mcp<'a>(
    store: &'a Arc<Mutex<MemoryStore>>,
    op: &str,
) -> Result<MutexGuard<'a, MemoryStore>, rmcp::ErrorData> {
    store
        .lock()
        .map_err(|e| McpError::internal_error(&poison_message(e, op)))
}

/// Acquire the shared store for the methods that answer in the library `Error`.
///
/// `Error::Database` is the internal-style variant of the #255 surface:
/// `From<Error> for rmcp::ErrorData` renders it as `McpError::internal_error`,
/// i.e. the same `data.type == "internal_error"` a `lock_mcp` failure produces.
/// It must not become `Error::DatabaseLocked` — that variant tells an MCP
/// client to retry a lock that will never free.
pub(super) fn lock_store<'a>(
    store: &'a Arc<Mutex<MemoryStore>>,
    op: &str,
) -> Result<MutexGuard<'a, MemoryStore>, Error> {
    store.lock().map_err(|e| Error::Database {
        message: poison_message(e, op),
    })
}

/// The message behind both helpers' poisoned answers, kept in one place so the
/// two return types cannot drift in what they tell the client. It always
/// *starts* with this crate's wording: the `Error::Database` frame that
/// `From<Error> for rmcp::ErrorData` prepends to the `lock_store` paths comes
/// before it, but nothing after that point is ours — std's `PoisonError`
/// Display closes the message, so a test pins the prefix and no more.
fn poison_message(e: std::sync::PoisonError<MutexGuard<'_, MemoryStore>>, op: &str) -> String {
    format!("store lock poisoned ({op}): {e}")
}
