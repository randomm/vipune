//! Agent lifecycle hooks (issue #191, task-b / issue #213).
//!
//! Zero-LLM hook pipeline for Claude Code lifecycle events.
//! The hook path never loads the ONNX model; it inserts placeholder embeddings
//! that classify as `Mock` (L2 norm strictly > 2.0) so a later `reindex` run
//! can backfill real vectors.

mod embedding;
mod extract;
pub mod payload;
mod run;

pub use payload::HookEvent;
pub use run::{read_stdin, run_hook_event};
