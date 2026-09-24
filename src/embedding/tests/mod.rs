//! Tests for the embedding engine.
//!
//! Model-free tests (no model download) and the `#[ignore]`d real-model
//! tests (run with `cargo test -- --ignored`) live in separate files so each
//! stays under the project 500-line cap.

mod model_free;
mod real_model;

