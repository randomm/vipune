# vipune

A minimal memory layer for AI agents.

**Status: Work in progress**

https://github.com/randomm/vipune

## Architecture

Single binary CLI built in Rust with:
- SQLite with FTS5 for text storage
- ONNX embeddings (bge-small-en-v1.5, 384 dimensions)
- Custom cosine similarity for vector search
- Fully synchronous - no tokio, no async runtime, no network dependencies at runtime

Apache-2.0 License by Janni Turunen.