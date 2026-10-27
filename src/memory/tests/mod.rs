//! Tests for the memory store.

#![cfg(test)]

pub(in crate::memory) mod crud;
mod embedding_integrity;
mod identity_refusal;
mod ingest;
mod ingest_boundary;
mod role_prefix;
mod search;
