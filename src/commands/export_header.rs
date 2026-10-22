//! The JSONL export header — the wire-format contract between
//! `vipune export` and `vipune import` (issue #229).
//!
//! Neither command owns this schema: the writer (`commands::export`)
//! serialises it and the reader (`commands::import::validate_header`)
//! deserialises it, and any future consumer of the format (an `mcp/`
//! import path, a backup verifier) talks to the format through this
//! module alone. Key names exist here exactly once. Issue #229 was what
//! happens when they don't: the writer spelled the version key
//! `format_version` inside its own `json!` literal while the reader
//! hand-looked-up `"version"`, so every release from v0.10.0 to v0.15.0
//! aborted on its own exports.

use crate::embedding::EMBEDDING_DIMS;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Format version of the JSONL export, written in the header line.
pub const EXPORT_FORMAT_VERSION: u32 = 1;

/// Deserialize a field that merely has to be PRESENT, explicit JSON
/// `null` included. Why presence is the contract: [`DeclaredVersion`].
fn any_present<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Value::deserialize(deserializer).map(Some)
}

/// The JSONL header line — the ONE schema both the writer (`write_jsonl`)
/// and the reader (`commands::import::validate_header`) go through.
///
/// `kind` and `rows` are required fields: a header missing either fails
/// deserialisation, so "a parsed `ExportHeader` is a valid export header"
/// is a type-level invariant, not a promise held only by the reader's
/// checks. `model_id` / `model_revision` stay optional (pre-#217 backups
/// carry none), and the version key is presence-typed — the contract is
/// stated once at [`DeclaredVersion`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportHeader {
    /// Discriminator, always `"export"`. JSON key `type`; the Rust field
    /// is `kind` because `type` is a keyword. The row-level `type` column
    /// is renamed `memory_type` so the two never collide.
    #[serde(rename = "type")]
    pub kind: String,
    /// JSONL format version. Serialises as `format_version` (what
    /// `write_jsonl` emits); the legacy spelling `version` deserialises
    /// into this same field. A header carrying BOTH spellings is refused
    /// explicitly by the reader (`validate_header`) — see
    /// [`DeclaredVersion::NonNumeric`]'s sibling rule for the state split.
    #[serde(alias = "version", default, deserialize_with = "any_present")]
    pub format_version: Option<Value>,
    /// Embedding dimensions the exported BLOBs were written for.
    #[serde(default)]
    pub embedding_dims: Option<u32>,
    /// Model id the source store recorded, or `None` on a pre-#217 export.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Model revision the source store recorded, or `None` on a pre-#217 export.
    #[serde(default)]
    pub model_revision: Option<String>,
    /// RFC3339 timestamp of the export run.
    #[serde(default)]
    pub exported_at: Option<String>,
    /// Data-row count declared by the writer (the header line is not counted).
    pub rows: u64,
}

/// The state of the header's version key after deserialisation.
///
/// This enum is the single statement of the version key's contract.
/// PRESENCE, not shape, is the compatibility rule from issue #229: a
/// backup whose header this build did not author must keep importing as
/// long as the key is there, and an explicit JSON `null` counts as
/// present (a plain `Option<Value>`/`Option<u32>` field folds `null` into
/// `None` and would reject the backups that carry it — recreating the
/// very bug #229 fixed). The non-`Absent` states are distinct so the
/// reader's forward-compat gate cannot be bypassed by spelling the
/// version any other way: every state must be named at the call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredVersion {
    /// Key absent under both spellings — the reader refuses.
    Absent,
    /// Key present with an explicit JSON `null` — the tolerated legacy
    /// backup shape; satisfies presence, declares no number.
    AcceptedNull,
    /// Key present as a non-negative integer.
    Numeric(u64),
    /// Key present as anything else (`"2"`, `-1`, `1.5`, `true`, `{}`) —
    /// refused; no released writer ever emitted one, so such a file is
    /// hand-edited and ambiguous about the schema it declares.
    NonNumeric(Value),
}

impl ExportHeader {
    /// The header `write_jsonl` emits. The invariant fields come from the
    /// format constants (not a struct literal at the call site), so the
    /// writer cannot build a header its own importer would refuse.
    pub fn new(rows: usize, exported_at: &str, model_id: &str, model_revision: &str) -> Self {
        Self {
            kind: "export".to_string(),
            format_version: Some(Value::from(EXPORT_FORMAT_VERSION)),
            embedding_dims: Some(EMBEDDING_DIMS as u32),
            model_id: Some(model_id.to_string()),
            model_revision: Some(model_revision.to_string()),
            exported_at: Some(exported_at.to_string()),
            rows: rows as u64,
        }
    }

    /// The version key's state — the reader's single entry point for both
    /// #229 rules (see [`DeclaredVersion`] for what each state means).
    pub fn declared_version(&self) -> DeclaredVersion {
        match self.format_version.as_ref() {
            None => DeclaredVersion::Absent,
            Some(Value::Null) => DeclaredVersion::AcceptedNull,
            Some(value) => match value.as_u64() {
                Some(n) => DeclaredVersion::Numeric(n),
                None => DeclaredVersion::NonNumeric(value.clone()),
            },
        }
    }
}
