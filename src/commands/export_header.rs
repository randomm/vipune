//! The JSONL export header — the wire-format contract between
//! `vipune export` and `vipune import` (issue #229).
//!
//! Neither command owns this schema: the writer (`commands::export`)
//! serialises it and the reader (`commands::import::validate_header`)
//! deserialises it, and any future consumer of the format (an `mcp/`
//! import path, a backup verifier) talks to the format through this
//! module alone. Production code spells the header's key names nowhere
//! else — every reader check is a field access on this type, so renaming a
//! key reaches both sides at once. (Test fixtures deliberately spell the
//! literals: their job is to pin what the writer emits, not to stay in
//! step with it.) Issue #229 was what happens when they don't: the writer
//! spelled the version key `format_version` inside its own `json!` literal
//! while the reader hand-looked-up `"version"`, so every release from
//! v0.10.0 to v0.15.0 aborted on its own exports.
//!
//! # This type is a wire type, not a validated one
//!
//! Every field deserialises optionally. That is deliberate: serde
//! requiredness would decide *which* malformed header produces *which*
//! error, and the reader — not the derive — owns those refusals and their
//! messages (see [`ExportHeader::is_export`], [`ExportHeader::declared_rows`],
//! [`ExportHeader::has_version_key`]). A "required" field here would be a
//! requiredness claim the reader then has to be careful not to contradict.

use crate::embedding::EMBEDDING_DIMS;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Format version of the JSONL export, written in the header line.
///
/// WRITTEN, NEVER COMPARED. The reader's only version rule is presence
/// ([`ExportHeader::has_version_key`]), so a file written by a future build
/// still imports into this one — which is what #229's acceptance criteria
/// require. Comparing it would refuse the very restores this ticket exists
/// to unblock, and could not be enforced anyway (a quoted `"2"` skips any
/// numeric coercion).
pub const EXPORT_FORMAT_VERSION: u32 = 1;

/// JSON key of the canonical version field, as emitted by `write_jsonl`.
pub const FORMAT_VERSION_KEY: &str = "format_version";
/// JSON key of the LEGACY version field, accepted on read only.
pub const LEGACY_VERSION_KEY: &str = "version";

/// Deserialize a field that merely has to be PRESENT, explicit JSON
/// `null` included. Why presence is the version-key contract:
/// [`ExportHeader::has_version_key`].
fn any_present<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Value::deserialize(deserializer).map(Some)
}

/// The JSONL header line — the ONE schema both the writer (`write_jsonl`)
/// and the reader (`commands::import::validate_header`) go through.
///
/// Fields are PRIVATE. Every read of a header goes through an accessor
/// below, so a consumer cannot reach the wire fields and skip the rules the
/// accessors encode — that is what keeps "the contract module is the whole
/// contract" true by construction rather than by convention. The only ways
/// in are [`ExportHeader::new`] (writer) and serde (reader).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportHeader {
    /// Discriminator, always `"export"` on a file this writer produced.
    /// JSON key `type`; the Rust field is `kind` because `type` is a
    /// keyword. The row-level `type` column is renamed `memory_type` so the
    /// two never collide. Optional here so a header missing it reaches the
    /// reader's own "missing type=export" message; validated by
    /// [`ExportHeader::is_export`].
    #[serde(rename = "type", default)]
    kind: Option<String>,
    /// JSONL format version — the canonical spelling `write_jsonl` emits.
    /// Presence-typed: see [`ExportHeader::has_version_key`].
    #[serde(default, deserialize_with = "any_present")]
    format_version: Option<Value>,
    /// The LEGACY spelling of the version key, JSON `version`, accepted on
    /// read only.
    ///
    /// A separate field rather than a `#[serde(alias)]` on `format_version`
    /// precisely because an `alias` makes serde's duplicate-field check
    /// refuse a header carrying BOTH spellings — and such a header exists
    /// in the wild: #229's own workaround was hand-adding `"version": 1` to
    /// a real export, so those files sit on disk with both keys side by
    /// side and must keep importing.
    ///
    /// `skip_serializing` keeps it out of the writer's output, so
    /// `write_jsonl` emits only the canonical `format_version`.
    #[serde(
        rename = "version",
        default,
        deserialize_with = "any_present",
        skip_serializing
    )]
    legacy_version: Option<Value>,
    /// Embedding dimensions the exported BLOBs were written for.
    #[serde(default)]
    embedding_dims: Option<u32>,
    /// Model id the source store recorded, or `None` on a pre-#217 export.
    #[serde(default)]
    model_id: Option<String>,
    /// Model revision the source store recorded, or `None` on a pre-#217 export.
    #[serde(default)]
    model_revision: Option<String>,
    /// RFC3339 timestamp of the export run.
    #[serde(default)]
    exported_at: Option<String>,
    /// Data-row count declared by the writer (the header line is not
    /// counted). Present on every released export, and required by the
    /// reader ([`ExportHeader::declared_rows`]) because that is what the
    /// shipped reader refused without — but its VALUE is advisory:
    /// `run_import` counts the rows it actually parses and never trusts this
    /// number, so a stale count is not a refusal. Cross-checking it against
    /// the parsed rows would reject files that import today, which is
    /// outside #229's scope.
    ///
    /// Typed loosely (`Value`, coerced by `as_u64`) to stay exactly as
    /// permissive as the shipped reader, which read it as
    /// `obj.get("rows").and_then(as_u64)` — a `1.0` counted, a string did
    /// not. A `u64` field would have rejected shapes the released importer
    /// accepted.
    #[serde(default)]
    rows: Option<Value>,
}

impl ExportHeader {
    /// The header `write_jsonl` emits. The invariant fields come from the
    /// format constants (not a struct literal at the call site), so the
    /// writer cannot build a header its own importer would refuse.
    pub fn new(rows: usize, exported_at: &str, model_id: &str, model_revision: &str) -> Self {
        Self {
            kind: Some("export".to_string()),
            format_version: Some(Value::from(EXPORT_FORMAT_VERSION)),
            legacy_version: None,
            embedding_dims: Some(EMBEDDING_DIMS as u32),
            model_id: Some(model_id.to_string()),
            model_revision: Some(model_revision.to_string()),
            exported_at: Some(exported_at.to_string()),
            rows: Some(Value::from(rows as u64)),
        }
    }

    /// The `type` discriminator. `None` = the key is absent, which the
    /// reader refuses with the shipped "missing type=export" message rather
    /// than serde's `missing field` error.
    pub fn kind(&self) -> Option<&str> {
        self.kind.as_deref()
    }

    /// The declared data-row count, or `None` when the key is absent or not
    /// a non-negative integer — the reader's refusal condition, stated here
    /// so the contract module is the whole contract. Coerced with `as_u64`,
    /// matching the shipped reader exactly.
    pub fn declared_rows(&self) -> Option<u64> {
        self.rows.as_ref().and_then(Value::as_u64)
    }

    /// Whether this is an export file. The reader refuses anything else.
    pub fn is_export(&self) -> bool {
        self.kind() == Some("export")
    }

    /// The recorded `model_id`, or `None` on a pre-#217 export.
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// The recorded `model_revision`, or `None` on a pre-#217 export.
    pub fn model_revision(&self) -> Option<&str> {
        self.model_revision.as_deref()
    }

    /// Whether the header carries the version key under EITHER spelling,
    /// under any value. This is the whole version rule, and presence is
    /// deliberately the only part of it:
    ///
    /// - Backups from v0.10.0 through v0.15.0 carry `format_version`,
    ///   `version`, or (hand-patched per #229's workaround) both, and all of
    ///   them must keep importing.
    /// - An explicit JSON `null` counts as present. A plain `Option<Value>`
    ///   field without `any_present` folds `null` into `None` and would
    ///   reject exactly the backups that carry it — recreating the bug #229
    ///   fixed.
    /// - The value is never compared against [`EXPORT_FORMAT_VERSION`]; see
    ///   that constant for why.
    pub fn has_version_key(&self) -> bool {
        self.format_version.is_some() || self.legacy_version.is_some()
    }

    /// The two spellings disagreeing about the version — the one thing the
    /// reader refuses about the version key, and the only genuine ambiguity
    /// in it. `{"format_version":1,"version":1}` (a #229 hand-patch) is NOT
    /// a conflict and imports; `{"format_version":1,"version":2}` declares
    /// two different schemas at once, and picking one silently is worse than
    /// refusing and naming the remedy.
    ///
    /// `(canonical, legacy)` when they differ, `None` when at most one
    /// spelling is present or they agree.
    pub fn version_conflict(&self) -> Option<(&Value, &Value)> {
        match (self.format_version.as_ref(), self.legacy_version.as_ref()) {
            (Some(canonical), Some(legacy)) if canonical != legacy => Some((canonical, legacy)),
            _ => None,
        }
    }
}
