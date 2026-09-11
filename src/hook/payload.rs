//! Claude Code hook payload parsing (issue #213).
//!
//! The hook path receives JSON on stdin shaped per-event:
//!
//! - `SessionStart`:   `{ "session_id": ..., "cwd": ... }` (no extraction)
//! - `UserPromptSubmit`: `{ "prompt": ..., "cwd": ... }`
//! - `PreToolUse`:     `{ "tool_name": ..., "tool_input": ..., "cwd": ... }`
//! - `PostToolUse`:    `{ "tool_response": ..., "cwd": ... }`
//! - `PreCompact`:     `{ "conversation": ..., "cwd": ... }`
//!
//! Field shapes are heterogeneous: `tool_input` and `tool_response` may be
//! nested JSON objects, `conversation` may be a list of message objects, or
//! the field may be missing entirely. The parser extracts the raw JSON value
//! and leaves serialisation to the extractor (`src/hook/extract.rs`).

use serde_json::Value;
use std::path::PathBuf;

/// Discriminates which event a hook invocation is for.
///
/// Carried by the caller (the command surface) rather than inferred from the
/// payload, because the Claude Code hook contract does not include an
/// `event_type` field in the payload itself — the event is identified by
/// *which* subcommand Claude Code invokes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PreCompact,
}

/// Parsed hook payload.
///
/// Only the fields the hook pipeline needs are extracted; the rest of the
/// payload is ignored.
#[derive(Debug, Clone)]
pub struct HookPayload {
    /// Working directory Claude Code reports for the session. `None` if the
    /// payload did not include `cwd` (the hook then has no project scope to
    /// insert under, so the candidate is silently dropped).
    pub cwd: Option<PathBuf>,
    /// Raw JSON value of the event-specific payload field (or `Null` if the
    /// field was absent). The extractor (`src/hook/extract.rs`) turns this
    /// into a candidate string.
    pub candidate_source: Value,
}

/// Parse a Claude Code hook JSON payload for the given event.
///
/// Returns `None` when:
/// - the payload is not a JSON object (e.g. the caller passed an array or a
///   bare string), or
/// - the event is `SessionStart` (no candidate extraction; the caller
///   should simply exit 0).
///
/// Returns `Some(HookPayload)` on success. The `candidate_source` field is
/// `Value::Null` when the event-specific field is absent from the payload.
pub fn parse_hook_payload(event: HookEvent, input: &str) -> Option<HookPayload> {
    let value: Value = serde_json::from_str(input).ok()?;
    let obj = value.as_object()?;

    // SessionStart is a no-op: no candidate extraction.
    if event == HookEvent::SessionStart {
        return None;
    }

    // `cwd` is present in every Claude Code hook payload per the docs, but if
    // it's missing we treat it as "no project scope" and the caller drops the
    // candidate silently rather than falling back to the vipune process's cwd
    // (which would leak the wrong project's DB into the agent's session).
    let cwd = match obj.get("cwd") {
        Some(Value::String(s)) => Some(PathBuf::from(s)),
        _ => None,
    };

    let candidate_source = match event {
        HookEvent::UserPromptSubmit => obj.get("prompt").cloned().unwrap_or(Value::Null),
        HookEvent::PreToolUse => obj.get("tool_input").cloned().unwrap_or(Value::Null),
        HookEvent::PostToolUse => obj.get("tool_response").cloned().unwrap_or(Value::Null),
        HookEvent::PreCompact => obj.get("conversation").cloned().unwrap_or(Value::Null),
        // SessionStart handled above; unreachable here.
        HookEvent::SessionStart => Value::Null,
    };

    Some(HookPayload {
        cwd,
        candidate_source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event_payload(event: HookEvent, json: &str) -> Option<HookPayload> {
        parse_hook_payload(event, json)
    }

    #[test]
    fn parse_user_prompt_submit_extracts_prompt_and_cwd() {
        let p = make_event_payload(
            HookEvent::UserPromptSubmit,
            r#"{"prompt": "hello world", "cwd": "/tmp/proj", "session_id": "x"}"#,
        )
        .expect("should parse");
        assert_eq!(
            p.cwd,
            Some(std::path::PathBuf::from("/tmp/proj")),
            "cwd must be extracted"
        );
        assert_eq!(
            p.candidate_source,
            Value::String("hello world".into()),
            "prompt field must be the candidate source"
        );
    }

    #[test]
    fn parse_pre_tool_use_extracts_tool_input_object() {
        let p = make_event_payload(
            HookEvent::PreToolUse,
            r#"{"tool_name": "Bash", "tool_input": {"command": "ls -la", "timeout": 30}, "cwd": "/tmp/proj"}"#,
        )
        .expect("should parse");
        assert!(
            p.candidate_source.is_object(),
            "tool_input may be an object; the extractor must handle it"
        );
        assert_eq!(
            p.candidate_source["command"],
            Value::String("ls -la".into())
        );
    }

    #[test]
    fn parse_post_tool_use_extracts_tool_response() {
        let p = make_event_payload(
            HookEvent::PostToolUse,
            r#"{"tool_response": {"stdout": "ok\n"}, "cwd": "/tmp/proj"}"#,
        )
        .expect("should parse");
        assert!(p.candidate_source.is_object());
    }

    #[test]
    fn parse_pre_compact_extracts_conversation() {
        let p = make_event_payload(
            HookEvent::PreCompact,
            r#"{"conversation": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "cwd": "/tmp/proj"}"#,
        )
        .expect("should parse");
        assert!(p.candidate_source.is_array());
    }

    #[test]
    fn parse_session_start_returns_none() {
        let p = make_event_payload(
            HookEvent::SessionStart,
            r#"{"session_id": "x", "cwd": "/tmp/proj"}"#,
        );
        assert!(
            p.is_none(),
            "SessionStart is a no-op — no payload extraction"
        );
    }

    #[test]
    fn parse_missing_cwd_yields_none_cwd() {
        let p = make_event_payload(HookEvent::UserPromptSubmit, r#"{"prompt": "hi"}"#)
            .expect("should parse");
        assert!(
            p.cwd.is_none(),
            "missing cwd → None (caller silently drops the candidate)"
        );
        assert_eq!(p.candidate_source, Value::String("hi".into()));
    }

    #[test]
    fn parse_missing_candidate_field_yields_null() {
        let p = make_event_payload(HookEvent::UserPromptSubmit, r#"{"cwd": "/tmp/proj"}"#)
            .expect("should parse");
        assert_eq!(
            p.candidate_source,
            Value::Null,
            "missing prompt field → Value::Null"
        );
    }

    #[test]
    fn parse_non_object_json_returns_none() {
        assert!(parse_hook_payload(HookEvent::UserPromptSubmit, "[1,2,3]").is_none());
        assert!(parse_hook_payload(HookEvent::UserPromptSubmit, "not json").is_none());
    }
}
