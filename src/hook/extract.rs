//! Zero-LLM candidate extraction and credential blocking (issue #213).
//!
//! The extractor turns the raw JSON value from the payload parser into a
//! plain-string candidate suitable for `content_hash` and storage. It also
//! performs a hard credential block: if the candidate contains any known
//! credential pattern (JWT, AWS access key, bearer token, etc.), the entire
//! candidate is dropped — not redacted.

use serde_json::Value;

/// Maximum candidate length the hook will store. Longer extractions are
/// silently dropped — hook candidates are short conversational fragments and
/// a pathological payload (e.g. a full file dumped into `tool_response`)
/// would otherwise bloat the memories table.
///
/// This is the only size guard on the hook path: the hook never calls the
/// embedder, so the token-count limit used by `add` does not apply.
const MAX_CANDIDATE_CHARS: usize = 8000;

/// Credential patterns that hard-block the entire candidate.
///
/// The block is HARD: matching content means the candidate is dropped
/// entirely — not redacted, not stored. The rationale is that partial
/// redaction (e.g. blanking the JWT but storing the rest) still leaks
/// context around the credential (which API, which user, which action),
/// which defeats the purpose of the block.
///
/// Patterns are case-insensitive substring matches. The `eyJ...` JWT pattern
/// uses a bounded prefix check (three base64url-encoded segments) rather than
/// a regex to keep the extractor dependency-free.
const CREDENTIAL_PATTERNS: &[&str] = &[
    // Common key-value credential forms
    "api_key",
    "api-key",
    "apikey",
    "access_key",
    "access-key",
    "secret_key",
    "secret-key",
    "private_key",
    "private-key",
    "password=",
    "passwd=",
    "token=",
    // Bearer tokens
    "bearer ",
    // AWS access key ID format
    "akia",
];

/// Check whether the candidate contains a credential pattern.
///
/// Returns `true` when the candidate must be hard-blocked (dropped entirely).
fn contains_credential(content: &str) -> bool {
    let lower = content.to_lowercase();

    // JWT: `eyJ` (base64url for `{"`) followed by 12+ base64url chars, then
    // `.` and more segments. JWTs are 3 base64url segments; `eyJ` is the
    // base64url encoding of `{"`.
    // Note: we search in the ORIGINAL (non-lowercased) string because `eyJ`
    // is case-sensitive base64url — lowercasing it to `eyj` would break the
    // match. We then check the structural constraints (dots, length) on the
    // lowercased remainder to be case-insensitive for the rest of the token.
    if let Some(idx) = content.find("eyJ") {
        let after_eyj = &content[idx..];
        // Need at least two dots (three segments) to be a JWT, not just "eyJ"
        // followed by other text. Use the lowercased remainder for the dot
        // count (dots are case-insensitive) and a length check on the raw
        // remainder.
        let after_lower = &lower[idx.min(lower.len())..];
        let dots = after_lower.chars().filter(|&c| c == '.').count();
        if dots >= 2 && after_eyj.len() >= 20 {
            return true;
        }
    }

    // AWS access key ID: AKIA followed by exactly 16 uppercase alphanumeric.
    for window in lower.as_bytes().windows(20) {
        if &window[0..4] == b"akia" && window[4..].iter().all(|b| b.is_ascii_alphanumeric()) {
            return true;
        }
    }

    // Substring patterns (case-insensitive).
    for pattern in CREDENTIAL_PATTERNS {
        if lower.contains(pattern) {
            return true;
        }
    }

    false
}

/// Extract a candidate string from the raw JSON payload value.
///
/// Handles the heterogeneous shapes Claude Code sends:
/// - `Value::String` → the string itself
/// - `Value::Object` → `serde_json::to_string` (compact)
/// - `Value::Array` → each element stringified and joined with `\n`
/// - `Value::Null` or other → `None` (no candidate)
///
/// Returns `None` when:
/// - the value is null, or
/// - the stringified result is empty after trimming, or
/// - the stringified result exceeds [`MAX_CANDIDATE_CHARS`], or
/// - the stringified result contains a credential pattern (hard block).
pub fn extract_candidate(value: &Value) -> Option<String> {
    let raw = stringify_value(value)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() > MAX_CANDIDATE_CHARS {
        return None;
    }
    if contains_credential(trimmed) {
        return None;
    }
    Some(trimmed.to_string())
}

/// Convert a JSON value to a plain string for storage.
fn stringify_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => {
            if s.trim().is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        Value::Object(_) | Value::Array(_) => serde_json::to_string(value).ok(),
        // Numbers, booleans — unlikely in hook payloads but stringify them.
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- stringify / basic extraction ----

    #[test]
    fn extract_string_value() {
        let v = Value::String("hello world".to_string());
        assert_eq!(extract_candidate(&v), Some("hello world".to_string()));
    }

    #[test]
    fn extract_null_returns_none() {
        assert_eq!(extract_candidate(&Value::Null), None);
    }

    #[test]
    fn extract_empty_string_returns_none() {
        assert_eq!(extract_candidate(&Value::String("   ".to_string())), None);
    }

    #[test]
    fn extract_object_stringifies_compact() {
        let v = json!({"command": "ls -la"});
        let result = extract_candidate(&v).expect("object should stringify");
        assert!(result.contains("\"command\":\"ls -la\""));
    }

    #[test]
    fn extract_array_joins_elements() {
        let v = json!([{"role": "user", "content": "hi"}]);
        let result = extract_candidate(&v).expect("array should stringify");
        assert!(result.contains("hi"));
    }

    #[test]
    fn extract_long_content_dropped() {
        let long = "a".repeat(MAX_CANDIDATE_CHARS + 1);
        assert_eq!(extract_candidate(&Value::String(long)), None);
    }

    // ---- credential hard block ----

    #[test]
    fn block_jwt_three_segments() {
        // Minimal JWT shape: eyJ... (three base64url segments separated by dots)
        let jwt =
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozj9NabvVH5hU0R3qTc";
        let v = Value::String(format!("token is {jwt} here"));
        assert_eq!(
            extract_candidate(&v),
            None,
            "JWT-bearing content must be hard-blocked"
        );
    }

    #[test]
    fn block_aws_access_key() {
        let v = Value::String("key=AKIAIOSFODNN7EXAMPLE end".to_string());
        assert_eq!(
            extract_candidate(&v),
            None,
            "AWS access key ID must be hard-blocked"
        );
    }

    #[test]
    fn block_bearer_token() {
        let v = Value::String("Authorization: Bearer abc123def456".to_string());
        assert_eq!(
            extract_candidate(&v),
            None,
            "Bearer token must be hard-blocked"
        );
    }

    #[test]
    fn block_api_key_pattern() {
        let v = Value::String("my api_key=supersecret123".to_string());
        assert_eq!(extract_candidate(&v), None, "api_key= must be hard-blocked");
    }

    #[test]
    fn block_password_pattern() {
        let v = Value::String("login password=hunter2".to_string());
        assert_eq!(
            extract_candidate(&v),
            None,
            "password= must be hard-blocked"
        );
    }

    #[test]
    fn no_false_positive_for_eyj_without_dots() {
        // "eyJ" alone (or with one dot) is not a JWT
        let v = Value::String("the user said eyJhello there".to_string());
        assert!(
            extract_candidate(&v).is_some(),
            "eyJ without enough structure must not block"
        );
    }

    #[test]
    fn no_false_positive_for_normal_text() {
        let v = Value::String("Alice works at Microsoft and uses Rust".to_string());
        assert_eq!(
            extract_candidate(&v),
            Some("Alice works at Microsoft and uses Rust".to_string())
        );
    }
}
