//! Tests for `vipune hook install` / `uninstall` (issue #213, task-d).
//!
//! All tests use a temp directory as a fake home so the real
//! `~/.claude/settings.json` is never touched. The `handle_hook_install_at`
//! / `handle_hook_uninstall_at` variants accept a `home` override for this.
//!
//! Because `handle_hook_install_at` resolves the vipune binary path via
//! `std::env::current_exe()`, the tests do not assert on the exact command
//! string — they assert on structure (entry count, foreign entries preserved,
//! idempotency) and on the state sidecar file content instead.

#![cfg(test)]

use crate::commands::hook_install::{
    InstallState, handle_hook_install_at, handle_hook_uninstall_at,
};
use serde_json::Value;
use std::path::Path;

const STATE_REL: &str = ".vipune/hook-install/state.json";

fn temp_home() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    (dir, home)
}

fn write_settings(home: &Path, raw: &str) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(claude_dir.join("settings.json"), raw).unwrap();
}

fn read_settings_value(home: &Path) -> Value {
    let raw = std::fs::read_to_string(home.join(".claude").join("settings.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

/// Assert the settings.json has exactly the given vipune events present in
/// the `hooks` map, each with at least one entry, and that the total group
/// count for `PreToolUse` matches expectations (foreign + vipune).
fn assert_vipune_groups_present(value: &Value, expected_pre_tool_groups: usize) {
    let hooks = value
        .get("hooks")
        .expect("hooks key must exist")
        .as_object()
        .unwrap();
    for event in [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PreCompact",
    ] {
        let arr = hooks
            .get(event)
            .unwrap_or_else(|| panic!("event {event} missing from hooks map"))
            .as_array()
            .unwrap_or_else(|| panic!("event {event} is not an array"));
        assert!(!arr.is_empty(), "event {event} array must not be empty");
    }
    let pre_tool = hooks.get("PreToolUse").unwrap().as_array().unwrap();
    assert_eq!(
        pre_tool.len(),
        expected_pre_tool_groups,
        "PreToolUse group count mismatch"
    );
}

#[test]
fn test_fresh_install_creates_settings_and_all_events() {
    let (_dir, home) = temp_home();
    handle_hook_install_at(false, Some(&home)).unwrap();

    let value = read_settings_value(&home);
    assert_vipune_groups_present(&value, 1); // no foreign entries, so 1 group per event
}

#[test]
fn test_install_preserves_foreign_top_level_keys() {
    let (_dir, home) = temp_home();
    let foreign = serde_json::json!({
        "model": "test-model",
        "permissions": { "defaultMode": "auto" },
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Grep|Glob",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "/opt/foreign-tool --check",
                            "timeout": 5
                        }
                    ]
                }
            ]
        }
    });
    write_settings(&home, &foreign.to_string());

    handle_hook_install_at(false, Some(&home)).unwrap();

    let value = read_settings_value(&home);
    assert_eq!(value.get("model").unwrap().as_str().unwrap(), "test-model");
    assert!(value.get("permissions").is_some());
    assert_vipune_groups_present(&value, 2); // 1 foreign + 1 vipune group for PreToolUse

    // Foreign PreToolUse entry (index 0) must be the foreign one
    let pre_tool = value
        .get("hooks")
        .unwrap()
        .as_object()
        .unwrap()
        .get("PreToolUse")
        .unwrap()
        .as_array()
        .unwrap();
    let foreign_cmd = pre_tool[0].get("hooks").unwrap().as_array().unwrap()[0]
        .get("command")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(foreign_cmd, "/opt/foreign-tool --check");
}

#[test]
fn test_install_is_idempotent_byte_stable() {
    let (_dir, home) = temp_home();
    let foreign = serde_json::json!({
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        { "type": "command", "command": "/opt/other-tool" }
                    ]
                }
            ]
        }
    });
    write_settings(&home, &foreign.to_string());

    handle_hook_install_at(false, Some(&home)).unwrap();
    let first = std::fs::read_to_string(home.join(".claude").join("settings.json")).unwrap();

    handle_hook_install_at(false, Some(&home)).unwrap();
    let second = std::fs::read_to_string(home.join(".claude").join("settings.json")).unwrap();

    assert_eq!(first, second, "re-run must be byte-stable");

    // Exactly 2 groups in UserPromptSubmit (foreign + vipune), not 3
    let value: Value = serde_json::from_str(&first).unwrap();
    let arr = value
        .get("hooks")
        .unwrap()
        .as_object()
        .unwrap()
        .get("UserPromptSubmit")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(arr.len(), 2);
}

#[test]
fn test_uninstall_removes_only_vipune_entries() {
    let (_dir, home) = temp_home();
    let foreign = serde_json::json!({
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        { "type": "command", "command": "/opt/foreign-tool" }
                    ]
                }
            ],
            "UserPromptSubmit": [
                {
                    "hooks": [
                        { "type": "command", "command": "/opt/other-tool" }
                    ]
                }
            ]
        }
    });
    write_settings(&home, &foreign.to_string());

    handle_hook_install_at(false, Some(&home)).unwrap();
    handle_hook_uninstall_at(false, Some(&home)).unwrap();

    let value = read_settings_value(&home);
    let hooks = value
        .get("hooks")
        .expect("hooks key must still exist (foreign entries remain)")
        .as_object()
        .unwrap();

    // Foreign PreToolUse entry must still be there
    let pre_tool = hooks.get("PreToolUse").unwrap().as_array().unwrap();
    assert_eq!(pre_tool.len(), 1);
    let cmd = pre_tool[0].get("hooks").unwrap().as_array().unwrap()[0]
        .get("command")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(cmd, "/opt/foreign-tool");

    // The vipune UserPromptSubmit group is gone; the foreign /opt/other-tool
    // entry (a separate group) must be preserved.
    assert!(
        hooks.contains_key("UserPromptSubmit"),
        "foreign UserPromptSubmit entry must be preserved"
    );
    let ups = hooks.get("UserPromptSubmit").unwrap().as_array().unwrap();
    assert_eq!(ups.len(), 1);
    let ups_cmd = ups[0].get("hooks").unwrap().as_array().unwrap()[0]
        .get("command")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(ups_cmd, "/opt/other-tool");
}

#[test]
fn test_uninstall_with_no_settings_file() {
    let (_dir, home) = temp_home();
    // No settings.json exists at all
    handle_hook_uninstall_at(false, Some(&home)).unwrap();
    assert!(!home.join(".claude").join("settings.json").exists());
}

#[test]
fn test_state_sidecar_written_on_install() {
    let (_dir, home) = temp_home();
    let original = serde_json::json!({ "model": "x" });
    write_settings(&home, &original.to_string());

    handle_hook_install_at(false, Some(&home)).unwrap();

    let state_path = home.join(STATE_REL);
    assert!(
        state_path.exists(),
        "state sidecar must exist after install"
    );
    let state_raw = std::fs::read_to_string(&state_path).unwrap();
    let state: InstallState = serde_json::from_str(&state_raw).unwrap();

    // The pre_mutation content must round-trip to the original JSON object
    let pre: Value = serde_json::from_str(state.pre_mutation.as_deref().unwrap()).unwrap();
    assert_eq!(
        pre, original,
        "pre_mutation must equal the original settings"
    );
}

#[test]
fn test_state_sidecar_removed_on_uninstall() {
    let (_dir, home) = temp_home();
    write_settings(&home, &serde_json::json!({}).to_string());
    handle_hook_install_at(false, Some(&home)).unwrap();
    handle_hook_uninstall_at(false, Some(&home)).unwrap();

    assert!(!home.join(STATE_REL).exists());
}

#[test]
fn test_install_creates_missing_claude_dir() {
    let (_dir, home) = temp_home();
    // No .claude dir at all
    assert!(!home.join(".claude").exists());
    handle_hook_install_at(false, Some(&home)).unwrap();
    assert!(home.join(".claude").join("settings.json").exists());
}
