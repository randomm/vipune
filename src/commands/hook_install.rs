//! `vipune hook install` / `vipune hook uninstall` — Claude Code
//! settings.json installer (issue #213, task-d).
//!
//! Merges vipune's hook entries into `~/.claude/settings.json` without
//! clobbering foreign entries (merge-not-clobber), is idempotent on re-run,
//! and records pre-mutation state to a vipune-owned sidecar file
//! (`~/.vipune/hook-install/state.json`) so `uninstall` can remove only
//! vipune's entries precisely.
//!
//! Tests live in the companion module `hook_install_tests.rs`.

use crate::errors::Error;
use crate::output::print_json;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const STATE_DIR_NAME: &str = ".vipune/hook-install";
const STATE_FILE_NAME: &str = "state.json";

/// Event names (as they appear in settings.json's `hooks` map) that vipune
/// installs entries for. `SessionStart` is installed but is a no-op at
/// runtime (the hook run path exits 0 without any DB insert for it).
pub(crate) const VIPUNE_EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PreCompact",
];

/// Pre-mutation state: a snapshot of the settings.json bytes as they existed
/// before vipune's install mutation, plus metadata about what vipune
/// installed.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct InstallState {
    /// Raw settings.json bytes before vipune's mutation; `null` if
    /// settings.json did not exist at install time.
    pub(crate) pre_mutation: Option<String>,
    /// RFC3339 timestamp of when the state was recorded.
    pub(crate) installed_at: String,
    /// The vipune binary path used in the installed command strings.
    pub(crate) vipune_bin_path: String,
}

/// True if `command` looks like a vipune hook invocation: `<vipune bin> hook <event>`.
///
/// Accepts both an absolute binary path (`/usr/bin/vipune hook X`) and the
/// bare name (`vipune hook X`). The binary's basename must be exactly
/// `vipune` and the first argument must start with `hook `.
pub(crate) fn is_vipune_entry(command: &str) -> bool {
    let (bin, args) = match command.split_once(' ') {
        Some((b, a)) => (b, a),
        None => return false,
    };
    if !args.starts_with("hook ") {
        return false;
    }
    let bin_name = bin.rsplit('/').next().unwrap_or(bin);
    bin_name == "vipune"
}

fn home_dir() -> Result<PathBuf, Error> {
    dirs::home_dir().ok_or_else(|| Error::Config("Could not determine home directory".to_string()))
}

/// Resolve the pre-mutation state sidecar path.
pub(crate) fn state_path(home: Option<&Path>) -> Result<PathBuf, Error> {
    let base = match home {
        Some(h) => h.to_path_buf(),
        None => home_dir()?,
    };
    Ok(base.join(STATE_DIR_NAME).join(STATE_FILE_NAME))
}

/// Resolve the settings.json path.
pub(crate) fn settings_path(home: Option<&Path>) -> Result<PathBuf, Error> {
    let base = match home {
        Some(h) => h.to_path_buf(),
        None => home_dir()?,
    };
    Ok(base.join(".claude").join("settings.json"))
}

/// Write `contents` to `path` atomically: write to a sibling temp file, then
/// rename over the target. Avoids partial-write corruption if the process is
/// interrupted mid-write. Creates parent directories as needed.
pub(crate) fn write_atomic(path: &Path, contents: &str) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            Error::Config(format!(
                "Failed to create directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    let tmp = path.with_extension(".tmp");
    fs::write(&tmp, contents)
        .map_err(|e| Error::Config(format!("Failed to write temp file {}: {e}", tmp.display())))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        Error::Config(format!(
            "Failed to rename {} to {}: {e}",
            tmp.display(),
            path.display()
        ))
    })?;
    Ok(())
}

/// True if `entries` contains a hook whose command equals `vipune_command`.
fn group_has_command(entries: Option<&serde_json::Value>, vipune_command: &str) -> bool {
    entries
        .and_then(|h| h.as_array())
        .map(|arr| {
            arr.iter()
                .any(|e| e.get("command").and_then(|c| c.as_str()) == Some(vipune_command))
        })
        .unwrap_or(false)
}

/// One vipune hook group (as it appears in `hooks.<Event>[]): a single inner
/// entry running `vipune hook <event>`.
fn vipune_group(vipune_bin: &str, event: &str) -> serde_json::Value {
    serde_json::json!({
        "hooks": [
            {
                "type": "command",
                "command": format!("{vipune_bin} hook {event}")
            }
        ]
    })
}

/// True if the vipune command for `event` already appears in any group's
/// inner hook entries.
fn vipune_group_present(groups: &[serde_json::Value], vipune_command: &str) -> bool {
    groups
        .iter()
        .any(|group| group_has_command(group.get("hooks"), vipune_command))
}

/// Merge vipune's hook groups into the existing `hooks` map. For each event,
/// a new single-group array is inserted if the key is absent; if present and
/// an array, vipune's group is pushed only when no existing group already
/// carries the exact same vipune command (idempotent). Non-array values
/// under `hooks.<event>` are left untouched.
pub(crate) fn merge_vipune_groups(
    existing: &serde_json::Map<String, serde_json::Value>,
    vipune_bin: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut result = existing.clone();
    for event in VIPUNE_EVENTS {
        let vipune_command = format!("{vipune_bin} hook {event}");
        let key = event.to_string();
        match result.get_mut(&key) {
            None => {
                result.insert(key, serde_json::json!([vipune_group(vipune_bin, event)]));
            }
            Some(val) => {
                if let Some(arr) = val.as_array() {
                    if !vipune_group_present(arr, &vipune_command) {
                        if let Some(arr_mut) = val.as_array_mut() {
                            arr_mut.push(vipune_group(vipune_bin, event));
                        }
                    }
                }
            }
        }
    }
    result
}

/// Remove vipune's hook groups from the `hooks` map. Per event array:
/// - A group whose inner entries are ALL vipune entries is dropped.
/// - A mixed group keeps only its foreign entries.
/// - A group that becomes empty is dropped; an event array that ends up empty
///   (or had no foreign content) drops the event key entirely.
///
/// Non-array values are preserved as-is.
pub(crate) fn remove_vipune_groups(
    existing: &serde_json::Map<String, serde_json::Value>,
    identity_bin: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut result = serde_json::Map::new();
    for (key, val) in existing {
        if let Some(arr) = val.as_array() {
            let key_str = key.clone();
            let identity = identity_bin.to_string();
            let is_vipune_cmd = |cmd: &str| -> bool {
                is_vipune_entry(cmd) || cmd == format!("{identity} hook {key_str}")
            };
            let new_arr = filter_group_array(arr, is_vipune_cmd);
            if !new_arr.is_empty() {
                result.insert(key_str, serde_json::Value::Array(new_arr));
            }
        } else {
            result.insert(key.clone(), val.clone());
        }
    }
    result
}

/// Remove vipune entries from one group array, returning the surviving
/// groups (all-vipune groups are dropped; mixed groups keep foreign entries).
fn filter_group_array(
    arr: &[serde_json::Value],
    is_vipune_cmd: impl Fn(&str) -> bool,
) -> Vec<serde_json::Value> {
    let mut new_arr: Vec<serde_json::Value> = Vec::new();
    for group in arr {
        if !group.is_object() {
            new_arr.push(group.clone());
            continue;
        }
        let Some(entries) = group.get("hooks").and_then(|h| h.as_array()) else {
            new_arr.push(group.clone());
            continue;
        };
        let filtered: Vec<serde_json::Value> = entries
            .iter()
            .filter(|e| {
                let cmd = e.get("command").and_then(|c| c.as_str()).unwrap_or("");
                !is_vipune_cmd(cmd)
            })
            .cloned()
            .collect();
        if filtered.is_empty() {
            continue; // drop the entire group
        }
        if filtered.len() == entries.len() {
            new_arr.push(group.clone());
        } else {
            let mut new_group = group.clone();
            if let Some(g) = new_group.as_object_mut() {
                g.insert("hooks".to_string(), serde_json::Value::Array(filtered));
            }
            new_arr.push(new_group);
        }
    }
    new_arr
}

/// Parse a settings value into a top-level object map, rejecting non-objects.
fn settings_object(
    value: serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>, Error> {
    match value {
        serde_json::Value::Object(m) => Ok(m),
        other => Err(Error::Config(format!(
            "Unexpected settings.json shape (expected object, got {other:?})"
        ))),
    }
}

/// Extract the `hooks` map from a settings object (empty if absent).
fn hooks_map(
    settings_map: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>, Error> {
    match settings_map.get("hooks") {
        Some(serde_json::Value::Object(m)) => Ok(m.clone()),
        Some(_) => Err(Error::Config(
            "Existing 'hooks' key in settings.json is not an object".to_string(),
        )),
        None => Ok(serde_json::Map::new()),
    }
}

/// Read the pre-mutation settings (raw bytes) from `settings_path`, if the
/// file exists.
fn read_existing_settings(settings_path: &Path) -> Result<Option<String>, Error> {
    if !settings_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(settings_path)
        .map_err(|e| Error::Config(format!("Failed to read {}: {e}", settings_path.display())))?;
    Ok(Some(raw))
}

/// Parse raw settings JSON into a top-level map (empty object if `None`).
fn parse_settings(
    existing_raw: &Option<String>,
) -> Result<serde_json::Map<String, serde_json::Value>, Error> {
    let value: serde_json::Value = match existing_raw {
        Some(raw) => serde_json::from_str(raw)
            .map_err(|e| Error::Config(format!("Failed to parse settings.json: {e}")))?,
        None => serde_json::json!({}),
    };
    settings_object(value)
}

/// Install vipune hook entries into `~/.claude/settings.json`.
///
/// Records the pre-mutation state sidecar BEFORE writing anything, then
/// merges vipune's groups into the `hooks` map and writes atomically.
///
/// # Arguments
///
/// * `json` - If true, emit JSON response; else human-readable.
pub fn handle_hook_install(json: bool) -> Result<ExitCode, Error> {
    handle_hook_install_at(json, None)
}

/// Install hook entries, using `home` as the base directory for
/// settings.json and the state sidecar (instead of the real user home).
/// `home = None` (the production path) uses `dirs::home_dir()`.
pub(crate) fn handle_hook_install_at(json: bool, home: Option<&Path>) -> Result<ExitCode, Error> {
    let settings_path = settings_path(home)?;
    let state_path = state_path(home)?;
    let vipune_bin = std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "vipune".to_string());

    let existing_raw = read_existing_settings(&settings_path)?;
    let mut settings_map = parse_settings(&existing_raw)?;

    // Record pre-mutation state BEFORE writing anything, so an interrupted
    // install still leaves the sidecar present with the original bytes.
    let state = InstallState {
        pre_mutation: existing_raw,
        installed_at: chrono::Utc::now().to_rfc3339(),
        vipune_bin_path: vipune_bin.clone(),
    };
    write_atomic(&state_path, &serde_json::to_string_pretty(&state)?)?;

    let new_hooks = merge_vipune_groups(&hooks_map(&settings_map)?, &vipune_bin);
    settings_map.insert("hooks".to_string(), serde_json::Value::Object(new_hooks));

    let new_raw = serde_json::to_string_pretty(&serde_json::Value::Object(settings_map))?;
    write_atomic(&settings_path, &new_raw)?;

    let installed: Vec<String> = VIPUNE_EVENTS.iter().map(|e| e.to_string()).collect();
    emit_install_result(json, &settings_path, &state_path, &installed)?;
    Ok(ExitCode::SUCCESS)
}

/// Events whose arrays contain at least one vipune entry (the ones
/// uninstall will actually remove), for the human/JSON report.
fn find_removed_events(
    hooks: &serde_json::Map<String, serde_json::Value>,
    identity_bin: &str,
) -> Vec<String> {
    let mut removed: Vec<String> = Vec::new();
    for event in VIPUNE_EVENTS {
        let key = event.to_string();
        let identity = format!("{identity_bin} hook {key}");
        let has_vipune = hooks
            .get(&key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().any(|group| {
                    group
                        .get("hooks")
                        .and_then(|h| h.as_array())
                        .is_some_and(|entries| {
                            entries.iter().any(|e| {
                                let cmd = e.get("command").and_then(|c| c.as_str()).unwrap_or("");
                                cmd == identity || is_vipune_entry(cmd)
                            })
                        })
                })
            })
            .unwrap_or(false);
        if has_vipune {
            removed.push(key);
        }
    }
    removed
}

/// The binary path to use for identifying vipune entries during uninstall.
///
/// Prefers the exact path recorded in the state sidecar at install time — the
/// install binary may have moved since (reinstall, PATH change), so runtime
/// `current_exe()` alone is not a reliable identity. Falls back to the
/// runtime path if no sidecar exists.
fn uninstall_identity_bin(state_path: &Path) -> String {
    std::fs::read_to_string(state_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<InstallState>(&raw).ok())
        .map(|s| s.vipune_bin_path)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "vipune".to_string())
        })
}

fn emit_install_result(
    json: bool,
    settings_path: &Path,
    state_path: &Path,
    installed: &[String],
) -> Result<(), Error> {
    if json {
        print_json(&serde_json::json!({
            "settings_path": settings_path.display().to_string(),
            "state_path": state_path.display().to_string(),
            "installed": installed,
        }));
    } else {
        println!(
            "Installed vipune hooks into {} ({} events: {})",
            settings_path.display(),
            installed.len(),
            installed.join(", ")
        );
    }
    Ok(())
}

/// Uninstall vipune hook entries from `~/.claude/settings.json`.
///
/// Surgically removes only vipune's entries (identified by command string),
/// preserving foreign entries added after install. The state sidecar is
/// removed on success.
///
/// # Arguments
///
/// * `json` - If true, emit JSON response; else human-readable.
pub fn handle_hook_uninstall(json: bool) -> Result<ExitCode, Error> {
    handle_hook_uninstall_at(json, None)
}

/// Uninstall hook entries, using `home` as the base directory (instead of
/// the real user home). `home = None` (the production path) uses
/// `dirs::home_dir()`.
pub(crate) fn handle_hook_uninstall_at(json: bool, home: Option<&Path>) -> Result<ExitCode, Error> {
    let settings_path = settings_path(home)?;
    let state_path = state_path(home)?;

    if !settings_path.exists() {
        // Nothing to do — settings.json doesn't exist at all.
        let _ = fs::remove_file(&state_path); // clean up any stale sidecar
        if json {
            print_json(&serde_json::json!({
                "settings_path": settings_path.display().to_string(),
                "removed": []
            }));
        } else {
            println!("Nothing to uninstall (settings.json not found).");
        }
        return Ok(ExitCode::SUCCESS);
    }

    let raw = fs::read_to_string(&settings_path)
        .map_err(|e| Error::Config(format!("Failed to read {}: {e}", settings_path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| Error::Config(format!("Failed to parse {}: {e}", settings_path.display())))?;
    let mut settings_map = settings_object(value)?;
    let existing_hooks = hooks_map(&settings_map)?;
    let identity_bin = uninstall_identity_bin(&state_path);
    let removed_events = find_removed_events(&existing_hooks, &identity_bin);

    let new_hooks = remove_vipune_groups(&existing_hooks, &identity_bin);
    if new_hooks.is_empty() {
        settings_map.remove("hooks");
    } else {
        settings_map.insert("hooks".to_string(), serde_json::Value::Object(new_hooks));
    }

    let new_raw = serde_json::to_string_pretty(&serde_json::Value::Object(settings_map))?;
    write_atomic(&settings_path, &new_raw)?;

    let _ = fs::remove_file(&state_path);

    if json {
        print_json(&serde_json::json!({
            "settings_path": settings_path.display().to_string(),
            "removed": removed_events,
        }));
    } else if removed_events.is_empty() {
        println!("No vipune hooks found to remove.");
    } else {
        println!(
            "Removed vipune hooks from {} ({} events: {})",
            settings_path.display(),
            removed_events.len(),
            removed_events.join(", ")
        );
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_vipune_entry() {
        assert!(is_vipune_entry("/usr/bin/vipune hook UserPromptSubmit"));
        assert!(is_vipune_entry("/opt/homebrew/bin/vipune hook PreToolUse"));
        assert!(is_vipune_entry("vipune hook SessionStart"));
        assert!(!is_vipune_entry("/opt/foreign-tool --check"));
        assert!(!is_vipune_entry("/usr/bin/vipune something-else"));
        assert!(!is_vipune_entry("vipune"));
        assert!(!is_vipune_entry("notvipune hook Foo"));
    }

    fn fake_home() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        (dir, home)
    }

    fn install_at(home: &Path, bin: &str) -> Result<(), Error> {
        let sp = settings_path(Some(home))?;
        let stp = state_path(Some(home))?;
        let existing_raw = read_existing_settings(&sp)?;
        let mut map = parse_settings(&existing_raw)?;
        let state = InstallState {
            pre_mutation: existing_raw,
            installed_at: "test".to_string(),
            vipune_bin_path: bin.to_string(),
        };
        write_atomic(&stp, &serde_json::to_string_pretty(&state)?)?;
        let hooks = merge_vipune_groups(&hooks_map(&map)?, bin);
        map.insert("hooks".to_string(), serde_json::Value::Object(hooks));
        write_atomic(
            &sp,
            &serde_json::to_string_pretty(&serde_json::Value::Object(map))?,
        )?;
        Ok(())
    }

    #[test]
    fn test_merge_preserves_foreign_and_appends_vipune() {
        let existing = serde_json::json!({
            "PreToolUse": [
                {
                    "matcher": "Grep|Glob",
                    "hooks": [{ "type": "command", "command": "/opt/foreign" }]
                }
            ]
        })
        .as_object()
        .unwrap()
        .clone();

        let merged = merge_vipune_groups(&existing, "/fake/vipune");
        let pre = merged.get("PreToolUse").unwrap().as_array().unwrap();
        assert_eq!(pre.len(), 2);
        assert_eq!(
            pre[0].get("matcher").unwrap().as_str().unwrap(),
            "Grep|Glob"
        );

        // Second merge must not duplicate
        let merged2 = merge_vipune_groups(&merged, "/fake/vipune");
        assert_eq!(
            merged2.get("PreToolUse").unwrap().as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn test_remove_keeps_mixed_group_foreign_entries() {
        let existing = serde_json::json!({
            "UserPromptSubmit": [
                {
                    "hooks": [
                        { "type": "command", "command": "/opt/foreign" },
                        { "type": "command", "command": "/fake/vipune hook UserPromptSubmit" }
                    ]
                }
            ]
        })
        .as_object()
        .unwrap()
        .clone();

        let removed = remove_vipune_groups(&existing, "/fake/vipune");
        let arr = removed.get("UserPromptSubmit").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let entries = arr[0].get("hooks").unwrap().as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].get("command").unwrap().as_str().unwrap(),
            "/opt/foreign"
        );
    }

    #[test]
    fn test_remove_drops_event_key_when_only_vipune() {
        let existing = serde_json::json!({
            "PreCompact": [
                { "hooks": [{ "type": "command", "command": "/fake/vipune hook PreCompact" }] }
            ]
        })
        .as_object()
        .unwrap()
        .clone();

        let removed = remove_vipune_groups(&existing, "/fake/vipune");
        assert!(!removed.contains_key("PreCompact"));
    }

    #[test]
    fn test_install_then_uninstall_roundtrip_with_fake_bin() {
        let (_dir, home) = fake_home();
        let original = serde_json::json!({ "model": "x" });
        let sp = home.join(".claude").join("settings.json");
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::write(&sp, original.to_string()).unwrap();

        install_at(&home, "/fake/vipune").unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sp).unwrap()).unwrap();
        assert!(value.get("model").is_some());
        assert!(
            value
                .get("hooks")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("SessionStart")
        );

        // Uninstall via the production-shaped path, but the installed command
        // uses /fake/vipune, which is_vipune_entry still recognises.
        handle_hook_uninstall_at(false, Some(&home)).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sp).unwrap()).unwrap();
        // All vipune events must be gone (only vipune entries existed for them).
        let hooks = value
            .get("hooks")
            .map(|h| h.as_object().cloned().unwrap_or_default())
            .unwrap_or_default();
        for event in VIPUNE_EVENTS {
            let key = event.to_string();
            assert!(
                !hooks.contains_key(&key),
                "{event} should have been removed"
            );
        }
        assert_eq!(value.get("model").unwrap().as_str().unwrap(), "x");
    }
}
