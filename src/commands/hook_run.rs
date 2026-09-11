//! Hook event handler adapter (issue #191, task-a; completed in issue #213).
//!
//! Thin command-surface module that reads the Claude Code JSON payload from
//! stdin, delegates to the `crate::hook` module (task-b), and always exits 0
//! with empty stdout on any non-fatal failure.
//!
//! The hook path must never load the ONNX model. The adapter reads stdin
//! once and passes the event type (from the subcommand) to the pipeline.

use std::process::ExitCode;

use crate::hook::{HookEvent, read_stdin, run_hook_event};

/// Map a `HookEvent` subcommand variant to its pipeline counterpart.
///
/// The command surface carries the event type via the subcommand variant
/// (Claude Code invokes a different `vipune hook <event>` for each event).
/// The pipeline receives the enum directly.
pub fn handle_hook_event(
    config: &crate::config::Config,
    _json: bool,
    event: HookEvent,
) -> Result<ExitCode, crate::errors::Error> {
    let input = read_stdin();
    run_hook_event(config, event, &input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hook::run_hook_event;

    /// The adapter must never surface an error to the agent. Even on
    /// catastrophic failure (e.g. unreadable stdin), the exit code is
    /// `SUCCESS`. This test exercises the happy path only; the full
    /// non-fatal-failure matrix lives in `crate::hook::run::tests`.
    #[test]
    fn handle_hook_event_delegates_to_pipeline() {
        let config = crate::config::Config::default();
        // The adapter reads from the real stdin — in a test context, stdin
        // is whatever the test harness provides (often empty). An empty
        // payload is a non-fatal failure that maps to exit 0.
        let result =
            handle_hook_event(&config, false, HookEvent::UserPromptSubmit).expect("adapter ok");
        assert_eq!(result, ExitCode::SUCCESS);
    }

    #[test]
    fn pipeline_returns_success_on_garbage_input() {
        let config = crate::config::Config::default();
        let exit = run_hook_event(&config, HookEvent::UserPromptSubmit, "not json")
            .expect("garbage stdin must not error");
        assert_eq!(exit, ExitCode::SUCCESS);
    }
}
