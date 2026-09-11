//! `vipune promote` — candidate → active promotion.
//!
//! Implements sub-issue 4 of issue #194: a **pure evaluator** that decides
//! whether a memory should be promoted from `candidate` to `active`, plus an
//! **apply path** that runs the evaluator over a project's candidates and
//! promotes the eligible ones through the existing status-update path.
//!
//! Eligibility (all must hold, mirroring the epic's "pure evaluator plus
//! apply path" criterion):
//! - `status == candidate` — the promotion precondition; `superseded` and
//!   `deprecated` rows are NEVER promoted, and `active` rows are a no-op.
//! - `retrieval_count >= threshold` — **strict `>=`** boundary: a candidate
//!   with `retrieval_count` one below the threshold (4 with the default 5)
//!   is NOT promoted, a candidate at the threshold (5) IS. The threshold
//!   defaults to 5 and is overridable via the `VIPUNE_PROMOTION_THRESHOLD`
//!   environment variable, mirroring the `VIPUNE_RECENCY_WEIGHT` pattern
//!   used by the recency knob in `src/config`.
//!
//! The retrieval telemetry (`retrieval_count`) that drives the decision is
//! populated by `touch_memories` at the two sites that represent user
//! retrieval: `search` with an explicit `--status candidate` (or
//! `--include-candidates`) filter and direct `get` by id. `list
//! --include-candidates` is output-only and does NOT increment the counter,
//! so a candidate left at zero retrievals never promotes — that is correct,
//! not a bug.
//!
//! The apply path issues `UPDATE status='active'` per eligible row via
//! `Database::update` with `UpdateOptions { status: Some("active"), .. }` —
//! the same path `vipune update --status` uses — and never deletes or
//! re-embeds anything.

use crate::errors::Error;
use crate::sqlite::{Database, UpdateOptions};
use std::process::ExitCode;

/// Default promotion threshold: a candidate promotes when
/// `retrieval_count >= PROMOTION_THRESHOLD_DEFAULT` (i.e. 5).
pub(crate) const PROMOTION_THRESHOLD_DEFAULT: i64 = 5;

/// Environment variable that overrides the promotion threshold, mirroring the
/// `VIPUNE_RECENCY_WEIGHT` env-override pattern used by the recency knob.
pub(crate) const PROMOTION_THRESHOLD_ENV: &str = "VIPUNE_PROMOTION_THRESHOLD";

/// A single row the pure evaluator needs to make a promotion decision.
///
/// Kept minimal and free of `&str` borrows so the evaluator is a trivially
/// testable pure function that does not touch the database.
pub(crate) struct PromotionCandidate {
    /// Memory id (used by the apply path; ignored by the pure evaluator).
    pub id: String,
    /// Current lifecycle status. Only `candidate` rows are ever promoted.
    pub status: String,
    /// Number of times this memory has been retrieved via search/get.
    pub retrieval_count: i64,
}

/// Pure promotion evaluator.
///
/// Returns `true` only when the row is a `candidate` (the promotion
/// precondition) AND its `retrieval_count` meets or exceeds `threshold`.
///
/// This is a pure function — no database, no I/O, no time — so the strict
/// `>=` boundary and the status exclusions are each directly testable:
/// - a `candidate` with `retrieval_count` below `threshold` is NOT promoted;
/// - a `candidate` with `retrieval_count` exactly at `threshold` IS promoted;
/// - any `superseded`, `deprecated`, or `active` row is NOT promoted,
///   regardless of `retrieval_count`.
pub(crate) fn should_promote(status: &str, retrieval_count: i64, threshold: i64) -> bool {
    status == "candidate" && retrieval_count >= threshold
}

/// Resolve the promotion threshold, honouring the `VIPUNE_PROMOTION_THRESHOLD`
/// environment override.
///
/// Mirrors `config::env_parser::apply_recency_weight_override`: if the
/// variable is set to a non-negative integer it wins, otherwise the default
/// threshold is returned. A malformed or negative value is an
/// [`Error::InvalidInput`] that names the offending value (the same shape the
/// recency parser uses for `VIPUNE_RECENCY_WEIGHT`).
pub(crate) fn resolve_promotion_threshold() -> Result<i64, Error> {
    let Ok(raw) = std::env::var(PROMOTION_THRESHOLD_ENV) else {
        return Ok(PROMOTION_THRESHOLD_DEFAULT);
    };
    let parsed: i64 = raw.trim().parse().map_err(|_| {
        Error::InvalidInput(format!(
            "Invalid {} '{}'; expected a non-negative integer",
            PROMOTION_THRESHOLD_ENV, raw
        ))
    })?;
    if parsed < 0 {
        return Err(Error::InvalidInput(format!(
            "Invalid {} '{}'; must be >= 0",
            PROMOTION_THRESHOLD_ENV, raw
        )));
    }
    Ok(parsed)
}

/// Collect the rows the evaluator needs for a project's candidates.
///
/// Fetches `(id, status, retrieval_count)` for every `candidate` row in the
/// project. Only candidates are fetched because only candidates can promote —
/// active/superseded/deprecated rows are structurally ineligible and skipping
/// them keeps the read cheap.
pub(crate) fn fetch_candidates(
    db: &Database,
    project_id: &str,
) -> Result<Vec<PromotionCandidate>, Error> {
    let mut stmt = db
        .conn()
        .prepare("SELECT id, status, retrieval_count FROM memories WHERE project_id = ? AND status = 'candidate'")
        .map_err(Error::from)?;

    let rows = stmt
        .query_map([project_id], |row| {
            Ok(PromotionCandidate {
                id: row.get::<_, String>(0)?,
                status: row.get::<_, String>(1)?,
                retrieval_count: row.get::<_, i64>(2)?,
            })
        })
        .map_err(Error::from)?
        .collect::<Result<Vec<_>, rusqlite::Error>>()
        .map_err(Error::from)?;

    Ok(rows)
}

/// Run the promotion apply path over a project's candidates.
///
/// Fetches the project's candidates, evaluates each via [`should_promote`],
/// and for every eligible row issues `UPDATE status='active'` through
/// `Database::update` (the existing status-update path, exactly what
/// `vipune update --status active` uses). Returns the number of rows
/// promoted.
///
/// # Errors
///
/// Returns an error if the threshold override is invalid, the candidate
/// fetch fails, or any status update fails (e.g. the row vanished between
/// fetch and update).
pub(crate) fn run_promotion(db: &Database, project_id: &str) -> Result<usize, Error> {
    let threshold = resolve_promotion_threshold()?;
    let candidates = fetch_candidates(db, project_id)?;

    let mut promoted = 0usize;
    for c in &candidates {
        if should_promote(&c.status, c.retrieval_count, threshold) {
            db.update(
                &c.id,
                project_id,
                UpdateOptions {
                    content: None,
                    embedding: None,
                    metadata: None,
                    memory_type: None,
                    status: Some("active"),
                    importance: None,
                },
            )
            .map_err(Error::from)?;
            promoted += 1;
        }
    }

    Ok(promoted)
}

/// Handle the `vipune promote` command.
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database (already resolved after the
///   `--db-path` override is applied).
/// * `project_id` - Project to scan for promotable candidates.
/// * `json` - If true, emit the JSON response; otherwise human-readable.
///
/// # Errors
///
/// Returns an error if the database cannot be opened, the threshold override
/// is invalid, or a status update fails.
pub fn handle_promote(
    db_path: &std::path::Path,
    project_id: &str,
    json: bool,
) -> Result<ExitCode, Error> {
    let db = Database::open(db_path).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string(),
            );
        }
        Error::Config(msg)
    })?;

    let threshold = resolve_promotion_threshold()?;
    let promoted = run_promotion(&db, project_id)?;

    let response = serde_json::json!({
        "project_id": project_id,
        "threshold": threshold,
        "promoted": promoted,
    });

    if json {
        crate::output::print_json(&response);
    } else {
        println!(
            "Promoted {} candidate memor{} to active (threshold retrieval_count >= {})",
            promoted,
            if promoted == 1 { "y" } else { "ies" },
            threshold
        );
    }

    Ok(ExitCode::SUCCESS)
}
