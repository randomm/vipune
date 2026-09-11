//! `vipune prune` handler (issue #194, sub-issue 5).
//!
//! Prune demotes stale candidate memories to `deprecated`. It **never deletes**:
//! demotions are issued as `UPDATE status = 'deprecated'` through the existing
//! update path, so the total row count of the database is unchanged after any
//! run — only the status column changes.
//!
//! # Eligibility (final, deterministic)
//!
//! A row is eligible for demotion if and only if ALL of the following hold:
//!
//! 1. `status = 'candidate'` (active, superseded and deprecated rows are never
//!    touched; `superseded` is checked twice for defense in depth)
//! 2. `retrieval_count < N` (N configurable, default 3)
//! 3. `age(created_at) > T` (age measured from `created_at`, consistent with
//!    the decay model; T configurable, default 14 days)
//!
//! # Hard exclusions (checked in the query, applied per row)
//!
//! - `type = 'guard'`: guard-type memories are never demoted.
//! - `importance IN ('high', 'critical')`: high/critical importance is a HARD
//!   exclusion — such rows are never demoted even when every eligibility
//!   criterion is satisfied.
//!
//! # Configuration
//!
//! N and T are configurable via TOML (`prune_count`, `prune_age_days` in
//! `~/.config/vipune/config.toml`) with `VIPUNE_` env overrides
//! (`VIPUNE_PRUNE_COUNT`, `VIPUNE_PRUNE_AGE_DAYS`) mirroring how
//! `VIPUNE_RECENCY_WEIGHT` is config-driven. The config plumbing lives in the
//! config workstream; this module exposes them as parameters so the handler
//! stays a pure function of its inputs (and stays testable without a real
//! config file).

use chrono::Utc;
use serde::Serialize;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration as StdDuration;

use crate::errors::Error;
use crate::memory::lifecycle::MemoryStatus;
use crate::sqlite::{Database, Memory, UpdateOptions};

/// Default retrieval-count threshold N: candidates with `retrieval_count < N`
/// are prune-eligible (subject to the age rule and hard exclusions).
pub const DEFAULT_PRUNE_COUNT: i64 = 3;

/// Default age threshold T: candidates older than T days are prune-eligible
/// (subject to the count rule and hard exclusions).
pub const DEFAULT_PRUNE_AGE_DAYS: i64 = 14;

/// Response for `vipune prune`: the rows that were demoted and, for the
/// never-deletes invariant, the database row count before and after the run.
#[derive(Debug, Serialize)]
pub struct PruneResponse {
    /// Number of rows demoted from `candidate` to `deprecated`.
    pub demoted: usize,
    /// Memory id of each demoted row.
    pub demoted_ids: Vec<String>,
    /// Total row count in the database before the run (all projects, all
    /// statuses). Prune never deletes, so this must equal `rows_after`.
    pub rows_before: usize,
    /// Total row count in the database after the run.
    pub rows_after: usize,
}

/// Pure eligibility evaluator for prune: selects exactly the rows matching
/// `status = 'candidate' AND retrieval_count < N AND age(created_at) > T`,
/// with guard-type and high/critical-importance rows excluded.
///
/// "Pure" in the sense that it contains no database write: selection is a
/// read-only query, and the demotion itself is applied separately by
/// [`PruneEvaluator::apply`] through the existing update path.
pub struct PruneEvaluator {
    /// Retrieval-count threshold N (`retrieval_count < N` to be eligible).
    pub count_threshold: i64,
    /// Age threshold T (age measured from `created_at`).
    pub age_threshold: StdDuration,
}

impl PruneEvaluator {
    /// Build an evaluator with the given N (count) and T (age, in days).
    pub fn new(count_threshold: i64, age_threshold_days: i64) -> Self {
        Self {
            count_threshold,
            age_threshold: StdDuration::from_secs(
                u64::try_from(age_threshold_days.saturating_mul(86_400)).unwrap_or(u64::MAX),
            ),
        }
    }

    /// Select exactly the rows matching the prune eligibility predicate:
    ///
    /// `status = 'candidate' AND type <> 'guard'
    ///   AND NOT (importance = 'high' OR importance = 'critical')
    ///   AND retrieval_count < N AND created_at < (now - T)`
    ///
    /// The query is scoped to one project because the demotion path
    /// (`Database::update`) is project-scoped. Run once per project id to
    /// cover the whole database.
    ///
    /// # Errors
    ///
    /// Returns error if the database query fails.
    pub fn select_candidates(
        &self,
        db: &mut Database,
        project_id: &str,
    ) -> Result<Vec<Memory>, Error> {
        let cutoff_delta = chrono::Duration::from_std(self.age_threshold)
            .unwrap_or_else(|_| chrono::Duration::days(i64::MAX / 2));
        let cutoff = Utc::now() - cutoff_delta;
        let cutoff_rfc3339 = cutoff.to_rfc3339();

        let sql = "SELECT id, project_id, content, metadata, embedding, created_at, updated_at, type, status, superseded_by, retrieval_count, last_retrieved_at, importance
             FROM memories
             WHERE project_id = ?1
               AND status = 'candidate'
               AND type <> 'guard'
               AND NOT (importance = 'high' OR importance = 'critical')
               AND retrieval_count < ?2
               AND created_at < ?3";

        let mut stmt = db.conn().prepare(sql)?;
        let mut results: Vec<Memory> = Vec::new();
        for row_result in stmt.query_map(
            rusqlite::params![project_id, self.count_threshold, &cutoff_rfc3339],
            crate::sqlite::map_row_to_memory,
        )? {
            results.push(row_result.map_err(Error::from)?);
        }
        Ok(results)
    }

    /// Apply the demotion to the selected rows: `UPDATE status = 'deprecated'`
    /// through the existing update path (`Database::update`), never a DELETE.
    ///
    /// Rows already in a terminal status are skipped defensively (a row that
    /// changed status between selection and apply is left alone).
    ///
    /// # Errors
    ///
    /// Returns error if any underlying database update fails.
    pub fn apply(&self, db: &mut Database, candidates: &[&Memory]) -> Result<Vec<String>, Error> {
        let mut demoted: Vec<String> = Vec::new();
        for candidate in candidates {
            // Defense in depth: only ever demote rows that are still candidate.
            if candidate.status != MemoryStatus::Candidate.as_str() {
                continue;
            }
            db.update(
                &candidate.id,
                &candidate.project_id,
                UpdateOptions {
                    content: None,
                    embedding: None,
                    metadata: None,
                    memory_type: None,
                    status: Some(MemoryStatus::Deprecated.as_str()),
                    importance: None,
                },
            )?;
            demoted.push(candidate.id.clone());
        }
        Ok(demoted)
    }

    /// Select candidates for a single project and immediately demote them.
    ///
    /// Convenience combination of [`select_candidates`](Self::select_candidates)
    /// and [`apply`](Self::apply).
    ///
    /// # Errors
    ///
    /// Returns error if the selection or any demotion fails.
    pub fn run_for_project(
        &self,
        db: &mut Database,
        project_id: &str,
    ) -> Result<Vec<String>, Error> {
        let candidates = self.select_candidates(db, project_id)?;
        let refs: Vec<&Memory> = candidates.iter().collect();
        self.apply(db, &refs)
    }
}

/// Count total rows in the database (all projects, all statuses).
fn count_total_rows(db: &Database) -> Result<usize, Error> {
    let count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .map_err(Error::from)?;
    Ok(count as usize)
}

/// Run the prune operation across all projects in the database.
///
/// # Behavior
///
/// For every distinct project id in the database, selects the eligible
/// candidate rows (see [`PruneEvaluator`]) and demotes them to `deprecated`
/// via the existing update path. No row is ever deleted: the response reports
/// `rows_before` and `rows_after` so the never-deletes invariant is
/// observable in the output (they are always equal on success).
///
/// # Arguments
///
/// * `db_path` - Path to the SQLite database
/// * `count_threshold` - N: prune candidates with `retrieval_count < N`
/// * `age_threshold_days` - T: prune candidates with age (from `created_at`)
///   greater than T days
/// * `json` - If true, output JSON; otherwise human-readable
///
/// # Errors
///
/// Returns error if the database cannot be opened or the prune fails.
pub fn handle_prune(
    db_path: &Path,
    count_threshold: i64,
    age_threshold_days: i64,
    json: bool,
) -> Result<ExitCode, Error> {
    // Open database
    let mut db = Database::open(db_path).map_err(|e| {
        let err_msg = e.to_string();
        if err_msg.contains("database is locked") {
            return Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            );
        }
        Error::Config(err_msg)
    })?;

    // Set busy timeout to 0ms for fast-fail behavior on database locks
    wrap_busy(db.set_busy_timeout(StdDuration::ZERO).map_err(Error::from))?;

    let rows_before = count_total_rows(&db)?;

    let evaluator = PruneEvaluator::new(count_threshold, age_threshold_days);

    // Run per project: the demotion path (Database::update) is project-scoped.
    let project_ids = wrap_busy(db.list_all_project_ids().map_err(Error::from))?;
    let mut demoted: Vec<String> = Vec::new();
    for project_id in &project_ids {
        let demoted_here = wrap_busy(evaluator.run_for_project(&mut db, project_id))?;
        demoted.extend(demoted_here);
    }

    let rows_after = count_total_rows(&db)?;

    let response = PruneResponse {
        demoted: demoted.len(),
        demoted_ids: demoted,
        rows_before,
        rows_after,
    };

    if json {
        crate::output::print_json(&response);
    } else {
        println!(
            "Pruned {} candidate memory(ies) to deprecated (N={}, T={}d); {} row(s) in database before, {} after",
            response.demoted,
            count_threshold,
            age_threshold_days,
            response.rows_before,
            response.rows_after,
        );
    }

    Ok(ExitCode::SUCCESS)
}

/// Wrap a database error, converting SQLITE_BUSY into the actionable MCP-server message.
fn wrap_busy<T>(result: Result<T, Error>) -> Result<T, Error> {
    match result {
        Ok(v) => Ok(v),
        Err(Error::SqliteModule(msg)) if msg.contains("database is locked") => {
            Err(Error::Config(
                "Database is locked. Another process (likely the MCP server) is holding a lock. Stop the MCP server and retry.".to_string()
            ))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::crud::test_fake_embedder;

    fn create_test_db() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        Database::open(&path).unwrap();
        (dir, path)
    }

    fn count_rows(db: &Database) -> usize {
        db.conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap() as usize
    }

    fn status_of(db: &Database, id: &str) -> String {
        db.conn()
            .query_row("SELECT status FROM memories WHERE id = ?", [id], |row| {
                row.get::<_, String>(0)
            })
            .unwrap()
    }

    fn type_of(db: &Database, id: &str) -> String {
        db.conn()
            .query_row("SELECT type FROM memories WHERE id = ?", [id], |row| {
                row.get::<_, String>(0)
            })
            .unwrap()
    }

    /// Insert a row with a known `created_at` (age controllable) and telemetry
    /// controllable via a follow-up UPDATE.
    fn seed(
        db: &Database,
        project_id: &str,
        content: &str,
        created_at: &str,
        memory_type: &str,
        status: &str,
    ) -> String {
        let emb = test_fake_embedder(content).unwrap();
        db.insert_with_time(
            project_id,
            content,
            &emb,
            None,
            created_at,
            created_at,
            memory_type,
            status,
        )
        .unwrap()
    }

    fn set_retrieval(db: &Database, id: &str, count: i64) {
        db.conn()
            .execute(
                "UPDATE memories SET retrieval_count = ? WHERE id = ?",
                (count, id),
            )
            .unwrap();
    }

    fn set_importance(db: &Database, id: &str, importance: &str) {
        db.conn()
            .execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (importance, id),
            )
            .unwrap();
    }

    /// True when the *already-opened* `db` has an `importance` column.
    #[allow(dead_code)] // used only when the importance column exists (sub-issue 2)
    fn has_importance_column_open(db: &Database) -> bool {
        db.conn()
            .prepare("SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'importance'")
            .ok()
            .and_then(|mut stmt| stmt.query_row([], |r| r.get::<_, i64>(0)).ok())
            .map(|c| c > 0)
            .unwrap_or(false)
    }

    const OLD: &str = "2020-01-01T00:00:00Z"; // ~5+ years ago → always age > T for sane T
    /// Recent timestamp: a few minutes before "now" (computed at call time).
    /// Using a fixed date is unreliable because the test must be recent
    /// *relative to the moment the test runs*.
    fn recent_rfc3339() -> String {
        (Utc::now() - chrono::Duration::minutes(5)).to_rfc3339()
    }

    fn eval_small() -> PruneEvaluator {
        // N=3, T=7 days: default-ish small thresholds for tests.
        PruneEvaluator::new(3, 7)
    }

    #[test]
    fn test_prune_demotes_eligible_candidate() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let id = seed(&db, "p", "stale candidate", OLD, "fact", "candidate");
        set_retrieval(&db, &id, 1); // 1 < 3

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert_eq!(demoted, vec![id.clone()]);
        assert_eq!(status_of(&db, &id), "deprecated");
    }

    #[test]
    fn test_prune_never_deletes_row_count_unchanged() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();

        // Seed a corpus with a mix of eligible and non-eligible rows.
        let eligible1 = seed(&db, "p", "eligible one", OLD, "fact", "candidate");
        set_retrieval(&db, &eligible1, 0);
        let eligible2 = seed(&db, "p", "eligible two", OLD, "preference", "candidate");
        set_retrieval(&db, &eligible2, 2);
        let too_hot = seed(&db, "p", "too hot", OLD, "fact", "candidate");
        set_retrieval(&db, &too_hot, 5); // 5 >= 3 → not eligible
        let too_new = seed(&db, "p", "too new", &recent_rfc3339(), "fact", "candidate");
        set_retrieval(&db, &too_new, 0); // age <= 7d → not eligible
        let active = seed(&db, "p", "active row", OLD, "fact", "active");
        set_retrieval(&db, &active, 0);

        let before = count_rows(&db);
        assert_eq!(before, 5);

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        let after = count_rows(&db);
        assert_eq!(
            before, after,
            "prune must never delete: row count must be unchanged"
        );
        assert_eq!(demoted, vec![eligible1.clone(), eligible2.clone()]);

        // Only status changed for the demoted rows.
        assert_eq!(status_of(&db, &eligible1), "deprecated");
        assert_eq!(status_of(&db, &eligible2), "deprecated");
        // Non-eligible rows keep their status.
        assert_eq!(status_of(&db, &too_hot), "candidate");
        assert_eq!(status_of(&db, &too_new), "candidate");
        assert_eq!(status_of(&db, &active), "active");
    }

    #[test]
    fn test_prune_guard_type_never_demoted() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let id = seed(&db, "p", "guard memory", OLD, "guard", "candidate");
        set_retrieval(&db, &id, 0); // fully eligible on count + age, but guard

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert!(demoted.is_empty(), "guard-type must never be demoted");
        assert_eq!(status_of(&db, &id), "candidate");
        assert_eq!(count_rows(&db), 1);
    }

    #[test]
    fn test_prune_superseded_never_demoted() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let id = seed(&db, "p", "superseded memory", OLD, "fact", "superseded");
        set_retrieval(&db, &id, 0);

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert!(demoted.is_empty(), "superseded rows are never demoted");
        assert_eq!(status_of(&db, &id), "superseded");
    }

    #[test]
    fn test_prune_deprecated_never_demoted_again() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let id = seed(&db, "p", "already deprecated", OLD, "fact", "deprecated");
        set_retrieval(&db, &id, 0);

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert!(demoted.is_empty());
        assert_eq!(status_of(&db, &id), "deprecated");
    }

    #[test]
    fn test_prune_count_threshold_boundary() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        // N=3: retrieval_count 2 is eligible (< 3), retrieval_count 3 is not.
        let below = seed(&db, "p", "count two", OLD, "fact", "candidate");
        set_retrieval(&db, &below, 2);
        let at = seed(&db, "p", "count three", OLD, "fact", "candidate");
        set_retrieval(&db, &at, 3);

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert_eq!(
            demoted,
            vec![below.clone()],
            "retrieval_count < N is strict"
        );
        assert_eq!(status_of(&db, &below), "deprecated");
        assert_eq!(status_of(&db, &at), "candidate");
    }

    #[test]
    fn test_prune_age_boundary() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        // T = 7 days. A row created 3 days before "now" is not old enough.
        let recent = Utc::now() - chrono::Duration::days(3);
        let recent_rfc = recent.to_rfc3339();
        let id = seed(&db, "p", "recent enough", &recent_rfc, "fact", "candidate");
        set_retrieval(&db, &id, 0);

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert!(
            demoted.is_empty(),
            "age must strictly exceed T to be eligible"
        );
        assert_eq!(status_of(&db, &id), "candidate");
    }

    #[test]
    fn test_prune_is_idempotent_second_run_zero_changes() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let id = seed(&db, "p", "stale", OLD, "fact", "candidate");
        set_retrieval(&db, &id, 0);

        let ev = eval_small();
        let first = ev.run_for_project(&mut db, "p").unwrap();
        assert_eq!(first, vec![id.clone()]);

        let second = ev.run_for_project(&mut db, "p").unwrap();
        assert_eq!(
            second,
            Vec::<String>::new(),
            "second run must demote nothing"
        );
        assert_eq!(status_of(&db, &id), "deprecated");
    }

    #[test]
    fn test_prune_multiple_projects() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let a = seed(&db, "a", "stale a", OLD, "fact", "candidate");
        set_retrieval(&db, &a, 0);
        let b = seed(&db, "b", "stale b", OLD, "fact", "candidate");
        set_retrieval(&db, &b, 0);

        let ev = eval_small();
        let mut all = Vec::new();
        for p in ["a", "b"] {
            all.extend(ev.run_for_project(&mut db, p).unwrap());
        }
        assert_eq!(all.len(), 2);
        assert!(all.contains(&a));
        assert!(all.contains(&b));
    }

    #[test]
    fn test_prune_defaults_are_3_and_14() {
        assert_eq!(DEFAULT_PRUNE_COUNT, 3);
        assert_eq!(DEFAULT_PRUNE_AGE_DAYS, 14);
    }

    #[test]
    fn test_prune_response_serializes() {
        let response = PruneResponse {
            demoted: 2,
            demoted_ids: vec!["id-1".to_string(), "id-2".to_string()],
            rows_before: 10,
            rows_after: 10,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"demoted\":2"));
        assert!(json.contains("\"rows_before\":10"));
        assert!(json.contains("\"rows_after\":10"));
    }

    #[test]
    fn test_prune_importance_high_never_demoted() {
        // Requires the `importance` column, which is added by a sibling epic
        // sub-issue (sub-issue 2). Skipped when the column is not yet present
        // in the test database so this workstream is green both before and
        // after the migration lands.
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        if !has_importance_column_open(&db) {
            return;
        }
        let id = seed(&db, "p", "high importance", OLD, "fact", "candidate");
        set_retrieval(&db, &id, 0);
        set_importance(&db, &id, "high");

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert!(
            demoted.is_empty(),
            "high-importance rows are a HARD exclusion — never demoted"
        );
        assert_eq!(status_of(&db, &id), "candidate");
    }

    #[test]
    fn test_prune_importance_critical_never_demoted() {
        // Requires the `importance` column (sibling sub-issue 2); skipped
        // until the migration lands.
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        if !has_importance_column_open(&db) {
            return;
        }
        let id = seed(&db, "p", "critical importance", OLD, "fact", "candidate");
        set_retrieval(&db, &id, 0);
        set_importance(&db, &id, "critical");

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert!(
            demoted.is_empty(),
            "critical-importance rows are a HARD exclusion — never demoted"
        );
        assert_eq!(status_of(&db, &id), "candidate");
    }

    #[test]
    fn test_prune_importance_medium_and_low_still_eligible() {
        // Requires the `importance` column (sibling sub-issue 2); skipped
        // until the migration lands.
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        if !has_importance_column_open(&db) {
            return;
        }
        let med = seed(&db, "p", "medium", OLD, "fact", "candidate");
        set_retrieval(&db, &med, 0);
        set_importance(&db, &med, "medium");
        let low = seed(&db, "p", "low", OLD, "fact", "candidate");
        set_retrieval(&db, &low, 0);
        set_importance(&db, &low, "low");

        let ev = eval_small();
        let demoted = ev.run_for_project(&mut db, "p").unwrap();

        assert_eq!(
            demoted.len(),
            2,
            "medium and low importance remain eligible"
        );
    }

    #[test]
    fn test_evaluator_new_uses_duration() {
        let ev = PruneEvaluator::new(5, 30);
        assert_eq!(ev.count_threshold, 5);
        assert_eq!(ev.age_threshold, StdDuration::from_secs(30 * 86_400));
    }

    #[test]
    fn test_type_unchanged_after_prune() {
        let (_dir, db_path) = create_test_db();
        let mut db = Database::open(&db_path).unwrap();
        let id = seed(&db, "p", "type check", OLD, "procedure", "candidate");
        set_retrieval(&db, &id, 0);

        let ev = eval_small();
        ev.run_for_project(&mut db, "p").unwrap();

        assert_eq!(
            type_of(&db, &id),
            "procedure",
            "prune must not touch memory_type"
        );
        assert_eq!(status_of(&db, &id), "deprecated");
    }
}
