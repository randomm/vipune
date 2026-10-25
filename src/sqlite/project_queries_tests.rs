//! Tests for the project-scan aggregate in `src/sqlite/project_queries.rs`.
//!
//! `project_queries.rs` was split out of `sqlite/mod.rs` to stay under the
//! 500-line cap, and its two helpers had no direct consumer test — the
//! handlers reached them through `commands/doctor.rs` only, so the aggregate's
//! documented zero-row behaviour could drift unnoticed (issue #255 review).
#[cfg(test)]
mod tests {
    use crate::sqlite::Database;
    use std::collections::HashMap;

    fn open_db() -> Database {
        Database::open(std::path::Path::new(":memory:")).unwrap()
    }

    fn insert_rows(db: &Database, project_id: &str, n: usize) {
        for i in 0..n {
            db.insert(
                project_id,
                &format!("{project_id} row {i}"),
                &vec![0.1f32; 384],
                None,
                "fact",
                "active",
            )
            .unwrap();
        }
    }

    #[test]
    fn row_counts_by_project_omits_zero_row_projects() {
        // The documented contract: a project with zero `memories` rows does not
        // appear at all, because `GROUP BY` emits only groups that exist. The
        // deleted per-project `count_rows_for_project` helper reported such a
        // project as `Some(0)` when handed its id, so the difference is real and
        // callers must not be surprised by it.
        let db = open_db();
        insert_rows(&db, "with-rows", 3);
        // A project that exists only in the dedup index / never inserted: the
        // aggregate can only know about projects that have rows.
        let counts: HashMap<String, usize> =
            db.row_counts_by_project().unwrap().into_iter().collect();
        assert_eq!(counts.get("with-rows"), Some(&3));
        assert_eq!(
            counts.len(),
            1,
            "only the project with rows is listed: {counts:?}"
        );
        assert!(
            !counts.contains_key("never-inserted"),
            "a zero-row project must be absent, got {counts:?}"
        );
    }

    #[test]
    fn row_counts_by_project_uses_the_project_index() {
        // The aggregate must stay on `idx_memories_project` rather than fall
        // back to a full-table scan (the query pins it with `INDEXED BY`; this
        // asserts the plan so a future edit that drops the hint — or drops the
        // index — fails loudly).
        let db = open_db();
        insert_rows(&db, "indexed", 2);
        let mut stmt = db
            .conn()
            .prepare("EXPLAIN QUERY PLAN SELECT project_id, COUNT(*) FROM memories INDEXED BY idx_memories_project GROUP BY project_id")
            .unwrap();
        let plan: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        let joined = plan.join(" | ");
        assert!(
            joined.contains("idx_memories_project"),
            "the aggregate must be satisfied by the project index, plan was: {joined}"
        );
    }

    #[test]
    fn list_all_project_ids_matches_the_aggregate_keys() {
        // The two helpers must agree on the project set for a populated store:
        // `list_all_project_ids` is the DISTINCT pass and `row_counts_by_project`
        // is the aggregate, and callers mix them (doctor --projects uses both).
        let db = open_db();
        insert_rows(&db, "proj-b", 1);
        insert_rows(&db, "proj-a", 2);
        let listed = db.list_all_project_ids().unwrap();
        let counted: Vec<String> = {
            let mut v: Vec<String> = db
                .row_counts_by_project()
                .unwrap()
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            v.sort();
            v
        };
        assert_eq!(listed, vec!["proj-a".to_string(), "proj-b".to_string()]);
        assert_eq!(counted, listed, "both helpers enumerate the same projects");
    }
}
