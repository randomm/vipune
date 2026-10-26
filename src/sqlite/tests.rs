#[cfg(test)]
#[path = "crud_tests.rs"]
mod crud;

#[cfg(test)]
#[path = "fts_tests.rs"]
mod fts;

#[cfg(test)]
#[path = "fts_migration_tests.rs"]
mod fts_migration;

#[cfg(test)]
#[path = "fts_error_shape_tests.rs"]
mod fts_error_shape;

#[cfg(test)]
#[path = "migration_tests.rs"]
mod migration;

#[cfg(test)]
#[path = "project_queries_tests.rs"]
mod project_queries;
