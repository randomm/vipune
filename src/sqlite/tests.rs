#[cfg(test)]
mod tests {
    #[path = "crud_tests.rs"]
    mod crud;
    #[path = "migration_tests.rs"]
    mod migration;
}
