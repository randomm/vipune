mod embedding;
mod errors;
mod memory;
mod project;
mod sqlite;

use clap::Parser;

/// vipune - A minimal memory layer for AI agents
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    // Placeholder for CLI arguments
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        println!("vipune initialized");
    }
}

#[cfg(test)]
mod test_utils {
    use tempfile::TempDir;

    pub fn test_db() -> crate::sqlite::Database {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = crate::sqlite::Database::open(&path).unwrap();
        std::mem::forget(dir);
        db
    }

    pub fn test_embedding(value: f32) -> Vec<f32> {
        vec![value; 384]
    }

    pub fn orthogonal_embeddings() -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0f32; 384];
        let mut b = vec![0.0f32; 384];
        a[0] = 1.0;
        b[1] = 1.0;
        (a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlite::{blob_to_vec, cosine_similarity, vec_to_blob, Database, Error};

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::parse_from(&["vipune", "--verbose"]);
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_default() {
        let cli = Cli::parse_from(&["vipune"]);
        assert!(!cli.verbose);
    }

    #[test]
    fn test_blob_size() {
        let embedding = test_utils::test_embedding(0.5);
        let blob = vec_to_blob(&embedding).unwrap();
        assert_eq!(blob.len(), 1536);
    }

    #[test]
    fn test_blob_round_trip() {
        let original = test_utils::test_embedding(0.123);
        let blob = vec_to_blob(&original).unwrap();
        let decoded = blob_to_vec(&blob).unwrap();
        assert_eq!(original.len(), decoded.len());
        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!((o - d).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_invalid_blob_size() {
        let too_short = vec![0u8; 10];
        let result = blob_to_vec(&too_short);
        assert!(matches!(result, Err(Error::InvalidBlobSize { .. })));
    }

    #[test]
    fn test_invalid_embedding_length() {
        let wrong_length = vec![0.5f32; 100]; // Should be 384
        let result = vec_to_blob(&wrong_length);
        assert!(matches!(result, Err(Error::MismatchedDimensions { .. })));
    }

    #[test]
    fn test_cosine_similarity_nan() {
        let mut a = vec![0.0f32; 384];
        a[0] = f32::NAN;
        let b = vec![1.0f32; 384];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(Error::InvalidEmbedding(_))));
    }

    #[test]
    fn test_cosine_similarity_infinity() {
        let mut a = vec![1.0f32; 384];
        a[0] = f32::INFINITY;
        let b = vec![1.0f32; 384];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(Error::InvalidEmbedding(_))));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = test_utils::test_embedding(1.0);
        let b = test_utils::test_embedding(1.0);
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let (a, b) = test_utils::orthogonal_embeddings();
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = test_utils::test_embedding(1.0);
        let b = test_utils::test_embedding(-1.0);
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty_vector() {
        let a = vec![];
        let b = vec![1.0f32];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(Error::EmptyVector)));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0f32; 384];
        let b = vec![1.0f32; 384];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_insert_and_get() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.5);
        let id = db
            .insert("proj-1", "test content", &embedding, Some("meta"))
            .unwrap();

        let memory = db.get(&id).unwrap().unwrap();
        assert_eq!(memory.id, id);
        assert_eq!(memory.project_id, "proj-1");
        assert_eq!(memory.content, "test content");
        assert_eq!(memory.metadata, Some("meta".to_string()));
    }

    #[test]
    fn test_insert_without_metadata() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.7);
        let id = db
            .insert("proj-2", "no metadata content", &embedding, None)
            .unwrap();

        let memory = db.get(&id).unwrap().unwrap();
        assert_eq!(memory.metadata, None);
    }

    #[test]
    fn test_get_nonexistent() {
        let db = test_utils::test_db();
        let result = db.get("does-not-exist").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_list_empty() {
        let db = test_utils::test_db();
        let memories = db.list("proj-1", 10).unwrap();
        assert!(memories.is_empty());
    }

    #[test]
    fn test_list_with_data() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.3);

        db.insert("proj-1", "first", &embedding, None).unwrap();
        db.insert("proj-1", "second", &embedding, None).unwrap();
        db.insert("proj-2", "should not appear", &embedding, None)
            .unwrap();

        let memories = db.list("proj-1", 10).unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].content, "second");
        assert_eq!(memories[1].content, "first");
    }

    #[test]
    fn test_list_limit() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.9);

        for i in 0..10 {
            db.insert("proj-1", &format!("item {}", i), &embedding, None)
                .unwrap();
        }

        let memories = db.list("proj-1", 5).unwrap();
        assert_eq!(memories.len(), 5);
    }

    #[test]
    fn test_invalid_limit_zero() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.5);

        db.insert("proj-1", "test", &embedding, None).unwrap();

        let result = db.search("proj-1", &embedding, 0);
        assert!(matches!(result, Err(Error::InvalidLimit(_))));
    }

    #[test]
    fn test_invalid_limit_too_large() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.5);

        db.insert("proj-1", "test", &embedding, None).unwrap();

        let result = db.search("proj-1", &embedding, 10_001);
        assert!(matches!(result, Err(Error::InvalidLimit(_))));
    }

    #[test]
    fn test_update() {
        let db = test_utils::test_db();
        let embedding1 = test_utils::test_embedding(0.2);
        let embedding2 = test_utils::test_embedding(0.8);

        let id = db
            .insert("proj-1", "original content", &embedding1, None)
            .unwrap();

        db.update(&id, "updated content", &embedding2).unwrap();

        let memory = db.get(&id).unwrap().unwrap();
        assert_eq!(memory.content, "updated content");
        assert_ne!(memory.created_at, memory.updated_at);
    }

    #[test]
    fn test_delete() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.6);

        let id = db.insert("proj-1", "to delete", &embedding, None).unwrap();
        assert!(db.delete(&id).unwrap());

        let memory = db.get(&id).unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let db = test_utils::test_db();
        let result = db.delete("does-not-exist").unwrap();
        assert!(!result);
    }

    #[test]
    fn test_search() {
        let db = test_utils::test_db();
        let embedding_a = test_utils::test_embedding(1.0);
        let embedding_b = test_utils::test_embedding(0.0);

        let id_a = db.insert("proj-1", "match A", &embedding_a, None).unwrap();
        let _id_b = db.insert("proj-1", "match B", &embedding_b, None).unwrap();

        let results = db.search("proj-1", &embedding_a, 5).unwrap();
        assert_eq!(results.len(), 2);

        let top = &results[0];
        assert_eq!(top.id, id_a);
        assert!((top.similarity.unwrap() - 1.0).abs() < 1e-6);
        assert!(top.similarity.unwrap() >= results[1].similarity.unwrap());
    }

    #[test]
    fn test_search_negative_similarity() {
        let db = test_utils::test_db();
        let embedding_pos = test_utils::test_embedding(1.0);
        let embedding_neg = test_utils::test_embedding(-1.0);

        db.insert("proj-1", "positive", &embedding_pos, None)
            .unwrap();
        let id_neg = db
            .insert("proj-1", "negative", &embedding_neg, None)
            .unwrap();

        let results = db.search("proj-1", &embedding_pos, 10).unwrap();
        assert_eq!(results.len(), 2);

        let negative_result = results.iter().find(|m| m.id == id_neg).unwrap();
        assert!((negative_result.similarity.unwrap() - (-1.0)).abs() < 1e-6);

        assert!(results[0].similarity.unwrap() > results[1].similarity.unwrap());
    }

    #[test]
    fn test_search_project_filter() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.5);

        db.insert("proj-1", "from proj 1", &embedding, None)
            .unwrap();
        db.insert("proj-2", "from proj 2", &embedding, None)
            .unwrap();

        let results = db.search("proj-1", &embedding, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project_id, "proj-1");
    }

    #[test]
    fn test_search_limit() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.5);

        for i in 0..10 {
            db.insert("proj-1", &format!("item {}", i), &embedding, None)
                .unwrap();
        }

        let results = db.search("proj-1", &embedding, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_find_similar() {
        let db = test_utils::test_db();
        let embedding_exact = test_utils::test_embedding(1.0);
        let embedding_dissimilar = test_utils::test_embedding(-1.0);
        let embedding_orthogonal = {
            let mut v = vec![0.0f32; 384];
            v[0] = 1.0;
            v
        };

        db.insert("proj-1", "exact match", &embedding_exact, None)
            .unwrap();
        db.insert("proj-1", "dissimilar", &embedding_dissimilar, None)
            .unwrap();
        db.insert("proj-1", "orthogonal", &embedding_orthogonal, None)
            .unwrap();

        let results = db.find_similar("proj-1", &embedding_exact, 0.8).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "exact match");
    }

    #[test]
    fn test_find_similar_no_results() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.5);

        db.insert("proj-1", "some memory", &embedding, None)
            .unwrap();

        let results = db
            .find_similar("proj-1", &test_utils::test_embedding(0.0), 0.95)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts_trigger_on_insert() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.4);

        db.insert("proj-1", "searchable content", &embedding, None)
            .unwrap();

        let count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_fts_trigger_on_update() {
        let db = test_utils::test_db();
        let embedding_old = test_utils::test_embedding(0.3);
        let embedding_new = test_utils::test_embedding(0.8);

        let id = db
            .insert("proj-1", "old content", &embedding_old, None)
            .unwrap();

        db.update(&id, "new content", &embedding_new).unwrap();

        let content: String = db
            .conn
            .query_row("SELECT content FROM memories_fts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(content, "new content");
    }

    #[test]
    fn test_fts_trigger_on_delete() {
        let db = test_utils::test_db();
        let embedding = test_utils::test_embedding(0.6);

        let id = db
            .insert("proj-1", "to be deleted", &embedding, None)
            .unwrap();
        db.delete(&id).unwrap();

        let count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_database_reopen() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");

        {
            let db = Database::open(&path).unwrap();
            let embedding = test_utils::test_embedding(0.5);
            db.insert("proj-1", "persistent", &embedding, None).unwrap();
        }

        {
            let db = Database::open(&path).unwrap();
            let memories = db.list("proj-1", 10).unwrap();
            assert_eq!(memories.len(), 1);
            assert_eq!(memories[0].content, "persistent");
        }
    }

    #[test]
    fn test_error_display() {
        let err = Error::InvalidBlobSize {
            expected: 1536,
            actual: 10,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("expected"));
        assert!(msg.contains("1536"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn test_embedding_dims_constant() {
        let embedding = test_utils::test_embedding(0.5);
        assert_eq!(embedding.len(), 384);
    }
}
