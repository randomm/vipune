//! Content hashing for dedup (issue #191).
//!
//! Provides a single shared, dependency-free 64-bit FNV-1a hash over
//! normalised content (lowercased, whitespace-collapsed). Used by:
//! - `Database::insert_with_hash` (the hook insert path)
//! - The dedup migration backfill (issue #191, task-c)
//!
//! The hash function and the normalisation step must remain identical
//! across all call sites or dedup will silently break.

/// Normalise content for hashing: lowercase, collapse all whitespace
/// runs (spaces, tabs, newlines) into single spaces, trim the result.
#[allow(dead_code)] // used by hook path (task-b) and migration backfill (task-c)
pub fn normalize_content(content: &str) -> String {
    let lower = content.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut in_space = false;
    for c in lower.chars() {
        if c.is_whitespace() {
            if !in_space && !out.is_empty() {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out.trim().to_string()
}

/// 64-bit FNV-1a hash of `input`, hex-encoded (16 lowercase chars).
/// No external crate — hand-rolled per project std-first philosophy.
#[allow(dead_code)] // used by content_hash; content_hash used by hook + migration
pub fn fnv1a64_hex(input: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    format!("{:016x}", hash)
}

/// Compute the dedup hash for a content string:
/// normalise (lowercase + collapse whitespace) then FNV-1a 64-bit hex.
#[allow(dead_code)] // used by hook insert path (task-b) and migration (task-c)
pub fn content_hash(content: &str) -> String {
    let normalized = normalize_content(content);
    fnv1a64_hex(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_lowercases() {
        assert_eq!(normalize_content("Hello World"), "hello world");
    }

    #[test]
    fn test_normalize_collapses_whitespace() {
        assert_eq!(normalize_content("Hello   World"), "hello world");
        assert_eq!(normalize_content("a\tb\nc\nd"), "a b c d");
    }

    #[test]
    fn test_normalize_trims_leading_trailing() {
        assert_eq!(normalize_content("  hello world  "), "hello world");
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_case_and_whitespace_insensitive() {
        assert_eq!(content_hash("Hello   World"), content_hash("hello world"));
    }

    #[test]
    fn test_content_hash_differs_for_different_content() {
        assert_ne!(content_hash("foo bar"), content_hash("bar foo"));
    }

    #[test]
    fn test_fnv1a64_hex_output_format() {
        let h = fnv1a64_hex("x");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // --- Parity test: hash.rs::content_hash vs the deleted migrations.rs::content_hash_for ---
    //
    // Both implementations used the same FNV-1a-64 algorithm over the same
    // normalised content (lowercase + whitespace-collapsed). This test pins
    // the exact hash values produced by the shared implementation so any future
    // change to `normalize_content` or `fnv1a64_hex` that breaks parity with
    // the values already stored in production databases will be caught here.
    //
    // The expected hex values were computed by the original
    // `migrations.rs::content_hash_for` before consolidation (issue #213).

    #[test]
    fn test_parity_with_deleted_content_hash_for() {
        // Each (input, expected_hash) pair was verified against the original
        // migrations.rs implementation (FNV-1a-64 over normalised content).
        // These are the exact values already stored in production databases
        // by the migration 4 backfill — any drift here means dedup breaks.
        let cases: &[(&str, &str)] = &[
            ("hello world", "779a65e7023cd2e7"),
            ("leading and trailing", "0502c48fed08ef80"),
            ("foo", "dcb27518fed9d577"),
            ("bar", "003934191339461a"),
            ("test content", "26877eb147d6f408"),
            ("same input", "b94119c1d5202297"),
            ("shared content", "74c6a290029c9309"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &content_hash(input),
                expected,
                "parity broken for input: {input:?}"
            );
        }

        // Normalisation parity: the old migrations.rs used trim_end while
        // hash.rs uses trim. Verify they produce identical output for the
        // whitespace-collapsed normalisation (the collapsing loop never
        // produces leading whitespace, so trim == trim_end here).
        assert_eq!(
            normalize_content("  leading and  trailing  "),
            normalize_content("leading and trailing"),
            "trim vs trim_end must not diverge for normalised output"
        );
    }
}
