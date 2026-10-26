//! Drift lock for the Breaking Changes changelog rule (issue #285).
//!
//! `cliff.toml` routes breaking commits by *parser position* — git-cliff stops
//! at the first matching parser — and release-plz only reads that file because
//! `release-plz.toml` names it. Both facts are one careless edit away from a
//! changelog that silently drops breaking changes, and no compiler is involved,
//! so the lock is a test. Like `tests/msrv_drift.rs` it reads config only.
//!
//! `toml::from_str` (not `str::parse::<toml::Value>`, which parses a TOML
//! *value*, not a document) deserialises the file into `toml::Value`; arrays
//! keep source order, so index 0 is the first parser.

#[test]
fn test_cliff_config_keeps_breaking_commits_in_their_own_group() {
    let read = |name: &str| {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(name))
            .unwrap_or_else(|e| panic!("read {name}: {e}"))
    };

    let cliff: toml::Value = toml::from_str(&read("cliff.toml")).expect("cliff.toml is valid TOML");
    assert_eq!(
        cliff["git"]["protect_breaking_commits"].as_bool(),
        Some(true),
        "git.protect_breaking_commits must stay true, or a skip parser can drop a \
         breaking commit before it reaches its group"
    );

    let parsers = cliff["git"]["commit_parsers"]
        .as_array()
        .expect("git.commit_parsers must be an array");
    let first = &parsers[0];
    assert_eq!(first["field"].as_str(), Some("breaking"));
    assert_eq!(first["pattern"].as_str(), Some("^true$"));
    let group = first["group"].as_str().expect("first parser needs a group");
    assert!(
        group.starts_with("<!-- 0 -->") && group.contains("Breaking Changes"),
        "the first parser's group must be the Breaking Changes group, sorted first by \
         its `<!-- 0 -->` prefix (striptags removes it from the heading), got {group:?}"
    );

    // Every setting lives inside release-plz's `[workspace]` table.
    let plz: toml::Value =
        toml::from_str(&read("release-plz.toml")).expect("release-plz.toml is valid TOML");
    assert_eq!(
        plz["workspace"]["changelog_config"].as_str(),
        Some("cliff.toml"),
        "release-plz must generate the changelog from cliff.toml, else the config above \
         is dead and breaking commits land wherever release-plz chooses"
    );
}
