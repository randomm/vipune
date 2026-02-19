# Remory to Vipune Migration Guide

> **⚠️ TEMPORARY DOCUMENT**: This guide is for one-time migration only. The import feature will be removed from vipune after migrations are complete.

## Prerequisites

1. Build and install vipune:
   ```bash
   cargo build --release
   cp target/release/vipune ~/.cargo/bin/
   ```

2. Locate your remory database (usually `~/.remory/remory.db`)

## Migration Steps

### Step 1: Import remory memories

```bash
vipune import ~/.remory/remory.db
```

This will:
- Import all memories from remory
- Compute ONNX embeddings (takes ~30-50 minutes for ~20,000 memories)
- Preserve remory's UUID project IDs

**Note**: The import will report duplicate memories. This is expected — vipune's conflict detection skips memories that already exist.

### Step 2: Fix project IDs (CRITICAL)

**Problem**: remory uses UUID project IDs, but vipune auto-detects git-URL-style IDs. Imported memories won't be visible in project directories until you fix the project IDs.

**Solution**: Run SQL UPDATEs to map UUIDs to git-URLs:

```bash
# Map UUIDs to git-URL project IDs
sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/fiona' WHERE project_id = 'ed11c8dc-2a40-5593-87a0-d66eb55a1190';"
sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/fiona' WHERE project_id = 'a9e91972-3b42-53dc-83e5-994e91edbbc7';"
sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/fiona' WHERE project_id = '812cf80f-5cc1-5305-b984-69db1563a22b';"

sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/vipune' WHERE project_id = 'c0c3f980-d339-5d2b-8506-8b912a462e88';"

sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/remory' WHERE project_id = 'dfe10a20-0016-5357-95dd-b9fe842e403a';"

sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/oh-my-singularity' WHERE project_id = 'e381fac2-3bde-5de9-8b1f-1bbc6b75f791';"

sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/opencode' WHERE project_id = 'e18fc021-6f0b-53ca-bbe4-f29071108c89';"
sqlite3 "$HOME/.vipune/memories.db" "UPDATE memories SET project_id = 'randomm/opencode' WHERE project_id = 'b76438ab-9949-5822-a40e-8d565abfa71e';"
```

### Step 3: Verify

Test that memories are accessible in project directories:

```bash
cd ~/projects/fiona
vipune list --limit 5
vipune search "handler"

cd ~/projects/opencode  
vipune list --limit 5
vipune search "agent"
```

## Project Mapping Reference

| Remory UUID | Vipune Project | Memories |
|-------------|----------------|----------|
| `ed11c8dc-2a40-5593-87a0-d66eb55a1190` | randomm/fiona | 5,134 |
| `a9e91972-3b42-53dc-83e5-994e91edbbc7` | randomm/fiona | 12 |
| `812cf80f-5cc1-5305-b984-69db1563a22b` | randomm/fiona | 4 |
| `c0c3f980-d339-5d2b-8506-8b912a462e88` | randomm/vipune | 1,316 |
| `dfe10a20-0016-5357-95dd-b9fe842e403a` | randomm/remory | 1,196 |
| `e381fac2-3bde-5de9-8b1f-1bbc6b75f791` | randomm/oh-my-singularity | 806 |
| `e18fc021-6f0b-53ca-bbe4-f29071108c89` | randomm/opencode | 19 |
| `b76438ab-9949-5822-a40e-8d565abfa71e` | randomm/opencode | 1 |

## Troubleshooting

**Empty results after import**: The project IDs haven't been updated yet. Run the SQL UPDATEs in Step 2.

**"No such file or directory"**: Check that `~/.remory/remory.db` exists and vipune is installed at `~/.cargo/bin/vipune`.

**Import takes forever**: Expected — each memory needs ONNX embedding computation (~2-3 seconds per memory).

## After Migration

Once migration is complete on all machines, the `vipune import` command will be removed from the codebase.

---

**Last updated**: February 2026  
**Status**: Temporary — will be removed after migrations complete
