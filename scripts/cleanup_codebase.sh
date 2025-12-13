#!/bin/bash
# Codebase Cleanup Script
# Review each section before running!
# 
# SAFETY: Run with --dry-run first to see what would be deleted
# Usage: ./scripts/cleanup_codebase.sh [--dry-run]

set -e
cd "$(dirname "$0")/.."

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No files will be deleted"
fi

delete_files() {
    if [ "$DRY_RUN" = true ]; then
        echo "  Would delete: $@"
    else
        rm -rf "$@"
        echo "  Deleted: $@"
    fi
}

echo "=============================================="
echo "CODEBASE CLEANUP"
echo "=============================================="
echo ""

# ============================================
# 1. DELETE: Intermediate experiment files
# ============================================
echo "📁 [1/7] Intermediate experiment files (39 files)"
for f in data/experiments/*.intermediate.json; do
    [ -f "$f" ] && delete_files "$f"
done

# ============================================
# 2. DELETE: Old backup
# ============================================
echo ""
echo "📁 [2/7] Old processed backup"
[ -d "data/processed_old_backup" ] && delete_files "data/processed_old_backup"

# ============================================
# 3. DELETE: Python cache
# ============================================
echo ""
echo "📁 [3/7] Python cache files"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "  Cleaned __pycache__, *.pyc, .pytest_cache"

# ============================================
# 4. DELETE: LaTeX build artifacts
# ============================================
echo ""
echo "📁 [4/7] LaTeX build artifacts"
for ext in aux log out fls fdb_latexmk synctex.gz bbl blg; do
    for f in paper/*.${ext}; do
        [ -f "$f" ] && delete_files "$f"
    done
done

# ============================================
# 5. DELETE: Log files
# ============================================
echo ""
echo "📁 [5/7] Log files"
[ -d "logs" ] && delete_files "logs"
for f in data/experiments/*.log data/experiments/*.txt; do
    [ -f "$f" ] && delete_files "$f"
done

# ============================================
# 6. ARCHIVE: Old experiments (keep only final_* and recent)
# ============================================
echo ""
echo "📁 [6/7] Archiving old experiments"
if [ "$DRY_RUN" = false ]; then
    mkdir -p data/experiments/archive
fi
# Keep: final_exp*, exp_text_caption_improved*, exp_text_caption_filtered*
# Archive: exp0-13, validation_*, exp_metrics_test, exp_validate_fixes, exp_test_*
for f in data/experiments/exp[0-9]*.json \
         data/experiments/validation_*.json \
         data/experiments/exp_metrics_test.json \
         data/experiments/exp_validate_fixes.json \
         data/experiments/exp_test_*.json; do
    if [ -f "$f" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  Would archive: $f"
        else
            mv "$f" data/experiments/archive/
            echo "  Archived: $f"
        fi
    fi
done

# ============================================
# 7. ARCHIVE: Debug/test configs (keep production ones)
# ============================================
echo ""
echo "📁 [7/7] Archiving debug configs"
if [ "$DRY_RUN" = false ]; then
    mkdir -p config/archive
fi
# Keep: queries_50_mixed.json (main), queries_validation_25.json
# Archive: debug, test, single, subset configs
for f in config/debug_*.json \
         config/query_debug.json \
         config/query_single.json \
         config/test_*.json \
         config/*_subset.json \
         config/ablation_*.json; do
    if [ -f "$f" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  Would archive: $f"
        else
            mv "$f" config/archive/
            echo "  Archived: $f"
        fi
    fi
done

echo ""
echo "=============================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE - No changes made"
    echo "Run without --dry-run to apply changes"
else
    echo "CLEANUP COMPLETE"
fi
echo "=============================================="

# Show final structure
echo ""
echo "📊 Final experiment files:"
ls -1 data/experiments/*.json 2>/dev/null | grep -v archive | head -10 || echo "  (none)"
echo ""
echo "📊 Final config files:"
ls -1 config/*.json 2>/dev/null | grep -v archive || echo "  (none)"

