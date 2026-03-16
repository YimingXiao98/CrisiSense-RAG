#!/usr/bin/env bash
# =============================================================================
# Experiment Runner — 2026-03-09
# =============================================================================
# Runs all experiments needed for the paper revision:
#   1. No-tweets ablation (R1-C6) — 3 models × 207 queries
#   2. Zero-shot no-retrieval baseline (R1-C3) — 3 models × 207 queries
#   3. Historical mean baseline (R1-C3) — already computed, no API calls
#   4. V2 re-fusion (improved multimodal) — already computed offline
#
# Prerequisites:
#   - conda activate harvey-rag
#   - .env file with GEMINI_API_KEY, OPENAI_API_KEY set
#
# Usage:
#   bash scripts/run_experiments_2026-03-09.sh [--dry-run] [EXPERIMENT_NAME]
#
# Examples:
#   bash scripts/run_experiments_2026-03-09.sh                # run all
#   bash scripts/run_experiments_2026-03-09.sh no_tweets      # run only no-tweets
#   bash scripts/run_experiments_2026-03-09.sh zero_shot      # run only zero-shot
#   bash scripts/run_experiments_2026-03-09.sh --dry-run      # print commands only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Must run from project root so Python can find the app module
cd "$PROJECT_ROOT"
CONFIG="$PROJECT_ROOT/config/queries_complete_map.json"
OUT_DIR="$PROJECT_ROOT/data/experiments/2026-03-10"
RUN_SCRIPT="$SCRIPT_DIR/run_baseline_experiment.py"

DRY_RUN=false
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) FILTER="$arg" ;;
    esac
done

mkdir -p "$OUT_DIR"

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="
    echo "  Command: conda run -n harvey-rag python $RUN_SCRIPT $*"
    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN — skipping]"
        return
    fi
    PYTHONPATH="$PROJECT_ROOT" conda run -n harvey-rag python "$RUN_SCRIPT" "$@"
    echo "  DONE: $name"
}

# ─── 1. No-Tweets Ablation (R1-C6) ──────────────────────────────────────────
# Tests: what happens without social media data?
# Runs for: Gemini (default), Llama, Qwen

if [ -z "$FILTER" ] || [ "$FILTER" = "no_tweets" ]; then

    run_experiment "No-Tweets / Gemini" \
        --config "$CONFIG" \
        --output "$OUT_DIR/exp_no_tweets_gemini.json" \
        --name "no_tweets_gemini" \
        --no_tweets \
        --text-model "google/gemini-2.5-flash"

    run_experiment "No-Tweets / Llama" \
        --config "$CONFIG" \
        --output "$OUT_DIR/exp_no_tweets_llama.json" \
        --name "no_tweets_llama" \
        --no_tweets \
        --text-model "meta-llama/llama-3.3-70b-instruct"

    run_experiment "No-Tweets / Qwen" \
        --config "$CONFIG" \
        --output "$OUT_DIR/exp_no_tweets_qwen.json" \
        --name "no_tweets_qwen" \
        --no_tweets \
        --text-model "qwen/qwen-2.5-72b-instruct"
fi

# ─── 2. Zero-Shot No-Retrieval Baseline (R1-C3) ─────────────────────────────
# Tests: what can the LLM do with zero retrieved context?
# Runs for: Gemini (default), Llama, Qwen

if [ -z "$FILTER" ] || [ "$FILTER" = "zero_shot" ]; then

    run_experiment "Zero-Shot / Gemini" \
        --config "$CONFIG" \
        --output "$OUT_DIR/exp_zero_shot_gemini.json" \
        --name "zero_shot_gemini" \
        --no_retrieval \
        --text-model "google/gemini-2.5-flash"

    run_experiment "Zero-Shot / Llama" \
        --config "$CONFIG" \
        --output "$OUT_DIR/exp_zero_shot_llama.json" \
        --name "zero_shot_llama" \
        --no_retrieval \
        --text-model "meta-llama/llama-3.3-70b-instruct"

    run_experiment "Zero-Shot / Qwen" \
        --config "$CONFIG" \
        --output "$OUT_DIR/exp_zero_shot_qwen.json" \
        --name "zero_shot_qwen" \
        --no_retrieval \
        --text-model "qwen/qwen-2.5-72b-instruct"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  EXPERIMENT PLAN SUMMARY"
echo "=========================================="
echo ""
echo "  Output directory: $OUT_DIR"
echo ""
echo "  Already completed (offline, no API calls):"
echo "    - exp_historical_mean_baseline.json  (global mean baseline)"
echo "    - exp_v2_gemini_multimodal.json      (V2 re-fusion: text-only flood, no visual)"
echo "    - exp_v2_llama_multimodal.json       (V2 re-fusion)"
echo "    - exp_v2_qwen_multimodal.json        (V2 re-fusion)"
echo ""
echo "  Experiments requiring API calls:"
if [ -z "$FILTER" ] || [ "$FILTER" = "no_tweets" ]; then
    echo "    - exp_no_tweets_gemini.json          (no-tweets ablation)"
    echo "    - exp_no_tweets_llama.json           (no-tweets ablation)"
    echo "    - exp_no_tweets_qwen.json            (no-tweets ablation)"
fi
if [ -z "$FILTER" ] || [ "$FILTER" = "zero_shot" ]; then
    echo "    - exp_zero_shot_gemini.json          (zero-shot no-retrieval)"
    echo "    - exp_zero_shot_llama.json           (zero-shot no-retrieval)"
    echo "    - exp_zero_shot_qwen.json            (zero-shot no-retrieval)"
fi
echo ""
echo "  Each API experiment: ~207 queries × ~\$0.01-0.05/query"
echo "  Estimated total cost: ~\$20-60 depending on model pricing"
echo ""
