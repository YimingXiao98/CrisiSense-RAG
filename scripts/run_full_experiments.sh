#!/usr/bin/env bash
# =============================================================================
# Full Experiment Runner — two configs, 4 modes × 3 models
# =============================================================================
# Experimental design:
#
#   MAIN RESULTS (Table 3) — 110-ZIP imagery-covered config:
#     All 4 modes run on the same 110 ZIPs where imagery is available.
#     Fair ablation: modes actually differ on every query.
#     Multimodal is the headline result.
#
#   SUPPLEMENTARY — 207-ZIP full config:
#     text_only only, showing broader geographic coverage.
#     Reported as a supplement to demonstrate system scalability.
#
# Modes (all on 110-ZIP config):
#   text_only     : --no_visual --no_captions
#   text_caption  : --no_visual                 (captions as imagery proxy)
#   multimodal    : --no_captions               (text + VLM; no double-counting)
#   no_tweets     : --no_captions --no_tweets   (full multimodal minus social media)
#
# Models:
#   gemini25      : models/gemini-2.5-flash     (Google AI SDK)
#   qwen35        : qwen3.5-397b               (OpenRouter)
#   gpt5mini      : openai/gpt-5-mini          (OpenRouter)
#
# Visual model: same as text model (each model uses its own vision capability)
#
# Total: 12 runs × 110 queries + 3 runs × 207 queries = 1,941 API calls
#
# Usage:
#   bash scripts/run_full_experiments.sh [--dry-run] [MODEL] [MODE]
#
# Examples:
#   bash scripts/run_full_experiments.sh                           # run all
#   bash scripts/run_full_experiments.sh --dry-run                 # print only
#   bash scripts/run_full_experiments.sh gemini25                  # all modes, gemini25
#   bash scripts/run_full_experiments.sh gemini25 multimodal       # single run
#   bash scripts/run_full_experiments.sh -- suppl                  # supplementary only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG_110="$PROJECT_ROOT/config/queries_imagery_covered_110.json"
CONFIG_97="$PROJECT_ROOT/config/queries_no_imagery_97.json"   # text_only suppl (97 ZIPs w/o imagery)
OUT_DIR="$PROJECT_ROOT/data/experiments/full_207_3-15"
RUN_SCRIPT="$SCRIPT_DIR/run_baseline_experiment.py"

# Vision model IDs — each model uses itself for visual analysis
GEMINI25_MODEL="models/gemini-2.5-flash"
QWEN35_MODEL="qwen3.5-397b"
GPT5MINI_MODEL="openai/gpt-5-mini"

DRY_RUN=false
MODEL_FILTER=""
MODE_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        gemini25|qwen35|gpt5mini) MODEL_FILTER="$arg" ;;
        text_only|text_caption|multimodal|no_tweets|suppl) MODE_FILTER="$arg" ;;
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

should_run() {
    local model="$1"
    local mode="$2"
    if [ -n "$MODEL_FILTER" ] && [ "$MODEL_FILTER" != "$model" ]; then
        return 1
    fi
    if [ -n "$MODE_FILTER" ] && [ "$MODE_FILTER" != "$mode" ]; then
        return 1
    fi
    return 0
}

# =============================================================================
# MAIN RESULTS — 110-ZIP imagery-covered config
# =============================================================================

# ── Gemini-2.5-Flash ──────────────────────────────────────────────────────────

if should_run gemini25 text_only; then
    run_experiment "Gemini-2.5-Flash / text_only [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gemini25_text_only.json" \
        --name "gemini25_text_only" \
        --no_visual --no_captions \
        --text-model "models/gemini-2.5-flash"
fi

if should_run gemini25 text_caption; then
    run_experiment "Gemini-2.5-Flash / text_caption [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gemini25_text_caption.json" \
        --name "gemini25_text_caption" \
        --no_visual \
        --text-model "models/gemini-2.5-flash"
fi

if should_run gemini25 multimodal; then
    run_experiment "Gemini-2.5-Flash / multimodal [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gemini25_multimodal.json" \
        --name "gemini25_multimodal" \
        --no_captions \
        --text-model "$GEMINI25_MODEL" \
        --vision-model "$GEMINI25_MODEL"
fi

if should_run gemini25 no_tweets; then
    run_experiment "Gemini-2.5-Flash / no_tweets [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gemini25_no_tweets.json" \
        --name "gemini25_no_tweets" \
        --no_captions --no_tweets \
        --text-model "$GEMINI25_MODEL" \
        --vision-model "$GEMINI25_MODEL"
fi

# ── Qwen3.5-397B ──────────────────────────────────────────────────────────────

if should_run qwen35 text_only; then
    run_experiment "Qwen3.5-397B / text_only [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_qwen35_text_only.json" \
        --name "qwen35_text_only" \
        --no_visual --no_captions \
        --text-model "qwen3.5-397b"
fi

if should_run qwen35 text_caption; then
    run_experiment "Qwen3.5-397B / text_caption [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_qwen35_text_caption.json" \
        --name "qwen35_text_caption" \
        --no_visual \
        --text-model "qwen3.5-397b"
fi

if should_run qwen35 multimodal; then
    run_experiment "Qwen3.5-397B / multimodal [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_qwen35_multimodal.json" \
        --name "qwen35_multimodal" \
        --no_captions \
        --text-model "$QWEN35_MODEL" \
        --vision-model "$QWEN35_MODEL"
fi

if should_run qwen35 no_tweets; then
    run_experiment "Qwen3.5-397B / no_tweets [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_qwen35_no_tweets.json" \
        --name "qwen35_no_tweets" \
        --no_captions --no_tweets \
        --text-model "$QWEN35_MODEL" \
        --vision-model "$QWEN35_MODEL"
fi

# ── GPT-5-mini ────────────────────────────────────────────────────────────────

if should_run gpt5mini text_only; then
    run_experiment "GPT-5-mini / text_only [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gpt5mini_text_only.json" \
        --name "gpt5mini_text_only" \
        --no_visual --no_captions \
        --text-model "openai/gpt-5-mini"
fi

if should_run gpt5mini text_caption; then
    run_experiment "GPT-5-mini / text_caption [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gpt5mini_text_caption.json" \
        --name "gpt5mini_text_caption" \
        --no_visual \
        --text-model "openai/gpt-5-mini"
fi

if should_run gpt5mini multimodal; then
    run_experiment "GPT-5-mini / multimodal [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gpt5mini_multimodal.json" \
        --name "gpt5mini_multimodal" \
        --no_captions \
        --text-model "$GPT5MINI_MODEL" \
        --vision-model "$GPT5MINI_MODEL"
fi

if should_run gpt5mini no_tweets; then
    run_experiment "GPT-5-mini / no_tweets [110]" \
        --config "$CONFIG_110" \
        --output "$OUT_DIR/exp_gpt5mini_no_tweets.json" \
        --name "gpt5mini_no_tweets" \
        --no_captions --no_tweets \
        --text-model "$GPT5MINI_MODEL" \
        --vision-model "$GPT5MINI_MODEL"
fi

# =============================================================================
# SUPPLEMENTARY — 97 ZIPs without imagery, text_only only
# Combine output with exp_*_text_only.json (110 ZIPs) for full 207-ZIP result.
# =============================================================================

if [ -z "$MODE_FILTER" ] || [ "$MODE_FILTER" = "suppl" ]; then

    if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "gemini25" ]; then
        run_experiment "Gemini-2.5-Flash / text_only [97 suppl]" \
            --config "$CONFIG_97" \
            --output "$OUT_DIR/suppl_gemini25_text_only_97.json" \
            --name "gemini25_text_only_97" \
            --no_visual --no_captions \
            --text-model "$GEMINI25_MODEL"
    fi

    if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "qwen35" ]; then
        run_experiment "Qwen3.5-397B / text_only [97 suppl]" \
            --config "$CONFIG_97" \
            --output "$OUT_DIR/suppl_qwen35_text_only_97.json" \
            --name "qwen35_text_only_97" \
            --no_visual --no_captions \
            --text-model "$QWEN35_MODEL"
    fi

    if [ -z "$MODEL_FILTER" ] || [ "$MODEL_FILTER" = "gpt5mini" ]; then
        run_experiment "GPT-5-mini / text_only [97 suppl]" \
            --config "$CONFIG_97" \
            --output "$OUT_DIR/suppl_gpt5mini_text_only_97.json" \
            --name "gpt5mini_text_only_97" \
            --no_visual --no_captions \
            --text-model "$GPT5MINI_MODEL"
    fi

fi

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "=========================================="
echo "  Output directory: $OUT_DIR"
echo ""
