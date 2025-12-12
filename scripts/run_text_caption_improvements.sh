#!/bin/bash
# Test Improved Text+Caption Methods
# Usage: ./scripts/run_text_caption_improvements.sh [LIMIT]
# Example: ./scripts/run_text_caption_improvements.sh 15 (runs 15 queries per experiment)

# Force CPU execution to avoid GPU OOM/contention
export CUDA_VISIBLE_DEVICES=""
echo "⚠️  Running in CPU-ONLY mode (CUDA_VISIBLE_DEVICES='')"

# Handle optional limit argument
LIMIT=$1
LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    echo "⚠️  Limiting experiments to a RANDOM selection of $LIMIT queries (Seed: 42)"
    LIMIT_FLAG="--limit $LIMIT --shuffle --seed 42"
else
    echo "running full query set (no limit)"
fi

# Ensure data directory exists
mkdir -p data/experiments

echo "🚀 Testing Improved Text+Caption Methods..."
echo "Logs will be written to data/experiments/"
echo ""
echo "Methods being tested:"
echo "  1. Text+Caption (Improved Prompt) - Same retrieval, better LLM guidance"
echo "  2. Text+Caption (Filtered) - Remove 'no damage' captions at retrieval"
echo ""

# 1. Text+Caption with Improved Prompt (no filtering)
# Uses the updated prompts.py with temporal context instructions
echo "[1/2] Text+Caption (Improved Prompt Only)..."
export HARVEY_FILTER_NEGATIVE_CAPTIONS=false
conda run -n harvey-rag python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/exp_text_caption_improved_prompt.json \
  --name exp_text_caption_improved_prompt \
  --no_visual \
  $LIMIT_FLAG

echo ""

# 2. Text+Caption with Filtered Captions
# Removes captions that only say "no flooding/no damage"
echo "[2/2] Text+Caption (With Negative Caption Filter)..."
export HARVEY_FILTER_NEGATIVE_CAPTIONS=true
conda run -n harvey-rag python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/exp_text_caption_filtered.json \
  --name exp_text_caption_filtered \
  --no_visual \
  $LIMIT_FLAG

echo ""
echo "✅ All experiments completed!"
echo ""
echo "📊 To compare results, run:"
echo "   python scripts/compare_experiments.py data/experiments/final_exp1_text_only.json data/experiments/exp_text_caption_improved_prompt.json data/experiments/exp_text_caption_filtered.json"

