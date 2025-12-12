#!/bin/bash
# Run Ablation Study Experiments (Text-Only, Text+Caption, Multimodal)
# Usage: ./scripts/run_ablation_study.sh [LIMIT]
# Example: ./scripts/run_ablation_study.sh 10 (runs 10 queries per experiment)

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

# Ensure data dictionary exists
mkdir -p data/experiments

echo "🚀 Starting Ablation Study (3 Parallel Experiments)..."
echo "Logs will be written to data/experiments/"

# 1. Text-Only Baseline
# Uses: Tweets, 311 Calls, Sensors, FEMA Data
# Excludes: All imagery and captions
echo "[1/3] Launching Text-Only Baseline (PID will follow)..."
nohup conda run -n harvey-rag python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/final_exp1_text_only.json \
  --name final_exp1_text_only \
  --no_visual --no_captions \
  $LIMIT_FLAG \
  > data/experiments/final_exp1.log 2>&1 &
echo "      PID: $!"

# 2. Text + Captions (Caption Bridge)
# Uses: Text-Only sources + Image Captions
# Excludes: Direct visual analysis
echo "[2/3] Launching Text + Captions (PID will follow)..."
nohup conda run -n harvey-rag python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/final_exp2_text_caption.json \
  --name final_exp2_text_caption \
  --no_visual \
  $LIMIT_FLAG \
  > data/experiments/final_exp2.log 2>&1 &
echo "      PID: $!"

# 3. Full Multimodal (Visual Additive)
# Uses: Everything + Visual Analysis (with "Additive" logic fix)
echo "[3/3] Launching Full Multimodal (PID will follow)..."
nohup conda run -n harvey-rag python scripts/run_baseline_experiment.py \
  --config config/queries_50_mixed.json \
  --output data/experiments/final_exp3_multimodal.json \
  --name final_exp3_multimodal \
  $LIMIT_FLAG \
  > data/experiments/final_exp3.log 2>&1 &
echo "      PID: $!"

echo "✅ All experiments execution started in background."
echo "   Monitor progress with: tail -f data/experiments/final_exp*.log"
