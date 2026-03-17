#!/usr/bin/env bash
# Retry failed Gemini 3 queries for MM, TO, and TC configurations.
# Runs only the 15 ZIPs that failed across those three experiments,
# then merges the results back into the original JSON files.

set -euo pipefail
cd "$(dirname "$0")/.."

RETRY_CFG="config/queries_gemini3_retry.json"
OUT_DIR="data/experiments/gemini3_3-15"
SCRIPT="scripts/run_baseline_experiment.py"
MODEL="models/gemini-3-flash-preview"

echo "=== Retry: Gemini 3 Multimodal (3 failed ZIPs) ==="
conda run -n harvey-rag python "$SCRIPT" \
    --config "$RETRY_CFG" \
    --output "$OUT_DIR/retry_gemini3_multimodal.json" \
    --name gemini3_multimodal_retry \
    --text-model "$MODEL" \
    --vision-model "$MODEL" \
    2>&1 | tee "$OUT_DIR/retry_multimodal.log"

echo ""
echo "=== Retry: Gemini 3 Text-Only (2 failed ZIPs) ==="
conda run -n harvey-rag python "$SCRIPT" \
    --config "$RETRY_CFG" \
    --output "$OUT_DIR/retry_gemini3_text_only.json" \
    --name gemini3_text_only_retry \
    --text-model "$MODEL" \
    --no_visual \
    --no_captions \
    2>&1 | tee "$OUT_DIR/retry_text_only.log"

echo ""
echo "=== Retry: Gemini 3 Text+Caption (13 failed ZIPs) ==="
conda run -n harvey-rag python "$SCRIPT" \
    --config "$RETRY_CFG" \
    --output "$OUT_DIR/retry_gemini3_text_caption.json" \
    --name gemini3_text_caption_retry \
    --text-model "$MODEL" \
    --no_visual \
    2>&1 | tee "$OUT_DIR/retry_text_caption.log"

echo ""
echo "=== Merging retry results into original experiment files ==="
conda run -n harvey-rag python - <<'EOF'
import json, os

out_dir = "data/experiments/gemini3_3-15"

merges = [
    ("exp_gemini3_multimodal.json",  "retry_gemini3_multimodal.json",  ["77099", "77026", "77042"]),
    ("exp_gemini3_text_only.json",   "retry_gemini3_text_only.json",   ["77099", "77042"]),
    ("exp_gemini3_text_caption.json","retry_gemini3_text_caption.json",["77059","77067","77587","77096","77039","77062","77547","77036","77042","77088","77025","77053","77035"]),
]

for orig_name, retry_name, failed_zips in merges:
    orig_path  = os.path.join(out_dir, orig_name)
    retry_path = os.path.join(out_dir, retry_name)

    if not os.path.exists(retry_path):
        print(f"SKIP {retry_name} — file not found")
        continue

    with open(orig_path)  as f: orig  = json.load(f)
    with open(retry_path) as f: retry = json.load(f)

    # Build lookup of successful retry records by ZIP
    retry_by_zip = {
        r["query"]["zip"]: r
        for r in retry["records"]
        if "error" not in r
    }

    replaced = 0
    new_records = []
    for r in orig["records"]:
        zip_code = r.get("query", {}).get("zip", "")
        if zip_code in failed_zips and zip_code in retry_by_zip:
            new_records.append(retry_by_zip[zip_code])
            replaced += 1
        else:
            new_records.append(r)

    # Also append any retry ZIPs that weren't in orig at all (shouldn't happen)
    existing_zips = {r.get("query",{}).get("zip") for r in orig["records"]}
    for zip_code, r in retry_by_zip.items():
        if zip_code not in existing_zips:
            new_records.append(r)
            replaced += 1

    orig["records"] = new_records

    # Recompute summary stats
    successful = [r for r in new_records if "error" not in r]
    ext_errs = [abs(r["model_response"]["flood_extent_pct"] - r["ground_truth"]["flooded_pct"]) for r in successful]
    dmg_all  = [abs(r["model_response"]["damage_severity_pct"] - r["ground_truth"]["pde_damage_score"]) for r in successful]
    pde_only = [r for r in successful if r["ground_truth"].get("pde_damage_score", 0.0) > 0.0]
    dmg_pde  = [abs(r["model_response"]["damage_severity_pct"] - r["ground_truth"]["pde_damage_score"]) for r in pde_only]

    orig["metadata"]["summary_stats"] = {
        "successful_queries": len(successful),
        "failed_queries":     len(new_records) - len(successful),
        "extent_mae":         round(sum(ext_errs)/len(ext_errs), 2),
        "damage_mae":         round(sum(dmg_all)/len(dmg_all),   2),
        "damage_mae_pde_only":round(sum(dmg_pde)/len(dmg_pde),   2),
        "n_pde":              len(pde_only),
    }

    with open(orig_path, "w") as f:
        json.dump(orig, f, indent=2, default=str)

    print(f"{orig_name}: replaced {replaced} records, N={len(successful)}, "
          f"ext_mae={orig['metadata']['summary_stats']['extent_mae']}, "
          f"dmg_mae_pde={orig['metadata']['summary_stats']['damage_mae_pde_only']}")

print("Done.")
EOF
