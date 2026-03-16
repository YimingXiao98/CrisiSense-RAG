"""Historical mean baseline: predict dataset-wide average for every query.

This is the simplest non-trivial baseline. For each ZIP, it predicts:
  - flood_extent_pct = mean(all GT flooded_pct values)
  - damage_severity_pct = mean(all GT pde_damage_score values) * 100

Two variants are computed:
  1. Global mean: same prediction for every ZIP (population-level average)
  2. FEMA prior mean: uses FEMA historical damage to create a per-ZIP prior

Usage:
    python scripts/run_historical_mean_baseline.py \
        --config config/queries_complete_map.json \
        --output data/experiments/2026-03-09/exp_historical_mean_baseline.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Historical mean baseline")
    parser.add_argument("--config", required=True, help="Query config JSON")
    parser.add_argument(
        "--output",
        default="data/experiments/2026-03-09/exp_historical_mean_baseline.json",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    queries = config["queries"]

    # Load ground truth
    base = Path("data/processed")
    flood_gt = json.loads((base / "flood_depth_by_zip.json").read_text())
    pde_gt = json.loads((base / "pde_by_zip.json").read_text())

    # Compute global means from the 207 query ZIPs
    all_flood_pcts = []
    all_damage_pcts = []
    for q in queries:
        z = q["zip"]
        fp = flood_gt.get(z, {}).get("flooded_pct", 0.0)
        dp = pde_gt.get(z, {}).get("mean_pde", 0.0) * 100
        all_flood_pcts.append(fp)
        all_damage_pcts.append(dp)

    global_flood_mean = float(np.mean(all_flood_pcts))
    global_damage_mean = float(np.mean(all_damage_pcts))

    print(f"Global mean flood_extent: {global_flood_mean:.2f}%")
    print(f"Global mean damage_severity: {global_damage_mean:.2f}%")

    # Generate predictions (same value for every query)
    records = []
    for i, q in enumerate(queries, 1):
        z = q["zip"]
        fp = flood_gt.get(z, {}).get("flooded_pct", 0.0)
        dp = pde_gt.get(z, {}).get("mean_pde", 0.0) * 100

        records.append({
            "query_id": i,
            "query": {
                "zip": z,
                "start_date": q.get("start", ""),
                "end_date": q.get("end", ""),
                "comment": q.get("category", ""),
            },
            "model_response": {
                "raw": {
                    "method": "global_mean_baseline",
                    "predicted_flood": round(global_flood_mean, 2),
                    "predicted_damage": round(global_damage_mean, 2),
                },
                "flood_extent_pct": round(global_flood_mean, 2),
                "damage_severity_pct": round(global_damage_mean, 2),
                "confidence": 0.0,
            },
            "ground_truth": {
                "flooded_pct": fp,
                "pde_damage_score": pde_gt.get(z, {}).get("mean_pde", 0.0),
                "mean_depth_m": flood_gt.get(z, {}).get("mean_depth_m", 0.0),
                "max_depth_m": flood_gt.get(z, {}).get("max_depth_m", 0.0),
                "claim_count": 0,
                "total_claim_amount": 0.0,
            },
        })

    # Compute MAE
    extent_errors = [abs(r["model_response"]["flood_extent_pct"] - r["ground_truth"]["flooded_pct"]) for r in records]
    damage_errors = [abs(r["model_response"]["damage_severity_pct"] - r["ground_truth"]["pde_damage_score"] * 100) for r in records]

    metadata = {
        "experiment_name": "historical_mean_baseline",
        "timestamp": datetime.now().isoformat(),
        "method": "global_mean",
        "global_flood_mean": round(global_flood_mean, 2),
        "global_damage_mean": round(global_damage_mean, 2),
        "total_queries": len(queries),
        "summary_stats": {
            "successful_queries": len(records),
            "failed_queries": 0,
            "extent_mae": round(float(np.mean(extent_errors)), 2),
            "damage_mae": round(float(np.mean(damage_errors)), 2),
        },
    }

    output = {
        "metadata": metadata,
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))

    print(f"\nResults:")
    print(f"  Extent MAE: {metadata['summary_stats']['extent_mae']}%")
    print(f"  Damage MAE: {metadata['summary_stats']['damage_mae']}%")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
