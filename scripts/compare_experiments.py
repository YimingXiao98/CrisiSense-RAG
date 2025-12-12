#!/usr/bin/env python3
"""Compare multiple experiment results side by side."""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_experiment(path: Path) -> Dict:
    """Load experiment JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Paths to experiment JSON files",
    )
    args = parser.parse_args()

    results = []
    for exp_path in args.experiments:
        if not exp_path.exists():
            print(f"⚠️  File not found: {exp_path}")
            continue
        
        data = load_experiment(exp_path)
        meta = data.get("metadata", {})
        stats = meta.get("summary_stats", {})
        
        results.append({
            "name": meta.get("experiment_name", exp_path.stem),
            "extent_mae": stats.get("extent_mae", "N/A"),
            "damage_mae": stats.get("damage_mae", "N/A"),
            "queries": meta.get("total_queries", "N/A"),
            "settings": meta.get("settings", {}),
        })

    if not results:
        print("No valid experiment files found.")
        return

    # Print comparison table
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"\n{'Experiment':<40} {'Extent MAE':<12} {'Damage MAE':<12} {'Queries':<8}")
    print("-" * 80)
    
    # Rows
    baseline_extent = None
    baseline_damage = None
    for i, r in enumerate(results):
        extent = r["extent_mae"]
        damage = r["damage_mae"]
        
        # Calculate delta from first experiment (baseline)
        extent_str = f"{extent:.2f}%" if isinstance(extent, (int, float)) else str(extent)
        damage_str = f"{damage:.2f}%" if isinstance(damage, (int, float)) else str(damage)
        
        if i == 0:
            baseline_extent = extent if isinstance(extent, (int, float)) else None
            baseline_damage = damage if isinstance(damage, (int, float)) else None
        else:
            if baseline_extent and isinstance(extent, (int, float)):
                delta = extent - baseline_extent
                extent_str += f" ({delta:+.2f})"
            if baseline_damage and isinstance(damage, (int, float)):
                delta = damage - baseline_damage
                damage_str += f" ({delta:+.2f})"
        
        print(f"{r['name']:<40} {extent_str:<12} {damage_str:<12} {r['queries']:<8}")
    
    print("-" * 80)
    print("\nNote: Numbers in parentheses show delta from first experiment (baseline)")
    
    # Print settings differences
    print("\n" + "=" * 80)
    print("SETTINGS")
    print("=" * 80)
    for r in results:
        print(f"\n{r['name']}:")
        for k, v in r.get("settings", {}).items():
            print(f"  {k}: {v}")
    
    print("\n")


if __name__ == "__main__":
    main()

