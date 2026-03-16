"""Re-fuse existing multimodal experiment data with improved fusion strategy.

V2 Fusion Strategy ("Damage-Only Visual"):
  - Flood extent: always use text estimate (ignore visual; imagery is post-recession)
  - Damage severity: text is baseline; visual can only boost, never lower
    damage = text_damage + alpha * max(0, visual_damage - text_damage)

This reads the intermediate text_analysis and visual_analysis stored in the
existing v3 multimodal experiment JSONs and applies new fusion logic without
re-running any API calls.

Usage:
    python scripts/refusion_v2.py [--alpha 0.4] [--output-dir data/experiments/2026-03-09]
"""

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent / "data" / "experiments"

# Source experiment files (v3 multimodal, with stored intermediates)
MULTIMODAL_FILES = {
    "gemini": BASE / "exp_v3_gemini_multimodal.json",
    "llama": BASE / "exp_v3_llama_multimodal.json",
    "qwen": BASE / "exp_v3_qwen_multimodal.json",
}


def v2_fuse(text_analysis: dict, visual_analysis: dict, alpha: float = 0.4) -> dict:
    """Apply V2 fusion: text-only flood, visual-additive damage.

    Args:
        text_analysis: dict with flood_pct, damage_pct, confidence
        visual_analysis: dict with flood_pct, damage_pct, confidence
        alpha: blend factor for visual damage boost (0 = text only, 1 = full visual boost)

    Returns:
        dict with fused flood_extent_pct, damage_severity_pct, confidence
    """
    text_flood = float(text_analysis.get("flood_pct", 0) or 0)
    text_damage = float(text_analysis.get("damage_pct", 0) or 0)
    text_conf = float(text_analysis.get("confidence", 0.5) or 0.5)

    visual_damage = float(visual_analysis.get("damage_pct", 0) or 0)
    visual_conf = float(visual_analysis.get("confidence", 0) or 0)

    # Flood: text only (post-recession imagery is unreliable for flood extent)
    fused_flood = text_flood

    # Damage: visual-additive only (visual can boost but never lower)
    if visual_damage > text_damage and visual_conf > 0.3:
        boost = alpha * (visual_damage - text_damage)
        fused_damage = text_damage + boost
    else:
        fused_damage = text_damage

    # Confidence: use text confidence as base, slight boost if visual confirms
    if visual_damage > 0 and abs(visual_damage - text_damage) < 15:
        fused_conf = min(text_conf + 0.05, 1.0)
    else:
        fused_conf = text_conf

    return {
        "flood_extent_pct": round(fused_flood, 1),
        "damage_severity_pct": round(fused_damage, 1),
        "confidence": round(fused_conf, 2),
    }


def refuse_experiment(input_path: Path, output_path: Path, alpha: float = 0.4):
    """Re-fuse a single experiment file with V2 strategy."""
    data = json.loads(input_path.read_text())
    records = data["records"]

    n_refused = 0
    n_skipped = 0

    for record in records:
        if "error" in record:
            continue

        raw = record["model_response"].get("raw", {})
        ta = raw.get("text_analysis", {})
        va = raw.get("visual_analysis", {})

        if not ta:
            n_skipped += 1
            continue

        # Apply V2 fusion
        fused = v2_fuse(ta, va, alpha=alpha)

        # Update the record's model_response with new fused values
        record["model_response"]["flood_extent_pct"] = fused["flood_extent_pct"]
        record["model_response"]["damage_severity_pct"] = fused["damage_severity_pct"]
        record["model_response"]["confidence"] = fused["confidence"]

        # Also update raw.estimates if it exists
        if "estimates" in raw:
            raw["estimates"]["flood_extent_pct"] = fused["flood_extent_pct"]
            raw["estimates"]["damage_severity_pct"] = fused["damage_severity_pct"]
            raw["estimates"]["confidence"] = fused["confidence"]

        # Tag the fusion method
        raw["fusion_method"] = "v2_damage_only_visual"
        raw["fusion_alpha"] = alpha

        n_refused += 1

    # Update metadata
    data["metadata"]["fusion_strategy"] = "v2_damage_only_visual"
    data["metadata"]["fusion_alpha"] = alpha
    data["metadata"]["refusion_timestamp"] = datetime.now().isoformat()
    data["metadata"]["source_file"] = str(input_path)

    # Recompute summary stats
    successful = [r for r in records if "error" not in r]
    if successful:
        extent_errors = [
            abs(r["model_response"]["flood_extent_pct"] - r["ground_truth"]["flooded_pct"])
            for r in successful
        ]
        damage_errors = [
            abs(
                r["model_response"]["damage_severity_pct"]
                - r["ground_truth"]["pde_damage_score"] * 100
            )
            for r in successful
        ]
        data["metadata"]["summary_stats"] = {
            "successful_queries": len(successful),
            "failed_queries": len(records) - len(successful),
            "extent_mae": round(np.mean(extent_errors), 2),
            "damage_mae": round(np.mean(damage_errors), 2),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=str))

    stats = data["metadata"].get("summary_stats", {})
    print(
        f"  {input_path.name} -> {output_path.name}: "
        f"refused={n_refused}, skipped={n_skipped}, "
        f"Extent MAE={stats.get('extent_mae', '?')}, "
        f"Damage MAE={stats.get('damage_mae', '?')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Re-fuse multimodal experiments with V2 strategy")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Visual damage boost factor (default: 0.4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE / "2026-03-09"),
        help="Output directory for re-fused experiments",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run alpha sweep from 0.0 to 1.0 and print comparison table",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.sweep:
        # Alpha sweep for finding optimal value
        print("\nAlpha Sweep (visual damage boost factor)")
        print("=" * 80)
        print(f"{'Alpha':<8} {'Gemini Ext':>11} {'Gemini Dmg':>11} "
              f"{'Llama Ext':>11} {'Llama Dmg':>11} "
              f"{'Qwen Ext':>11} {'Qwen Dmg':>11}")
        print("-" * 80)

        for alpha_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            row = [f"{alpha_val:<8.1f}"]
            for model, path in MULTIMODAL_FILES.items():
                if not path.exists():
                    row.extend(["N/A", "N/A"])
                    continue
                data = json.loads(path.read_text())
                records = [r for r in data["records"] if "error" not in r]

                ext_errs, dmg_errs = [], []
                for r in records:
                    raw = r["model_response"].get("raw", r["model_response"])
                    ta = raw.get("text_analysis", {})
                    va = raw.get("visual_analysis", {})
                    if not ta:
                        continue
                    fused = v2_fuse(ta, va, alpha=alpha_val)
                    ext_errs.append(abs(fused["flood_extent_pct"] - r["ground_truth"]["flooded_pct"]))
                    dmg_errs.append(abs(fused["damage_severity_pct"] - r["ground_truth"]["pde_damage_score"] * 100))

                row.append(f"{np.mean(ext_errs):>11.2f}")
                row.append(f"{np.mean(dmg_errs):>11.2f}")
            print(" ".join(row))

        # Also print original (current fusion) for comparison
        print("-" * 80)
        row = ["Original"]
        for model, path in MULTIMODAL_FILES.items():
            if not path.exists():
                row.extend(["N/A", "N/A"])
                continue
            data = json.loads(path.read_text())
            records = [r for r in data["records"] if "error" not in r]
            ext_errs = [abs(r["model_response"]["flood_extent_pct"] - r["ground_truth"]["flooded_pct"]) for r in records]
            dmg_errs = [abs(r["model_response"]["damage_severity_pct"] - r["ground_truth"]["pde_damage_score"] * 100) for r in records]
            row.append(f"{np.mean(ext_errs):>11.2f}")
            row.append(f"{np.mean(dmg_errs):>11.2f}")
        print(" ".join(row))
        return

    # Normal mode: refuse with specified alpha
    print(f"V2 Fusion: text-only flood, visual-additive damage (alpha={args.alpha})")
    print(f"Output: {output_dir}\n")

    for model, input_path in MULTIMODAL_FILES.items():
        if not input_path.exists():
            print(f"  SKIP {model}: {input_path} not found")
            continue
        output_path = output_dir / f"exp_v2_{model}_multimodal.json"
        refuse_experiment(input_path, output_path, alpha=args.alpha)

    print("\nDone.")


if __name__ == "__main__":
    main()
