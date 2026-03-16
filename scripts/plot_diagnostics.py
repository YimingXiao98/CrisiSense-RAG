"""Generate diagnostic plots and statistics for reviewer response.

Produces:
  1. Scatter plots: predicted vs ground truth (extent + damage) with R², Pearson r
  2. Signed residual histograms
  3. IQR and summary stats table (printed to stdout)

Usage:
    python scripts/plot_diagnostics.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Experiment files (best config per model = text-only for Llama/Qwen, text+caption for Gemini) ──
# But for the scatter/IQR we want the FULL picture across all configs.
# Use the best-performing config per model as the "main" result for scatter plots,
# and report IQR across ALL configs.

BASE = Path(__file__).parent.parent / "data" / "experiments"

EXPERIMENTS = {
    # Gemini: exp_complete_map.json matches paper's "Text-Only" row (28.40, 18.66)
    "Gemini Text-Only": BASE / "exp_complete_map.json",
    "Gemini Text+Caption": BASE / "exp_complete_map_text_caption.json",
    "Gemini Multimodal": BASE / "exp_v3_gemini_multimodal.json",
    "Llama Text-Only": BASE / "exp_llama_text_only.json",
    "Llama Text+Caption": BASE / "exp_llama_text_caption.json",
    "Llama Multimodal": BASE / "exp_v3_llama_multimodal.json",
    "Qwen Text-Only": BASE / "exp_qwen_text_only.json",
    "Qwen Text+Caption": BASE / "exp_qwen_text_caption.json",
    "Qwen Multimodal": BASE / "exp_v3_qwen_multimodal.json",
}

# Best config per model (used for scatter plots)
BEST = {
    "Gemini 2.5 Flash": "Gemini Text+Caption",
    "Llama 3.3 70B": "Llama Text-Only",
    "Qwen 2.5 72B": "Qwen Text-Only",
}

OUTPUT_DIR = Path(__file__).parent.parent / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_records(path):
    """Load experiment records, skipping errors."""
    data = json.loads(path.read_text())
    return [r for r in data["records"] if "error" not in r]


def extract_arrays(records):
    """Extract prediction and ground truth arrays."""
    extent_pred, extent_gt = [], []
    damage_pred, damage_gt = [], []

    for r in records:
        ep = r["model_response"]["flood_extent_pct"]
        eg = r["ground_truth"]["flooded_pct"]
        extent_pred.append(ep)
        extent_gt.append(eg)

        dp = r["model_response"]["damage_severity_pct"]
        dg = r["ground_truth"]["pde_damage_score"] * 100  # 0-1 -> 0-100
        damage_pred.append(dp)
        damage_gt.append(dg)

    return (
        np.array(extent_pred), np.array(extent_gt),
        np.array(damage_pred), np.array(damage_gt),
    )


def print_iqr_table():
    """Print IQR and summary stats for all experiments."""
    print("\n" + "=" * 90)
    print(f"{'Configuration':<25} {'Metric':<10} {'MAE':>6} {'Median AE':>10} "
          f"{'IQR':>12} {'P10':>6} {'P90':>6}")
    print("=" * 90)

    for name, path in EXPERIMENTS.items():
        records = load_records(path)
        ep, eg, dp, dg = extract_arrays(records)

        extent_errors = np.abs(ep - eg)
        damage_errors = np.abs(dp - dg)

        for metric, errors in [("Extent", extent_errors), ("Damage", damage_errors)]:
            mae = np.mean(errors)
            median = np.median(errors)
            q25, q75 = np.percentile(errors, [25, 75])
            p10, p90 = np.percentile(errors, [10, 90])
            iqr = q75 - q25
            print(f"{name:<25} {metric:<10} {mae:6.2f} {median:10.2f} "
                  f"[{q25:5.2f}, {q75:5.2f}] {p10:6.2f} {p90:6.2f}")
        print("-" * 90)


def plot_scatter(ax, gt, pred, title, color):
    """Plot predicted vs ground truth scatter on a given axis."""
    ax.scatter(gt, pred, alpha=0.35, s=18, c=color, edgecolors="none")

    # Identity line
    lims = [0, max(np.max(gt), np.max(pred)) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=0.8)

    # Stats
    r_val, p_val = stats.pearsonr(gt, pred)
    rho, _ = stats.spearmanr(gt, pred)
    mae = np.mean(np.abs(pred - gt))

    # R^2
    ss_res = np.sum((pred - gt) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.text(0.05, 0.95,
            f"MAE = {mae:.1f}%\n$R^2$ = {r2:.3f}\n$r$ = {r_val:.3f}\n"
            f"$\\rho$ = {rho:.3f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Ground Truth (%)")
    ax.set_ylabel("Predicted (%)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")


def make_scatter_plots():
    """Create 3x2 scatter plot grid (3 models x 2 metrics)."""
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    colors_extent = "#2166ac"
    colors_damage = "#b2182b"

    for i, (model_label, config_key) in enumerate(BEST.items()):
        records = load_records(EXPERIMENTS[config_key])
        ep, eg, dp, dg = extract_arrays(records)

        plot_scatter(axes[i, 0], eg, ep,
                     f"{model_label} - Flood Extent", colors_extent)
        plot_scatter(axes[i, 1], dg, dp,
                     f"{model_label} - Damage Severity", colors_damage)

    fig.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "scatter_pred_vs_gt.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def make_residual_histograms():
    """Create signed residual histograms for best config per model."""
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    for i, (model_label, config_key) in enumerate(BEST.items()):
        records = load_records(EXPERIMENTS[config_key])
        ep, eg, dp, dg = extract_arrays(records)

        extent_resid = ep - eg
        damage_resid = dp - dg

        for j, (resid, metric, color) in enumerate([
            (extent_resid, "Flood Extent", "#2166ac"),
            (damage_resid, "Damage Severity", "#b2182b"),
        ]):
            ax = axes[i, j]
            ax.hist(resid, bins=30, color=color, alpha=0.7, edgecolor="white")
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            ax.axvline(np.mean(resid), color="red", linestyle="-", alpha=0.7,
                       label=f"Mean = {np.mean(resid):.1f}")
            ax.axvline(np.median(resid), color="orange", linestyle="-", alpha=0.7,
                       label=f"Median = {np.median(resid):.1f}")
            ax.set_xlabel("Signed Residual (Predicted - GT) [pp]")
            ax.set_ylabel("Count")
            ax.set_title(f"{model_label} - {metric}", fontsize=10)
            ax.legend(fontsize=7)

    fig.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "residual_histograms.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    # Check all files exist
    missing = [n for n, p in EXPERIMENTS.items() if not p.exists()]
    if missing:
        print(f"Missing experiment files: {missing}", file=sys.stderr)
        sys.exit(1)

    print_iqr_table()
    make_scatter_plots()
    make_residual_histograms()
    print("\nDone.")


if __name__ == "__main__":
    main()
