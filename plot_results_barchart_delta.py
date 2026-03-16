"""
plot_results_barchart_delta.py
Figure 3 (alternative): delta MAE relative to Text-Only baseline.
Negative = improvement, positive = degradation.
X-axis: 4 models. Bars: Text+Caption, No-Tweets, Multimodal (3 configs vs. baseline).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Georgia"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

BASE_OUTPUT_DIR = "paper/figures"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename_base):
    for fmt in ["pdf", "jpg"]:
        out_path = os.path.join(BASE_OUTPUT_DIR, f"{filename_base}.{fmt}")
        kw = {"dpi": 300, "bbox_inches": "tight"} if fmt == "jpg" else {"bbox_inches": "tight"}
        fig.savefig(out_path, **kw)
        print(f"Saved {out_path}")


# ── Raw MAE [Text-Only, Text+Caption, No-Tweets, Multimodal] ─────────────────
raw = {
    "Gemini25": {
        "extent": [12.88, 12.54, 17.42, 12.98],
        "damage": [13.23, 13.75, 11.32, 13.59],
    },
    "Gemini3": {
        "extent": [11.23, 11.18, 17.70, 11.08],
        "damage": [10.03, 11.24, 10.13, 10.10],
    },
    "Qwen": {
        "extent": [22.06, 22.93, 27.10, 19.64],
        "damage": [12.52, 12.71, 13.50, 12.02],
    },
    "GPT": {
        "extent": [17.48, 18.41, 14.68,  8.86],
        "damage": [12.12, 12.34, 12.63, 15.52],
    },
}

# ── Delta = config - text_only (index 0 is baseline → always 0) ──────────────
configs = ["Text+Caption", "No-Tweets", "Multimodal"]  # skip Text-Only (=0)
models  = ["Gemini 2.5 Flash", "Gemini 3 Flash", "Qwen 3.5 397B", "GPT-5-mini"]
keys    = ["Gemini25", "Gemini3", "Qwen", "GPT"]

delta_ext, delta_dmg = {}, {}
for k in keys:
    base_e = raw[k]["extent"][0]
    base_d = raw[k]["damage"][0]
    delta_ext[k] = [raw[k]["extent"][i] - base_e for i in range(1, 4)]
    delta_dmg[k] = [raw[k]["damage"][i] - base_d for i in range(1, 4)]

# ── Colors per config ─────────────────────────────────────────────────────────
config_colors = ["#3498db", "#e67e22", "#e74c3c"]  # blue, orange, red

x = np.arange(len(models))
n = len(configs)
width = 0.22

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
fig.suptitle(
    r"MAE Change Relative to Text-Only Baseline ($\Delta$ pp, $N=110$)",
    fontsize=14, fontweight="bold", y=1.02,
)

for j, (cfg, color) in enumerate(zip(configs, config_colors)):
    offset = (j - (n - 1) / 2) * width

    d_ext = [delta_ext[k][j] for k in keys]
    d_dmg = [delta_dmg[k][j] for k in keys]

    # Extent panel
    bars = ax1.bar(x + offset, d_ext, width, label=cfg, color=color,
                   alpha=0.80, edgecolor=color, linewidth=1.0)
    for bar, val in zip(bars, d_ext):
        va = "bottom" if val >= 0 else "top"
        ypos = val + 0.15 if val >= 0 else val - 0.15
        ax1.annotate(f"{val:+.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4 if val >= 0 else -4),
                     textcoords="offset points",
                     ha="center", va=va, fontsize=7.5, fontweight="bold")

    # Damage panel
    bars = ax2.bar(x + offset, d_dmg, width, label=cfg, color=color,
                   alpha=0.80, edgecolor=color, linewidth=1.0)
    for bar, val in zip(bars, d_dmg):
        va = "bottom" if val >= 0 else "top"
        ax2.annotate(f"{val:+.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4 if val >= 0 else -4),
                     textcoords="offset points",
                     ha="center", va=va, fontsize=7.5, fontweight="bold")

for ax, title in [(ax1, "Flood Extent MAE Change (pp)"),
                  (ax2, "Damage Severity MAE Change (pp)")]:
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", label="Text-Only (baseline)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(r"$\Delta$ MAE vs. Text-Only (pp)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    # Annotate direction
    ymin, ymax = ax.get_ylim()
    ax.text(0.98, 0.04, "← better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="green", style="italic")
    ax.text(0.98, 0.96, "worse →", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="red", style="italic")

plt.tight_layout()
save_fig(fig, "results_barchart_delta")
plt.close()
print("Done.")
