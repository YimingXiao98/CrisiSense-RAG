"""
generate_residual_plots.py
Generates residual (error) distribution plots for Figure 4 (reviewer comment R1-C6).
Uses GPT-5-mini multimodal results (best flood extent model).
Outputs: paper/figures/[pdf|png|jpg]/residual_plots.{pdf,png,jpg}
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Georgia"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 12

RESULTS_PATH = "data/experiments/full_207_3-15/exp_gpt5mini_multimodal.json"
PDE_PATH = "data/processed/pde_by_zip.json"
BASE_OUTPUT_DIR = "paper/figures"
FORMATS = ["pdf", "png", "jpg"]
for fmt in FORMATS:
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, fmt), exist_ok=True)


def save_fig(fig, filename_base):
    for fmt in FORMATS:
        out_path = os.path.join(BASE_OUTPUT_DIR, fmt, f"{filename_base}.{fmt}")
        kw = {"dpi": 300, "bbox_inches": "tight"} if fmt in ("png", "jpg") else {"bbox_inches": "tight"}
        fig.savefig(out_path, **kw)
        print(f"Saved {out_path}")


# ── Load data ────────────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
with open(PDE_PATH) as f:
    pde_by_zip = json.load(f)
pde_zips = set(pde_by_zip.keys())

extent_pred, extent_gt = [], []
damage_pred, damage_gt = [], []

for r in results_data["records"]:
    z = r["query"]["zip"]
    ep = r["model_response"].get("flood_extent_pct", 0.0)
    eg = r["ground_truth"].get("flooded_pct", 0.0)
    extent_pred.append(ep)
    extent_gt.append(eg)

    if z in pde_zips and "pde_damage_score" in r["ground_truth"]:
        damage_pred.append(r["model_response"].get("damage_severity_pct", 0.0))
        damage_gt.append(r["ground_truth"]["pde_damage_score"])

extent_pred = np.array(extent_pred)
extent_gt   = np.array(extent_gt)
damage_pred = np.array(damage_pred)
damage_gt   = np.array(damage_gt)

ext_resid = extent_pred - extent_gt          # signed residual
dmg_resid = damage_pred - damage_gt

print(f"Extent  : N={len(ext_resid)}, mean={ext_resid.mean():.2f}, std={ext_resid.std():.2f}")
print(f"Damage  : N={len(dmg_resid)}, mean={dmg_resid.mean():.2f}, std={dmg_resid.std():.2f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Row 1: Flood Extent ──────────────────────────────────────────────────────

# Panel A – Residual histogram
ax_a = fig.add_subplot(gs[0, 0])
bins = np.arange(-80, 81, 5)
ax_a.hist(ext_resid, bins=bins, color="#3498db", edgecolor="white", linewidth=0.6, alpha=0.85)
ax_a.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax_a.axvline(ext_resid.mean(), color="#e74c3c", linewidth=1.4, linestyle="-",
             label=f"Mean = {ext_resid.mean():.1f} pp")
ax_a.set_xlabel("Residual (Pred $-$ GT, pp)", fontsize=11)
ax_a.set_ylabel("ZIP Code Count", fontsize=11)
ax_a.set_title("(A) Flood Extent Residuals", fontsize=12, fontweight="bold")
ax_a.legend(fontsize=10)
ax_a.set_xlim(-85, 85)

# Panel B – Scatter Pred vs GT
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(extent_gt, extent_pred, color="#3498db", alpha=0.55, s=25, edgecolors="none")
lim = max(extent_gt.max(), extent_pred.max()) + 5
ax_b.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="1:1 line")
ax_b.set_xlabel("Ground Truth (%)", fontsize=11)
ax_b.set_ylabel("Predicted (%)", fontsize=11)
ax_b.set_title("(B) Flood Extent: Pred vs GT", fontsize=12, fontweight="bold")
ax_b.set_xlim(0, lim); ax_b.set_ylim(0, lim)
ax_b.legend(fontsize=10)

# Panel C – Absolute error CDF
ax_c = fig.add_subplot(gs[0, 2])
abs_ext = np.abs(ext_resid)
xs = np.sort(abs_ext)
ys = np.arange(1, len(xs) + 1) / len(xs)
ax_c.plot(xs, ys, color="#3498db", linewidth=2)
for pct in [50, 75, 90]:
    v = np.percentile(abs_ext, pct)
    ax_c.axvline(v, linestyle=":", linewidth=1, color="gray")
    ax_c.text(v + 0.5, pct / 100 - 0.06, f"P{pct}={v:.1f}", fontsize=8, color="gray")
ax_c.set_xlabel("|Residual| (pp)", fontsize=11)
ax_c.set_ylabel("Cumulative Fraction", fontsize=11)
ax_c.set_title("(C) Flood Extent AE CDF", fontsize=12, fontweight="bold")
ax_c.set_xlim(0, None); ax_c.set_ylim(0, 1.02)

# ── Row 2: Damage Severity ───────────────────────────────────────────────────

# Panel D – Residual histogram
ax_d = fig.add_subplot(gs[1, 0])
ax_d.hist(dmg_resid, bins=bins, color="#e67e22", edgecolor="white", linewidth=0.6, alpha=0.85)
ax_d.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax_d.axvline(dmg_resid.mean(), color="#e74c3c", linewidth=1.4, linestyle="-",
             label=f"Mean = {dmg_resid.mean():.1f} pp")
ax_d.set_xlabel("Residual (Pred $-$ GT, pp)", fontsize=11)
ax_d.set_ylabel("ZIP Code Count", fontsize=11)
ax_d.set_title("(D) Damage Severity Residuals", fontsize=12, fontweight="bold")
ax_d.legend(fontsize=10)
ax_d.set_xlim(-85, 85)

# Panel E – Scatter Pred vs GT
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(damage_gt, damage_pred, color="#e67e22", alpha=0.55, s=25, edgecolors="none")
lim_d = max(damage_gt.max(), damage_pred.max()) + 5
ax_e.plot([0, lim_d], [0, lim_d], "k--", linewidth=1.2, label="1:1 line")
ax_e.set_xlabel("Ground Truth (%)", fontsize=11)
ax_e.set_ylabel("Predicted (%)", fontsize=11)
ax_e.set_title("(E) Damage Severity: Pred vs GT", fontsize=12, fontweight="bold")
ax_e.set_xlim(0, lim_d); ax_e.set_ylim(0, lim_d)
ax_e.legend(fontsize=10)

# Panel F – Absolute error CDF
ax_f = fig.add_subplot(gs[1, 2])
abs_dmg = np.abs(dmg_resid)
xs_d = np.sort(abs_dmg)
ys_d = np.arange(1, len(xs_d) + 1) / len(xs_d)
ax_f.plot(xs_d, ys_d, color="#e67e22", linewidth=2)
for pct in [50, 75, 90]:
    v = np.percentile(abs_dmg, pct)
    ax_f.axvline(v, linestyle=":", linewidth=1, color="gray")
    ax_f.text(v + 0.5, pct / 100 - 0.06, f"P{pct}={v:.1f}", fontsize=8, color="gray")
ax_f.set_xlabel("|Residual| (pp)", fontsize=11)
ax_f.set_ylabel("Cumulative Fraction", fontsize=11)
ax_f.set_title("(F) Damage Severity AE CDF", fontsize=12, fontweight="bold")
ax_f.set_xlim(0, None); ax_f.set_ylim(0, 1.02)

fig.suptitle(
    "Residual Analysis — GPT-5-mini Multimodal ($N=110$ ZIPs, PDE-covered subset for damage)",
    fontsize=13, fontweight="bold", y=1.01,
)

save_fig(fig, "residual_plots")
plt.close()
print("Done.")
