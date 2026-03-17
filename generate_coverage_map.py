"""
generate_coverage_map.py
Produces a single map showing the four evaluation groups across the 207-ZIP study area:
  - Imagery core + PDE     (N=104): main multimodal + damage evaluation
  - Imagery core, no PDE   (N=6):   main multimodal extent evaluation only
  - Peripheral + PDE       (N=35):  supplementary text-only + damage evaluation
  - Peripheral, no PDE     (N=62):  supplementary text-only extent evaluation only
"""
import json
import os
import shutil

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import box

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Georgia"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14

BASE_OUTPUT_DIR = "paper/figures"
FORMATS = ["pdf", "jpg"]
for fmt in FORMATS:
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, fmt), exist_ok=True)

# ── Load ZIP group membership ──────────────────────────────────────────────────
IMAGERY_MM   = "data/experiments/full_207_3-15/exp_gpt5mini_multimodal.json"
PERIPHERAL   = "data/experiments/full_207_3-15/suppl_gpt5mini_text_only_97.json"
PDE_PATH     = "data/processed/pde_by_zip.json"
GEOJSON_PATH = "data/raw/tx_zips.geojson"

with open(PDE_PATH) as f:
    pde_zips = set(json.load(f).keys())

with open(IMAGERY_MM) as f:
    imagery_zips = {r["query"]["zip"] for r in json.load(f)["records"]}

with open(PERIPHERAL) as f:
    peripheral_zips = {r["query"]["zip"] for r in json.load(f)["records"]}

all_zips = imagery_zips | peripheral_zips
print(f"Imagery ZIPs: {len(imagery_zips)}, Peripheral ZIPs: {len(peripheral_zips)}, Total: {len(all_zips)}")

# Categorise each ZIP
def categorise(z):
    if z in imagery_zips:
        return "Imagery core with PDE (N=104)" if z in pde_zips else "Imagery core, no PDE (N=6)"
    else:
        return "Peripheral with PDE (N=35)" if z in pde_zips else "Peripheral, no PDE (N=62)"

category_map = {z: categorise(z) for z in all_zips}
cats = list(set(category_map.values()))
print("Categories:", {c: sum(v == c for v in category_map.values()) for c in cats})

# Colour palette: 2 warm tones for imagery, 2 cool tones for peripheral
COLORS = {
    "Imagery core with PDE (N=104)": "#2166ac",   # strong blue
    "Imagery core, no PDE (N=6)":    "#92c5de",   # light blue
    "Peripheral with PDE (N=35)":    "#d6604d",   # medium red-orange
    "Peripheral, no PDE (N=62)":     "#f4a582",   # salmon-orange
}

# ── Load and merge GeoJSON ─────────────────────────────────────────────────────
gdf = gpd.read_file(GEOJSON_PATH)
zip_col = [c for c in gdf.columns if "zip" in c.lower() or "zcta" in c.lower()][0]

gdf_study = gdf[gdf[zip_col].isin(all_zips)].copy()
gdf_study["category"] = gdf_study[zip_col].map(category_map)

minx, miny, maxx, maxy = gdf_study.total_bounds
padding = 0.1
bbox = box(minx - padding, miny - padding, maxx + padding, maxy + padding)
gdf_bg = gdf[gdf.intersects(bbox) & ~gdf[zip_col].isin(all_zips)].copy()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 9))

# Background (non-study ZIPs)
gdf_bg.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.4)

# Plot each category with its colour
for cat, color in COLORS.items():
    subset = gdf_study[gdf_study["category"] == cat]
    if len(subset):
        subset.plot(ax=ax, color=color, edgecolor="white", linewidth=0.5)

ax.set_axis_off()
ax.set_xlim(minx - 0.05, maxx + 0.05)
ax.set_ylim(miny - 0.05, maxy + 0.05)

# Legend
legend_order = [
    "Imagery core with PDE (N=104)",
    "Imagery core, no PDE (N=6)",
    "Peripheral with PDE (N=35)",
    "Peripheral, no PDE (N=62)",
]
patches = [mpatches.Patch(color=COLORS[c], label=c) for c in legend_order]
ax.legend(
    handles=patches,
    loc="lower left",
    fontsize=12,
    framealpha=0.9,
    title="Evaluation Group",
    title_fontsize=12,
)

plt.tight_layout()

# Save
for fmt in FORMATS:
    out = os.path.join(BASE_OUTPUT_DIR, fmt, f"map_coverage_groups.{fmt}")
    kw = {"dpi": 300, "bbox_inches": "tight"} if fmt == "jpg" else {"bbox_inches": "tight"}
    fig.savefig(out, **kw)
    print(f"Saved {out}")
shutil.copy(
    os.path.join(BASE_OUTPUT_DIR, "pdf", "map_coverage_groups.pdf"),
    os.path.join(BASE_OUTPUT_DIR, "map_coverage_groups.pdf"),
)
plt.close()
print("Done.")
