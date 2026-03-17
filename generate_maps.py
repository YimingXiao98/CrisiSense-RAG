import geopandas as gpd
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
from shapely.geometry import box

# Set font to a professional serif font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Georgia"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 18  # Increase base font size

# Extent maps: GPT-5-mini multimodal (best flood extent, 8.86% MAE)
# Damage maps: Gemini 3 Flash multimodal (best damage, 10.14% MAE)
EXTENT_RESULTS_PATH = "data/experiments/full_207_3-15/exp_gpt5mini_multimodal.json"
DAMAGE_RESULTS_PATH = "data/experiments/gemini3_3-15/exp_gemini3_multimodal.json"
GEOJSON_PATH = "data/raw/tx_zips.geojson"
PDE_PATH = "data/processed/pde_by_zip.json"
BASE_OUTPUT_DIR = "paper/figures"

# Define formats to save
FORMATS = ["png", "pdf", "jpg"]
for fmt in FORMATS:
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, fmt), exist_ok=True)


def save_fig(fig, filename_base):
    """Helper to save figure in multiple formats in separate folders and root."""
    for fmt in FORMATS:
        out_path = os.path.join(BASE_OUTPUT_DIR, fmt, f"{filename_base}.{fmt}")
        if fmt == "jpg" or fmt == "png":
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    # Also copy PDF to root so LaTeX can find it directly
    import shutil
    shutil.copy(
        os.path.join(BASE_OUTPUT_DIR, "pdf", f"{filename_base}.pdf"),
        os.path.join(BASE_OUTPUT_DIR, f"{filename_base}.pdf"),
    )


print(f"Loading PDE ZIP coverage from {PDE_PATH}...")
with open(PDE_PATH, "r") as f:
    pde_by_zip = json.load(f)
pde_zips = set(pde_by_zip.keys())
print(f"PDE coverage ZIPs: {len(pde_zips)}")


def extract_records(results_path):
    print(f"Loading results from {results_path}...")
    with open(results_path, "r") as f:
        data = json.load(f)
    out = {}
    for r in data["records"]:
        z = r["query"]["zip"]
        try:
            extent_pred = r["model_response"]["flood_extent_pct"]
        except:
            extent_pred = 0.0
        try:
            extent_gt = r["ground_truth"]["flooded_pct"]
        except:
            extent_gt = 0.0
        try:
            damage_pred = r["model_response"]["damage_severity_pct"]
        except:
            damage_pred = 0.0
        damage_gt = (
            r["ground_truth"]["pde_damage_score"]
            if z in pde_zips and "pde_damage_score" in r["ground_truth"]
            else None
        )
        out[z] = {
            "ZIP": z,
            "Extent_Pred": extent_pred,
            "Extent_GT": extent_gt,
            "Damage_Pred": damage_pred,
            "Damage_GT": damage_gt,
        }
    return out


ext_records = extract_records(EXTENT_RESULTS_PATH)
dmg_records = extract_records(DAMAGE_RESULTS_PATH)

# Build combined df: extent from gpt5mini, damage from qwen35
records = []
for z, er in ext_records.items():
    dr = dmg_records.get(z, {})
    records.append({
        "ZIP": z,
        "Extent_Pred": er["Extent_Pred"],
        "Extent_GT": er["Extent_GT"],
        "Damage_Pred": dr.get("Damage_Pred", 0.0),
        "Damage_GT": er["Damage_GT"],  # GT is the same regardless of model
    })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} records. Missing Damage GT: {df['Damage_GT'].isna().sum()}")

# FILTER TO CORE AREA ONLY (ZIPs with PDE coverage)
print(f"\nFiltering to core area (ZIPs with PDE coverage)...")
df_core = df[df["ZIP"].isin(pde_zips)].copy()
print(f"Core area: {len(df_core)} ZIPs (filtered from {len(df)})")
df = df_core  # Use core area for all visualizations

print(f"Loading GeoJSON from {GEOJSON_PATH}...")
try:
    gdf = gpd.read_file(GEOJSON_PATH)
    print("GeoJSON loaded successfully.")

    # Filter for relevant ZIPs
    zip_col = [c for c in gdf.columns if "zip" in c.lower() or "zcta" in c.lower()][0]
    print(f"Using ZIP column: {zip_col}")

    # Create the Results GeoDataFrame
    gdf_merged = gdf[gdf[zip_col].isin(df["ZIP"])].copy()
    gdf_merged = gdf_merged.merge(df, left_on=zip_col, right_on="ZIP")

    # Create Background GeoDataFrame (Context)
    # Get bounding box of the results to define the "Study Area" view
    minx, miny, maxx, maxy = gdf_merged.total_bounds
    # Add some padding
    padding = 0.1  # degrees
    bbox = box(minx - padding, miny - padding, maxx + padding, maxy + padding)

    # Filter full GDF to this bounding box for the background layer
    gdf_bg = gdf[gdf.intersects(bbox) & ~gdf[zip_col].isin(df["ZIP"])].copy()

    print(f"Background layer has {len(gdf_bg)} ZIP codes.")

    # Plotting function
    def plot_map(column, title, filename_base):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot Background (Gray) - Context outside study area
        gdf_bg.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.5)

        # Plot Base Layer for Study Area (Darker Gray) - explicit "No Data" color
        gdf_merged.plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=0.5)

        # Plot Data (Color)
        gdf_merged.dropna(subset=[column]).plot(
            column=column,
            ax=ax,
            legend=True,
            legend_kwds={
                "label": "Percentage (%)",
                "orientation": "vertical",
                "location": "left",
                "shrink": 0.5,
                "pad": 0.02,
                "fraction": 0.04,
            },
            cmap="OrRd",
            edgecolor="black",
            linewidth=0.5,
            vmin=0,
            vmax=100,
        )

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_axis_off()

        # Set limits to the results bounding box + padding
        ax.set_xlim(minx - 0.05, maxx + 0.05)
        ax.set_ylim(miny - 0.05, maxy + 0.05)

        plt.tight_layout()
        save_fig(fig, filename_base)
        plt.close()

    # Generate Maps
    plot_map("Extent_GT", "Ground Truth Flood Extent (%)", "map_extent_gt_v2")
    plot_map("Extent_Pred", "Predicted Flood Extent (%)", "map_extent_pred_v2")
    plot_map("Damage_GT", "Ground Truth Damage Severity (%)", "map_damage_gt_v2")
    plot_map("Damage_Pred", "Predicted Damage Severity (%)", "map_damage_pred_v2")

    # Difference Maps
    gdf_merged["Extent_Diff"] = gdf_merged["Extent_Pred"] - gdf_merged["Extent_GT"]
    gdf_merged["Damage_Diff"] = gdf_merged["Damage_Pred"] - gdf_merged["Damage_GT"]

    def plot_diff_map(column, title, filename_base):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot Background
        gdf_bg.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.5)

        # Plot Data
        gdf_merged.dropna(subset=[column]).plot(
            column=column,
            ax=ax,
            legend=True,
            legend_kwds={
                "label": "Difference (pp)",
                "orientation": "vertical",
                "location": "left",
                "shrink": 0.5,
                "pad": 0.02,
                "fraction": 0.04,
            },
            cmap="RdBu_r",
            edgecolor="black",
            linewidth=0.5,
            vmin=-100,
            vmax=100,
        )

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_axis_off()
        ax.set_xlim(minx - 0.05, maxx + 0.05)
        ax.set_ylim(miny - 0.05, maxy + 0.05)

        plt.tight_layout()
        save_fig(fig, filename_base)
        plt.close()

    plot_diff_map(
        "Extent_Diff", "Extent Prediction Error (Pred - GT)", "map_extent_error_v2"
    )
    plot_diff_map(
        "Damage_Diff", "Damage Prediction Error (Pred - GT)", "map_damage_error_v2"
    )

except Exception as e:
    print(f"Error processing GeoJSON: {e}")
