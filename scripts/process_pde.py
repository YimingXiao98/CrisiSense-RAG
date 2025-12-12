"""Process PDE (Point Damage Estimate) data into ZIP-level aggregates."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

def process_pde():
    pde_path = Path("data/raw/pde/pde_junwei.shp")
    zip_path = Path("data/raw/tx_zips.geojson")
    output_path = Path("data/processed/pde_by_zip.json")

    logger.info(f"Loading PDE data from {pde_path}")
    pde = gpd.read_file(pde_path)
    logger.info(f"Loaded {len(pde)} damage points")

    logger.info(f"Loading ZIP boundaries from {zip_path}")
    zips = gpd.read_file(zip_path)
    logger.info(f"Loaded {len(zips)} ZIP polygons")

    # Ensure CRS match
    if pde.crs != zips.crs:
        logger.warning(f"CRS mismatch: PDE {pde.crs} vs ZIP {zips.crs}. Reprojecting...")
        zips = zips.to_crs(pde.crs)

    # Spatial Join
    logger.info("Performing spatial join...")
    joined = gpd.sjoin(pde, zips, how="inner", predicate="within")
    logger.info(f"Matched {len(joined)} points to ZIPs")

    # Inspect columns to find ZIP code column
    # Usually it's ZCTA5CE10 for TIGER/Line derived files
    zip_col = "ZCTA5CE10"
    if zip_col not in joined.columns:
        # Fallback search
        possible_cols = [c for c in joined.columns if "ZIP" in c.upper() or "ZCTA" in c.upper()]
        if possible_cols:
            zip_col = possible_cols[0]
            logger.info(f"Using inferred ZIP column: {zip_col}")
        else:
            raise ValueError(f"Could not find ZIP column in {joined.columns}")

    # Aggregate
    logger.info("Aggregating by ZIP...")
    stats = joined.groupby(zip_col)["pde"].agg(["mean", "count", "sum"])
    
    # Format for JSON
    output_dict = {}
    for zip_code, row in stats.iterrows():
        output_dict[str(zip_code)] = {
            "mean_pde": float(row["mean"]),
            "point_count": int(row["count"]),
            "sum_pde": float(row["sum"])
        }
    
    logger.info(f"Saving aggregates for {len(output_dict)} ZIPs to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

if __name__ == "__main__":
    process_pde()
