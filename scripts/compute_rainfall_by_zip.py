"""Compute peak Harvey rainfall per ZIP via IDW interpolation.

Reads the HCFCD 5-minute rainfall data (data/raw/2017-5min.csv) and
gage locations (data/raw/GageLocations.csv), sums cumulative rainfall
per station over Aug 25-Sep 1, then uses inverse-distance weighting
to interpolate that value to each Harris County ZIP centroid.

Output: data/processed/rainfall_peak_by_zip.json
  {
    "77096": {"peak_rain_in": 43.2, "nearest_gage": "105", "n_gages": 5},
    ...
  }

Usage:
    conda run -n harvey-rag python scripts/compute_rainfall_by_zip.py
"""

import json
import math
from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

HARVEY_START = "2017-08-25"
HARVEY_END = "2017-09-01"
IDW_POWER = 2          # Inverse distance weighting exponent
MAX_RADIUS_KM = 50     # Only use gages within this radius
MIN_GAGES = 1          # Use at least this many nearest gages


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def compute_rainfall_by_zip() -> None:
    # 1. Parse 5-min rainfall (wide format: row=time, col=site_id_sensor_id)
    logger.info("Parsing 5-min rainfall CSV …")
    df = pd.read_csv(RAW / "2017-5min.csv", skiprows=8, low_memory=False)
    df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Filter to Harvey peak window
    peak = df[(df["datetime"] >= HARVEY_START) & (df["datetime"] <= HARVEY_END)].copy()
    logger.info(f"Peak window rows: {len(peak)}")

    # Sum per sensor column (cumulative rainfall in inches)
    sensor_cols = [c for c in peak.columns if c != "datetime"]
    rain_totals = peak[sensor_cols].apply(pd.to_numeric, errors="coerce").sum()

    # Column names are like "100_100", "105_105" — site_id is the first part
    # (Some columns have different sensor on same site, take max per site_id)
    site_rain: dict[str, float] = {}
    for col, val in rain_totals.items():
        site_id = str(col).split("_")[0]
        site_rain[site_id] = max(site_rain.get(site_id, 0.0), float(val))

    logger.info(f"Stations with rainfall data: {len(site_rain)}")

    # 2. Load gage locations
    gages = pd.read_csv(RAW / "GageLocations.csv")
    gages["Site_Id"] = gages["Site_Id"].astype(str)
    gages = gages.drop_duplicates("Site_Id")[["Site_Id", "lat", "lng"]].copy()

    # Merge with rainfall totals
    gages["rain_in"] = gages["Site_Id"].map(site_rain)
    gages = gages[gages["rain_in"].notna() & (gages["rain_in"] > 0)].copy()
    logger.info(f"Gages with positive Harvey rainfall: {len(gages)}")
    logger.info(f"  Range: {gages['rain_in'].min():.1f} – {gages['rain_in'].max():.1f} inches")

    # 3. Load ZIP centroids (nested under "zip_codes" key)
    raw_coords = json.loads((PROCESSED / "zip_coordinates.json").read_text())
    zip_coords = raw_coords.get("zip_codes", raw_coords)
    logger.info(f"ZIP centroids: {len(zip_coords)}")

    # 4. IDW interpolation
    results: dict[str, dict] = {}
    gage_list = list(gages.itertuples())

    for zip_code, coords in zip_coords.items():
        if not isinstance(coords, dict):
            continue
        zlat = coords.get("lat") or coords.get("latitude")
        zlon = coords.get("lon") or coords.get("lng") or coords.get("longitude")
        if zlat is None or zlon is None:
            continue

        distances = [
            (haversine_km(zlat, zlon, g.lat, g.lng), g.rain_in, g.Site_Id)
            for g in gage_list
        ]
        distances.sort(key=lambda x: x[0])

        # Filter to radius, keep at least MIN_GAGES nearest
        within = [d for d in distances if d[0] <= MAX_RADIUS_KM]
        if len(within) < MIN_GAGES:
            within = distances[:MIN_GAGES]
        if not within:
            continue

        # IDW
        weights = [1.0 / max(d[0], 0.1) ** IDW_POWER for d in within]
        total_w = sum(weights)
        idw_rain = sum(w * d[1] for w, d in zip(weights, within)) / total_w

        results[zip_code] = {
            "peak_rain_in": round(idw_rain, 2),
            "nearest_gage": within[0][2],
            "nearest_dist_km": round(within[0][0], 1),
            "n_gages_used": len(within),
        }

    out_path = PROCESSED / "rainfall_peak_by_zip.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.success(f"Saved {len(results)} ZIP rainfall estimates → {out_path}")

    # Summary stats
    vals = [v["peak_rain_in"] for v in results.values()]
    logger.info(f"  Rainfall range: {min(vals):.1f} – {max(vals):.1f} inches")
    logger.info(f"  Mean: {sum(vals)/len(vals):.1f} inches")


if __name__ == "__main__":
    compute_rainfall_by_zip()
