"""Reverse-geocode geotagged tweets to ZIP codes.

106K tweets (0.4%) have lat/lon coordinates but no ZIP code.
This script:
1. Spatial-joins geotagged tweets against harris_zips.geojson boundaries
2. Saves tweet_id -> zip mapping to data/processed/tweet_zip_lookup.json
3. Updates data/processed/tweets.parquet with the new zip column
4. Patches data/processed/text_corpus.jsonl in-place with the new ZIPs
   (so the retriever picks them up without a full re-index)

Usage:
    conda run -n harvey-rag python scripts/geocode_tweets.py
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

PROCESSED = Path("data/processed")
RAW = Path("data/raw")


def geocode_tweets() -> None:
    logger.info("Loading tweets.parquet …")
    df = pd.read_parquet(PROCESSED / "tweets.parquet")
    logger.info(f"Total tweets: {len(df):,}")

    geotagged = df[df["lat"].notna() & df["lon"].notna()].copy()
    logger.info(f"Geotagged tweets: {len(geotagged):,}")

    logger.info("Loading ZIP boundary shapefile …")
    zips = gpd.read_file(RAW / "zip_boundaries" / "tl_2017_us_zcta510.shp")
    zips = zips.rename(columns={"ZCTA5CE10": "zip"})
    zips["zip"] = zips["zip"].astype(str).str.zfill(5)
    # Filter to Harris County area (rough bounding box)
    zips = zips.cx[-96.0:-94.8, 29.4:30.2]
    logger.info(f"ZIP polygons in Harris County area: {len(zips)}")

    # Build GeoDataFrame for geotagged tweets
    gdf = gpd.GeoDataFrame(
        geotagged[["tweet_id", "lat", "lon"]],
        geometry=gpd.points_from_xy(geotagged["lon"], geotagged["lat"]),
        crs="EPSG:4326",
    )
    if zips.crs is None:
        zips = zips.set_crs("EPSG:4326")
    else:
        zips = zips.to_crs("EPSG:4326")

    logger.info("Spatial joining tweets to ZIP boundaries …")
    joined = gpd.sjoin(gdf, zips[["zip", "geometry"]], how="left", predicate="within")
    joined = joined[["tweet_id", "zip"]].drop_duplicates("tweet_id")

    matched = joined["zip"].notna().sum()
    logger.info(f"Matched {matched:,}/{len(geotagged):,} geotagged tweets to a ZIP")

    # Build lookup keyed by NUMERIC tweet id (strip "tag:...:" prefix)
    def extract_numeric_id(raw_id: str) -> str:
        """Convert 'tag:search.twitter.com,2005:123456' → '123456'."""
        s = str(raw_id)
        return s.split(":")[-1] if ":" in s else s

    lookup_by_tag = {
        str(row.tweet_id): str(row.zip)
        for row in joined.itertuples()
        if pd.notna(row.zip)
    }
    # Also key by numeric ID for corpus matching
    lookup = {
        extract_numeric_id(tag_id): zip_code
        for tag_id, zip_code in lookup_by_tag.items()
    }
    lookup_path = PROCESSED / "tweet_zip_lookup.json"
    lookup_path.write_text(json.dumps(lookup, indent=2))
    logger.info(f"Saved lookup: {lookup_path} ({len(lookup):,} entries)")

    # Update tweets.parquet using original tag-format IDs
    zip_series = pd.Series(lookup_by_tag, name="zip_geo")
    df["tweet_id_str"] = df["tweet_id"].astype(str)
    df["zip"] = df["tweet_id_str"].map(zip_series)
    df = df.drop(columns=["tweet_id_str"])
    df.to_parquet(PROCESSED / "tweets.parquet", index=False)
    logger.info("Updated tweets.parquet")

    # Patch text_corpus.jsonl in-place using numeric IDs
    corpus_path = PROCESSED / "text_corpus.jsonl"
    patched = 0
    lines_out = []
    with corpus_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("source") == "tweet":
                # doc_id format: "tweet_123456789"
                numeric_id = str(doc.get("doc_id", "")).replace("tweet_", "")
                if numeric_id in lookup and not doc.get("zip"):
                    doc["zip"] = lookup[numeric_id]
                    if "payload" in doc and isinstance(doc["payload"], dict):
                        doc["payload"]["zip"] = lookup[numeric_id]
                    patched += 1
            lines_out.append(json.dumps(doc))

    corpus_path.write_text("\n".join(lines_out) + "\n")
    logger.info(f"Patched {patched:,} tweet docs in text_corpus.jsonl with ZIP codes")

    # Summary
    zip_coverage = joined.groupby("zip").size().sort_values(ascending=False)
    logger.info(f"\nTop 10 ZIPs by geotagged tweet count:")
    for zip_code, count in zip_coverage.head(10).items():
        logger.info(f"  {zip_code}: {count}")


if __name__ == "__main__":
    geocode_tweets()
