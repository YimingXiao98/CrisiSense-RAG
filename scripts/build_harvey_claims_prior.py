"""Build per-ZIP pre-event NFIP historical flood risk prior.

Filters claims.parquet to PRE-HARVEY claims only (dateOfLoss < 2017-08-01,
Harris County TX), aggregates by ZIP, and saves a structured JSON
vulnerability prior for the context packager.

Using only pre-event data avoids ground-truth leakage: in real deployment
during an active disaster, no claims from that event would yet be available.

Output: data/processed/harvey_claims_by_zip.json
  {
    "77096": {
      "claim_count": 9828,
      "years_span": 39,
      "avg_annual_claims": 252,
      "avg_building_payout": 52000.0,
      "risk_tier": "high"
    },
    ...
  }

Usage:
    conda run -n harvey-rag python scripts/build_harvey_claims_prior.py
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

PROCESSED = Path("data/processed")
RAW = Path("data/raw")


def build_harvey_claims_prior() -> None:
    logger.info("Loading claims.parquet …")
    df = pd.read_parquet(PROCESSED / "claims.parquet")
    logger.info(f"Total claims: {len(df):,}")

    df["dateOfLoss"] = pd.to_datetime(df["dateOfLoss"], errors="coerce")

    # PRE-HARVEY only — strictly before Aug 1 2017 to avoid leakage
    pre = df[
        (df["dateOfLoss"] < "2017-08-01")
        & (df["countyCode"].astype(str).str.strip() == "48201.0")
    ].copy()
    logger.info(f"Pre-Harvey Harris County claims: {len(pre):,}")
    logger.info(f"  Date range: {pre['dateOfLoss'].min().date()} to {pre['dateOfLoss'].max().date()}")

    # Years spanned (for annualised rate)
    min_year = pre["dateOfLoss"].dt.year.min()
    max_year = 2017  # up to (not including) Harvey
    years_span = max_year - min_year

    zip_col = "reportedZipCode"
    pre[zip_col] = pre[zip_col].astype(str).str.split(".").str[0].str.zfill(5)
    pre = pre[pre[zip_col].str.match(r"^\d{5}$")]

    agg = pre.groupby(zip_col).agg(
        claim_count=(zip_col, "count"),
        avg_building_payout=("amountPaidOnBuildingClaim", "mean"),
        median_water_depth_ft=("waterDepth", "median"),
    ).reset_index()

    # Risk tier thresholds (annual claim rate per ZIP)
    # High: >50 claims/yr, Moderate: 10-50, Low: <10
    def risk_tier(annual_rate: float) -> str:
        if annual_rate >= 50:
            return "high"
        if annual_rate >= 10:
            return "moderate"
        return "low"

    results: dict[str, dict] = {}
    for _, row in agg.iterrows():
        count = int(row["claim_count"])
        if count < 1:
            continue
        avg_payout = float(row["avg_building_payout"] or 0)
        median_depth = float(row["median_water_depth_ft"]) if pd.notna(row["median_water_depth_ft"]) else 0.0
        annual_rate = round(count / years_span, 1)
        results[row[zip_col]] = {
            "claim_count": count,
            "years_span": years_span,
            "avg_annual_claims": annual_rate,
            "avg_building_payout": round(avg_payout, 0),
            "median_water_depth_ft": round(median_depth, 1),
            "risk_tier": risk_tier(annual_rate),
        }

    out_path = PROCESSED / "harvey_claims_by_zip.json"
    out_path.write_text(json.dumps(results, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x)))
    logger.success(f"Saved {len(results)} ZIP pre-Harvey risk priors → {out_path}")

    counts = [v["claim_count"] for v in results.values()]
    tiers = {t: sum(1 for v in results.values() if v["risk_tier"] == t) for t in ["high","moderate","low"]}
    logger.info(f"  ZIPs with pre-Harvey data: {len(results)}")
    logger.info(f"  Claim count range: {min(counts)} – {max(counts)}")
    logger.info(f"  Risk tiers: {tiers}")


if __name__ == "__main__":
    build_harvey_claims_prior()
