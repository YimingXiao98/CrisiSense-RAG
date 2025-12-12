#!/usr/bin/env python
"""
Add 311 call counts as alternative ground truth for experiments.
Calculates call counts per ZIP for each query's time window.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger


def load_311_data(path: Path) -> pd.DataFrame:
    """Load 311 calls data."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Clean ZIP codes
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.replace(r"\.0$", "", regex=True)

    logger.info(f"Loaded {len(df)} 311 records")
    return df


def count_calls_for_query(df: pd.DataFrame, zip_code: str, start: str, end: str) -> int:
    """Count 311 calls for a specific ZIP and time window."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    mask = (
        (df["zip"] == str(zip_code))
        & (df["timestamp"] >= start_dt)
        & (df["timestamp"] <= end_dt)
    )
    return int(mask.sum())


def add_311_to_experiment(exp_path: Path, df_311: pd.DataFrame) -> None:
    """Add 311 call counts to an experiment file."""
    logger.info(f"Processing {exp_path}")

    with open(exp_path) as f:
        data = json.load(f)

    for record in data["records"]:
        query = record["query"]
        zip_code = query.get("zip")
        start = query.get("start_date") or query.get("start")
        end = query.get("end_date") or query.get("end")

        if zip_code and start and end:
            call_count = count_calls_for_query(df_311, zip_code, start, end)

            # Add to ground_truth
            if "ground_truth" not in record:
                record["ground_truth"] = {}
            record["ground_truth"]["call_311_count"] = call_count

            # Also add to retrieval_metadata for convenience
            if "retrieval_metadata" not in record:
                record["retrieval_metadata"] = {}
            record["retrieval_metadata"]["call_311_count"] = call_count

    # Save updated file
    with open(exp_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.success(f"Updated {exp_path}")


def main():
    # Load 311 data
    data_dir = Path("data")
    df_311 = load_311_data(data_dir / "processed" / "311.parquet")

    # Process all experiment files
    exp_dir = data_dir / "experiments"
    exp_files = list(exp_dir.glob("exp*.json"))

    for exp_path in exp_files:
        if "intermediate" in exp_path.name:
            continue
        try:
            add_311_to_experiment(exp_path, df_311)
        except Exception as e:
            logger.error(f"Failed to process {exp_path}: {e}")

    logger.info("Done adding 311 counts to all experiments!")

    # Calculate correlation summary
    logger.info("\n=== 311 Call Correlation Analysis ===")
    for exp_path in exp_files:
        if "intermediate" in exp_path.name:
            continue
        try:
            with open(exp_path) as f:
                data = json.load(f)

            preds = []
            calls = []
            for r in data["records"]:
                pred = r.get("model_response", {}).get("damage_pct", 0)
                call_count = r.get("ground_truth", {}).get("call_311_count", 0)
                preds.append(pred)
                calls.append(call_count)

            if sum(calls) > 0:  # Only if we have calls
                from scipy.stats import spearmanr

                corr, pval = spearmanr(preds, calls)
                logger.info(f"{exp_path.name}: Spearman ρ = {corr:.3f} (p={pval:.4f})")
            else:
                logger.warning(f"{exp_path.name}: No 311 calls in time windows")
        except Exception as e:
            logger.error(f"Failed correlation for {exp_path}: {e}")


if __name__ == "__main__":
    main()
