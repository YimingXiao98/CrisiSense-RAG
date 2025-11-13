"""Normalize FEMA NFIP claims exports into claim ground-truth records."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from dateutil import parser as date_parser
from loguru import logger

from app.core.dataio.loaders import save_parquet

CLAIM_FIELDS = (
    "id",
    "dateOfLoss",
    "reportedZipCode",
    "state",
    "countyCode",
    "latitude",
    "longitude",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "causeOfDamage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process FEMA NFIP claims CSV into claims.parquet")
    parser.add_argument("--input", required=True, help="Path to the FEMA NFIP CSV export")
    parser.add_argument("--output", required=True, help="Normalized output path (parquet-style JSON)")
    parser.add_argument("--state", help="Two-letter state filter (e.g., TX)")
    parser.add_argument("--county-codes", nargs="*", help="One or more FIPS county codes (e.g., 48201)")
    parser.add_argument("--zip-codes", nargs="*", help="Restrict to specific ZIP codes")
    parser.add_argument("--start-date", help="Earliest loss date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Latest loss date (YYYY-MM-DD)")
    return parser.parse_args()


def parse_date(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = date_parser.parse(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def parse_float(value: str | None) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def matches_filters(
    row: dict,
    *,
    state: Optional[str],
    county_codes: Optional[set[str]],
    zip_codes: Optional[set[str]],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> bool:
    if state and (row.get("state") or "").upper() != state.upper():
        return False
    if county_codes:
        county = (row.get("countyCode") or "").strip()
        if county and county not in county_codes:
            return False
    if zip_codes:
        reported_zip = (row.get("reportedZipCode") or "").strip()
        if reported_zip and reported_zip not in zip_codes:
            return False
    if start_dt or end_dt:
        loss_dt = parse_date(row.get("dateOfLoss"))
        if not loss_dt:
            return False
        if start_dt and loss_dt < start_dt:
            return False
        if end_dt and loss_dt > end_dt:
            return False
    return True


def normalize_row(row: dict) -> Optional[dict]:
    timestamp = parse_date(row.get("dateOfLoss"))
    if not timestamp:
        return None
    lat = parse_float(row.get("latitude"))
    lon = parse_float(row.get("longitude"))
    amount_components = [
        parse_float(row.get("amountPaidOnBuildingClaim")) or 0.0,
        parse_float(row.get("amountPaidOnContentsClaim")) or 0.0,
        parse_float(row.get("amountPaidOnIncreasedCostOfComplianceClaim")) or 0.0,
    ]
    total_amount = round(sum(amount_components), 2)
    return {
        "claim_id": row.get("id"),
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp.isoformat(),
        "severity": row.get("causeOfDamage") or row.get("floodEvent"),
        "zip": (row.get("reportedZipCode") or "").strip() or None,
        "amount": total_amount,
        "state": row.get("state"),
        "county_fips": row.get("countyCode"),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    county_codes = set(args.county_codes) if args.county_codes else None
    zip_codes = set(args.zip_codes) if args.zip_codes else None
    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)

    logger.info("Loading NFIP claims", path=input_path)
    normalized: list[dict] = []
    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(CLAIM_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV missing required fields: {missing}")
        for row in reader:
            if not matches_filters(
                row,
                state=args.state,
                county_codes=county_codes,
                zip_codes=zip_codes,
                start_dt=start_dt,
                end_dt=end_dt,
            ):
                continue
            record = normalize_row(row)
            if record:
                normalized.append(record)
    logger.info("Normalized claims", count=len(normalized))
    save_parquet(normalized, Path(args.output))
    logger.info("NFIP claims written", output=args.output)


if __name__ == "__main__":
    main()
