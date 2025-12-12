
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, date
from math import sqrt
from app.core.dataio.loaders import load_parquet_table
from app.core.dataio.storage import DataLocator

class ClaimsGroundTruth:
    """Aggregate insurance claims to estimate structural damage intensity."""

    def __init__(self, claims_path: Path) -> None:
        if not claims_path.exists():
            raise FileNotFoundError(f"Claims table not found: {claims_path}")
        raw = load_parquet_table(claims_path)
        print(f"DEBUG: Raw table loaded, {len(raw)} records.")
        self.by_zip = defaultdict(list)
        totals = defaultdict(float)
        for record in raw:
            # Use correct column names from parquet schema
            zip_val = record.get("reportedZipCode")
            if zip_val is None:
                continue
            # Handle float zip codes
            try:
                zip_code = str(int(float(zip_val))).zfill(5)
            except (ValueError, TypeError):
                continue
            
            timestamp = self._parse_ts(record.get("timestamp"))
            if not timestamp:
                continue
                
            amount = float(record.get("netBuildingPaymentAmount") or 0.0)
            self.by_zip[zip_code].append({"timestamp": timestamp, "amount": amount})
            totals[zip_code] += amount
        self.max_total = max(totals.values()) if totals else 0.0
        print(f"DEBUG: Loaded claims for {len(self.by_zip)} ZIPs. Max total: {self.max_total}")
        if "77063" in self.by_zip:
            print(f"DEBUG: 77063 has {len(self.by_zip['77063'])} records.")
        else:
            print("DEBUG: 77063 NOT FOUND in by_zip")

    def score(self, zip_code: str, start_date, end_date):
        records = self.by_zip.get(str(zip_code).zfill(5), [])
        if not records:
            return {"damage_pct": 0.0, "claim_count": 0, "total_amount": 0.0}

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        # Ensure timezone awareness matches (naive vs aware)
        # The parquet timestamps are likely aware (UTC). We need to make start/end aware or strip tz.
        # Simplest is to strip tz from record timestamp if present.
        
        total = 0.0
        count = 0
        for record in records:
            ts = record["timestamp"]
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)
                
            if start_dt <= ts <= end_dt:
                total += record["amount"]
                count += 1
        if total <= 0.0 or self.max_total <= 0:
            return {"damage_pct": 0.0, "claim_count": count, "total_amount": total}
        pct = min((total / self.max_total) * 100.0, 100.0)
        return {"damage_pct": round(pct, 2), "claim_count": count, "total_amount": total}

    @staticmethod
    def _parse_ts(value):
        if not value:
            return None
        try:
            # Handle pandas Timestamp objects directly if possible, or string conversion
            return datetime.fromisoformat(str(value).replace('Z', ''))
        except ValueError:
            return None

def main():
    # Load existing results
    results_path = Path("data/processed/eval_results_multimodal.json")
    data = json.loads(results_path.read_text())
    
    # Initialize Ground Truth
    locator = DataLocator(Path("data"))
    gt = ClaimsGroundTruth(locator.table_path("claims"))
    
    metrics_rows = []
    abs_errors = []
    sq_errors = []
    
    print("Rescoring results...")
    
    for record in data["records"]:
        # Parse dates
        start = date.fromisoformat(record["start"])
        end = date.fromisoformat(record["end"])
        
        # Get correct ground truth
        truth = gt.score(record["zip"], start, end)
        
        # Update record
        record["actual_damage_pct"] = truth["damage_pct"]
        record["claim_count"] = truth["claim_count"]
        record["total_claim_amount"] = round(truth["total_amount"], 2)
        
        # Recalculate error
        diff = record["pred_damage_pct"] - truth["damage_pct"]
        record["abs_error"] = round(abs(diff), 2)
        
        abs_errors.append(abs(diff))
        sq_errors.append(diff**2)
        metrics_rows.append(record)
        
        print(f"ZIP {record['zip']}: Pred={record['pred_damage_pct']}, Actual={truth['damage_pct']}, Error={record['abs_error']}")

    # Recalculate summary
    summary = {
        "mae": round(sum(abs_errors) / len(abs_errors), 3),
        "rmse": round(sqrt(sum(sq_errors) / len(sq_errors)), 3),
        "count": len(abs_errors),
    }
    
    output = {"summary": summary, "records": metrics_rows}
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nNew MAE: {summary['mae']}")

if __name__ == "__main__":
    main()
