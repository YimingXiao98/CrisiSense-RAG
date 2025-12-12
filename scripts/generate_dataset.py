"""Generate a stratified evaluation dataset from insurance claims."""
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.dataio.loaders import load_parquet_table

def generate_stratified_queries(claims_path, output_path, n_queries=100):
    """Generate stratified queries with balanced damage levels."""
    
    print(f"Loading claims from {claims_path}...")
    raw = load_parquet_table(Path(claims_path))
    
    # Aggregate claims by ZIP and time window
    zip_windows = defaultdict(lambda: defaultdict(float))
    
    # Harvey-specific constraints
    HARVEY_START = datetime(2017, 8, 25)
    HARVEY_END = datetime(2017, 9, 30)
    HARVEY_ZIP_PREFIXES = ["77", "78"]  # Houston and surrounding Texas areas
    
    for record in raw:
        zip_val = record.get("reportedZipCode")
        if zip_val is None:
            continue
        
        try:
            zip_code = str(int(float(zip_val))).zfill(5)
        except (ValueError, TypeError):
            continue
        
        # Filter for Harvey-area ZIPs
        if not zip_code.startswith(tuple(HARVEY_ZIP_PREFIXES)):
            continue
        
        timestamp_str = record.get("timestamp")
        if not timestamp_str:
            continue
        
        try:
            timestamp = datetime.fromisoformat(str(timestamp_str).replace('Z', ''))
            # Strip timezone if present
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)
        except ValueError:
            continue
        
        # Filter for Harvey timeframe (with some buffer for recovery period)
        if not (HARVEY_START <= timestamp <= HARVEY_END):
            continue
        
        amount = float(record.get("netBuildingPaymentAmount") or 0.0)
        
        # Create 3-day windows (shorter for Harvey, which was intense but brief)
        window_start = timestamp.date()
        week_key = window_start.strftime("%Y-%m-%d")
        
        zip_windows[zip_code][week_key] += amount
    
    # Normalize to damage percentages
    max_total = max(sum(windows.values()) for windows in zip_windows.values())
    
    all_queries = []
    for zip_code, windows in zip_windows.items():
        for week_start, total in windows.items():
            if total <= 0:
                continue
            
            damage_pct = (total / max_total) * 100.0
            start_date = datetime.strptime(week_start, "%Y-%m-%d")
            end_date = start_date + timedelta(days=6)
            
            all_queries.append({
                "zip": zip_code,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "damage_pct": damage_pct,
                "total_amount": total
            })
    
    # Add zero-damage queries from Harvey ZIPs in post-disaster period
    # Get some actual Harvey ZIPs that had claims
    harvey_zips = list(zip_windows.keys())[:20] if zip_windows else ["77002", "77089", "77096", "77063"]
    
    for zip_code in harvey_zips:
        # Add late recovery periods with zero damage
        for offset in [35, 42, 49]:  # October 2017
            start_date = datetime(2017, 8, 25) + timedelta(days=offset)
            if start_date > datetime(2017, 9, 30):
                continue
            end_date = start_date + timedelta(days=3)
            all_queries.append({
                "zip": zip_code,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "damage_pct": 0.0,
                "total_amount": 0.0
            })
    
    # Stratify
    high_damage = [q for q in all_queries if q["damage_pct"] > 20.0]
    low_damage = [q for q in all_queries if 0.1 <= q["damage_pct"] <= 20.0]
    no_damage = [q for q in all_queries if q["damage_pct"] < 0.1]
    
    print(f"Available: {len(high_damage)} high, {len(low_damage)} low, {len(no_damage)} zero")
    
    # Sample
    n_per_group = n_queries // 3
    sampled = []
    
    sampled.extend(random.sample(high_damage, min(n_per_group, len(high_damage))))
    sampled.extend(random.sample(low_damage, min(n_per_group, len(low_damage))))
    sampled.extend(random.sample(no_damage, min(n_queries - len(sampled), len(no_damage))))
    
    # Shuffle
    random.shuffle(sampled)
    
    # Format for eval
    queries = []
    for q in sampled[:n_queries]:
        queries.append({
            "zip": q["zip"],
            "start": q["start"],
            "end": q["end"],
            "k_tiles": 10,
            "n_text": 20,
            "comment": f"Damage: {q['damage_pct']:.2f}%, Amount: ${q['total_amount']:,.0f}"
        })
    
    # Save
    output = {
        "data_dir": "data",
        "provider": "gemini",
        "ground_truth": "claims",
        "queries": queries
    }
    
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"Saved {len(queries)} queries to {output_path}")
    
    # Print distribution
    high_count = sum(1 for q in sampled[:n_queries] if q["damage_pct"] > 20.0)
    low_count = sum(1 for q in sampled[:n_queries] if 0.1 <= q["damage_pct"] <= 20.0)
    zero_count = sum(1 for q in sampled[:n_queries] if q["damage_pct"] < 0.1)
    
    print(f"Distribution: {high_count} high, {low_count} low, {zero_count} zero")

if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    generate_stratified_queries(
        "data/processed/claims.parquet",
        "config/queries_100_stratified.json",
        n_queries=100
    )
