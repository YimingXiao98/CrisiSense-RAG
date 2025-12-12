#!/usr/bin/env python
"""
Generate a 50-query evaluation set:
- 35 queries from high 311 coverage ZIPs (urban Houston)
- 15 queries from suburban/outlying ZIPs
"""
import json
import pandas as pd
from datetime import date, timedelta
import random

random.seed(42)

# Harvey time periods (peak impact)
HARVEY_WINDOWS = [
    (date(2017, 8, 25), date(2017, 8, 31)),  # Peak flooding
    (date(2017, 8, 26), date(2017, 9, 1)),
    (date(2017, 8, 27), date(2017, 9, 2)),
    (date(2017, 8, 28), date(2017, 9, 3)),
    (date(2017, 8, 29), date(2017, 9, 4)),
    (date(2017, 9, 1), date(2017, 9, 7)),   # Post-peak
    (date(2017, 9, 4), date(2017, 9, 10)),
]

def generate_queries():
    # Load 311 data
    df = pd.read_parquet('data/processed/311.parquet')
    df['zip'] = df['zip'].astype(str).str.replace('.0', '', regex=False)
    
    # Get top 35 ZIPs with most 311 calls
    zip_counts = df[df['zip'] != 'None'].groupby('zip').size().sort_values(ascending=False)
    high_311_zips = zip_counts.head(35).index.tolist()
    
    # Suburban ZIPs (from existing validation set, not in 311)
    suburban_zips = ['77479', '77486', '77642', '77833', '77539', '78412', 
                     '77407', '77520', '77571', '77521', '77468', '77630',
                     '77386', '77058', '77546']  # Added a few more
    
    queries = []
    
    # Generate 35 queries for high-311 ZIPs
    for i, zip_code in enumerate(high_311_zips):
        window = HARVEY_WINDOWS[i % len(HARVEY_WINDOWS)]
        queries.append({
            "zip": zip_code,
            "start": str(window[0]),
            "end": str(window[1]),
            "category": "high_311_coverage"
        })
    
    # Generate 15 queries for suburban ZIPs
    for i, zip_code in enumerate(suburban_zips[:15]):
        window = HARVEY_WINDOWS[i % len(HARVEY_WINDOWS)]
        queries.append({
            "zip": zip_code,
            "start": str(window[0]),
            "end": str(window[1]),
            "category": "suburban"
        })
    
    return queries

def main():
    queries = generate_queries()
    
    config = {
        "data_dir": "data",
        "provider": "gemini",
        "enable_visual": True,
        "description": "50-query evaluation: 35 high-311-coverage ZIPs + 15 suburban ZIPs",
        "queries": queries
    }
    
    with open('config/queries_50_mixed.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated {len(queries)} queries")
    print(f"  High-311 coverage: {sum(1 for q in queries if q['category'] == 'high_311_coverage')}")
    print(f"  Suburban: {sum(1 for q in queries if q['category'] == 'suburban')}")
    print(f"Saved to config/queries_50_mixed.json")

if __name__ == "__main__":
    main()
