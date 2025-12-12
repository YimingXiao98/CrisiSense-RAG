#!/usr/bin/env python3
"""Process rainfall gauge data for integration into RAG pipeline.

Input files:
- data/raw/2017-5min.csv: 5-minute rainfall measurements (all sensors)
- data/raw/gauges.csv: Sensor metadata with ZIP codes

Output:
- data/processed/rainfall_by_zip.json: Daily rainfall totals per ZIP code
- Updates to text corpus with rainfall context
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict


def load_5min_data(filepath: Path) -> pd.DataFrame:
    """Load and clean the 5-minute rainfall data."""
    print(f"Loading 5-min data from {filepath}...")
    
    # Skip metadata rows (8 rows before header on row 9)
    df = pd.read_csv(filepath, skiprows=8)
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['data_time_utc'].iloc[0]} to {df['data_time_utc'].iloc[-1]}")
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['data_time_utc'], format='%m/%d/%y %H:%M', errors='coerce')
    
    # Filter to Harvey period (Aug 20 - Sep 15, 2017)
    harvey_start = pd.Timestamp('2017-08-20')
    harvey_end = pd.Timestamp('2017-09-15')
    
    df_harvey = df[(df['timestamp'] >= harvey_start) & (df['timestamp'] <= harvey_end)].copy()
    print(f"  Harvey period: {len(df_harvey)} rows")
    
    return df_harvey


def load_sensor_metadata(filepath: Path) -> dict:
    """Load sensor to ZIP code mapping from gauges.csv."""
    print(f"Loading sensor metadata from {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"  Found {len(df)} sensor records")
    
    # Get unique sensor_id -> zipcode mapping
    sensor_to_zip = {}
    for _, row in df.iterrows():
        sensor_id = str(int(row['sensor_id']))
        zipcode = str(int(row['zipcode'])) if pd.notna(row['zipcode']) else None
        if zipcode:
            sensor_to_zip[sensor_id] = zipcode
    
    print(f"  Mapped {len(sensor_to_zip)} sensors to ZIP codes")
    unique_zips = set(sensor_to_zip.values())
    print(f"  Covering {len(unique_zips)} unique ZIPs: {sorted(unique_zips)}")
    
    return sensor_to_zip


def aggregate_by_zip_day(df: pd.DataFrame, sensor_to_zip: dict) -> dict:
    """Aggregate rainfall by ZIP code and date."""
    print("Aggregating rainfall by ZIP and date...")
    
    # Melt the wide format to long
    id_vars = ['timestamp', 'data_time_utc']
    value_vars = [col for col in df.columns if col not in id_vars and '_' in col]
    
    results = defaultdict(lambda: defaultdict(list))
    
    for col in value_vars:
        # Column format: siteID_sensorID
        parts = col.split('_')
        sensor_id = parts[0]  # First part is sensor ID
        
        if sensor_id not in sensor_to_zip:
            continue
        
        zipcode = sensor_to_zip[sensor_id]
        
        # Get non-null values with timestamps
        for idx, row in df.iterrows():
            val = row[col]
            if pd.notna(val) and val != '' and val != 0:
                try:
                    rainfall = float(val)
                    date = row['timestamp'].date()
                    results[zipcode][str(date)].append(rainfall)
                except (ValueError, AttributeError):
                    pass
    
    # Compute daily totals
    daily_totals = {}
    for zipcode, date_data in results.items():
        daily_totals[zipcode] = {}
        for date, values in date_data.items():
            daily_totals[zipcode][date] = {
                'total_in': round(sum(values), 2),
                'max_5min_in': round(max(values), 2),
                'readings': len(values),
            }
    
    print(f"  Processed {len(daily_totals)} ZIP codes with rainfall data")
    
    return daily_totals


def compute_cumulative_rainfall(daily_totals: dict) -> dict:
    """Compute cumulative rainfall and peak statistics per ZIP."""
    summary = {}
    
    for zipcode, date_data in daily_totals.items():
        dates = sorted(date_data.keys())
        daily_values = [date_data[d]['total_in'] for d in dates]
        
        cumulative = sum(daily_values)
        peak_day = max(date_data.items(), key=lambda x: x[1]['total_in'])
        
        summary[zipcode] = {
            'cumulative_in': round(cumulative, 2),
            'peak_day': peak_day[0],
            'peak_day_in': peak_day[1]['total_in'],
            'days_with_rain': sum(1 for v in daily_values if v > 0),
            'daily_data': date_data,
        }
    
    return summary


def generate_text_summaries(rainfall_summary: dict) -> list:
    """Generate textual summaries for the text corpus."""
    docs = []
    
    for zipcode, data in rainfall_summary.items():
        # Create a summary document for this ZIP
        text = (
            f"Rainfall gauge data for ZIP code {zipcode} during Hurricane Harvey: "
            f"Total cumulative rainfall was {data['cumulative_in']} inches over "
            f"{data['days_with_rain']} days with measurable precipitation. "
            f"Peak rainfall of {data['peak_day_in']} inches occurred on {data['peak_day']}."
        )
        
        doc = {
            'doc_id': f'gauge_{zipcode}',
            'source': 'gauge',
            'text': text,
            'timestamp': f"{data['peak_day']}T12:00:00+00:00",  # Use peak day as timestamp
            'zip': zipcode,
            'lat': None,
            'lon': None,
            'category': 'rainfall',
            'payload': {
                'cumulative_in': data['cumulative_in'],
                'peak_day_in': data['peak_day_in'],
                'peak_day': data['peak_day'],
            }
        }
        docs.append(doc)
        
        # Also create per-day entries for high rainfall days (> 5 inches)
        for date, day_data in data['daily_data'].items():
            if day_data['total_in'] >= 5.0:
                text_daily = (
                    f"Heavy rainfall recorded in ZIP {zipcode} on {date}: "
                    f"{day_data['total_in']} inches total, with 5-minute peak of "
                    f"{day_data['max_5min_in']} inches."
                )
                
                doc_daily = {
                    'doc_id': f'gauge_{zipcode}_{date}',
                    'source': 'gauge',
                    'text': text_daily,
                    'timestamp': f"{date}T12:00:00+00:00",
                    'zip': zipcode,
                    'lat': None,
                    'lon': None,
                    'category': 'rainfall_daily',
                    'payload': {
                        'total_in': day_data['total_in'],
                        'max_5min_in': day_data['max_5min_in'],
                    }
                }
                docs.append(doc_daily)
    
    return docs


def main():
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Load data
    df = load_5min_data(raw_dir / '2017-5min.csv')
    sensor_to_zip = load_sensor_metadata(raw_dir / 'gauges.csv')
    
    # Aggregate by ZIP and date
    daily_totals = aggregate_by_zip_day(df, sensor_to_zip)
    
    # Compute summary statistics
    rainfall_summary = compute_cumulative_rainfall(daily_totals)
    
    # Save JSON summary
    output_path = processed_dir / 'rainfall_by_zip.json'
    with open(output_path, 'w') as f:
        json.dump(rainfall_summary, f, indent=2)
    print(f"\n✓ Saved rainfall summary to {output_path}")
    
    # Generate text documents for corpus
    text_docs = generate_text_summaries(rainfall_summary)
    print(f"✓ Generated {len(text_docs)} text documents for corpus")
    
    # Append to text corpus
    corpus_path = processed_dir / 'text_corpus.jsonl'
    existing_count = 0
    if corpus_path.exists():
        with open(corpus_path, 'r') as f:
            existing_count = sum(1 for _ in f)
    
    # Check if gauge docs already exist
    has_gauge_docs = False
    if corpus_path.exists():
        with open(corpus_path, 'r') as f:
            for line in f:
                if '"source": "gauge"' in line:
                    has_gauge_docs = True
                    break
    
    if has_gauge_docs:
        print("⚠ Gauge documents already exist in corpus, skipping append")
    else:
        with open(corpus_path, 'a') as f:
            for doc in text_docs:
                f.write(json.dumps(doc) + '\n')
        print(f"✓ Appended {len(text_docs)} gauge docs to {corpus_path}")
        print(f"  Corpus now has {existing_count + len(text_docs)} documents")
    
    # Print summary
    print("\n=== Rainfall Data Summary ===")
    for zipcode, data in sorted(rainfall_summary.items()):
        print(f"  {zipcode}: {data['cumulative_in']:5.1f} in total, "
              f"peak {data['peak_day_in']:5.1f} in on {data['peak_day']}")


if __name__ == '__main__':
    main()
