#!/usr/bin/env python3
"""
Process rainfall gauge data with proper ZIP code assignment via Census Geocoder.

Inputs:
- data/raw/GageLocations.csv (sensor metadata with lat/lng)
- data/raw/2017-5min.csv (5-minute rainfall readings)

Outputs:
- data/processed/sensor_locations.json (sensor_id -> {zip, lat, lng, location})
- data/processed/rainfall_by_sensor.json (sensor_id -> {zip, daily_data})
"""

import csv
import json
import requests
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOCATIONS_FILE = RAW_DIR / "GageLocations.csv"
RAINFALL_FILE = RAW_DIR / "2017-5min.csv"
OUTPUT_LOCATIONS = PROCESSED_DIR / "sensor_locations.json"
OUTPUT_RAINFALL = PROCESSED_DIR / "rainfall_by_sensor.json"


def load_zip_coordinates() -> Dict[str, tuple]:
    """Load ZIP code centroids from processed file."""
    zip_coords_file = PROCESSED_DIR / "zip_coordinates.json"
    with open(zip_coords_file, "r") as f:
        data = json.load(f)
    
    zip_coords = {}
    for zip_code, info in data.get("zip_codes", {}).items():
        zip_coords[zip_code] = (info["lat"], info["lon"])
    
    print(f"Loaded {len(zip_coords)} ZIP code centroids")
    return zip_coords


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers."""
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


def find_nearest_zip(lat: float, lng: float, zip_coords: Dict[str, tuple]) -> tuple:
    """Find the nearest ZIP code to a given coordinate.
    
    Returns:
        Tuple of (zip_code, distance_km)
    """
    best_zip = None
    best_dist = float('inf')
    
    for zip_code, (zlat, zlon) in zip_coords.items():
        dist = haversine_km(lat, lng, zlat, zlon)
        if dist < best_dist:
            best_dist = dist
            best_zip = zip_code
    
    return best_zip, best_dist


def assign_zips_to_sensors(sensors: Dict[str, dict], zip_coords: Dict[str, tuple]) -> Dict[str, str]:
    """
    Assign ZIP codes to sensors based on nearest ZIP centroid.
    
    Returns:
        Dict mapping sensor_id -> ZIP code
    """
    results = {}
    
    for sensor_id, sensor in sensors.items():
        lat, lng = sensor["lat"], sensor["lng"]
        zip_code, dist_km = find_nearest_zip(lat, lng, zip_coords)
        
        if zip_code and dist_km < 50:  # Max 50km to avoid edge cases
            results[sensor_id] = zip_code
            print(f"  Sensor {sensor_id}: ZIP {zip_code} ({dist_km:.1f} km)")
        else:
            print(f"  Sensor {sensor_id}: No ZIP found within 50km")
    
    return results


def load_rain_sensors(filepath: Path) -> Dict[str, dict]:
    """Load rain gauge sensor locations from CSV."""
    sensors = {}
    
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Sensor Type") == "Rain Increment":
                sensor_id = row["sensor_id"]
                sensors[sensor_id] = {
                    "site_id": row["Site_Id"],
                    "location": row["Location"],
                    "lat": float(row["lat"]),
                    "lng": float(row["lng"]),
                    "agency": row["Agency"],
                }
    
    print(f"Loaded {len(sensors)} rain gauge sensors")
    return sensors


def parse_5min_rainfall(filepath: Path, sensor_ids: set) -> Dict[str, Dict[str, float]]:
    """
    Parse 5-minute rainfall data and aggregate by sensor and date.
    
    Note: The CSV has 8 metadata rows before the actual header on row 9.
    Column format is "SiteId_SensorId" (e.g., "100_100", "105_105").
    
    Returns:
        Dict[sensor_id, Dict[date_str, daily_total_inches]]
    """
    rainfall = defaultdict(lambda: defaultdict(float))
    
    with open(filepath, "r", encoding="utf-8-sig") as f:
        # Skip 8 metadata rows
        for _ in range(8):
            next(f)
        
        reader = csv.reader(f)
        header = next(reader)
        
        # Parse column names: format is "SiteId_SensorId"
        # Map column index -> sensor_id for matching sensors
        col_map = {}
        for idx, col in enumerate(header):
            if "_" in col:
                parts = col.split("_")
                if len(parts) >= 2:
                    sensor_id = parts[-1]  # Take last part as sensor_id
                    if sensor_id in sensor_ids:
                        col_map[idx] = sensor_id
        
        print(f"Found {len(col_map)} rain sensor columns in data file")
        
        # Parse rows
        row_count = 0
        for row in reader:
            if not row or len(row) < 2:
                continue
            
            try:
                # Parse timestamp (format: "8/25/17 0:00" or similar)
                timestamp_str = row[0]
                # Handle various date formats
                for fmt in ["%m/%d/%y %H:%M", "%m/%d/%y", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        dt = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue  # Skip unparseable rows
                
                date_str = dt.strftime("%Y-%m-%d")
                
                # Aggregate rainfall values
                for col_idx, sensor_id in col_map.items():
                    if col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            try:
                                inches = float(val)
                                if inches > 0:
                                    rainfall[sensor_id][date_str] += inches
                            except ValueError:
                                pass
                
                row_count += 1
                
            except Exception as e:
                continue
        
        print(f"Processed {row_count} time intervals")
    
    return rainfall


def main():
    print("=" * 60)
    print("Sensor Data Processing with Nearest-ZIP Geocoding")
    print("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load sensor locations
    print("\n[1/4] Loading rain sensor locations...")
    sensors = load_rain_sensors(LOCATIONS_FILE)
    
    # Step 2: Load ZIP coordinates and assign ZIPs to sensors
    print("\n[2/4] Assigning ZIP codes to sensors (nearest centroid)...")
    zip_coords = load_zip_coordinates()
    zip_map = assign_zips_to_sensors(sensors, zip_coords)
    
    print(f"       Successfully assigned {len(zip_map)} sensors to ZIP codes")
    
    # Update sensors with ZIP codes
    for sid, zip_code in zip_map.items():
        sensors[sid]["zip"] = zip_code
    
    # Step 3: Parse rainfall data
    print("\n[3/4] Parsing 5-minute rainfall data...")
    sensor_ids = set(sensors.keys())
    rainfall = parse_5min_rainfall(RAINFALL_FILE, sensor_ids)
    
    # Step 4: Build output structures
    print("\n[4/4] Building output files...")
    
    # Sensor locations with ZIP
    OUTPUT_LOCATIONS.write_text(json.dumps(sensors, indent=2))
    print(f"       Saved sensor locations to {OUTPUT_LOCATIONS}")
    
    # Rainfall by sensor (only sensors with ZIP codes)
    rainfall_output = {}
    for sensor_id, daily_data in rainfall.items():
        if sensor_id in sensors and "zip" in sensors[sensor_id]:
            rainfall_output[sensor_id] = {
                "sensor_id": sensor_id,
                "zip": sensors[sensor_id]["zip"],
                "location": sensors[sensor_id]["location"],
                "lat": sensors[sensor_id]["lat"],
                "lng": sensors[sensor_id]["lng"],
                "daily_data": {
                    date: round(total, 3) for date, total in sorted(daily_data.items())
                }
            }
    
    OUTPUT_RAINFALL.write_text(json.dumps(rainfall_output, indent=2))
    print(f"       Saved rainfall data to {OUTPUT_RAINFALL}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rain sensors:     {len(sensors)}")
    print(f"Successfully geocoded:  {len(zip_map)}")
    print(f"Sensors with rainfall:  {len(rainfall_output)}")
    
    # Count unique ZIPs
    unique_zips = set(s.get("zip") for s in sensors.values() if s.get("zip"))
    print(f"Unique ZIP codes:       {len(unique_zips)}")
    
    # Sample output
    if rainfall_output:
        sample_sid = list(rainfall_output.keys())[0]
        sample = rainfall_output[sample_sid]
        print(f"\nSample sensor: {sample_sid}")
        print(f"  Location: {sample['location']}")
        print(f"  ZIP: {sample['zip']}")
        print(f"  Days with data: {len(sample['daily_data'])}")


if __name__ == "__main__":
    main()
