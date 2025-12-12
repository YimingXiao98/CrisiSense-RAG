#!/usr/bin/env python3
"""
Rebuild text corpus from raw data with proper timestamps.

Sources:
- 311 calls: data/raw/houston_311.csv
- Tweets: data/raw/twitter/GNIPHarvey/*.json.gz
- Image captions: data/processed/imagery_captions.json (optional, for Exp1)
"""

import argparse
import gzip
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as date_parser
from loguru import logger

# Paths
RAW_311 = Path("data/raw/houston_311.csv")
RAW_TWEETS_DIR = Path("data/raw/twitter/GNIPHarvey")
CAPTIONS_FILE = Path("data/processed/imagery_captions.json")
OUTPUT_CORPUS = Path("data/processed/text_corpus.jsonl")

# Date range for Harvey (restrict to relevant period) - timezone aware
HARVEY_START = datetime(2017, 8, 20, tzinfo=timezone.utc)
HARVEY_END = datetime(2017, 10, 15, tzinfo=timezone.utc)

# Keyword filters for tweets
ALLOW_KEYWORDS = {
    "flood",
    "flooding",
    "flooded",
    "hurricane",
    "storm",
    "rain",
    "underwater",
    "rescue",
    "trapped",
    "stuck",
    "help",
    "emergency",
    "911",
    "evacuate",
    "damage",
    "collapsed",
    "power",
    "outage",
    "road",
    "bridge",
    "bayou",
    "creek",
}

BLOCK_KEYWORDS = {
    "spotify",
    "music",
    "song",
    "album",
    "lyrics",
    "vote",
    "election",
    "trump",
    "biden",
    "president",
    "giveaway",
    "contest",
    "win",
    "sale",
    "shirt",
    "merch",
    "game",
    "nfl",
    "nba",
    "football",
    "baseball",
    "love",
    "heart",
    "tears",
}


def parse_datetime(ts_str):
    """Parse various timestamp formats, always returning timezone-aware datetime."""
    if not ts_str:
        return None
    try:
        dt = date_parser.parse(ts_str)
        # Make timezone-aware if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        return None


def load_311_calls(path: Path, max_records: int = 10000):
    """Load 311 calls from CSV with proper timestamps."""
    logger.info(f"Loading 311 calls from {path}")
    records = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = parse_datetime(row.get("timestamp"))
            if not ts:
                continue
            if ts < HARVEY_START or ts > HARVEY_END:
                continue

            # Get ZIP code
            zip_code = row.get("zip", "")
            if zip_code:
                zip_code = str(zip_code).replace(".0", "")

            record = {
                "doc_id": f"311_{row.get('record_id', '')}",
                "source": "311",
                "text": row.get("description", ""),
                "timestamp": ts.isoformat(),
                "zip": zip_code,
                "lat": float(row.get("lat", 0)) if row.get("lat") else None,
                "lon": float(row.get("lon", 0)) if row.get("lon") else None,
                "category": row.get("category", ""),
                "payload": {
                    "description": row.get("description", ""),
                    "category": row.get("category", ""),
                    "timestamp": ts.isoformat(),
                    "location": f"{row.get('lat', '')}, {row.get('lon', '')}",
                    "zip": zip_code,
                },
            }
            records.append(record)

            if len(records) >= max_records:
                break

    logger.info(f"Loaded {len(records)} 311 calls")
    return records


def load_tweets(tweets_dir: Path, max_records: int = 10000):
    """Load tweets from gzipped JSON files with proper timestamps.

    Filters applied:
    - Time window: Aug 20 - Oct 15, 2017
    - Minimum length: 10 characters
    - Skip retweets (RT @)
    - Skip spam (>3 hashtags or >2 URLs)
    - Keyword filter: must contain allow keyword, must not contain block keyword
    - Duplicate filter: skip identical text

    Uses random file sampling to get diverse tweets across the dataset.
    """
    import random

    logger.info(f"Loading tweets from {tweets_dir}")
    records = []
    seen_texts = set()  # For duplicate detection

    gz_files = list(tweets_dir.glob("*.json.gz"))
    logger.info(f"Found {len(gz_files)} gzipped JSON files")

    # Shuffle files for random sampling across the dataset
    random.seed(42)  # Reproducible
    random.shuffle(gz_files)

    # Track filter stats
    stats = {
        "total": 0,
        "skipped_time": 0,
        "skipped_short": 0,
        "skipped_rt": 0,
        "skipped_spam": 0,
        "skipped_no_allow": 0,
        "skipped_blocked": 0,
        "skipped_duplicate": 0,
        "accepted": 0,
    }

    for gz_file in gz_files:
        if len(records) >= max_records:
            break

        try:
            with gzip.open(gz_file, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(records) >= max_records:
                        break
                    try:
                        tweet = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    stats["total"] += 1

                    if stats["total"] % 10000 == 0:
                        logger.info(
                            f"Scanned {stats['total']} tweets | Accepted: {stats['accepted']} | Stats: {stats}"
                        )

                    # Parse timestamp
                    ts = parse_datetime(tweet.get("postedTime"))
                    if not ts:
                        continue
                    if ts < HARVEY_START or ts > HARVEY_END:
                        stats["skipped_time"] += 1
                        continue

                    # Get text content
                    text = tweet.get("body", "")
                    if not text or len(text) < 10:
                        stats["skipped_short"] += 1
                        continue

                    # Skip retweets
                    if text.startswith("RT @") or text.startswith("rt @"):
                        stats["skipped_rt"] += 1
                        continue

                    # Skip spam: too many hashtags (>3) or URLs (>2)
                    hashtag_count = text.count("#")
                    url_count = text.lower().count("http")
                    if hashtag_count > 3 or url_count > 2:
                        stats["skipped_spam"] += 1
                        continue

                    # Keyword filter: must contain at least one allow keyword
                    text_lower = text.lower()
                    has_allow = any(kw in text_lower for kw in ALLOW_KEYWORDS)
                    if not has_allow:
                        stats["skipped_no_allow"] += 1
                        continue

                    # Block keyword filter: discard if any block keyword present
                    has_block = any(kw in text_lower for kw in BLOCK_KEYWORDS)
                    if has_block:
                        stats["skipped_blocked"] += 1
                        continue

                    # Duplicate filter: skip if we've seen this exact text
                    text_hash = hash(text)
                    if text_hash in seen_texts:
                        stats["skipped_duplicate"] += 1
                        continue
                    seen_texts.add(text_hash)

                    # Get location if available
                    geo = tweet.get("geo")
                    lat, lon = None, None
                    if geo and geo.get("coordinates"):
                        coords = geo["coordinates"]
                        if isinstance(coords, list) and len(coords) >= 2:
                            lat, lon = coords[1], coords[0]  # GeoJSON is [lon, lat]

                    # Get user location
                    user_location = ""
                    if tweet.get("actor", {}).get("location", {}).get("displayName"):
                        user_location = tweet["actor"]["location"]["displayName"]

                    tweet_id = tweet.get("id", "").replace(
                        "tag:search.twitter.com,2005:", ""
                    )

                    record = {
                        "doc_id": f"tweet_{tweet_id}",
                        "source": "tweet",
                        "text": text,
                        "timestamp": ts.isoformat(),
                        "zip": "",  # Tweets don't have ZIP
                        "lat": lat,
                        "lon": lon,
                        "payload": {
                            "text": text,
                            "timestamp": ts.isoformat(),
                            "location": user_location,
                            "tweet_id": tweet_id,
                        },
                    }
                    records.append(record)
                    stats["accepted"] += 1

        except Exception as e:
            logger.warning(f"Error reading {gz_file}: {e}")

    logger.info(f"Loaded {len(records)} tweets")
    logger.info(f"Filter stats: {stats}")
    return records


def load_captions(captions_path: Path):
    """Load image captions as text documents for hybrid search."""
    if not captions_path.exists():
        logger.warning(f"Captions file not found: {captions_path}")
        return []

    logger.info(f"Loading captions from {captions_path}")
    records = []

    data = json.loads(captions_path.read_text())
    captions = data.get("captions", [])

    for cap in captions:
        ts = parse_datetime(cap.get("timestamp"))
        if not ts:
            ts = datetime(2017, 8, 31, tzinfo=timezone.utc)  # Default to Aug 31

        # Create a searchable text from the caption
        caption_text = cap.get("caption", "")
        if not caption_text:
            continue

        record = {
            "doc_id": f"caption_{cap.get('tile_id', '')}",
            "source": "caption",
            "text": caption_text,
            "timestamp": ts.isoformat(),
            "zip": cap.get("zip", ""),
            "lat": cap.get("lat"),
            "lon": cap.get("lon"),
            "payload": {
                "tile_id": cap.get("tile_id", ""),
                "caption": caption_text,
                "uri": cap.get("uri", ""),
                "timestamp": ts.isoformat(),
            },
        }
        records.append(record)

    logger.info(f"Loaded {len(records)} captions")
    return records


def main(include_captions: bool = False, max_records: int = 10000):
    """Build the text corpus."""
    logger.info("Building text corpus with proper timestamps...")

    # Load data
    calls_311 = load_311_calls(RAW_311, max_records=max_records)
    tweets = load_tweets(RAW_TWEETS_DIR, max_records=max_records)

    # Combine all records
    all_records = calls_311 + tweets

    # Optionally add captions
    if include_captions:
        captions = load_captions(CAPTIONS_FILE)
        all_records.extend(captions)
        logger.info(f"Added {len(captions)} captions to corpus")
    logger.info(f"Total records: {len(all_records)}")

    # Analyze date distribution
    from collections import Counter

    date_counts = Counter()
    for rec in all_records:
        ts = rec.get("timestamp", "")[:10]
        date_counts[ts] += 1

    logger.info("Date distribution:")
    for date, count in sorted(date_counts.items())[:20]:
        logger.info(f"  {date}: {count}")

    # Write corpus
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    logger.success(f"Wrote {len(all_records)} records to {OUTPUT_CORPUS}")

    # Summary by source
    source_counts = Counter(r["source"] for r in all_records)
    for source, count in source_counts.items():
        logger.info(f"  {source}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild text corpus")
    parser.add_argument(
        "--include-captions",
        action="store_true",
        help="Include image captions in the corpus (for Exp1)",
    )
    parser.add_argument(
        "--max-records", type=int, default=10000, help="Max records per source"
    )
    args = parser.parse_args()

    main(include_captions=args.include_captions, max_records=args.max_records)
