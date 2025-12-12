import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger


def parse_datetime(ts_str):
    """Parse various timestamp formats, always returning timezone-aware datetime."""
    # Simplified for this specific task as we know the format in captions
    try:
        if not ts_str:
            return None
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Append captions to text corpus")
    parser.add_argument(
        "--captions", type=Path, default="data/processed/imagery_captions.json"
    )
    parser.add_argument(
        "--corpus", type=Path, default="data/processed/text_corpus.jsonl"
    )
    args = parser.parse_args()

    if not args.captions.exists():
        logger.error(f"Captions file not found: {args.captions}")
        return

    logger.info(f"Loading captions from {args.captions}")
    data = json.loads(args.captions.read_text())
    captions = data.get("captions", [])

    new_records = []
    for cap in captions:
        ts_str = cap.get("timestamp")
        # Ensure timezone awareness
        if not ts_str:
            ts = datetime(2017, 8, 31, tzinfo=timezone.utc)
        else:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except:
                ts = datetime(2017, 8, 31, tzinfo=timezone.utc)

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
        new_records.append(record)

    logger.info(f"Appending {len(new_records)} captions to {args.corpus}")

    with open(args.corpus, "a", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record) + "\n")

    logger.success("Done!")


if __name__ == "__main__":
    main()
