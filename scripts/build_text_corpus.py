"""Aggregate text documents from multiple modalities into a single corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from loguru import logger

from app.core.dataio.loaders import load_parquet_table
from app.core.dataio.storage import DataLocator

DEFAULT_OUTPUT = Path("data/processed/text_corpus.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified text corpus for hybrid retrieval")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path (default: data/processed/text_corpus.jsonl)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Optional cap on number of documents per source (newest first)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locator = DataLocator(Path(args.data_dir))
    docs: List[dict] = []
    docs.extend(_build_tweet_docs(locator, args.max_per_source))
    docs.extend(_build_311_docs(locator, args.max_per_source))
    docs.extend(_build_fema_docs(locator, args.max_per_source))
    docs.extend(_build_claim_docs(locator, args.max_per_source))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Text corpus written", path=str(output_path), docs=len(docs))


def _resolve_table(locator: DataLocator, name: str) -> Path | None:
    processed = locator.table_path(name, example=False)
    if processed.exists():
        return processed
    example = locator.table_path(name, example=True)
    if example.exists():
        return example
    logger.warning("Skipping source with missing table", source=name)
    return None


def _build_tweet_docs(locator: DataLocator, limit: int | None) -> List[dict]:
    path = _resolve_table(locator, "tweets")
    if not path:
        return []
    table = load_parquet_table(path)
    docs = []
    for row in table:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        doc = _make_doc(
            doc_id=f"tweet_{row.get('tweet_id')}",
            text=f"Tweet: {text}",
            timestamp=row.get("timestamp"),
            zip_code=row.get("zip"),
            source="tweet",
            payload=row,
        )
        docs.append(doc)
    return _apply_limit(docs, limit)


def _build_311_docs(locator: DataLocator, limit: int | None) -> List[dict]:
    path = _resolve_table(locator, "311")
    if not path:
        return []
    table = load_parquet_table(path)
    docs = []
    for row in table:
        desc = " ".join(filter(None, [row.get("category"), row.get("description")])).strip()
        if not desc:
            continue
        doc = _make_doc(
            doc_id=f"311_{row.get('record_id')}",
            text=f"311 report: {desc}",
            timestamp=row.get("timestamp"),
            zip_code=row.get("zip"),
            source="311",
            payload=row,
        )
        docs.append(doc)
    return _apply_limit(docs, limit)


def _build_fema_docs(locator: DataLocator, limit: int | None) -> List[dict]:
    path = _resolve_table(locator, "fema_kb")
    if not path:
        return []
    table = load_parquet_table(path)
    docs = []
    for row in table:
        zip_code = row.get("zip")
        year = row.get("year")
        loss = row.get("loss_mean")
        text = f"FEMA knowledge base: average loss ${loss} in zip {zip_code} during {year}."
        timestamp = f"{year}-01-01T00:00:00"
        doc = _make_doc(
            doc_id=f"fema_{zip_code}_{year}",
            text=text,
            timestamp=timestamp,
            zip_code=zip_code,
            source="fema_kb",
            payload=row,
        )
        docs.append(doc)
    return _apply_limit(docs, limit)


def _build_claim_docs(locator: DataLocator, limit: int | None) -> List[dict]:
    claims_path = Path(locator.base_dir) / "processed" / "claims.parquet"
    if not claims_path.exists():
        example = locator.examples / "claims.parquet"
        claims_path = example if example.exists() else None
    if not claims_path:
        logger.warning("Skipping claims corpus; file missing")
        return []
    table = load_parquet_table(claims_path)
    docs = []
    for row in table:
        amount = row.get("amount")
        severity = row.get("severity")
        text = (
            f"NFIP claim payout ${amount} in zip {row.get('zip')} on {row.get('timestamp')} "
            f"due to {severity}"
        )
        doc = _make_doc(
            doc_id=f"claim_{row.get('claim_id')}",
            text=text,
            timestamp=row.get("timestamp"),
            zip_code=row.get("zip"),
            source="claim",
            payload=row,
        )
        docs.append(doc)
    return _apply_limit(docs, limit)


def _apply_limit(docs: List[dict], limit: int | None) -> List[dict]:
    if limit is None or len(docs) <= limit:
        return docs
    docs.sort(key=lambda d: d.get("timestamp") or "", reverse=True)
    return docs[:limit]


def _make_doc(doc_id: str, text: str, timestamp: str | None, zip_code: str | None, source: str, payload: dict) -> dict:
    cleaned_text = text.strip()
    tokens = cleaned_text.lower().split()
    return {
        "doc_id": doc_id,
        "text": cleaned_text,
        "tokens": tokens,
        "timestamp": timestamp,
        "zip": zip_code,
        "source": source,
        "payload": payload,
    }


if __name__ == "__main__":
    main()
