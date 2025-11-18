"""Check that processed corpus/index assets exist before running the API."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List

from loguru import logger

from app.config import get_settings


def gather_required_paths(data_dir: Path) -> List[Path]:
    processed = data_dir / "processed"
    settings = get_settings()
    requirements = [
        Path(settings.hybrid_corpus_path),
        Path(settings.hybrid_index_path),
        Path(settings.hybrid_ids_path),
        Path(settings.hybrid_meta_path),
        processed / "imagery_tiles.parquet",
        processed / "gauges.parquet",
        processed / "311.parquet",
        processed / "tweets.parquet",
    ]
    return requirements


def validate(data_dir: Path) -> int:
    logger.info("Validating hybrid retriever assets", data_dir=str(data_dir))
    paths = gather_required_paths(data_dir)
    missing = [path for path in paths if not path.exists()]
    if missing:
        for path in missing:
            logger.error("Missing required asset", path=str(path))
        logger.error("Run the ingestion + corpus/index scripts before starting the API")
        return 1
    logger.success("All required assets found", files=len(paths))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory (defaults to DATA_DIR env)")
    args = parser.parse_args()
    settings = get_settings()
    data_dir = args.data_dir or Path(settings.data_dir)
    raise SystemExit(validate(data_dir))


if __name__ == "__main__":
    main()
