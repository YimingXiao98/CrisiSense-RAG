"""Pre-convert NOAA aerial TIF tiles to JPEG for faster model inference.

Converts all .tif files under data/raw/imagery/ to JPEG at 1024px max
dimension (matching the VisualAnalysisClient's default resize), saving to
data/processed/imagery_jpg/ with identical relative paths.

After running this script, the VisualAnalysisClient will use the pre-converted
JPEGs instead of loading and resizing 9351x9351 TIFs on every API call.

Usage:
    conda run -n harvey-rag python scripts/preprocess_imagery.py
    conda run -n harvey-rag python scripts/preprocess_imagery.py --max-size 2048  # higher quality
    conda run -n harvey-rag python scripts/preprocess_imagery.py --dry-run
"""

import argparse
from pathlib import Path

from loguru import logger
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check for large TIFs

RAW_DIR = Path("data/raw/imagery")
OUT_DIR = Path("data/processed/imagery_jpg")


def convert_tif(src: Path, dst: Path, max_size: int, quality: int) -> bool:
    try:
        with Image.open(src) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Failed {src.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-convert TIF imagery to JPEG")
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Max dimension in pixels (default: 1024, matches VisualAnalysisClient)")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG quality 1-95 (default: 85)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be converted without doing it")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing JPEGs (default: skip if exists)")
    args = parser.parse_args()

    tif_files = sorted(RAW_DIR.rglob("*.tif"))
    logger.info(f"Found {len(tif_files)} TIF files under {RAW_DIR}")
    logger.info(f"Output directory: {OUT_DIR}")
    logger.info(f"Max size: {args.max_size}px, Quality: {args.quality}")

    converted = skipped = failed = 0

    for tif in tif_files:
        rel = tif.relative_to(RAW_DIR)
        jpg = (OUT_DIR / rel).with_suffix(".jpg")

        if jpg.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            logger.info(f"Would convert: {rel} → {jpg.relative_to(OUT_DIR)}")
            converted += 1
            continue

        ok = convert_tif(tif, jpg, args.max_size, args.quality)
        if ok:
            converted += 1
            if converted % 100 == 0:
                logger.info(f"Progress: {converted}/{len(tif_files)-skipped} converted")
        else:
            failed += 1

    logger.success(f"Done. Converted={converted}, Skipped={skipped}, Failed={failed}")
    logger.info(f"Output: {OUT_DIR} ({sum(1 for _ in OUT_DIR.rglob('*.jpg'))} JPEGs)")


if __name__ == "__main__":
    main()
