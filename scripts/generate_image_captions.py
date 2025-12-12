"""Generate image captions for imagery tiles using Gemini Flash.

This script:
1. Loads imagery tiles from the CSV metadata
2. Generates captions using Gemini 2.5 Flash
3. Saves results to data/processed/imagery_captions.json

Usage:
    python scripts/generate_image_captions.py [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from loguru import logger
import google.generativeai as genai


# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

genai.configure(api_key=GEMINI_API_KEY)


CAPTION_PROMPT = """Describe this aerial/satellite image in 2-3 sentences for disaster assessment.
Focus on:
1. Land use (residential, commercial, industrial, rural)
2. Any visible flooding, water bodies, or water damage
3. Any visible structural damage or debris
4. Road conditions and accessibility
5. Vegetation (green areas, parks, flooded fields)

Be specific about what you observe. If you cannot see clear evidence of damage or flooding, say so.
Do NOT make assumptions beyond what is visible in the image."""


def convert_tiff_to_jpeg(tiff_path: Path, temp_dir: Path = None) -> Optional[Path]:
    """Convert TIFF image to JPEG for Gemini compatibility."""
    try:
        from PIL import Image
        import tempfile

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())

        jpg_path = temp_dir / f"{tiff_path.stem}.jpg"

        # Open and convert TIFF to RGB JPEG
        with Image.open(tiff_path) as img:
            # Handle different modes
            if img.mode in ("RGBA", "LA", "PA"):
                # Convert alpha to white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to 1024 max dimension (faster upload, Gemini compatible)
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            img.save(jpg_path, "JPEG", quality=85)

        return jpg_path
    except Exception as e:
        logger.warning(f"Failed to convert TIFF {tiff_path}: {e}")
        return None


def generate_caption_gemini(
    image_path: Path,
    model_name: str = "models/gemini-2.0-flash",
    max_retries: int = 3,
    temp_dir: Path = None,
) -> Optional[str]:
    """Generate caption for an image using Gemini Flash."""
    from PIL import Image
    import tempfile

    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None

    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    # Load and prepare image, save as JPEG
    try:
        jpg_path = temp_dir / f"{image_path.stem}_caption.jpg"
        with Image.open(image_path) as img:
            # Handle different modes
            if img.mode in ("RGBA", "LA", "PA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                pil_image = background
            elif img.mode != "RGB":
                pil_image = img.convert("RGB")
            else:
                pil_image = img.copy()

            # Resize to 1024 max dimension
            max_dim = 1024
            if max(pil_image.size) > max_dim:
                ratio = max_dim / max(pil_image.size)
                new_size = (
                    int(pil_image.size[0] * ratio),
                    int(pil_image.size[1] * ratio),
                )
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            pil_image.save(jpg_path, "JPEG", quality=85)
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None

    model = genai.GenerativeModel(model_name)
    caption = None

    for attempt in range(max_retries):
        try:
            # Upload file to Gemini
            file = genai.upload_file(str(jpg_path))

            response = model.generate_content(
                [file, CAPTION_PROMPT],
                generation_config=genai.GenerationConfig(
                    max_output_tokens=256,
                    temperature=0.2,
                ),
            )

            caption = response.text.strip()

            # Clean up uploaded file
            try:
                genai.delete_file(file.name)
            except:
                pass

            break  # Success

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {image_path.name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            continue

    # Clean up temp file
    try:
        jpg_path.unlink()
    except:
        pass

    return caption


def load_existing_captions(output_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing captions for resume support."""
    if output_path.exists():
        data = json.loads(output_path.read_text())
        return {item["tile_id"]: item for item in data.get("captions", [])}
    return {}


def generate_captions(
    tiles_csv: Path,
    output_path: Path,
    base_dir: Path,
    limit: Optional[int] = None,
    resume: bool = True,
    batch_save_interval: int = 50,
) -> None:
    """Generate captions for all imagery tiles."""

    # Load tiles metadata
    tiles = []
    with open(tiles_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tiles.append(row)

    logger.info(f"Loaded {len(tiles)} tiles from {tiles_csv}")

    if limit:
        tiles = tiles[:limit]
        logger.info(f"Limiting to {limit} tiles")

    # Load existing captions if resuming
    existing = {}
    if resume:
        existing = load_existing_captions(output_path)
        if existing:
            logger.info(f"Resuming: Found {len(existing)} existing captions")

    # Generate captions
    captions = list(existing.values())  # Start with existing
    skipped = 0
    generated = 0
    failed = 0

    for i, tile in enumerate(tiles):
        tile_id = tile["tile_id"]

        # Skip if already captioned
        if tile_id in existing:
            skipped += 1
            continue

        # Resolve image path
        uri = tile.get("uri", "")
        if not uri:
            logger.warning(f"No URI for tile {tile_id}")
            failed += 1
            continue

        image_path = base_dir / uri
        if not image_path.exists():
            # Try with filename
            filename = tile.get("filename", "")
            if filename:
                alt_paths = list(base_dir.glob(f"**/{filename}"))
                if alt_paths:
                    image_path = alt_paths[0]
                else:
                    logger.warning(f"Image not found: {image_path}")
                    failed += 1
                    continue

        # Generate caption
        logger.info(f"[{i+1}/{len(tiles)}] Captioning {tile_id}...")
        caption = generate_caption_gemini(image_path)

        if caption:
            caption_entry = {
                "tile_id": tile_id,
                "caption": caption,
                "uri": uri,
                "timestamp": tile.get("timestamp", ""),
                "zip": tile.get("zip", ""),
                "lat": float(tile.get("lat", 0)) if tile.get("lat") else None,
                "lon": float(tile.get("lon", 0)) if tile.get("lon") else None,
            }
            captions.append(caption_entry)
            existing[tile_id] = caption_entry
            generated += 1

            logger.info(f"  Caption: {caption[:100]}...")
        else:
            failed += 1

        # Save periodically
        if generated > 0 and generated % batch_save_interval == 0:
            _save_captions(output_path, captions)
            logger.info(f"Saved {len(captions)} captions (checkpoint)")

        # Rate limiting (Gemini has ~15 RPM for free tier, 1000 for paid)
        time.sleep(0.5)

    # Final save
    _save_captions(output_path, captions)

    logger.success(f"✓ Caption generation complete!")
    logger.info(f"  Generated: {generated}")
    logger.info(f"  Skipped (existing): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total captions: {len(captions)}")


def _save_captions(output_path: Path, captions: List[Dict]) -> None:
    """Save captions to JSON file."""
    output_data = {
        "total_captions": len(captions),
        "captions": captions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions")
    parser.add_argument(
        "--tiles-csv",
        default="data/processed/imagery_tiles.csv",
        help="Path to imagery tiles CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/imagery_captions.json",
        help="Output path for captions JSON",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for resolving image paths",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tiles to process",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing captions",
    )

    args = parser.parse_args()

    generate_captions(
        tiles_csv=Path(args.tiles_csv),
        output_path=Path(args.output),
        base_dir=Path(args.base_dir),
        limit=args.limit,
        resume=not args.no_resume,
    )
