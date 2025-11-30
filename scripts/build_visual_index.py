"""Build visual index from imagery tiles using CLIP embeddings."""

import argparse
from pathlib import Path
import sys

from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.dataio.storage import DataLocator
from app.core.dataio.loaders import load_parquet_table
from app.core.indexing.clip_indexer import CLIPIndexer
from app.core.indexing.visual_index import VisualIndex


def main(
    data_dir: str = "data", model_name: str = "ViT-B/32", force_rebuild: bool = False
):
    """
    Build CLIP visual index from imagery tiles.

    Args:
        data_dir: Path to data directory
        model_name: CLIP model variant to use
        force_rebuild: Rebuild even if index exists
    """
    locator = DataLocator(Path(data_dir))

    # Output paths
    index_dir = locator.processed / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = index_dir / "visual_embeddings.npy"
    metadata_path = index_dir / "visual_metadata.json"

    # Check if index already exists
    if embeddings_path.exists() and metadata_path.exists() and not force_rebuild:
        logger.info(f"Visual index already exists at {index_dir}")
        logger.info("Use --force to rebuild")

        # Load and display stats
        index = VisualIndex.load(embeddings_path, metadata_path)
        logger.info(f"Loaded index with {len(index.metadata)} tiles")
        return

    # Load imagery tiles metadata
    imagery_table_path = locator.table_path("imagery_tiles")
    logger.info(f"Loading imagery metadata from {imagery_table_path}")

    tiles = load_parquet_table(imagery_table_path)  # Already returns list of dicts

    logger.info(f"Found {len(tiles)} imagery tiles")

    # Initialize CLIP
    clip_indexer = CLIPIndexer(model_name=model_name)

    # Build visual index
    visual_index = VisualIndex()

    # Determine image base directory
    # Use base_dir/raw/imagery as the image base
    image_base = locator.base_dir / "raw" / "imagery"

    logger.info(f"Building visual index from {image_base}...")
    visual_index.build_from_tiles(
        tiles=tiles, clip_indexer=clip_indexer, image_dir=image_base, batch_size=32
    )

    # Save index
    visual_index.save(embeddings_path, metadata_path)

    logger.info(f"Visual index built successfully!")
    logger.info(f"Embeddings: {embeddings_path}")
    logger.info(f"Metadata: {metadata_path}")

    # Test search
    logger.info("\nTesting semantic search...")
    test_queries = ["flooded streets", "damaged buildings", "aerial view of city"]

    for query in test_queries:
        results = visual_index.search(query, clip_indexer, top_k=3)
        logger.info(f"\nQuery: '{query}'")
        for tile_id, score, metadata in results:
            logger.info(f"  {tile_id}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP visual index")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if index exists"
    )

    args = parser.parse_args()

    main(data_dir=args.data_dir, model_name=args.model, force_rebuild=args.force)
