"""CLIP-based image embedding generator for semantic visual search."""

from __future__ import annotations

import torch
import clip
from PIL import Image
from pathlib import Path
from typing import Union, List
from loguru import logger
import numpy as np


class CLIPIndexer:
    """
    Wrapper for OpenAI CLIP model to generate image and text embeddings.

    Supports semantic image search by encoding images and queries into a shared embedding space.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model variant. Options: "ViT-B/32", "ViT-B/16", "ViT-L/14"
            device: Device to run model on. Auto-detects CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model {model_name} on {self.device}")

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info(f"CLIP model loaded successfully")

    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Encode an image to a CLIP embedding vector.

        Args:
            image_path: Path to the image file

        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

            return image_features.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query to a CLIP embedding vector.

        Args:
            text: Natural language query (e.g., "flooded streets", "damaged buildings")

        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            text_tokens = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

            return text_features.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Failed to encode text '{text}': {e}")
            raise

    def encode_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode multiple images in batches for efficiency.

        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once

        Returns:
            Array of shape (N, embedding_dim) with embeddings for all images
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(self.preprocess(image))
                except Exception as e:
                    logger.warning(f"Skipping {path}: {e}")
                    continue

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(batch_features.cpu().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    @staticmethod
    def compute_similarity(image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """
        Compute cosine similarity between image and text embeddings (already normalized).

        Args:
            image_emb: Image embedding vector
            text_emb: Text embedding vector

        Returns:
            Similarity score in range [0, 1]
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = np.dot(image_emb, text_emb)
        # Convert from [-1, 1] to [0, 1] range
        return float((similarity + 1) / 2)
