"""Create FAISS embedding index from the unified text corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import faiss
from loguru import logger
from sentence_transformers import SentenceTransformer

DEFAULT_CORPUS = Path("data/processed/text_corpus.jsonl")
DEFAULT_INDEX = Path("data/processed/text_embeddings.faiss")
DEFAULT_IDS = Path("data/processed/text_embeddings_ids.json")
DEFAULT_META = Path("data/processed/text_embeddings_meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index hybrid corpus embeddings")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Input corpus JSONL path")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Output FAISS index path")
    parser.add_argument("--ids", default=str(DEFAULT_IDS), help="Output doc-id list path")
    parser.add_argument("--meta", default=str(DEFAULT_META), help="Output metadata path")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found: {corpus_path}")

    docs = _load_corpus(corpus_path)
    if not docs:
        raise SystemExit("Corpus contains no documents")

    model = SentenceTransformer(args.model)
    texts = [doc["text"] for doc in docs]
    logger.info("Encoding documents", count=len(texts), model=args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    ids_path = Path(args.ids)
    ids_path.write_text(json.dumps([doc["doc_id"] for doc in docs], ensure_ascii=False, indent=2))

    meta_path = Path(args.meta)
    meta = {"model": args.model, "dimension": dim, "normalize": True}
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Embedding index written", index=str(index_path), docs=len(docs))


def _load_corpus(path: Path) -> List[dict]:
    docs: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs


if __name__ == "__main__":
    main()
