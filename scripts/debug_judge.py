"""Debug script to manually inspect judge's faithfulness evaluation."""

import json
import sys
from pathlib import Path
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.retrieval.retriever import Retriever
from app.core.retrieval.selector import select_candidates
from app.core.retrieval.context_packager import package_context
from app.core.models.split_client import SplitPipelineClient
from app.core.eval.eval_generation import GenerationEvaluator


def inspect_query(zip_code: str, start: str, end: str, provider: str = "openai"):
    """Manually inspect a single query's faithfulness evaluation."""

    # Setup
    locator = DataLocator(Path("data"))
    retriever = Retriever(locator)
    client = SplitPipelineClient(provider=provider)
    evaluator = GenerationEvaluator(client)

    # Create query
    query = RAGQuery(zip=zip_code, start=start, end=end, k_tiles=10, n_text=20)

    print(f"\n{'='*80}")
    print(f"QUERY: ZIP {zip_code}, {start} to {end}")
    print(f"{'='*80}\n")

    # Retrieve context
    print("📥 Retrieving context...")
    result = retriever.retrieve(query)
    candidates = select_candidates(result, query.k_tiles, query.n_text)
    context = package_context(candidates)

    # Generate answer
    print("🤖 Generating answer...")
    answer = client.infer(
        zip_code=query.zip,
        time_window={"start": str(query.start), "end": str(query.end)},
        imagery_tiles=context["imagery_tiles"],
        text_snippets=context["text_snippets"],
        sensor_table=context["sensor_table"],
        kb_summary=context["kb_summary"],
        tweets=context.get("tweets", []),
        calls=context.get("calls", []),
        sensors=context.get("sensors", []),
        fema=context.get("fema", []),
    )

    print("\n" + "=" * 80)
    print("GENERATED ANSWER:")
    print("=" * 80)
    print(json.dumps(answer, indent=2))

    # Prepare judge input
    print("\n" + "=" * 80)
    print("JUDGE INPUT (CONTEXT):")
    print("=" * 80)

    snippets = "\n".join(context.get("text_snippets", []))
    kb_summary = context.get("kb_summary", "")
    image_ids = [t.get("tile_id", "unknown") for t in context.get("imagery_tiles", [])]
    imagery_context = f"Available Imagery Tiles: {', '.join(image_ids)}"
    query_context = (
        f"Query Parameters: ZIP={query.zip}, Start={query.start}, End={query.end}"
    )

    print(f"{query_context}\n{imagery_context}\n")
    print("Text Snippets:")
    print(snippets[:1000] + "..." if len(snippets) > 1000 else snippets)
    print(f"\nKB Summary: {kb_summary}")

    # Run judge
    print("\n" + "=" * 80)
    print("RUNNING FAITHFULNESS JUDGE...")
    print("=" * 80)

    query_text = f"Assess impact for {query.zip} from {query.start} to {query.end}"
    query_params = {"zip": query.zip, "start": str(query.start), "end": str(query.end)}

    gen_metrics = evaluator.evaluate(
        query=query_text,
        context=context,
        answer=json.dumps(answer),
        query_params=query_params,
    )

    print("\n" + "=" * 80)
    print("JUDGE OUTPUT:")
    print("=" * 80)
    print(f"Faithfulness: {gen_metrics['faithfulness']}")
    print(f"Relevance: {gen_metrics['relevance']}")

    return answer, gen_metrics


if __name__ == "__main__":
    # Example usage - modify these parameters to inspect different queries
    if len(sys.argv) >= 4:
        zip_code = sys.argv[1]
        start = sys.argv[2]
        end = sys.argv[3]
    else:
        # Default: first query from eval config
        zip_code = "77002"
        start = "2017-08-26"
        end = "2017-08-30"

    inspect_query(zip_code, start, end)
