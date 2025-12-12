"""Generate full validation data with multiple LLM judge scores.

This script:
1. Runs the RAG pipeline for 25 validation queries
2. Evaluates each response with 6 different LLM judges
3. Saves full context + responses + all judge scores for human validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file if exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from loguru import logger

from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.models.split_client import SplitPipelineClient
from app.core.retrieval.context_packager import package_context
from app.core.retrieval.retriever import Retriever
from app.core.retrieval.selector import select_candidates
from app.core.eval.ground_truth import ClaimsGroundTruth
from app.core.eval.multi_judge import JudgeClient, get_available_judges


# Configure judges to use
JUDGES_TO_RUN = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "llama-3.2-90b",
    "gemma-3-27b",
    "qwen3-vl-235b",
]


def format_answer_for_judge(answer: Dict[str, Any]) -> str:
    """Format the VLM answer dict as readable text for judge evaluation."""
    parts = []

    # Natural language summary
    if summary := answer.get("natural_language_summary"):
        parts.append(f"Summary: {summary}")

    # Estimates
    if estimates := answer.get("estimates"):
        parts.append(
            f"Damage Estimate: {estimates.get('structural_damage_pct', 'N/A')}%"
        )
        parts.append(f"Confidence: {estimates.get('confidence', 'N/A')}")
        if roads := estimates.get("roads_affected"):
            parts.append(f"Roads Affected: {roads}")

    # Evidence references
    if refs := answer.get("evidence_refs"):
        if tweet_ids := refs.get("tweet_ids"):
            parts.append(f"Cited Tweets: {', '.join(tweet_ids[:5])}")
        if call_ids := refs.get("call_311_ids"):
            parts.append(f"Cited 311 Calls: {', '.join(call_ids[:5])}")

    # Reasoning (if detailed)
    if reasoning := answer.get("reasoning"):
        parts.append(f"Reasoning: {reasoning}")

    return "\n".join(parts) if parts else str(answer)


def extract_context_for_human(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract human-readable context from packaged retrieval results.

    Now extracts full metadata including timestamps and locations for verification.
    """
    result = {
        "tweets": [],
        "call_311": [],
        "imagery_tiles": [],
        "sensor_data": None,
        "kb_summary": context.get("kb_summary", ""),
    }

    # Extract tweets with full metadata (timestamps, locations)
    for tweet in context.get("tweets", []):
        if isinstance(tweet, dict):
            result["tweets"].append(
                {
                    "text": tweet.get("text", ""),
                    "timestamp": tweet.get("timestamp", tweet.get("date", "N/A")),
                    "location": tweet.get("location", tweet.get("zip", "N/A")),
                    "id": tweet.get("id", tweet.get("tweet_id", "unknown")),
                }
            )
        else:
            # Fallback for string-only tweets (from text_snippets)
            result["tweets"].append(
                {"text": str(tweet), "timestamp": "N/A", "location": "N/A"}
            )

    # Extract 311 calls with full metadata
    for call in context.get("calls", []):
        if isinstance(call, dict):
            result["call_311"].append(
                {
                    "description": call.get("description", call.get("text", "")),
                    "timestamp": call.get("timestamp", call.get("date", "N/A")),
                    "location": call.get(
                        "location", call.get("address", call.get("zip", "N/A"))
                    ),
                    "id": call.get("id", call.get("call_id", "unknown")),
                    "type": call.get("type", call.get("category", "N/A")),
                }
            )
        else:
            result["call_311"].append(
                {"description": str(call), "timestamp": "N/A", "location": "N/A"}
            )

    # If tweets/calls are empty, fall back to text_snippets (for backward compatibility)
    if not result["tweets"] and not result["call_311"]:
        for snippet in context.get("text_snippets", []):
            if "tweet" in snippet.lower() or "@" in snippet:
                result["tweets"].append(
                    {"text": snippet, "timestamp": "N/A", "location": "N/A"}
                )
            elif "311" in snippet.lower() or "call" in snippet.lower():
                result["call_311"].append(
                    {"description": snippet, "timestamp": "N/A", "location": "N/A"}
                )
            else:
                result["tweets"].append(
                    {"text": snippet, "timestamp": "N/A", "location": "N/A"}
                )

    # Imagery tiles with timestamp extracted from tile_id
    for tile in context.get("imagery_tiles", []):
        tile_id = tile.get("tile_id", "unknown")
        # Extract date from tile_id (format: 20170901bC0953430w293730n)
        date_from_id = (
            tile_id[:8] if len(tile_id) >= 8 and tile_id[:8].isdigit() else "N/A"
        )
        if date_from_id != "N/A":
            date_from_id = f"{date_from_id[:4]}-{date_from_id[4:6]}-{date_from_id[6:8]}"

        result["imagery_tiles"].append(
            {
                "tile_id": tile_id,
                "date": tile.get("timestamp", tile.get("date", date_from_id)),
                "flight": tile.get("flight_name", tile.get("source", "N/A")),
                "location": (
                    f"({tile.get('lat', 'N/A')}, {tile.get('lon', 'N/A')})"
                    if tile.get("lat")
                    else tile.get("zip", "N/A")
                ),
            }
        )

    # Sensor table - preserve full content
    if sensor := context.get("sensor_table"):
        result["sensor_data"] = sensor

    return result


def run_multi_judge_evaluation(
    config_path: str,
    output_path: str,
    judges_to_run: List[str] = None,
    skip_rag: bool = False,
) -> None:
    """Run evaluation with multiple judges and save full context."""

    judges_to_run = judges_to_run or JUDGES_TO_RUN

    # Load config
    config = json.loads(Path(config_path).read_text())
    queries = [RAGQuery(**q) for q in config["queries"]]

    logger.info(
        f"Running multi-judge evaluation for {len(queries)} queries with {len(judges_to_run)} judges"
    )

    # Initialize components
    locator = DataLocator(Path(config.get("data_dir", "data")))
    retriever = Retriever(locator)
    client = SplitPipelineClient(
        provider=config.get("provider", "gemini"),
        enable_visual=config.get("enable_visual", True),
        visual_provider=config.get("visual_provider"),
        visual_model=config.get("visual_model"),
    )
    gt = ClaimsGroundTruth(locator.table_path("claims"))

    # Initialize judges
    judges: Dict[str, JudgeClient] = {}
    for judge_name in judges_to_run:
        try:
            judges[judge_name] = JudgeClient(judge_name)
            logger.info(f"✓ Initialized judge: {judge_name}")
        except Exception as e:
            logger.warning(f"✗ Failed to initialize judge {judge_name}: {e}")

    if not judges:
        raise RuntimeError("No judges could be initialized!")

    # Process each query
    records = []
    for i, query in enumerate(queries, 1):
        logger.info(
            f"[{i}/{len(queries)}] Processing {query.zip} ({query.start} to {query.end})"
        )

        # Step 1: Retrieve context
        result = retriever.retrieve(query)
        candidates = select_candidates(result, query.k_tiles, query.n_text)
        context = package_context(candidates)

        # Step 2: Generate response (using Split Pipeline)
        answer = client.infer(
            zip_code=query.zip,
            time_window={"start": str(query.start), "end": str(query.end)},
            imagery_tiles=context["imagery_tiles"],
            text_snippets=context["text_snippets"],
            sensor_table=context["sensor_table"],
            kb_summary=context["kb_summary"],
            project_root=locator.base_dir.parent,
        )
        logger.info(f"VLM Response: {json.dumps(answer, default=str)[:300]}...")

        # Step 3: Get ground truth
        truth = gt.score(query.zip, query.start, query.end)

        # Step 4: Format answer for judges
        answer_text = format_answer_for_judge(answer)
        query_text = f"Assess impact for {query.zip} from {query.start} to {query.end}"
        query_params = {
            "zip": query.zip,
            "start": str(query.start),
            "end": str(query.end),
        }

        # Step 5: Get scores from all judges
        judge_scores = {}
        for judge_name, judge_client in judges.items():
            try:
                logger.info(f"  Evaluating with {judge_name}...")
                result = judge_client.evaluate(
                    query=query_text,
                    context=context,
                    answer=answer_text,
                    query_params=query_params,
                )
                judge_scores[judge_name] = {
                    "faithfulness": result["faithfulness"],
                    "relevance": result["relevance"],
                    "faithfulness_reasoning": result["faithfulness_detail"].get(
                        "reasoning", ""
                    ),
                    "relevance_reasoning": result["relevance_detail"].get(
                        "reasoning", ""
                    ),
                }
                logger.info(
                    f"    {judge_name}: F={result['faithfulness']:.2f}, R={result['relevance']:.2f}"
                )
            except Exception as e:
                logger.warning(f"    {judge_name} failed: {e}")
                judge_scores[judge_name] = {
                    "faithfulness": None,
                    "relevance": None,
                    "error": str(e),
                }

        # Build record
        record = {
            "query_id": i,
            "query": {
                "zip": query.zip,
                "start_date": str(query.start),
                "end_date": str(query.end),
                "description": getattr(query, "comment", f"ZIP {query.zip}"),
            },
            "retrieved_context": extract_context_for_human(context),
            "model_response": {
                "raw": answer,
                "formatted": answer_text,
                "damage_pct": float(
                    answer.get("estimates", {}).get("structural_damage_pct", 0.0)
                ),
                "confidence": float(answer.get("estimates", {}).get("confidence", 0.0)),
            },
            "ground_truth": {
                "actual_damage_pct": truth["damage_pct"],
                "claim_count": truth["claim_count"],
                "total_claim_amount": round(truth["total_amount"], 2),
            },
            "judge_scores": judge_scores,
            "human_scores": {
                "faithfulness": None,
                "relevance": None,
                "notes": "",
            },
        }
        records.append(record)

        # Save intermediate results
        intermediate_path = Path(output_path).with_suffix(".intermediate.json")
        intermediate_path.write_text(
            json.dumps(
                {
                    "total_queries": len(queries),
                    "completed": i,
                    "judges": list(judges.keys()),
                    "records": records,
                },
                indent=2,
                default=str,
            )
        )

    # Save final results
    output_data = {
        "instructions": "Evaluate model responses using the same criteria as the LLM judges. See INSTRUCTIONS.md for detailed rubric.",
        "total_queries": len(records),
        "judges": list(judges.keys()),
        "records": records,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2, default=str))

    logger.success(f"✓ Saved multi-judge validation to {output_file}")
    logger.info(f"  Total queries: {len(records)}")
    logger.info(f"  Judges used: {', '.join(judges.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-judge evaluation")
    parser.add_argument("--config", required=True, help="Path to query config JSON")
    parser.add_argument(
        "--output",
        default="data/validation/coworker_task/multi_judge_validation_25.json",
    )
    parser.add_argument(
        "--judges", nargs="+", default=None, help="Specific judges to run"
    )
    args = parser.parse_args()

    run_multi_judge_evaluation(
        config_path=args.config,
        output_path=args.output,
        judges_to_run=args.judges,
    )
