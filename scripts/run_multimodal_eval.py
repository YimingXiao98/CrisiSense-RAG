import json
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.eval.eval_runner import EvalRunner
from app.core.retrieval.context_packager import package_context
from app.core.retrieval.selector import select_candidates
from app.core.models.split_client import SplitPipelineClient


class MultimodalEvalRunner(EvalRunner):
    def __init__(
        self, locator, provider=None, ground_truth="claims", retrieval_gt=None, device="cpu"
    ):
        super().__init__(locator, provider, ground_truth, retrieval_gt, device=device)
        # Initialize SplitPipelineClient with visual analysis enabled
        self.client = SplitPipelineClient(
            provider=provider,
            enable_visual=True,
            use_llm_fusion=True
        )
        # Re-initialize generation_evaluator with the new client
        from app.core.eval.eval_generation import GenerationEvaluator

        self.generation_evaluator = GenerationEvaluator(self.client)

    def run(self, queries):
        metrics_rows = []
        abs_errors = []
        sq_errors = []
        for query in queries:
            logger.info(f"Evaluating query (Multimodal): ZIP {query.zip}")
            
            # Enable visual search in retrieval
            query.enable_visual_search = True
            result = self.retriever.retrieve(query)

            retrieval_metrics = {}
            if self.retrieval_evaluator:
                query_id = f"{query.zip}_{query.start}_{query.end}"
                retrieval_metrics = self.retrieval_evaluator.evaluate(query_id, result)

            candidates = select_candidates(result, query.k_tiles, query.n_text)
            context = package_context(candidates)

            # Run inference with visual analysis
            answer = self.client.infer(
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
                project_root=self.locator.base_dir.parent.parent # Resolve to project root
            )
            logger.info(f"Multimodal Response: {answer.get('natural_language_summary')[:100]}...")

            row = {
                "zip": query.zip,
                "start": str(query.start),
                "end": str(query.end),
                "pred_damage_pct": float(
                    answer.get("estimates", {}).get("structural_damage_pct", 0.0)
                ),
                "confidence": float(answer.get("estimates", {}).get("confidence", 0.0)),
                "fusion_method": answer.get("fusion_method", "unknown"),
                "visual_damage_pct": float(answer.get("visual_analysis", {}).get("damage_pct", 0.0)),
                "text_damage_pct": float(answer.get("text_analysis", {}).get("damage_pct", 0.0)),
            }

            # Generation Metrics (Faithfulness, Relevance)
            query_text = (
                f"Assess impact for {query.zip} from {query.start} to {query.end}"
            )
            query_params = {
                "zip": query.zip,
                "start": str(query.start),
                "end": str(query.end),
            }
            gen_metrics = self.generation_evaluator.evaluate(
                query=query_text,
                context=context,
                answer=json.dumps(answer),
                query_params=query_params,
            )
            row.update(gen_metrics)

            row.update(retrieval_metrics)
            if self.gt:
                truth = self.gt.score(query.zip, query.start, query.end)
                row.update(
                    actual_damage_pct=truth["damage_pct"],
                    claim_count=truth["claim_count"],
                    total_claim_amount=round(truth["total_amount"], 2),
                )
                diff = row["pred_damage_pct"] - truth["damage_pct"]
                abs_errors.append(abs(diff))
                sq_errors.append(diff**2)
                row["abs_error"] = round(abs(diff), 2)
            metrics_rows.append(row)

        summary = {}
        if abs_errors:
            summary = {
                "mae": round(sum(abs_errors) / len(abs_errors), 3),
                "rmse": round((sum(sq_errors) / len(sq_errors)) ** 0.5, 3),
                "count": len(abs_errors),
            }
        
        output = {"summary": summary, "records": metrics_rows}
        output_path = self.locator.processed / "eval_results_multimodal.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        return {"results": output_path}


def main(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    queries = [RAGQuery(**q) for q in config["queries"]]
    locator = DataLocator(Path(config.get("data_dir", "data")))
    
    # Force provider to openai for visual analysis if not specified
    provider = config.get("provider", "openai")
    
    runner = MultimodalEvalRunner(
        locator,
        provider=provider,
        ground_truth=config.get("ground_truth", "claims"),
        retrieval_gt=config.get("retrieval_gt"),
        device="cpu",
    )
    outputs = runner.run(queries)
    logger.info(
        "Multimodal Pipeline Evaluation complete",
        outputs={k: str(v) for k, v in outputs.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal pipeline evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
