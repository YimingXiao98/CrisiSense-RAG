import json
import argparse
from pathlib import Path
from loguru import logger
from app.core.dataio.schemas import RAGQuery
from app.core.dataio.storage import DataLocator
from app.core.eval.eval_runner import EvalRunner
from app.core.retrieval.context_packager import package_context
from app.core.retrieval.selector import select_candidates
from app.core.models.split_client import SplitPipelineClient


class SplitEvalRunner(EvalRunner):
    def __init__(
        self, locator, provider=None, ground_truth="claims", retrieval_gt=None
    ):
        super().__init__(locator, provider, ground_truth, retrieval_gt)
        # Override client with SplitPipelineClient
        # provider arg is kept for compatibility but SplitPipelineClient defaults to gemini
        self.client = SplitPipelineClient(provider=provider)
        # Re-initialize generation_evaluator with the new client so it uses _generate_text
        from app.core.eval.eval_generation import GenerationEvaluator

        self.generation_evaluator = GenerationEvaluator(self.client)

    def run(self, queries):
        metrics_rows = []
        abs_errors = []
        sq_errors = []
        for query in queries:
            logger.info("Evaluating query (Split-Pipeline)", zip=query.zip)
            result = self.retriever.retrieve(query)

            retrieval_metrics = {}
            if self.retrieval_evaluator:
                query_id = f"{query.zip}_{query.start}_{query.end}"
                retrieval_metrics = self.retrieval_evaluator.evaluate(query_id, result)

            candidates = select_candidates(result, query.k_tiles, query.n_text)
            context = package_context(candidates)

            # Note: SplitPipelineClient.infer will strip imagery internally for text analysis,
            # but we pass it here in case future VisualAnalysisClient needs it.

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
            )
            logger.info(f"Split Pipeline Response: {answer}")

            row = {
                "zip": query.zip,
                "start": str(query.start),
                "end": str(query.end),
                "pred_damage_pct": float(
                    answer.get("estimates", {}).get("structural_damage_pct", 0.0)
                ),
                "confidence": float(answer.get("estimates", {}).get("confidence", 0.0)),
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
                answer=json.dumps(answer),  # Pass full JSON answer for context
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
        output_path = self.locator.processed / "eval_results_split.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        return {"results": output_path}


def main(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    queries = [RAGQuery(**q) for q in config["queries"]]
    locator = DataLocator(Path(config.get("data_dir", "data")))
    runner = SplitEvalRunner(
        locator,
        provider=config.get("provider"),
        ground_truth=config.get("ground_truth", "claims"),
        retrieval_gt=config.get("retrieval_gt"),
    )
    outputs = runner.run(queries)
    logger.info(
        "Split Pipeline Evaluation complete",
        outputs={k: str(v) for k, v in outputs.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run split pipeline evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
