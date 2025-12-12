"""Generate detailed validation file with full responses for human annotation."""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.eval.eval_runner import EvalRunner, EvalConfig
from app.core.dataio.locators import DataLocator
from loguru import logger

def generate_detailed_validation(config_path: str, output_path: str):
    """Run evaluation and save detailed responses for human annotation."""
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create eval config
    eval_config = EvalConfig(
        data_dir=config_dict.get('data_dir', 'data'),
        provider=config_dict.get('provider', 'gemini'),
        visual_provider=config_dict.get('visual_provider'),
        visual_model=config_dict.get('visual_model'),
        ground_truth=config_dict.get('ground_truth', 'claims'),
        queries=config_dict['queries']
    )
    
    # Initialize runner
    locator = DataLocator(Path(eval_config.data_dir))
    runner = EvalRunner(
        locator=locator,
        provider=eval_config.provider,
        visual_provider=eval_config.visual_provider,
        visual_model=eval_config.visual_model,
        ground_truth_source=eval_config.ground_truth
    )
    
    # Store detailed results
    detailed_results = []
    
    logger.info(f"Generating detailed validation for {len(eval_config.queries)} queries...")
    
    for i, query in enumerate(eval_config.queries, 1):
        logger.info(f"[{i}/{len(eval_config.queries)}] Processing {query['zip']} {query['start']} to {query['end']}")
        
        # Run retrieval
        context = runner.retriever.retrieve(
            zip_code=query['zip'],
            start_date=query['start'],
            end_date=query['end'],
            k_tiles=query.get('k_tiles', 10),
            n_text=query.get('n_text', 20)
        )
        
        # Get model response
        response = runner.client.infer(
            zip_code=query['zip'],
            start_date=query['start'],
            end_date=query['end'],
            retrieved_context=context,
            project_root=locator.base_dir.parent
        )
        
        # Get ground truth
        gt = runner.gt_source.score(query['zip'], query['start'], query['end'])
        
        # Format retrieved context for readability
        retrieved_evidence = {
            "tweets": [],
            "call_311": [],
            "imagery_tiles": [],
            "sensors": []
        }
        
        # Extract tweets
        for doc in context.get('text_docs', []):
            if 'tweet' in doc.get('doc_type', '').lower():
                retrieved_evidence["tweets"].append({
                    "id": doc.get('id'),
                    "text": doc.get('text', doc.get('content', '')),
                    "timestamp": doc.get('timestamp', 'N/A')
                })
        
        # Extract 311 calls
        for doc in context.get('text_docs', []):
            if '311' in doc.get('doc_type', '').lower():
                retrieved_evidence["call_311"].append({
                    "id": doc.get('id'),
                    "type": doc.get('call_type', 'N/A'),
                    "description": doc.get('text', doc.get('content', '')),
                    "timestamp": doc.get('timestamp', 'N/A')
                })
        
        # Extract imagery
        for tile in context.get('imagery_tiles', []):
            retrieved_evidence["imagery_tiles"].append({
                "id": tile.get('tile_id'),
                "date": tile.get('date', 'N/A'),
                "flight": tile.get('flight_name', 'N/A')
            })
        
        # Build detailed record
        detailed_record = {
            "query_id": i,
            "query": {
                "zip": query['zip'],
                "start_date": query['start'],
                "end_date": query['end'],
                "comment": query.get('comment', '')
            },
            "retrieved_context": retrieved_evidence,
            "model_response": {
                "reasoning": response.get('reasoning', ''),
                "natural_language_summary": response.get('natural_language_summary', ''),
                "estimates": response.get('estimates', {}),
                "evidence_refs": response.get('evidence_refs', {})
            },
            "ground_truth": {
                "actual_damage_pct": gt['damage_pct'],
                "claim_count": gt['claim_count'],
                "total_amount": gt['total_amount']
            },
            "human_scores": {
                "faithfulness": None,
                "relevance": None,
                "notes": ""
            }
        }
        
        detailed_results.append(detailed_record)
    
    # Save detailed results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    logger.success(f"Saved detailed validation to {output_file}")
    logger.info(f"Total queries: {len(detailed_results)}")

if __name__ == "__main__":
    generate_detailed_validation(
        config_path="config/queries_validation_25.json",
        output_path="data/validation/coworker_task/detailed_validation_25.json"
    )
