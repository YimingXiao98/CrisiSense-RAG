"""Extract detailed validation data from summary results and rebuild with structure."""
import json
from pathlib import Path

def create_detailed_validation_from_summary():
    """
    Create detailed validation file from summary results.
    Note: This simplified version uses the summary data and creates a structure
    suitable for human annotation, though it won't have full retrieved context text.
    """
    
    # Load summary results
    summary_path = Path("data/validation/coworker_task/validation_results_25.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load query configs
    config_path = Path("data/validation/coworker_task/queries_validation_25.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create detailed records
    detailed_records = []
    
    for i, (record, query) in enumerate(zip(summary['records'], config['queries']), 1):
        detailed_record = {
            "query_id": i,
            "query": {
                "zip": record['zip'],
                "start_date": record['start'],
                "end_date": record['end'],
                "description": query.get('comment', f"ZIP {record['zip']}, {record['start']} to {record['end']}")
            },
            "model_prediction": {
                "predicted_damage_pct": record['pred_damage_pct'],
                "confidence": record['confidence'],
                "interpretation": interpret_prediction(record['pred_damage_pct'], record['confidence'])
            },
            "ground_truth": {
                "actual_damage_pct": record['actual_damage_pct'],
                "claim_count": record['claim_count'],
                "total_claim_amount": record['total_claim_amount'],
                "interpretation": interpret_ground_truth(record['actual_damage_pct'], record['claim_count'])
            },
            "gpt4_automated_scores": {
                "faithfulness": record['faithfulness'],
                "relevance": record['relevance'],
                "note": "These are GPT-4's automated scores for reference"
            },
            "your_human_scores": {
                "faithfulness": None,
                "relevance": None,
                "notes": ""
            },
            "annotation_guidance": {
                "faithfulness_question": "Does the model's prediction seem supported by the ground truth data?",
                "relevance_question": "Would this prediction be useful for disaster response?",
                "consider": [
                    f"Model predicted {record['pred_damage_pct']}%, actual was {record['actual_damage_pct']}%",
                    f"Model confidence: {record['confidence']}",
                    f"{record['claim_count']} insurance claims filed (${record['total_claim_amount']:,.0f} total)"
                ]
            }
        }
        
        detailed_records.append(detailed_record)
    
    # Save detailed validation
    output_path = Path("data/validation/coworker_task/detailed_validation_25.json")
    with open(output_path, 'w') as f:
        json.dump({
            "instructions": "Score each query's faithfulness and relevance (0.0-1.0). See INSTRUCTIONS.md for detailed rubric.",
            "total_queries": len(detailed_records),
            "records": detailed_records
        }, f, indent=2)
    
    print(f"✅ Created {output_path}")
    print(f"   Total queries: {len(detailed_records)}")
    print(f"   Ready for human annotation!")

def interpret_prediction(pred_pct, confidence):
    """Generate human-readable interpretation of model prediction."""
    if pred_pct == 0.0:
        return f"No damage detected (confidence: {confidence})"
    elif pred_pct < 5:
        return f"Minimal damage ({pred_pct}%, confidence: {confidence})"
    elif pred_pct < 20:
        return f"Moderate damage ({pred_pct}%, confidence: {confidence})"
    else:
        return f"Severe damage ({pred_pct}%, confidence: {confidence})"

def interpret_ground_truth(actual_pct, claim_count):
    """Generate human-readable interpretation of ground truth."""
    if actual_pct == 0.0:
        return f"No verified damage (0 claims or minimal)"
    elif actual_pct < 5:
        return f"Minor impact ({actual_pct}%, {claim_count} claims)"
    elif actual_pct < 20:
        return f"Moderate impact ({actual_pct}%, {claim_count} claims)"
    else:
        return f"Severe impact ({actual_pct}%, {claim_count} claims)"

if __name__ == "__main__":
    create_detailed_validation_from_summary()
