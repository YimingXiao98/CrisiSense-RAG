"""Generate a validation subset for human annotation."""
import json
import random
from pathlib import Path

def create_validation_subset(source_config, output_config, n_samples=10):
    """Sample queries for human validation."""
    
    # Load source config
    with open(source_config, 'r') as f:
        config = json.load(f)
    
    # Set seed for reproducibility
    random.seed(123)
    
    # Sample queries
    all_queries = config['queries']
    sampled = random.sample(all_queries, min(n_samples, len(all_queries)))
    
    # Create validation config
    validation_config = {
        "data_dir": config.get("data_dir", "data"),
        "provider": config.get("provider", "gemini"),
        "ground_truth": config.get("ground_truth", "claims"),
        "queries": sampled
    }
    
    # Save
    with open(output_config, 'w') as f:
        json.dump(validation_config, f, indent=2)
    
    print(f"Sampled {len(sampled)} queries for validation")
    print(f"Saved to {output_config}")
    
    # Print summary
    for i, q in enumerate(sampled, 1):
        print(f"{i}. ZIP {q['zip']}: {q['start']} to {q['end']} - {q.get('comment', 'N/A')}")

if __name__ == "__main__":
    create_validation_subset(
        "config/queries_100_stratified.json",
        "config/queries_validation_10.json",
        n_samples=10
    )
