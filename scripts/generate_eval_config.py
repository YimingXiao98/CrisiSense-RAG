import json
from pathlib import Path
from datetime import datetime


def generate_config():
    gt_path = Path("data/annotations/retrieval_gt.json")
    if not gt_path.exists():
        print(f"Error: {gt_path} not found")
        return

    gt_data = json.loads(gt_path.read_text())

    queries = []
    for key in gt_data.keys():
        # Key format: ZIP_START_END
        parts = key.split("_")
        if len(parts) != 3:
            print(f"Skipping invalid key: {key}")
            continue

        zip_code, start_date, end_date = parts
        queries.append(
            {
                "zip": zip_code,
                "start": start_date,
                "end": end_date,
                "k_tiles": 10,
                "n_text": 20,
            }
        )

    config = {
        "queries": queries,
        "provider": None,  # No VLM needed for retrieval eval
        "ground_truth": "claims",
        "retrieval_gt": str(gt_path.absolute()),
    }

    output_path = Path("data/examples/eval_config_retrieval.json")
    output_path.write_text(json.dumps(config, indent=2))
    print(f"Generated config with {len(queries)} queries at {output_path}")


if __name__ == "__main__":
    generate_config()
