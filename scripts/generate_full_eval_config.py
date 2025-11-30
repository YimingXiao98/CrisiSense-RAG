import json
from pathlib import Path


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

    # Limit to 25 queries if there are more
    queries = queries[:25]

    config = {
        "queries": queries,
        "provider": "gemini-1.5-pro",
        "ground_truth": "claims",
        "retrieval_gt": str(gt_path.absolute()),
    }

    output_path = Path("data/examples/eval_config.json")
    output_path.write_text(json.dumps(config, indent=2))
    print(f"Generated config with {len(queries)} queries at {output_path}")


if __name__ == "__main__":
    generate_config()
