import json

with open("data/experiments/exp0_baseline.json") as f:
    data = json.load(f)
    print("=== RECORD 1 ===")
    r = data["records"][0]
    print("Keys:", r.keys())
    print("\nGT:", json.dumps(r.get("ground_truth"), indent=2))
    print("\nResponse:", json.dumps(r.get("model_response"), indent=2))
