import json
import pandas as pd
from pathlib import Path


def main():
    # Load Exp0 (Baseline)
    with open("data/experiments/exp0_baseline.json") as f:
        exp0_data = json.load(f)

    # Load Exp1 (Final)
    interim_path = Path("data/experiments/exp1_caption_bridge.json")
    if not interim_path.exists():
        interim_path = Path("data/experiments/exp1_caption_bridge.intermediate.json")

    if not interim_path.exists():
        print("No results file found for Exp1.")
        return

    with open(interim_path) as f:
        exp1_data = json.load(f)

    # Convert to dictionaries keyed by zip/query_id
    # Assuming records have 'query' field which contains 'zip'

    def get_results_dict(records):
        res = {}
        for r in records:
            # Try to find zip
            zip_code = r.get("query", {}).get("zip")
            if not zip_code:
                # Fallback to parsing query text or matching order if stable (but matching by zip is safer)
                continue

            # Get MAE or error for this query
            # Actual and Predicted are in 'model_response' or similar?
            # Exp0 metadata says "mae", but per-record error is needed.
            # Let's look at the structure. Usually "evaluation" or "ground_truth" vs "response".
            # The previous log output showed "Predicted: 0.00%, Actual: 0.05%, Diff: 0.05%"
            # So let's try to extract that.

            # Parsing from "natural_language_summary" or "estimates" in model output?
            # Or maybe the record already has 'metrics'?

            # Let's assume there is a 'metrics' key or we calculate it.
            # Wait, run_baseline_experiment saves records.
            # Let's look at one record from the log or assume standard structure.
            # "estimates": {'flood_impact_pct': 0.0, ...}
            # "ground_truth": ...

            # Try to get error from metrics
            metrics = r.get("metrics", {})
            error = metrics.get("error_structural_damage")

            if error is None:
                # Debug if first record
                if r.get("query_id") == 1:
                    print(f"Debug Rec 1 Keys: {r.keys()}")
                    print(f"Debug Rec 1 Metrics: {metrics}")

                # Try to calculate from estimates and GT
                try:
                    # model_response -> damage_pct (top level formatted)
                    # OR model_response -> raw -> estimates -> flood_impact_pct

                    pred = r.get("model_response", {}).get("damage_pct")
                    if pred is None:
                        pred = (
                            r.get("model_response", {})
                            .get("raw", {})
                            .get("estimates", {})
                            .get("flood_impact_pct")
                        )
                    if pred is None:
                        pred = (
                            r.get("model_response", {})
                            .get("estimates", {})
                            .get("flood_impact_pct", 0.0)
                        )

                    # GT key is 'actual_damage_pct'
                    actual = r.get("ground_truth", {}).get("actual_damage_pct")

                    if pred is not None and actual is not None:
                        error = abs(pred - actual)
                except Exception as e:
                    print(f"Calc error: {e}")
                    error = None

            if error is not None:
                res[zip_code] = error
            else:
                # print(f"Could not calc error for {zip_code}")
                pass
        return res

    exp0_results = get_results_dict(exp0_data["records"])
    exp1_results = get_results_dict(exp1_data["records"])

    # Find common zips
    common_zips = set(exp1_results.keys()) & set(exp0_results.keys())

    if not common_zips:
        print("No common queries found to compare (or parsing failed).")
        # Print keys to debug
        print(f"Exp0 Zips sample: {list(exp0_results.keys())[:5]}")
        print(f"Exp1 Zips sample: {list(exp1_results.keys())[:5]}")
        return

    print(f"Comparing {len(common_zips)} queries:")

    exp0_errors = []
    exp1_errors = []

    improvements = 0
    degradations = 0
    tied = 0

    print(f"{'ZIP':<10} | {'Exp0 Err':<10} | {'Exp1 Err':<10} | {'Diff':<10}")
    print("-" * 46)

    for z in sorted(common_zips):
        e0 = exp0_results[z]
        e1 = exp1_results[z]
        diff = e1 - e0  # Negative means improvement (lower error)

        exp0_errors.append(e0)
        exp1_errors.append(e1)

        if diff < -0.001:
            improvements += 1
            status = "BETTER"
        elif diff > 0.001:
            degradations += 1
            status = "WORSE"
        else:
            tied += 1
            status = "SAME"

        print(f"{z:<10} | {e0:>9.2f}% | {e1:>9.2f}% | {diff:>9.2f}% ({status})")

    avg_mae0 = sum(exp0_errors) / len(exp0_errors)
    avg_mae1 = sum(exp1_errors) / len(exp1_errors)

    print("-" * 46)
    print(f"Interim MAE (n={len(common_zips)}):")
    print(f"Exp0 Baseline: {avg_mae0:.4f}%")
    print(f"Exp1 Caption:  {avg_mae1:.4f}%")
    print(f"Change:        {avg_mae1 - avg_mae0:.4f}%")
    print(f"Wins: {improvements}, Losses: {degradations}, Ties: {tied}")


if __name__ == "__main__":
    main()
