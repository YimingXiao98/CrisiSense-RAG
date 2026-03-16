import os
import shutil
from pathlib import Path

# The 9 experiment files explicitly referenced in the final paper metrics (plot_diagnostics.py)
TARGET_EXPERIMENTS = [
    "data/experiments/exp_complete_map_text_only.json",
    "data/experiments/exp_complete_map_text_caption.json",
    "data/experiments/exp_complete_map.json",
    "data/experiments/exp_llama_text_only.json",
    "data/experiments/exp_llama_text_caption.json",
    "data/experiments/exp_llama_multimodal.json",
    "data/experiments/exp_qwen_text_only.json",
    "data/experiments/exp_qwen_text_caption.json",
    "data/experiments/exp_qwen_multimodal.json",
]

TARGET_DIR = Path("actual_exps")

def main():
    if not TARGET_DIR.exists():
        TARGET_DIR.mkdir(parents=True)
        print(f"Created directory: {TARGET_DIR}")
    
    success_count = 0
    missing_files = []

    for file_path_str in TARGET_EXPERIMENTS:
        src = Path(file_path_str)
        if src.exists():
            dst = TARGET_DIR / src.name
            shutil.copy2(src, dst)
            print(f"Copied: {src.name}")
            success_count += 1
        else:
            missing_files.append(file_path_str)
            print(f"Missing: {src}")

    print("\n--- Summary ---")
    print(f"Successfully copied {success_count} experiment files used in the paper.")
    if missing_files:
        print(f"WARNING: Could not find {len(missing_files)} files. Did they get renamed?")

if __name__ == "__main__":
    main()
