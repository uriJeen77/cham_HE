import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add project root to sys.path to allow absolute imports
root_path = str(Path(__file__).parent.parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from benchmarks.human_eval.human_eval import run_humaneval_pipeline

# Ensure the results directory exists
script_dir = Path(__file__).parent
output_dir = str(script_dir / "results")
os.makedirs(output_dir, exist_ok=True)

# Run a small test on 5 samples
if __name__ == "__main__":
    print(f"🚀 Starting test run of 5 tasks on the consolidated pipeline...")
    results = run_humaneval_pipeline(
        data_path="benchmarks/human_eval/data/HumanEval.jsonl.gz",
        output_dir=output_dir,
        miu=0.6,
        limit=3,
        timeout=12.0
    )

    print("\n✅ Test completed!")
    print(f"Total tasks processed: {results.get('total_tasks')}")
    print(f"Original Pass@1: {results.get('original_metrics', {}).get('pass@1', 0):.2%}")
    print(f"Distorted Pass@1: {results.get('distorted_metrics', {}).get('pass@1', 0):.2%}")

    # Export to CSV
    print("\n📊 Exporting results to CSV...")
    results_jsonl = Path(output_dir) / "tasks_complete.jsonl"
    results_csv = Path(output_dir) / "results.csv"

    if results_jsonl.exists():
        tasks = []
        with open(results_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        
        df = pd.DataFrame(tasks)
        df.to_csv(results_csv, index=False, encoding='utf-8-sig')
        print(f"✨ Results exported to {results_csv}")
    else:
        print("⚠️ tasks_complete.jsonl not found, CSV export skipped.")

