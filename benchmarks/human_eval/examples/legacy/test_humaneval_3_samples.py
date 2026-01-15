"""
test run of HumanEvalBenchmark with 3 samples only"""

import sys
from pathlib import Path

# add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from chameleon.benchmarks import get_benchmark

# paths
DATA_PATH = r"C:\Users\urile\OneDrive - BGU\cham_HE\cham_HE\Projects\HumanEvalV3\original_data\human-eval-v2-20210705.jsonl"
OUTPUT_DIR = r"C:\Users\urile\OneDrive - BGU\cham_HE\cham_HE\Projects\HumanEvalV3\test_results"

# config
config = {
    "timeout": 3.0,
    "distortion_model": "mistral-large-latest",
    "validation_model": "mistral-large-latest",
    "generation_model": "mistral-large-latest",
    "k_values": [1],
    "project_path": project_root
}

if __name__ == "__main__":
    print("🧪 Starting test run with 3 samples...\n")
    
    try:
        # Create the benchmark
        benchmark = get_benchmark("human_eval", config)
        
        # Run full pipeline with limit=3
        results = benchmark.run_full_pipeline(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            miu=0.6,  # Medium distortion level
            limit=3   # Only 3 samples!
        )
        
        print("\n✅ Test completed successfully!")
        print(f"📊 Results saved in: {OUTPUT_DIR}")
        print(f"\n📈 Summary:")
        print(results)
        
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        import traceback
        traceback.print_exc()
