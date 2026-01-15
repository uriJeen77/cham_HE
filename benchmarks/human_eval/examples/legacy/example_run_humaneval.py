"""
דוגמה לשימוש ב-HumanEvalBenchmark
"""

from chameleon.benchmarks import get_benchmark
from pathlib import Path

# נתיבים
DATA_PATH = "C:\\Users\\urile\\OneDrive - BGU\\cham_HE\\cham_HE\\Projects\\HumanEvalV3\\original_data\\human-eval-v2-20210705.jsonl"
OUTPUT_DIR = "C:\\Users\\urile\\OneDrive - BGU\\cham_HE\\cham_HE\\Projects\\HumanEvalV3\\results"

# הגדרות
config = {
    "timeout": 3.0,
    "distortion_model": "mistral-large-latest",
    "validation_model": "mistral-large-latest",
    "generation_model": "mistral-large-latest",
    "k_values": [1],
    "project_path": Path("C:\\Users\\urile\\OneDrive - BGU\\cham_HE\\cham_HE")
}

# יצירת הבנצ'מרק
if __name__ == "__main__":
    benchmark = get_benchmark("human_eval", config)

    # הרצת הצינור המלא
    print("🚀 מתחיל הרצה של HumanEvalV3...")
    results = benchmark.run_full_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        miu=0.6  # רמת עיוות בינונית
    )

    print("\n✅ הושלם!")
    print(f"📊 תוצאות נשמרו ב: {OUTPUT_DIR}")

