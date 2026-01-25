"""
Run VQAv2 pipeline on 3 samples and verify outputs exist.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.VQAV2.vqav2 import run_vqav2_pipeline  # noqa: E402


def main() -> None:
    data_path = REPO_ROOT / "benchmarks" / "VQAV2" / "data" / "vqav2_sampled_optimized.json"
    output_dir = REPO_ROOT / "benchmarks" / "VQAV2" / "results"

    run_vqav2_pipeline(
        data_path=str(data_path),
        output_dir=str(output_dir),
        miu=0.6,
        limit=3,
    )

    summary_path = output_dir / "summary.json"
    tasks_path = output_dir / "tasks_complete.jsonl"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json at {summary_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks_complete.jsonl at {tasks_path}")

    print("OK: VQAv2 pipeline produced summary.json and tasks_complete.jsonl")


if __name__ == "__main__":
    main()
