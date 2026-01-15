"""
Batch Processor for Benchmark-Agnostic Model Evaluation.

This module provides a generic way to evaluate models across different benchmarks
by using the BaseBenchmark interface.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from benchmarks import get_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """
    Configuration for benchmark evaluation.
    """
    project_dir: Path
    model: str
    benchmark_type: str = "human_eval"
    api_key: str = ""

    @classmethod
    def from_project(cls, project_name: str, projects_dir: str = "Projects") -> "EvaluationConfig":
        """
        Create config from project settings.
        """
        import yaml

        project_dir = Path(projects_dir) / project_name
        config_path = project_dir / "project_config.yaml"
        if not config_path.exists():
            # Fallback to config.yaml if project_config.yaml doesn't exist
            config_path = project_dir / "config.yaml"
            
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found in: {project_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        env_path = project_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        target_cfg = cfg.get("target_model", {})
        vendor = target_cfg.get("vendor", "openai")
        api_key = os.getenv(f"{vendor.upper()}_API_KEY", "")
        
        model_name = target_cfg.get("name", cfg.get("model_name", "unknown-model"))
        benchmark_type = cfg.get("benchmark_type", "human_eval")

        return cls(
            project_dir=project_dir,
            model=model_name,
            benchmark_type=benchmark_type,
            api_key=api_key,
        )


class BatchProcessor:
    """
    Generic processor that evaluates model completions using a specific benchmark.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.benchmark = get_benchmark(config.benchmark_type, {"timeout": 3.0})

    def evaluate(
        self,
        sample_file: str,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on a samples file.
        """
        logger.info(f"Starting evaluation for project: {self.config.project_dir.name}")
        logger.info(f"Benchmark: {self.config.benchmark_type}")
        logger.info(f"Samples file: {sample_file}")

        if not Path(sample_file).exists():
            raise FileNotFoundError(f"Samples file not found: {sample_file}")

        # Load samples (model completions)
        samples = []
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # In a real scenario, we might want to load the original tasks 
        # to have access to tests/golden answers if not present in samples.
        # For simplicity here, we assume evaluate_completion handles it.
        
        results = []
        for sample in samples:
            # Note: This assumes the sample contains enough info for the benchmark to evaluate.
            # Usually 'task_id' and 'completion'.
            res = self.benchmark.evaluate_completion(sample, sample["completion"])
            results.append(res)

        metrics = self.benchmark.calculate_metrics(results)

        logger.info("Evaluation complete.")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

        return {
            "status": "complete",
            "metrics": metrics,
            "results_count": len(results)
        }


def run_evaluation(project_name: str, projects_dir: str = "Projects"):
    """Entry point for running evaluation."""
    config = EvaluationConfig.from_project(project_name, projects_dir)
    processor = BatchProcessor(config)
    
    # Logic to find samples file
    sample_file = config.project_dir / "results" / "samples.jsonl"
    if not sample_file.exists():
         # Check root of project
         sample_file = config.project_dir / "samples.jsonl"
    
    if not sample_file.exists():
        logger.error(f"Could not find samples file for project {project_name}")
        return

    return processor.evaluate(str(sample_file))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    args = parser.parse_args()
    
    run_evaluation(args.project, args.projects_dir)
