"""
VQAv2 Benchmark - Distortion/Generation pipeline (evaluation deferred).

This mirrors the structure of the HumanEval benchmark but adapts it to
vision-question answering. Evaluation/metrics are intentionally left as a
stub so they can be completed later.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from chameleon.core.base import BaseBenchmark
from chameleon.distortion.engine import DistortionEngine
from benchmarks.VQAV2.data.dataloader import load_entries
from benchmarks.VQAV2.prompts import build_generation_messages, build_validation_prompt

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - optional dependency
    Mistral = None


class VQAV2Benchmark(BaseBenchmark):
    """VQAv2 robustness benchmark pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self._load_env()
        self.logger = self._setup_logger()

    def _load_env(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env_path = repo_root / "chameleon" / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("VQAV2Benchmark")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    # ------------------------------------------------------------------ #
    # BaseBenchmark required methods
    # ------------------------------------------------------------------ #
    def load_tasks(self, data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        entries = load_entries(data_path, limit=limit)
        images_dir = Path(data_path).parent / "sampled_images_300"

        tasks: List[Dict[str, Any]] = []
        for item in entries:
            image_field = item.get("image")
            filename = Path(image_field).name if image_field else f"COCO_val2014_{int(item['image_id']):012d}.jpg"
            image_path = str(images_dir / filename)

            tasks.append(
                {
                    "question_id": str(item["question_id"]),
                    "question": item["question"],
                    "image_id": item.get("image_id"),
                    "image_path": image_path,
                    "answers": item.get("answers", []),
                    "multiple_choice_answer": item.get("multiple_choice_answer"),
                    "answer_type": item.get("answer_type"),
                    "raw_entry": item,
                }
            )

        self.logger.info(f"✓ Loaded {len(tasks)} tasks from {data_path}")
        return tasks

    def format_prompt(self, task: Dict[str, Any]) -> str:
        return f"Image: {task.get('image_path')}\nQuestion: {task.get('question')}\nAnswer:"

    def evaluate_completion(self, task: Dict[str, Any], completion: str) -> Dict[str, Any]:
        return {"passed": None, "result": completion, "status": "pending_eval"}

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"status": "pending", "note": "metrics not implemented"}

    # ------------------------------------------------------------------ #
    # Pipeline helpers
    # ------------------------------------------------------------------ #
    def distort_questions(self, tasks: List[Dict[str, Any]], miu: float) -> List[Dict[str, Any]]:
        distorter = DistortionEngine.create(
            {
                "engine_type": self.config.get("distortion_engine", "api"),
                "vendor": self.config.get("distortion_vendor", "mistral"),
                "model_name": self.config.get("distortion_model", "mistral-large-latest"),
                "api_key_env_var": self.config.get("distortion_api_key_env", "MISTRAL_API_KEY"),
            },
            project_path=self.config.get("project_path"),
        )

        self.logger.info(f"🔀 Distorting {len(tasks)} questions with miu={miu} ...")
        success_count = 0

        for task in tasks:
            try:
                res = distorter.distort_question(
                    question_id=task["question_id"],
                    question=task["question"],
                    miu=miu,
                )
                task["distorted_question"] = res.distorted_question
                task["distortion_success"] = res.success
                task["distortion_miu"] = miu
                task["distortion_latency_ms"] = res.latency_ms
                task["distortion_error"] = res.error
                if res.success:
                    success_count += 1
            except Exception as e:  # pragma: no cover - defensive
                task["distorted_question"] = None
                task["distortion_success"] = False
                task["distortion_error"] = str(e)

        self.logger.info(f"✅ Distorted {success_count}/{len(tasks)} tasks")
        return tasks

    def validate_distortions(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not Mistral:
            self.logger.warning("mistralai not installed - skipping validation")
            for task in tasks:
                task["validation_passed"] = None
            return tasks

        api_key = os.getenv(self.config.get("validation_api_key_env", "MISTRAL_API_KEY"))
        if not api_key:
            self.logger.warning("MISTRAL_API_KEY not found - skipping validation")
            for task in tasks:
                task["validation_passed"] = None
            return tasks

        client = Mistral(api_key=api_key)
        model = self.config.get("validation_model", "mistral-large-latest")

        self.logger.info(f"🔍 Validating {len(tasks)} distortions...")
        passed = 0
        for task in tasks:
            if not task.get("distortion_success"):
                task["validation_passed"] = False
                task["validation_reason"] = "distortion_failed"
                continue

            prompt = build_validation_prompt(task["question"], task.get("distorted_question", ""))
            try:
                response = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16,
                    temperature=0,
                )
                content = response.choices[0].message.content.strip().lower()
                is_yes = content.startswith("yes")
                task["validation_passed"] = is_yes
                task["validation_raw"] = content
                passed += 1 if is_yes else 0
            except Exception as e:  # pragma: no cover
                task["validation_passed"] = None
                task["validation_reason"] = str(e)

        self.logger.info(f"✅ Validation yes={passed}/{len(tasks)} (None means skipped)")
        return tasks

    def generate_answers(self, tasks: List[Dict[str, Any]], for_distorted: bool = False) -> List[Dict[str, Any]]:
        if not Mistral:
            self.logger.warning("mistralai not installed - skipping generation")
            return tasks

        api_key = os.getenv(self.config.get("generation_api_key_env", "MISTRAL_API_KEY"))
        if not api_key:
            self.logger.warning("MISTRAL_API_KEY not found - skipping generation")
            return tasks

        client = Mistral(api_key=api_key)
        model = self.config.get("generation_model", "mistral-large-latest")

        tag = "distorted" if for_distorted else "original"
        self.logger.info(f"💻 Generating answers for {tag} prompts...")

        for task in tasks:
            question_text = task.get("distorted_question" if for_distorted else "question")
            if not question_text:
                task[f"{tag}_answer"] = None
                task[f"{tag}_generation_success"] = False
                task[f"{tag}_generation_error"] = "missing_question"
                continue

            messages = build_generation_messages(question_text, task.get("image_path", ""))

            try:
                response = client.chat.complete(
                    model=model,
                    messages=messages,
                    temperature=self.config.get("generation_temperature", 0.0),
                    max_tokens=self.config.get("generation_max_tokens", 64),
                )
                content = response.choices[0].message.content.strip()
                task[f"{tag}_answer"] = content
                task[f"{tag}_generation_success"] = True
            except Exception as e:  # pragma: no cover
                task[f"{tag}_answer"] = None
                task[f"{tag}_generation_success"] = False
                task[f"{tag}_generation_error"] = str(e)

        return tasks

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def run_full_pipeline(
        self,
        data_path: str,
        output_dir: str,
        miu: float = 0.6,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("⭐  VQAv2 - Robustness Evaluation Pipeline  ⭐")
        self.logger.info("=" * 60)

        # 1) Load
        tasks = self.load_tasks(data_path, limit=limit)

        # 2) Distort
        tasks = self.distort_questions(tasks, miu=miu)

        # 3) Validate distortions
        tasks = self.validate_distortions(tasks)

        # 4a) Generate for original
        tasks = self.generate_answers(tasks, for_distorted=False)

        # 4b) Generate for distorted
        tasks = self.generate_answers(tasks, for_distorted=True)

        # 5) Evaluation stub (deferred)
        original_results = []
        distorted_results = []
        for task in tasks:
            if task.get("original_answer"):
                original_results.append(self.evaluate_completion(task, task["original_answer"]))
            if task.get("distorted_answer"):
                distorted_results.append(self.evaluate_completion(task, task["distorted_answer"]))

        original_metrics = self.calculate_metrics(original_results) if original_results else {"status": "pending"}
        distorted_metrics = self.calculate_metrics(distorted_results) if distorted_results else {"status": "pending"}

        summary = {
            "total_tasks": len(tasks),
            "distortion_successful": sum(1 for t in tasks if t.get("distortion_success")),
            "validation_passed": sum(1 for t in tasks if t.get("validation_passed") is True),
            "original_answers": sum(1 for t in tasks if t.get("original_answer")),
            "distorted_answers": sum(1 for t in tasks if t.get("distorted_answer")),
            "original_metrics": original_metrics,
            "distorted_metrics": distorted_metrics,
            "miu": miu,
            "evaluation_status": "pending",
        }

        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(output_path / "tasks_complete.jsonl", "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

        self.logger.info("📊 Summary saved to summary.json")
        self.logger.info("📝 Tasks saved to tasks_complete.jsonl")
        return summary


def run_vqav2_pipeline(
    data_path: str,
    output_dir: str,
    miu: float = 0.6,
    limit: Optional[int] = None,
    distortion_model: str = "mistral-large-latest",
    generation_model: str = "mistral-large-latest",
    validation_model: str = "mistral-large-latest",
    project_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for the VQAv2 pipeline.
    """
    config = {
        "distortion_model": distortion_model,
        "generation_model": generation_model,
        "validation_model": validation_model,
        "project_path": project_path,
    }
    benchmark = VQAV2Benchmark(config)
    return benchmark.run_full_pipeline(data_path, output_dir, miu=miu, limit=limit)
