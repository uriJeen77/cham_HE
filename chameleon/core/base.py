"""
Abstract base class for all Chameleon benchmarks.

To add a new benchmark, subclass AbstractBenchmark and implement the four
required methods.  Override the optional methods only for behaviour that
differs from the defaults.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.base.types import (
    BenchmarkMetrics,
    BenchmarkValidationResult,
    DistortionPrompt,
    EvalResult,
    Task,
)


# Generic system prompt used when a benchmark does not override get_generation_prompt.
_DEFAULT_GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following task as accurately as possible."
)


class AbstractBenchmark(ABC):
    """
    Base class for Chameleon benchmark implementations.

    Required methods (must be implemented by every benchmark):
        load_data            — load dataset items as Task objects
        get_distortion_prompt — build the distortion prompt pair for one task
        evaluate             — judge one (task, response) pair
        calculate_metrics    — aggregate EvalResult list into BenchmarkMetrics

    Optional methods (sensible defaults; override only what differs):
        get_field_to_distort  — name of the field in Task.data to distort
        get_mu_rules          — mapping of μ values to distortion rule strings
        validate_distortion   — check if a distortion is semantically acceptable
        get_generation_prompt — (system_prompt, user_prompt) for the target LLM
        supports_stage        — whether this benchmark participates in a stage
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # ------------------------------------------------------------------
    # Required — every benchmark must implement these four methods
    # ------------------------------------------------------------------

    @abstractmethod
    def load_data(self, data_path: str) -> List[Task]:
        """Load benchmark items from data_path and return them as Task objects."""

    @abstractmethod
    def get_distortion_prompt(self, task: Task, miu: float, n_distortions: int) -> DistortionPrompt:
        """
        Build the distortion prompt for one task at the given μ level.

        Returns a DistortionPrompt with both system and user prompt components.
        """

    @abstractmethod
    def evaluate(self, task: Task, response: str) -> EvalResult:
        """
        Evaluate a single model response for the given task.

        Returns an EvalResult with is_correct=True/False and optional metadata.
        """

    @abstractmethod
    def calculate_metrics(self, results: List[EvalResult]) -> BenchmarkMetrics:
        """
        Aggregate a list of EvalResults into benchmark-level metrics.

        Returns BenchmarkMetrics with overall_score, per_mu breakdown, and
        any benchmark-specific extras in metadata.
        """

    # ------------------------------------------------------------------
    # Optional — override only what differs from the defaults
    # ------------------------------------------------------------------

    def get_field_to_distort(self) -> str:
        """Name of the Task.data field that should be distorted. Default: 'prompt'."""
        return "prompt"

    def get_mu_rules(self) -> Dict[float, str]:
        """
        Mapping of μ values to distortion rule strings.

        Default: the shared MIU_RULES from chameleon.distortion.constants.
        Override to define benchmark-specific intensity rules.
        """
        from chameleon.distortion.constants import MIU_RULES
        return MIU_RULES

    def validate_distortion(
        self,
        original_text: str,
        distorted_text: str,
        task: Task,
    ) -> BenchmarkValidationResult:
        """
        Check whether a distortion is semantically acceptable.

        Default: always valid (no validation).  Override to add LLM-judge or
        rule-based checks.
        """
        return BenchmarkValidationResult(is_valid=True)

    def get_generation_prompt(self, task: Task) -> Tuple[str, str]:
        """
        Build the (system_prompt, user_prompt) pair sent to the target LLM.

        Default: generic system prompt + the raw content of get_field_to_distort().
        Override to add few-shot examples, special formatting, etc.
        """
        field = self.get_field_to_distort()
        user_prompt = task.distorted_text if task.distorted_text is not None else task.data.get(field, "")
        return _DEFAULT_GENERATION_SYSTEM_PROMPT, user_prompt

    def supports_stage(self, stage: str) -> bool:
        """
        Whether this benchmark participates in a given pipeline stage.

        stage is one of: "distort", "generate", "evaluate", "analyze".
        Default: True for all stages.  Override to opt out of stages that are
        not applicable (e.g. a benchmark that ships pre-distorted data can
        return False for "distort").
        """
        return True

    # ------------------------------------------------------------------
    # Backward-compatibility shims for code that still uses the old API
    # ------------------------------------------------------------------

    def load_tasks(self, data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Deprecated: use load_data() instead."""
        tasks = self.load_data(data_path)
        if limit is not None:
            tasks = tasks[:limit]
        return [{"task_id": t.task_id, **t.data} for t in tasks]

    def format_prompt(self, task: Dict[str, Any]) -> str:
        """Deprecated: use get_generation_prompt() instead."""
        wrapped = Task(task_id=task.get("task_id", ""), data=task)
        _, user_prompt = self.get_generation_prompt(wrapped)
        return user_prompt

    def evaluate_completion(self, task: Dict[str, Any], completion: str) -> Dict[str, Any]:
        """Deprecated: use evaluate() instead."""
        wrapped = Task(task_id=task.get("task_id", ""), data=task)
        result = self.evaluate(wrapped, completion)
        return {
            "task_id": result.task_id,
            "passed": result.is_correct,
            "result": result.metadata.get("result", ""),
        }


# Alias so existing imports of BaseBenchmark keep working.
BaseBenchmark = AbstractBenchmark
