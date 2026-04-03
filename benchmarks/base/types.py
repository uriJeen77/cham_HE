"""
Shared pipeline types for all Chameleon benchmarks.

These dataclasses form the contract between pipeline stages and benchmark implementations.
The framework populates Task fields progressively as data flows through each stage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Task:
    """
    A single benchmark item flowing through the pipeline.

    `task_id`  — unique identifier (e.g. "HumanEval/0", "MMLU/stem/42")
    `data`     — raw dataset record; keys are benchmark-specific
    The remaining fields are populated by the pipeline stages:
    `distorted_text`    — the distorted version of the target field
    `miu`               — distortion intensity used to produce distorted_text
    `distortion_index`  — which distortion variant (0-based) within a miu level
    """

    task_id: str
    data: Dict[str, Any]
    distorted_text: Optional[str] = None
    miu: Optional[float] = None
    distortion_index: Optional[int] = None


@dataclass
class DistortionPrompt:
    """
    A fully-formed prompt pair for the distortion engine.

    Separating system and user prompts allows different LLM backends
    (chat vs. completion) to be handled uniformly by the runner.
    """

    system_prompt: str
    user_prompt: str


@dataclass
class BenchmarkValidationResult:
    """
    Result of validating whether a distortion is semantically acceptable.

    Named BenchmarkValidationResult to avoid collision with
    chameleon.distortion.validator.ValidationResult.
    """

    is_valid: bool
    reason: str = ""


@dataclass
class EvalResult:
    """
    Result of evaluating a single (task, model-response) pair.

    `is_correct` — primary correctness signal consumed by the analysis stage
    `metadata`   — benchmark-specific details (e.g. execution trace, matched choice)
    """

    task_id: str
    is_correct: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """
    Aggregate metrics produced by a benchmark's calculate_metrics().

    `overall_score` — primary scalar metric (Pass@1, accuracy, F1, …)
    `per_mu`        — mapping from μ level → score, used for the degradation curve
    `metadata`      — benchmark-specific extras (e.g. pass@k for multiple k values)
    """

    overall_score: float
    per_mu: Dict[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
