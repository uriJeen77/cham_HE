# New Benchmark Integration Guide

This guide outlines the technical requirements for adding a new benchmark to the `cham_HE` project. To ensure seamless integration with the existing evaluation pipeline, please follow these specifications.

## 1. Class Structure & Interface

Your benchmark should be implemented as a class inheriting from [BaseBenchmark](file:///c:/Users/urile/OneDrive%20-%20BGU/cham_HE/cham_HE/chameleon/core/base.py#4-48).

```python
from chameleon.core.base import BaseBenchmark
from typing import List, Dict, Any, Optional

class MyNewBenchmark(BaseBenchmark):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with a config dictionary containing:
        - distortion_model, validation_model, generation_model
        - miu (distortion level)
        - project_path
        """
        super().__init__(config or {})
        # ... initialization ...

    def load_tasks(self, data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load tasks and return a list of dictionaries."""
        pass

    def format_prompt(self, task: Dict[str, Any]) -> str:
        """Return the formatted prompt/question from a task dictionary."""
        pass

    def evaluate_completion(self, task: Dict[str, Any], completion: str) -> Dict[str, Any]:
        """Run tests/evaluations and return {'passed': bool, 'result': str}."""
        pass

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics (e.g., pass@1, accuracy)."""
        pass
```

## 2. Pipeline Implementation

Implement a [run_full_pipeline](file:///c:/Users/urile/OneDrive%20-%20BGU/cham_HE/cham_HE/benchmarks/human_eval/human_eval.py#455-568) method that orchestrates the following steps:

1.  **Loading**: Load data via [load_tasks](file:///c:/Users/urile/OneDrive%20-%20BGU/cham_HE/cham_HE/chameleon/core/base.py#18-25).
2.  **Distortion**: Use `chameleon.distortion.engine.DistortionEngine` to distort the prompts.
3.  **Validation**: Implement a validation step to ensure semantic equivalence (using an LLM "judge").
4.  **Generation**: Generate responses/code for both original and distorted prompts.
5.  **Evaluation**: Run [evaluate_completion](file:///c:/Users/urile/OneDrive%20-%20BGU/cham_HE/cham_HE/benchmarks/human_eval/human_eval.py#372-394) for both sets.
6.  **Results**: Save results to `summary.json` and a detailed `tasks_complete.jsonl`.

## 3. Directory Structure

Place your implementation in `benchmarks/<benchmark_name>/`:
```text
benchmarks/<benchmark_name>/
├── __init__.py
├── <benchmark_name>.py (Main implementation)
├── engine/ (Custom processing logic, execution, etc.)
├── data/ (Sample datasets)
└── examples/ (Usage scripts)
```

## 4. Key Dependencies

- **Distorter**: Use the project's internal `DistortionEngine`.
- **LLM Models**: Default to `mistral-large-latest` unless specified otherwise in the config.
- **Environment Variables**: Use `MISTRAL_API_KEY` for API access.

## 5. Output Consistency

Ensure your metrics and summary format match the existing project structure to allow for consolidated reporting across different benchmarks.
