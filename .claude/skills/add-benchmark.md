---
name: add-benchmark
description: Use when a user wants to add a new benchmark dataset to the Chameleon project. Guides through interview, implementation checklist, smoke test, and unit test scaffolding.
---

# Add Benchmark Skill

## Rule
INTERVIEW FIRST. Write no code until Phase 1 is complete.
Then follow the Phase 2 checklist in strict order — announce each step number, implement it, mark it done, move on.

---

## Phase 1: Interview

Ask the user ALL of the following before writing any code. You may batch them into one message.

**Q1. Benchmark name** — What is the snake_case identifier?
(e.g. `mmlu`, `hebrew_qa`, `omni_math`)
→ Used as: directory name, registry key, class name prefix (TitleCase).

**Q2. Domain / description** — What kind of tasks does it contain?
(e.g. Python coding challenges, math reasoning, visual QA, Hebrew NLP, multiple-choice science)

**Q3. Dataset source** — How is the data stored or accessed?
- **A) Local file** in `benchmarks/<name>/data/` (JSONL, CSV, or `.jsonl.gz`)
- **B) HuggingFace** `datasets` library (ask for dataset name + split)
- **C) Custom API / scraper** (ask for endpoint description)

**Q4. Evaluation paradigm** — How should a model response be judged?
Based on the domain, recommend one of:
- **Code execution** (sandboxed subprocess) — best for coding tasks
- **Multiple choice** (exact match on A/B/C/D) — best for MCQ benchmarks
- **Exact match / F1** (string comparison) — best for short-answer QA
- **LLM-as-judge** (call Mistral/OpenAI to score) — best for open-ended text

State your recommendation and rationale. Then ask the user to confirm or choose differently.

**Q5. Field to distort** — Which field in each task record contains the problem text that will be semantically distorted?
Default: `"prompt"`. Confirm if the benchmark uses a different field name (e.g. `"question"`, `"description"`, `"problem"`).

**Q6. Smoke test size** — How many tasks to run during verification? (Default: 3)

**Then check the filesystem**: Does `benchmarks/<name>/` already exist?
- If **yes**: tell the user, then ask — extend the existing implementation or overwrite?
- If **no**: proceed to Phase 2.

---

## Phase 2: Implementation Checklist

Announce each step ("**Step N — <title>**") before executing it. Mark `[x]` when done before moving to the next.

---

### [ ] Step 1 — Create directory structure

Recommend the HumanEval layout. Simplify only if the benchmark is simple (no sandbox, single-file evaluation).

**Full layout (recommended for most benchmarks):**
```
benchmarks/<name>/
├── __init__.py
├── README.md          ← generate at Step 12
├── <name>.py          ← main class (Step 9)
├── config.py          (Step 8)
├── models.py          (Step 6)
├── prompts.py         (Step 7)
├── data/              ← dataset files
├── engine/
│   ├── __init__.py
│   ├── data.py        (Step 3)
│   ├── execution.py   (Step 4)
│   └── evaluation.py  (Step 5)
└── examples/
    └── test_consolidated.py
```

**Minimal layout (simple benchmarks, no subprocess sandbox):**
```
benchmarks/<name>/
├── __init__.py
├── <name>.py
├── config.py
├── models.py
├── prompts.py
└── data/
```

Create directories and empty `__init__.py` files.

---

### [ ] Step 2 — Place or document the dataset

- **Local file**: copy dataset into `benchmarks/<name>/data/`. Support `.jsonl`, `.csv`, `.jsonl.gz`.
- **HuggingFace**: document the `datasets.load_dataset()` call in `engine/data.py`. Do not store data locally.
- **Custom API**: document the endpoint and auth requirements in `engine/data.py`.

> **Note**: If the dataset file is >10 MB, add it to `.gitignore` and document how to obtain it in the README.

---

### [ ] Step 3 — Implement `engine/data.py`

Loads the dataset and returns `List[Task]`.

> ⚠️ **Warning**: The keys you put in `Task.data` here define the interface for ALL later methods. `evaluate()`, `get_distortion_prompt()`, and `prompts.py` must all reference the same keys consistently.

Pattern:
```python
from benchmarks.base.types import Task

def load_data(data_path: str) -> list[Task]:
    tasks = []
    # parse file / HF dataset / API → records
    for record in records:
        tasks.append(Task(
            task_id=record["task_id"],     # must be unique
            data={
                "prompt": record["..."],   # field to distort (confirmed in interview Q5)
                # ... other fields needed by evaluate()
            }
        ))
    return tasks
```

---

### [ ] Step 4 — Implement `engine/execution.py`

Runs one (task, response) pair and returns `{"passed": bool, "result": str}`.

Match the evaluation paradigm chosen in the interview:

**Code execution** (sandboxed subprocess):
```python
import subprocess, tempfile, os

def execute_code(task, response: str, timeout: float = 3.0) -> dict:
    code = response + "\n" + task.data["test"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=timeout
        )
        return {"passed": result.returncode == 0, "result": result.stdout or result.stderr}
    except subprocess.TimeoutExpired:
        return {"passed": False, "result": "timeout"}
    finally:
        os.unlink(path)
```

> ⚠️ **Warning**: Never use `exec()` or `eval()` on untrusted code. Always use subprocess.

**Multiple choice** (exact match):
```python
def check_answer(task, response: str) -> dict:
    expected = task.data["answer"].strip().upper()
    predicted = response.strip().upper()[0] if response.strip() else ""
    return {"passed": predicted == expected, "result": predicted}
```

**Exact match / F1**: implement string normalization + token overlap.

**LLM-as-judge**: call the configured judge model with a scoring prompt; parse binary or numeric score.

---

### [ ] Step 5 — Implement `engine/evaluation.py`

Orchestrates execution across all tasks.

> ⚠️ **Warning**: Results must include tasks where `task.miu == 0.0` (original undistorted problems). Without this baseline, the degradation curve has no reference point.

```python
def evaluate_all(tasks, responses, executor_fn) -> list:
    results = []
    for task, response in zip(tasks, responses):
        outcome = executor_fn(task, response)
        results.append({
            "task_id": task.task_id,
            "miu": task.miu,       # 0.0 for originals — required for degradation curve
            "is_correct": outcome["passed"],
            "metadata": outcome,
        })
    return results
```

---

### [ ] Step 6 — Implement `models.py`

Data classes specific to this benchmark:

```python
from dataclasses import dataclass, field

@dataclass
class <Name>Task:
    task_id: str
    prompt: str
    # benchmark-specific fields (answer, tests, choices, etc.)

@dataclass
class PipelineConfig:
    model: str = "mistral-small-latest"
    timeout: float = 3.0
    mu_levels: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    n_distortions: int = 1
```

---

### [ ] Step 7 — Implement `prompts.py`

All LLM prompts in one place. Adapt tone and content to the benchmark's domain.

```python
DISTORTION_SYSTEM_PROMPT = """You are a semantic paraphrasing engine. ..."""

GENERATION_SYSTEM_PROMPT = """You are an expert in <domain>. ..."""

def build_distortion_user_prompt(task, miu: float, n_distortions: int, mu_rules: dict) -> str:
    rule = mu_rules.get(miu, "")
    field_value = task.data.get("<field_to_distort>", "")
    return f"Paraphrase at intensity {miu}.\nRule: {rule}\n\n{field_value}"

def build_generation_user_prompt(task) -> str:
    return task.distorted_text or task.data.get("<field_to_distort>", "")
```

---

### [ ] Step 8 — Implement `config.py`

Constants for this benchmark:

```python
DEFAULT_MODEL = "mistral-small-latest"
TIMEOUT = 3.0
MU_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_DISTORTIONS = 1
K_VALUES = [1, 10, 100]   # for pass@k; remove if not applicable
```

---

### [ ] Step 9 — Implement `<name>.py` — the main class

Subclass `AbstractBenchmark` from `chameleon/core/base.py`. Wire together all engine pieces.

```python
from chameleon.core.base import AbstractBenchmark
from benchmarks.base.types import Task, DistortionPrompt, EvalResult, BenchmarkMetrics
from benchmarks.<name>.engine.data import load_data as _load_data
from benchmarks.<name>.engine.execution import execute_code
from benchmarks.<name>.prompts import (
    DISTORTION_SYSTEM_PROMPT, GENERATION_SYSTEM_PROMPT,
    build_distortion_user_prompt, build_generation_user_prompt,
)
from collections import defaultdict
from typing import List


class <Name>Benchmark(AbstractBenchmark):

    def load_data(self, data_path: str) -> List[Task]:
        return _load_data(data_path)

    def get_field_to_distort(self) -> str:
        return "<field_to_distort>"   # from interview Q5

    def get_distortion_prompt(self, task: Task, miu: float, n_distortions: int) -> DistortionPrompt:
        return DistortionPrompt(
            system_prompt=DISTORTION_SYSTEM_PROMPT,
            user_prompt=build_distortion_user_prompt(task, miu, n_distortions, self.get_mu_rules()),
        )

    def get_generation_prompt(self, task: Task):
        return GENERATION_SYSTEM_PROMPT, build_generation_user_prompt(task)

    def evaluate(self, task: Task, response: str) -> EvalResult:
        outcome = execute_code(task, response)
        return EvalResult(
            task_id=task.task_id,
            is_correct=outcome["passed"],
            metadata=outcome,
        )

    def calculate_metrics(self, results: List[EvalResult]) -> BenchmarkMetrics:
        by_mu = defaultdict(list)
        for r in results:
            mu = r.metadata.get("miu", 0.0)
            by_mu[mu].append(r.is_correct)
        per_mu = {mu: sum(v) / len(v) for mu, v in by_mu.items()}
        overall = sum(r.is_correct for r in results) / len(results) if results else 0.0
        return BenchmarkMetrics(overall_score=overall, per_mu=per_mu)
```

---

### [ ] Step 10 — Register in `benchmarks/__init__.py`

> ⚠️ **Warning**: This is the most commonly forgotten step. Without it, `get_benchmark()` raises `ValueError: Unknown benchmark '<name>'`.

Add one line to `BENCHMARK_REGISTRY`:

```python
BENCHMARK_REGISTRY: dict = {
    "human_eval": "benchmarks.human_eval.human_eval.HumanEvalBenchmark",
    "<name>": "benchmarks.<name>.<name>.<Name>Benchmark",   # ← add this
}
```

---

### [ ] Step 11 — Create `benchmarks/<name>/__init__.py`

```python
from benchmarks.<name>.<name> import <Name>Benchmark

__all__ = ["<Name>Benchmark"]
```

---

### [ ] Step 12 — Generate `benchmarks/<name>/README.md`

Cover:
- What this benchmark tests
- Dataset source and format
- Evaluation method
- Quick usage example (`benchmark.type: <name>` in config.yaml)
- Expected metrics output

---

### [ ] Step 13 — Smoke test

Verify end-to-end correctness on a small sample (use the count from interview Q6; default 3):

1. Instantiate the class and call `load_data()` — confirm tasks load.
2. Call `get_distortion_prompt()` on one task — confirm `DistortionPrompt` structure.
3. Call `evaluate()` with a sample response — confirm `EvalResult` is returned.
4. Call `calculate_metrics()` on dummy `EvalResult` list — confirm `BenchmarkMetrics` shape.

Report what passed and flag any failures before moving on.

---

### [ ] Step 14 — Scaffold unit tests

Create `tests/test_<name>.py`:

```python
import pytest
from benchmarks.<name>.<name> import <Name>Benchmark
from benchmarks.base.types import Task, EvalResult


@pytest.fixture
def benchmark():
    return <Name>Benchmark(config={})


def test_load_data(benchmark, tmp_path):
    # Write a minimal sample file, call load_data, assert Task list returned
    pass


def test_evaluate_correct(benchmark):
    task = Task(task_id="test/0", data={"prompt": "...", ...})
    result = benchmark.evaluate(task, "<correct response>")
    assert result.is_correct is True
    assert result.task_id == "test/0"


def test_evaluate_incorrect(benchmark):
    task = Task(task_id="test/1", data={"prompt": "...", ...})
    result = benchmark.evaluate(task, "<wrong response>")
    assert result.is_correct is False


def test_calculate_metrics(benchmark):
    results = [
        EvalResult(task_id="t0", is_correct=True, metadata={"miu": 0.0}),
        EvalResult(task_id="t1", is_correct=False, metadata={"miu": 0.5}),
    ]
    metrics = benchmark.calculate_metrics(results)
    assert 0.0 <= metrics.overall_score <= 1.0
    assert isinstance(metrics.per_mu, dict)
```

Fill in realistic task data from the actual dataset before committing.

---

## Done

Report to the user:
- Which steps completed successfully
- Any warnings that were triggered and how they were handled
- How to run the full pipeline: `python cli.py run --project <ProjectName>`
- How to run tests: `pytest tests/test_<name>.py -v`
