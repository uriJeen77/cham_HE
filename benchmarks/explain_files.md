# Benchmark Directory Structure Guide

Each benchmark under `benchmarks/` follows this layout. The description below uses `human_eval` as the reference — the same pattern applies to `omni_math`, `vaq`, and `hebrew`.

---

## Top-Level Files

| File | Purpose |
|------|---------|
| `__init__.py` | Python package marker — makes the directory importable |
| `README.md` | Human-facing documentation for this specific benchmark |
| `config.py` | **Benchmark-level constants**: default model names (`mistral-large-latest`), timeout, μ levels (`low/medium/high`), k values for pass@k, log format. Not runtime config — just named defaults the benchmark class reads. |
| `<benchmark>.py` | **The main class** — subclasses `BaseBenchmark` and implements all 6 pipeline steps: `load_data` → `distort_prompts` → `validate_distortions` → `generate_code` → `validate_syntax` → `evaluate`. Also exposes a `run_full_pipeline()` convenience method. This is the entry point for the benchmark. |
| `models.py` | **Typed dataclasses** for the benchmark's domain objects: the task struct (fields added progressively through each pipeline step), a `PipelineConfig` dataclass, and a `PipelineResult` summary dataclass. Replaces loose dict-based passing with type-safe objects. |
| `prompts.py` | **All LLM prompts** in one place — system prompts and user prompt builder functions for: distortion (single + batch + retry), validation (MegaJudge), and code generation. Centralised here so prompt tuning doesn't require touching the main class. |

---

## `data/`

Stores the benchmark dataset file (e.g., `HumanEval.jsonl.gz`). The `.gitkeep` placeholder in new benchmarks means: the directory exists but the actual dataset file is not committed — it gets downloaded or placed manually.

---

## `engine/`

The execution core — benchmark-agnostic enough to be reusable:

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `data.py` | Reads the dataset file (handles `.gz` and plain JSONL), returns a dict of tasks keyed by `task_id` |
| `execution.py` | **Sandboxed code runner** — executes generated Python code against unit tests in an isolated subprocess with a timeout. Returns `{passed: bool, result: str}`. |
| `evaluate_functional_correctness.py` | **Pass@k computation** — implements the unbiased estimator formula from the HumanEval paper: `1 - prod(1 - k/range(n-c+1, n+1))`. Computes correctness across multiple samples per task. |
| `evaluation.py` | **Orchestrator** — ties `data.py` + `execution.py` + `evaluate_functional_correctness.py` together; iterates tasks, dispatches to execution, aggregates metrics. |

---

## `examples/`

Usage demonstrations — not part of the pipeline itself:

| File | Purpose |
|------|---------|
| `README.md` | Explains how to run the examples |
| `test_consolidated.py` | A single runnable script that exercises the full pipeline end-to-end on a small subset of tasks |
| `legacy/custom_pipeline.py` | Early prototype of a custom pipeline wiring |
| `legacy/example_run_humaneval.py` | Simple standalone run script from before the CLI existed |
| `legacy/run_full_pipeline_method.py` | Demonstrates calling `run_full_pipeline()` directly |
| `legacy/test_humaneval_3_samples.py` | Quick smoke test on 3 tasks — useful for sanity checks |

---

## `test_consolidated/`

| File | Purpose |
|------|---------|
| `summary.json` | Output artifact from a pipeline run — contains counts (tasks loaded, distortions succeeded, validations passed) and final metrics (`pass@1` for original vs distorted). Written by `run_full_pipeline()`. |

---

## The Pattern for New Benchmarks

When implementing `omni_math.py`, `vaq.py`, or `hebrew.py`, the work is:

1. **`<benchmark>.py`** — create a subclass of `BaseBenchmark`, implement `load_data`, `get_field_to_distort`, `get_distortion_prompt`, `validate_distortion`, `get_generation_prompt`, `evaluate`, `calculate_metrics`
2. **`models.py`** — define the task dataclass with the fields that accumulate through your pipeline stages
3. **`prompts.py`** — write the system/user prompts specific to your benchmark's format and distortion constraints
4. **`config.py`** — set the defaults for your benchmark (timeout, models, μ levels)
5. **`engine/`** — adapt or reuse `execution.py`; `data.py` needs to know your dataset format; `evaluation.py` orchestrates them
