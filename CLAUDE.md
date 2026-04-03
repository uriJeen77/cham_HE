# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Chameleon** is an LLM Robustness Benchmark Framework that evaluates how well LLMs maintain functional correctness when coding problems are semantically distorted (paraphrased) at varying intensity levels (μ = 0.0 to 1.0). The core workflow: take HumanEval problems → generate semantic variants → run target LLM → execute code against unit tests → analyze degradation curve.

## Development Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional feature installs:
```bash
pip install -e ".[analysis]"      # matplotlib, seaborn, statsmodels, scikit-learn
pip install -e ".[openai]"        # OpenAI client
pip install -e ".[anthropic]"     # Anthropic client
pip install -e ".[dev]"           # pytest, black, ruff, mypy, pre-commit
```

Environment: copy `.env` with `MISTRAL_API_KEY` (used as distortion engine by default).

## CLI Commands

All user-facing commands go through `cli.py`:

```bash
python cli.py init                          # Interactive project wizard
python cli.py list                          # List existing projects
python cli.py distort --project MyProject   # Run distortion stage only
python cli.py generate --project MyProject  # Run LLM generation stage only
python cli.py evaluate --project MyProject  # Run evaluation stage only
python cli.py analyze --project MyProject   # Run analysis/visualization stage only
python cli.py run --project MyProject       # Run complete end-to-end pipeline
```

## Testing

```bash
pytest tests/ -v
pytest tests/test_analysis.py
pytest tests/test_core.py
```

Code style: `black` (100 char line length), `ruff` (E, W, F, I, B, C4, UP rules).

## Architecture

### Key Data Flow

```
Input JSONL (HumanEval format)
  → Distortion (chameleon/distortion/) — generate μ-scaled semantic variants using Mistral
  → Validation — verify distortions preserve underlying logic
  → Generation (chameleon/models/) — run target LLM on original + distorted prompts
  → Evaluation (chameleon/evaluation/) — execute code in sandbox against unit tests
  → Analysis (chameleon/analysis/) — compute metrics (Pass@k, CRI, Elasticity) + plots
```

### Core Modules

- **`cli.py` / `chameleon/cli/commands.py`** — Entry point and interactive CLI wizard
- **`chameleon/workflow.py`** — `ChameleonWorkflow` orchestrates the full pipeline; `WorkflowConfig` dataclass holds runtime config
- **`chameleon/core/config.py`** — `ChameleonConfig` global config; `chameleon/core/schemas.py` — Pydantic models and enums
- **`chameleon/distortion/`** — Distortion engine: `constants.py` defines μ-rules and temperature scaling, `runner.py` executes batch API calls, `engine.py` is the base abstraction
- **`chameleon/models/`** — Pluggable LLM backends via factory pattern in `registry.py`. Backends: `openai_backend.py`, `anthropic_backend.py`, `mlx_backend.py` (Apple Silicon), `cuda_backend.py`, `ollama_backend.py`, `dummy_backend.py`
- **`chameleon/evaluation/batch_processor.py`** — Benchmark-agnostic code execution runner
- **`chameleon/analysis/`** — Metrics, visualizations (matplotlib/seaborn), and reports; `mcnemar.py` for statistical tests
- **`benchmarks/human_eval/`** — HumanEval-specific pipeline with its own `runner.py`, `engine/` (sandbox execution + Pass@k), and `models.py` data classes

### Project Workspace Layout

Runtime projects are created under `Projects/<name>/`:
```
Projects/MyProject/
├── original_data/    # Input JSONL
├── distorted_data/   # Generated distortions
├── results/          # Execution logs & metrics
├── analysis/         # Visualizations & reports
└── config.yaml       # Project configuration (model, μ values, distortion settings)
```

### Adding a New LLM Backend

1. Subclass `chameleon/models/base.py` `ModelBackend`
2. Register in `chameleon/models/registry.py`

### μ (miu) Parameter

Controls distortion intensity: 0.0 = original problem unchanged, 1.0 = fully paraphrased. Rules for how each μ level transforms prompts are defined in `chameleon/distortion/constants.py`.

## Docker

```bash
docker build -t chameleon .
docker run -it --rm \
  -v $(pwd)/Projects:/app/Projects \
  -e MISTRAL_API_KEY=$MISTRAL_API_KEY \
  chameleon python cli.py --help
```
