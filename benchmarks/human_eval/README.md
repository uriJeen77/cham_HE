# HumanEval Modular Architecture

## Overview

This directory contains a **modular, refactored** implementation of the HumanEval robustness evaluation pipeline. The new architecture replaces the monolithic `HumanEvalBenchmark.run_full_pipeline()` method with composable, testable pipeline steps.

## Architecture

### Core Components

```
chameleon/benchmarks/human_eval/
├── models.py           # Data models (HumanEvalTask, PipelineConfig, PipelineResult)
├── config.py           # Configuration constants
├── runner.py           # Main pipeline orchestrator
├── steps/              # Modular pipeline steps
│   ├── base.py         # Abstract base class for all steps
│   ├── load.py         # Step 1: Load data
│   ├── distortion.py   # Step 2: Distort prompts
│   ├── validation.py   # Step 3: Validate semantic equivalence
│   ├── generation.py   # Step 4: Generate code
│   ├── evaluation.py   # Step 5: Evaluate and calculate metrics
│   └── factory.py      # Factory for creating steps
└── utils/              # Utility functions
```

### Pipeline Steps

Each step follows a consistent interface:

```python
class BaseStep(ABC):
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        ...
    
    def execute(self, tasks: List[HumanEvalTask]) -> List[HumanEvalTask]:
        """Process tasks and return updated list"""
        ...
```

**The 5 Pipeline Steps:**

1. **LoadStep**: Load HumanEval tasks from JSONL file
2. **DistortionStep**: Create semantically distorted versions of prompts
3. **ValidationStep**: Validate that distortions preserve logical meaning
4. **CodeGenerationStep**: Generate Python code using LLM (for both original and distorted)
5. **EvaluationStep**: Run unit tests and calculate pass@k metrics

## Benefits of Modular Architecture

✅ **Testability**: Each step can be tested in isolation  
✅ **Reusability**: Steps can be used in different pipelines  
✅ **Flexibility**: Easy to skip, reorder, or replace steps  
✅ **Maintainability**: Clear separation of concerns  
✅ **Extensibility**: Simple to add new steps  

## Usage

### Quick Start (Convenience Function)

```python
from chameleon.benchmarks.human_eval.runner import run_humaneval_pipeline

result = run_humaneval_pipeline(
    data_path="data/humaneval.jsonl",
    output_dir="results/test",
    miu=0.6,
    limit=3
)

print(f"Original pass@1: {result.original_metrics['pass@1']:.2%}")
print(f"Distorted pass@1: {result.distorted_metrics['pass@1']:.2%}")
```

### Custom Configuration

```python
from chameleon.benchmarks.human_eval.runner import ModularHumanEvalRunner
from chameleon.benchmarks.human_eval.models import PipelineConfig

# Create custom config
config = PipelineConfig(
    distortion_model="mistral-large-latest",
    generation_model="mistral-large-latest",
    miu=0.9,  # High distortion
    timeout=5.0,
    k_values=[1, 5, 10]
)

# Run pipeline
runner = ModularHumanEvalRunner(config)
result = runner.run_full_pipeline(
    data_path="data/humaneval.jsonl",
    output_dir="results/high_distortion",
    limit=10
)
```

### Custom Pipeline (Specific Steps Only)

```python
runner = ModularHumanEvalRunner(config)

# Run only distortion and validation
tasks = runner.run_custom_pipeline(
    step_names=["load", "distortion", "validation"],
    data_path="data/humaneval.jsonl",
    limit=5
)

# Check validation results
for task in tasks:
    print(f"{task.task_id}: {task.validation_passed}")
```

### Manual Step-by-Step Execution

```python
from chameleon.benchmarks.human_eval.steps import (
    LoadStep, DistortionStep, ValidationStep
)

# Create steps
load_step = LoadStep(config, logger)
distortion_step = DistortionStep(config, logger)
validation_step = ValidationStep(config, logger)

# Execute sequentially
tasks = load_step.load_from_path("data/humaneval.jsonl", limit=3)
tasks = distortion_step.execute(tasks)
tasks = validation_step.execute(tasks)

# Process results
for task in tasks:
    if task.validation_passed:
        print(f"✓ {task.task_id}")
```

## Data Models

### HumanEvalTask

Type-safe dataclass representing a single task:

```python
@dataclass
class HumanEvalTask:
    # Core fields
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str
    
    # Populated progressively by steps
    distorted_prompt: Optional[str] = None
    distortion_success: bool = False
    validation_passed: Optional[bool] = None
    original_code: Optional[str] = None
    distorted_code: Optional[str] = None
    original_syntax_valid: bool = False
    distorted_syntax_valid: bool = False
```

### PipelineConfig

Configuration for the pipeline:

```python
@dataclass
class PipelineConfig:
    distortion_model: str = "mistral-large-latest"
    validation_model: str = "mistral-large-latest"
    generation_model: str = "mistral-large-latest"
    timeout: float = 3.0
    k_values: List[int] = [1]
    miu: float = 0.6
    project_path: Optional[str] = None
```

### PipelineResult

Summary statistics:

```python
@dataclass
class PipelineResult:
    total_tasks: int
    distortion_successful: int
    validation_passed: int
    original_code_generated: int
    distorted_code_generated: int
    original_syntax_valid: int
    distorted_syntax_valid: int
    original_metrics: Dict[str, float]
    distorted_metrics: Dict[str, float]
    miu: float
```

## Output Structure

After running the pipeline, results are saved to the output directory:

```
results/
├── summary.json              # PipelineResult as JSON
└── tasks_complete.jsonl      # Full task data with all fields
```

## Examples

See `example_modular_pipeline.py` for comprehensive examples:

- **Example 1**: Simple run (3 samples, default settings)
- **Example 2**: Custom configuration (high distortion)
- **Example 3**: Partial pipeline (load + distort only)
- **Example 4**: Manual step-by-step execution

## Migration Guide

### Old API (Monolithic)

```python
benchmark = HumanEvalBenchmark(config)
summary = benchmark.run_full_pipeline(
    data_path="data.jsonl",
    output_dir="results",
    miu=0.6,
    limit=3
)
```

### New API (Modular)

```python
result = run_humaneval_pipeline(
    data_path="data.jsonl",
    output_dir="results",
    miu=0.6,
    limit=3
)
```

**Benefits**: Same interface, cleaner internals!

## Testing

Each step can be tested independently:

```python
def test_distortion_step():
    config = PipelineConfig(miu=0.6)
    logger = logging.getLogger("test")
    
    step = DistortionStep(config, logger)
    
    # Create test task
    task = HumanEvalTask(
        task_id="test/0",
        prompt="def add(a, b): ...",
        canonical_solution="return a + b",
        test="assert add(1,2)==3",
        entry_point="add"
    )
    
    # Execute
    result = step.execute([task])
    
    # Verify
    assert result[0].distortion_success
    assert result[0].distorted_prompt is not None
```

## Future Enhancements

Potential extensions to the modular architecture:

- [ ] **Caching Steps**: Cache intermediate results to disk
- [ ] **Parallel Execution**: Run steps in parallel when possible
- [ ] **Progress Bars**: Add tqdm progress bars to each step
- [ ] **Retry Logic**: Automatic retries for failed API calls
- [ ] **Alternative Steps**: Multiple implementations of the same step
- [ ] **Conditional Steps**: Skip steps based on previous results
- [ ] **Metrics Dashboard**: Real-time visualization of pipeline progress

## Contributing

When adding new steps:

1. Inherit from `BaseStep`
2. Implement `execute()`, `name`, and `description`
3. Add to `factory.py`
4. Export from `steps/__init__.py`
5. Update this README

## License

MIT License - see project root LICENSE for details.
