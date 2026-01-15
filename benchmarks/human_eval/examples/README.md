# Chameleon Examples & Test Scripts

This directory contains example scripts and test utilities for the Chameleon framework.

## 📁 Directory Structure

### Modular Architecture Examples (NEW ⭐)

**`modular/`** - Examples using the new modular HumanEval pipeline

- **`example_modular_pipeline.py`** - 4 comprehensive usage examples:
  1. Simple run with default settings
  2. Custom configuration (high distortion)
  3. Partial pipeline (specific steps only)
  4. Manual step-by-step execution
  
- **`test_modular_pipeline.py`** - Test script for the modular pipeline
  - Runs on 3 samples
  - Validates all components
  - Generates detailed report

### Legacy Examples

**`legacy/`** - Examples using the original monolithic architecture

- **`custom_pipeline.py`** - Custom pipeline implementation
- **`example_run_humaneval.py`** - Basic HumanEval example
- **`run_full_pipeline_method.py`** - Full pipeline demonstration
- **`test_humaneval_3_samples.py`** - Legacy test script
- **`test_architecture.py`** - Architecture test
- **`toy_workflow_demo.py`** - Workflow demonstration

## 🚀 Quick Start

### Run Modular Pipeline Test
```bash
cd examples/modular
python test_modular_pipeline.py
```

### Try Modular Examples
```bash
cd examples/modular
python example_modular_pipeline.py
```

## 📚 Learn More

- **Modular Architecture**: See `chameleon/benchmarks/human_eval/README.md`
- **Main Documentation**: See project root `README.md`

## 🔄 Migration Guide

If you're using legacy examples, we recommend migrating to the modular architecture:

**Old way:**
```python
benchmark = HumanEvalBenchmark(config)
benchmark.run_full_pipeline(...)
```

**New way:**
```python
from chameleon.benchmarks.human_eval.runner import run_humaneval_pipeline
result = run_humaneval_pipeline(data_path, output_dir, miu=0.6)
```

Benefits:
- ✅ Cleaner, more maintainable code
- ✅ Better testability
- ✅ More flexible (run partial pipelines)
- ✅ Rich logging and progress tracking
