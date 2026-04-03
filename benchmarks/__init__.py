"""
Benchmark registry for Chameleon.

To add a new benchmark:
  1. Implement AbstractBenchmark in benchmarks/<name>/<name>.py
  2. Add an entry to BENCHMARK_REGISTRY below

The registry maps benchmark names (as used in config.yaml) to dotted
import paths so new benchmarks can be added without editing framework code.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chameleon.core.base import AbstractBenchmark

BENCHMARK_REGISTRY: dict = {
    "human_eval": "benchmarks.human_eval.human_eval.HumanEvalBenchmark",
    # "mmlu": "benchmarks.mmlu.mmlu.MMLUBenchmark",
    # "gsm8k": "benchmarks.gsm8k.gsm8k.GSM8KBenchmark",
}


def get_benchmark(name: str, config: dict) -> "AbstractBenchmark":
    """
    Instantiate a benchmark by name.

    Args:
        name:   Registry key, e.g. "human_eval"
        config: Configuration dict passed to the benchmark constructor

    Returns:
        An AbstractBenchmark instance

    Raises:
        ValueError: if name is not in the registry
    """
    if name not in BENCHMARK_REGISTRY:
        registered = list(BENCHMARK_REGISTRY)
        raise ValueError(
            f"Unknown benchmark '{name}'. Registered benchmarks: {registered}"
        )

    dotted_path = BENCHMARK_REGISTRY[name]
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)
