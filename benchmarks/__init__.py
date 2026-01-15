def get_benchmark(name: str, config: dict):
    """
    Factory function to get a benchmark instance.
    Uses lazy imports to avoid circular dependencies.
    """
    if name == "human_eval":
        from .human_eval.human_eval import HumanEvalBenchmark
        return HumanEvalBenchmark(config)
    
    raise ValueError(f"Unknown benchmark: {name}")

