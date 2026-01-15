from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks in Chameleon.
    
    This interface defines the required methods for loading data,
    formatting prompts, and evaluating results across different benchmarks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark with its specific configuration.
        """
        self.config = config

    @abstractmethod
    def load_tasks(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load tasks from a given data path.
        Returns a list of task dictionaries.
        """
        pass

    @abstractmethod
    def format_prompt(self, task: Dict[str, Any]) -> str:
        """
        Given a task dictionary, return the formatted prompt for the LLM.
        """
        pass

    @abstractmethod
    def evaluate_completion(self, task: Dict[str, Any], completion: str) -> Dict[str, Any]:
        """
        Evaluate a single completion for a specific task.
        Returns a dictionary with result metrics (e.g., {'passed': True, 'result': '...'}).
        """
        pass

    @abstractmethod
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Given a list of results from evaluate_completion, calculate aggregate metrics
        (e.g., pass@k, accuracy, etc.).
        """
        pass
