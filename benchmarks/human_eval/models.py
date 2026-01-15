"""
Data Models for HumanEval Benchmark

Type-safe dataclasses that replace dictionary-based data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class HumanEvalTask:
    """
    Represents a single HumanEval task with all associated data.
    
    Fields are populated progressively as the task moves through the pipeline:
    1. Initial load: task_id, prompt, canonical_solution, test, entry_point
    2. After distortion: distorted_prompt, distortion_success, distortion_miu
    3. After validation: validation_passed, validation_reason, equivalence_level
    4. After code generation: original_code, distorted_code
    5. After syntax validation: original_syntax_valid, distorted_syntax_valid
    """
    # Core fields (loaded from data)
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str
    
    # Distortion fields (Step 2)
    distorted_prompt: Optional[str] = None
    distortion_success: bool = False
    distortion_miu: Optional[float] = None
    
    # Validation fields (Step 3)
    validation_passed: Optional[bool] = None
    validation_reason: str = ""
    equivalence_level: str = ""
    
    # Code generation fields (Step 4)
    original_code: Optional[str] = None
    distorted_code: Optional[str] = None
    
    # Syntax validation fields (Step 5)
    original_syntax_valid: bool = False
    distorted_syntax_valid: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id, #step 1 - load data
            "prompt": self.prompt, #step 1 - load data
            "canonical_solution": self.canonical_solution, #step 1 - load data
            "test": self.test, #step 1 - load data
            "entry_point": self.entry_point, #step 1 - load data
            "distorted_prompt": self.distorted_prompt, #step 2 - distortion
            "distortion_success": self.distortion_success, #step 2 - distortion
            "distortion_miu": self.distortion_miu, #step 2 - distortion
            "validation_passed": self.validation_passed, #step 3 - validation
            "validation_reason": self.validation_reason, #step 3 - validation
            "equivalence_level": self.equivalence_level, #step 3 - validation
            "original_code": self.original_code, #step 4 - code generation
            "distorted_code": self.distorted_code, #step 4 - code generation
            "original_syntax_valid": self.original_syntax_valid, #step 5 - syntax validation
            "distorted_syntax_valid": self.distorted_syntax_valid, #step 5 - syntax validation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HumanEvalTask":
        """Create from dictionary (for loading from JSON)."""
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            canonical_solution=data.get("canonical_solution", ""),
            test=data.get("test", ""),
            entry_point=data.get("entry_point", ""),
            distorted_prompt=data.get("distorted_prompt"),
            distortion_success=data.get("distortion_success", False),
            distortion_miu=data.get("distortion_miu"),
            validation_passed=data.get("validation_passed"),
            validation_reason=data.get("validation_reason", ""),
            equivalence_level=data.get("equivalence_level", ""),
            original_code=data.get("original_code"),
            distorted_code=data.get("distorted_code"),
            original_syntax_valid=data.get("original_syntax_valid", False),
            distorted_syntax_valid=data.get("distorted_syntax_valid", False),
        )


@dataclass
class PipelineConfig:
    """
    Configuration for the HumanEval evaluation pipeline.
    """
    # Model configurations
    distortion_model: str = "mistral-large-latest"
    validation_model: str = "mistral-large-latest"
    generation_model: str = "mistral-large-latest"
    
    # Evaluation configurations
    timeout: float = 3.0
    k_values: List[int] = field(default_factory=lambda: [1])
    
    # Distortion configuration
    miu: float = 0.6
    
    # Project path (for distortion engine)
    project_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "distortion_model": self.distortion_model,
            "validation_model": self.validation_model,
            "generation_model": self.generation_model,
            "timeout": self.timeout,
            "k_values": self.k_values,
            "miu": self.miu,
            "project_path": str(self.project_path) if self.project_path else None,
        }


@dataclass
class PipelineResult:
    """
    Results from running the complete pipeline.
    """
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_tasks": self.total_tasks,
            "distortion_successful": self.distortion_successful,
            "validation_passed": self.validation_passed,
            "original_code_generated": self.original_code_generated,
            "distorted_code_generated": self.distorted_code_generated,
            "original_syntax_valid": self.original_syntax_valid,
            "distorted_syntax_valid": self.distorted_syntax_valid,
            "original_metrics": self.original_metrics,
            "distorted_metrics": self.distorted_metrics,
            "miu": self.miu,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
HumanEval Pipeline Results:
  Total tasks: {self.total_tasks}
  Distortions successful: {self.distortion_successful}
  Validations passed: {self.validation_passed}
  Original pass@1: {self.original_metrics.get('pass@1', 0):.2%}
  Distorted pass@1: {self.distorted_metrics.get('pass@1', 0):.2%}
"""
