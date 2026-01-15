"""
Core schemas and data models for Chameleon.

Uses Pydantic for validation and serialization.
"""

from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    """Supported input modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class BackendType(str, Enum):
    """Supported LLM backend types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    MLX = "mlx"
    CUDA_LOCAL = "cuda_local"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    DUMMY = "dummy"


class DistortionEngineType(str, Enum):
    """Types of distortion engine backends."""
    API = "api"           # Use cloud API (OpenAI, Anthropic, Mistral API, etc.)
    LOCAL = "local"       # Use locally downloaded model
    OLLAMA = "ollama"     # Use Ollama for local inference
    VLLM = "vllm"         # Use vLLM for high-performance local inference
    HUGGINGFACE = "huggingface"  # Use HuggingFace transformers


class BenchmarkType(str, Enum):
    """Supported benchmark types."""
    HUMAN_EVAL = "human_eval"
    # Future benchmarks (e.g., BIG-bench, MBPP) can be added here


class DistortionEngineConfig(BaseModel):
    """
    Configuration for the distortion generation engine.
    
    Supports multiple backends:
    - API: Cloud-based (OpenAI, Anthropic, Mistral API, Google, etc.)
    - Local: Downloaded models (HuggingFace, custom paths)
    - Ollama: Local Ollama server
    - vLLM: High-performance local inference
    """
    
    # Engine type
    engine_type: DistortionEngineType = Field(default=DistortionEngineType.API)
    
    # Model identification
    vendor: str = Field(default="mistral")  # openai, anthropic, mistral, google, huggingface, local
    model_name: str = Field(default="mistral-large-latest")
    
    # For local models
    model_path: Optional[str] = Field(default=None)  # Path to downloaded model
    quantization: Optional[str] = Field(default=None)  # e.g., "4bit", "8bit", "awq", "gptq"
    
    # For API models
    api_key_env_var: Optional[str] = Field(default=None)  # e.g., "MISTRAL_API_KEY"
    api_base_url: Optional[str] = Field(default=None)  # Custom API endpoint
    
    # Performance settings
    max_workers: int = Field(default=4, ge=1, le=32)
    use_gpu: bool = Field(default=True)
    batch_size: int = Field(default=8, ge=1, le=64)
    max_memory_gb: Optional[float] = Field(default=None)  # GPU memory limit
    device: str = Field(default="auto")  # "auto", "cuda", "cuda:0", "cpu", "mps"
    
    # Generation settings
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    
    model_config = {'protected_namespaces': ()}

    @classmethod
    def from_preset(cls, preset: str) -> "DistortionEngineConfig":
        """Create config from a preset name."""
        presets = {
            "mistral-7b-local": cls(
                engine_type=DistortionEngineType.LOCAL,
                vendor="huggingface",
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                use_gpu=True,
                device="auto",
            ),
            "mistral-api": cls(
                engine_type=DistortionEngineType.API,
                vendor="mistral",
                model_name="mistral-small-latest",
                api_key_env_var="MISTRAL_API_KEY",
            ),
            "openai-gpt4": cls(
                engine_type=DistortionEngineType.API,
                vendor="openai",
                model_name="gpt-4o-mini",
                api_key_env_var="OPENAI_API_KEY",
            ),
            "anthropic-claude": cls(
                engine_type=DistortionEngineType.API,
                vendor="anthropic",
                model_name="claude-3-haiku-20240307",
                api_key_env_var="ANTHROPIC_API_KEY",
            ),
            "ollama-mistral": cls(
                engine_type=DistortionEngineType.OLLAMA,
                vendor="ollama",
                model_name="mistral",
                api_base_url="http://localhost:11434",
            ),
            "ollama-llama3": cls(
                engine_type=DistortionEngineType.OLLAMA,
                vendor="ollama",
                model_name="llama3",
                api_base_url="http://localhost:11434",
            ),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]


class DistortionConfig(BaseModel):
    """Configuration for distortion generation."""
    miu_values: List[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    distortions_per_question: int = Field(default=10, ge=1, le=100)
    
    # Distortion engine configuration
    engine: DistortionEngineConfig = Field(default_factory=DistortionEngineConfig)
    
    @field_validator('miu_values')
    @classmethod
    def validate_miu_values(cls, v):
        """Validate miu values are in [0, 1] range."""
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Miu value must be between 0.0 and 1.0: {val}")
        return sorted(list(set(v)))  # Remove duplicates and sort


class DataSchema(BaseModel):
    """Expected columns for data files."""
    original_data: Optional[List[str]] = Field(
        default=["question_id", "question", "answer_options", "correct_answer"]
    )
    distorted_data: Optional[List[str]] = Field(
        default=["question_id", "original_question", "distorted_question", 
                 "miu", "distortion_index", "answer_options", "correct_answer"]
    )
    results: Optional[List[str]] = Field(
        default=["question_id", "miu", "distorted_question", 
                 "model_answer", "is_correct"]
    )


class ProjectMetadata(BaseModel):
    """Project metadata."""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = Field(default="2.0")


class ProjectConfig(BaseModel):
    """Full project configuration stored in project_config.yaml."""
    project_name: str
    modality: Modality
    model_name: str
    backend_type: BackendType
    benchmark_type: BenchmarkType = Field(default=BenchmarkType.HUMAN_EVAL)
    description: Optional[str] = None
    
    # Paths (stored as strings for YAML serialization)
    project_dir: str
    original_data_dir: str
    distorted_data_dir: str
    results_dir: str
    analysis_dir: str
    
    # Distortion settings
    distortion_config: DistortionConfig = Field(default_factory=DistortionConfig)
    
    # Data schema
    data_schema: DataSchema = Field(default_factory=DataSchema)
    
    # Metadata
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    
    class Config:
        use_enum_values = True
        protected_namespaces = ()


class EvaluationRecord(BaseModel):
    """A single evaluation record."""
    question_id: str
    miu: float
    distortion_index: int
    original_question: str
    distorted_question: str
    answer_options: Optional[List[str]] = None
    correct_answer: str
    model_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    
    model_config = {'protected_namespaces': ()}


class BatchResult(BaseModel):
    """Results from a batch evaluation."""
    project_name: str
    model_name: str
    backend_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_questions: int
    completed: int
    failed: int
    records: List[EvaluationRecord]
    
    model_config = {'protected_namespaces': ()}
    
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        correct = sum(1 for r in self.records if r.is_correct)
        total = sum(1 for r in self.records if r.is_correct is not None)
        return correct / total if total > 0 else 0.0
    
    def accuracy_by_miu(self) -> Dict[float, float]:
        """Calculate accuracy grouped by miu level."""
        by_miu = {}
        for miu in set(r.miu for r in self.records):
            miu_records = [r for r in self.records if r.miu == miu and r.is_correct is not None]
            if miu_records:
                correct = sum(1 for r in miu_records if r.is_correct)
                by_miu[miu] = correct / len(miu_records)
        return by_miu
