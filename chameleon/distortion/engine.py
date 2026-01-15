"""
Distortion Engine (Backend Module).

PURPOSE:
This module provides the architecture for executing distortions, specifically designed for:
1. Local offline execution (HuggingFace/GPU)
2. Local server execution (Ollama)
3. Modular API interfaces

NOTE:
Currently, the main 'runner.py' operates independently (Standalone) using its own 
optimized Batch API implementation and does NOT import this file.

This file is preserved for:
- Future support of local/offline workflows
- Reference for modular engine architecture
"""

import os
import time
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch  # Local engines rely on torch; imported once at module level

from chameleon.distortion.constants import (
    DISTORTION_SYSTEM_PROMPT,
    get_distortion_prompt,
    get_batch_distortion_prompt,  # Kept for future batch support, even if unused today
)


@dataclass
class DistortionResult:
    """Result of a single distortion operation."""
    question_id: str
    original_question: str
    distorted_question: str
    miu: float
    distortion_index: int
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class BaseDistortionEngine(ABC):
    """
    Abstract base class for distortion engines.

    This class defines the minimal interface all engines must implement:
    - generate(): single-text generation primitive
    - is_available(): runtime capability probe / health check
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize engine with configuration.

        Args:
            config: Engine configuration dict. Common fields:
                - model_name: str
                - temperature: float
                - max_tokens: int
        """
        self.config: Dict[str, Any] = config
        self.model_name: str = config.get("model_name", "unknown")
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 512)

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        This is the primitive operation that all engines implement.
        Higher-level helpers (like distort_question) build on top of this.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the engine is available and ready.

        This should be a *cheap* capability probe (e.g., dependency check,
        pinging a local server, or verifying an API key exists).
        """
        raise NotImplementedError

    def _extract_single_distortion(self, raw_output: str) -> str:
        """
        Extract a single distorted prompt from the model output.

        The distortion prompts typically ask the model to return a numbered list
        (e.g. "1. ...", "2. ..."). For HumanEval-style usage we usually request
        exactly one distortion, so we:
        - Take the first non-empty line.
        - Strip common list markers like "1.", "1)", "- ", "* ".
        """
        if not raw_output:
            return raw_output

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return raw_output.strip()

        first = lines[0]
        # Remove common leading list markers: "1.", "1)", "- ", "* ", etc.
        first = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*]\s+)", "", first)
        return first.strip()

    def distort_question(
        self,
        question_id: str,
        question: str,
        miu: float,
        distortion_index: int = 0,
        custom_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        answer_options: Optional[str] = None,
        correct_answer: Optional[str] = None,
    ) -> DistortionResult:
        """
        Generate a single distorted version of a question / HumanEval prompt.

        Args:
            question_id: Unique identifier for the question
            question: Original prompt text (e.g., HumanEval description+code)
            miu: Distortion intensity (0.0 to 1.0)
            distortion_index: Index for multiple distortions of same question
            answer_options: Deprecated / unused (kept for backward compatibility)
            correct_answer: Deprecated / unused (kept for backward compatibility)

        Returns:
            DistortionResult with the distorted text
        """
        # Handle miu = 0 (no distortion: identity mapping)
        if miu == 0.0:
            return DistortionResult(
                question_id=question_id,
                original_question=question,
                distorted_question=question,
                miu=miu,
                distortion_index=distortion_index,
                latency_ms=0.0,
                success=True,
            )

        if custom_prompt:
            prompt = custom_prompt
        else:
            # Fallback for backward compatibility or default behavior
            # Now we use the version from constants (which we will keep for simple cases)
            from chameleon.distortion.constants import get_distortion_prompt
            prompt = get_distortion_prompt(
                question=question,
                miu=miu,
                n_distortions=1,
            )

        # === LIVE BLOCK START: Local/remote model generation ===
        try:
            start_time = time.time()
            distorted = self.generate(prompt)
            latency_ms = (time.time() - start_time) * 1000.0

            # Clean up the response and extract the single distortion line
            
            

            return DistortionResult(
                question_id=question_id,
                original_question=question,
                distorted_question=distorted,
                miu=miu,
                distortion_index=distortion_index,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as e:
            # Preserve legacy behavior: fall back to original question on failure
            return DistortionResult(
                question_id=question_id,
                original_question=question,
                distorted_question=question,
                miu=miu,
                distortion_index=distortion_index,
                success=False,
                error=str(e),
            )
        # === LIVE BLOCK END ===


class LocalHuggingFaceEngine(BaseDistortionEngine):
    """
    Distortion engine using local HuggingFace models.

    Optimized for GPU usage with quantization support.

    This engine is intended for fully offline/distconnected operation where
    a model is hosted on the local machine.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device: str = config.get("device", "auto")
        self.quantization: Optional[str] = config.get("quantization", "4bit")
        self.use_gpu: bool = config.get("use_gpu", True)
        self._loaded: bool = False

    def _load_model(self) -> None:
        """
        Lazily load the model and tokenizer.

        This is intentionally side-effectful and may allocate significant GPU/CPU
        memory. It is marked as a LIVE BLOCK to make it easy to spot for dry-run
        or unit-test scenarios.
        """
        if self._loaded:
            return

        # === LIVE BLOCK START: Local HF model & tokenizer loading ===
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"\n   🔄 Loading model: {self.model_name}")
            print(f"   Quantization: {self.quantization or 'none'}")
            print(f"   Device: {self.device}")

            # Configure quantization (only works with CUDA)
            quantization_config = None
            if self.quantization and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig

                    if self.quantization == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    elif self.quantization == "8bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                except ImportError:
                    print("   ⚠️ bitsandbytes not available, loading without quantization")
                    quantization_config = None
            elif self.quantization and not torch.cuda.is_available():
                print("   ⚠️ Quantization requires CUDA, loading without quantization (will use more RAM)")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load model with quantization
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "auto" else self.device,
                    trust_remote_code=True,
                    attn_implementation="eager",  # Avoid flash attention issues
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.use_gpu else torch.float32,
                    device_map="auto" if self.device == "auto" else self.device,
                    trust_remote_code=True,
                    attn_implementation="eager",  # Avoid flash attention issues
                    low_cpu_mem_usage=True,  # Better memory management
                )

            self._loaded = True
            print("   ✓ Model loaded successfully")

        except ImportError as e:
            self._loaded = False
            raise ImportError(
                "Required packages not installed. Run: "
                "pip install transformers torch bitsandbytes accelerate. "
                f"Error: {e}"
            )
        except Exception as e:
            self._loaded = False
            raise RuntimeError(f"Failed to load model: {e}")
        # === LIVE BLOCK END ===

    def generate(self, prompt: str) -> str:
        """
        Generate text using the local HuggingFace model.

        This is the single-text primitive used by BaseDistortionEngine.
        """
        self._load_model()

        # Format as chat message for instruct models
        messages = [
            {"role": "system", "content": DISTORTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = f"{DISTORTION_SYSTEM_PROMPT}\n\n{prompt}"

        # === LIVE BLOCK START: Local HF generation ===
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Generate with cache disabled to avoid DynamicCache issues
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,  # Greedy for speed and consistency
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to avoid DynamicCache issues
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # === LIVE BLOCK END ===

        return response

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for multiple prompts in a batch.

        This is a convenience helper for higher throughput; semantics are the
        same as calling generate() per prompt, but implemented as a single
        HF batch for efficiency.
        """
        self._load_model()

        # Format all prompts
        formatted_prompts: List[str] = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": DISTORTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = f"{DISTORTION_SYSTEM_PROMPT}\n\n{prompt}"
            formatted_prompts.append(formatted)

        # === LIVE BLOCK START: Local HF batch generation ===
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_lengths = [len(self.tokenizer.encode(p)) for p in formatted_prompts]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )

        responses: List[str] = []
        for output, input_len in zip(outputs, input_lengths):
            new_tokens = output[input_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(response)
        # === LIVE BLOCK END ===

        return responses

    def is_available(self) -> bool:
        """
        Check if required local packages and (optionally) GPU are available.

        This does NOT attempt to load the model; it only checks dependencies and
        hardware capabilities.
        """
        try:
            import transformers  # noqa: F401

            # Check GPU availability
            if self.use_gpu:
                if torch.cuda.is_available():
                    return True
                print("   ⚠️ CUDA not available. Will use CPU (slower).")
                self.use_gpu = False
                self.device = "cpu"
            return True

        except ImportError as e:
            print(f"   ❌ Missing package: {e}")
            print("   Install with: pip install torch transformers accelerate")
            return False


class APIDistortionEngine(BaseDistortionEngine):
    """
    Distortion engine using cloud APIs (OpenAI, Anthropic, Mistral or compatible).

    This engine is intentionally generic and only requires an OpenAI-style
    chat completion or vendor-specific client.
    """

    def __init__(self, config: Dict[str, Any], project_path: Optional[Path] = None):
        super().__init__(config)
        self.vendor: str = config.get("vendor", "openai")
        self.api_key_env_var: str = config.get("api_key_env_var", f"{self.vendor.upper()}_API_KEY")
        self.api_base_url: Optional[str] = config.get("api_base_url")
        self.project_path: Optional[Path] = project_path
        self._client: Any = None

        # Load env file on init (non-breaking side effect, same behavior as before)
        self._load_env_file()

    def _load_env_file(self) -> None:
        """Load API key from project's .env file and optional root .env."""
        # Try project-specific .env first
        if self.project_path:
            env_file = Path(self.project_path) / ".env"
            if env_file.exists():
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, value = line.split("=", 1)
                                key = key.strip()
                                value = value.strip()
                                if value:  # Only set if value is not empty
                                    os.environ[key] = value
                    print(f"   ✓ Loaded environment from {env_file}")
                except Exception as e:
                    print(f"   ⚠️ Could not load {env_file}: {e}")

        # Also try root .env
        root_env = Path(".env")
        if root_env.exists():
            try:
                with open(root_env, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            # Do not override already-set vars
                            if value and not os.getenv(key):
                                os.environ[key] = value
            except Exception:
                # Silent failure preserves previous behavior
                pass

    def _get_client(self) -> Any:
        """
        Get or create the API client.

        This is a LIVE BLOCK because it instantiates SDK clients that will be
        used for real network traffic to external providers.
        """
        if self._client:
            return self._client

        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                "API key not found. Set "
                f"{self.api_key_env_var} in:\n"
                f"  - Project .env file: {self.project_path / '.env' if self.project_path else 'N/A'}\n"
                f"  - Or system environment variable"
            )

        # === LIVE BLOCK START: External API client construction ===
        if self.vendor == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=self.api_base_url)
        elif self.vendor == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)
        elif self.vendor == "mistral":
            # Use new Mistral SDK
            from mistralai import Mistral

            self._client = Mistral(api_key=api_key)
        else:
            # Use OpenAI-compatible API
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=self.api_base_url)
        # === LIVE BLOCK END ===

        return self._client

    def generate(self, prompt: str) -> str:
        """
        Generate text using the configured cloud API.

        The exact call path depends on `self.vendor`, but the contract is the same:
        return a single string response.
        """
        client = self._get_client()

        # === LIVE BLOCK START: Network calls to vendor APIs ===
        if self.vendor == "anthropic":
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=DISTORTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        if self.vendor == "mistral":
            # New Mistral SDK format
            response = client.chat.complete(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": DISTORTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        # OpenAI-compatible API path (OpenAI, custom, etc.)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": DISTORTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
        # === LIVE BLOCK END ===

    def is_available(self) -> bool:
        """
        Check if API key is available.

        This does not perform any network I/O, it only validates configuration.
        """
        return bool(os.getenv(self.api_key_env_var))


class OllamaDistortionEngine(BaseDistortionEngine):
    """
    Distortion engine using local Ollama server.

    This engine assumes an Ollama instance running on the given base URL
    (typically http://localhost:11434).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base_url: str = config.get("api_base_url", "http://localhost:11434")

    def generate(self, prompt: str) -> str:
        """
        Generate text using a local Ollama server.

        This is a LIVE BLOCK that issues HTTP requests to the local Ollama API.
        """
        import requests

        # === LIVE BLOCK START: HTTP call to local Ollama ===
        response = requests.post(
            f"{self.api_base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": f"{DISTORTION_SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
        )
        response.raise_for_status()
        return response.json()["response"]
        # === LIVE BLOCK END ===

    def is_available(self) -> bool:
        """
        Check if Ollama server is running.

        This performs a lightweight GET /api/tags probe.
        """
        import requests

        # === LIVE BLOCK START: Local Ollama health check ===
        try:
            response = requests.get(f"{self.api_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
        # === LIVE BLOCK END ===


class DistortionEngine:
    """
    Factory class for creating distortion engines.

    Automatically selects the appropriate backend based on configuration.
    """

    @staticmethod
    def create(config: Dict[str, Any], project_path: Optional[Path] = None) -> BaseDistortionEngine:
        """
        Create a distortion engine based on configuration.

        Args:
            config: Engine configuration with 'engine_type' key
            project_path: Path to project directory (for loading .env)

        Returns:
            Appropriate distortion engine instance
        """
        engine_type = config.get("engine_type", "local")

        if engine_type == "local":
            return LocalHuggingFaceEngine(config)
        if engine_type == "api":
            return APIDistortionEngine(config, project_path=project_path)
        if engine_type == "ollama":
            return OllamaDistortionEngine(config)
        raise ValueError(f"Unknown engine type: {engine_type}")

    @staticmethod
    def from_project_config(
        project_config: Dict[str, Any],
        project_path: Optional[Path] = None,
    ) -> BaseDistortionEngine:
        """
        Create engine from project configuration.

        Args:
            project_config: Full project config dict
            project_path: Path to project directory

        Returns:
            Configured distortion engine
        """
        distortion_config = project_config.get("distortion_config", project_config.get("distortion", {}))
        engine_config = distortion_config.get("engine", {})

        return DistortionEngine.create(engine_config, project_path=project_path)
