"""
Default Configuration for HumanEval Benchmark
"""

from typing import Dict, List

# Model configurations
DEFAULT_MODELS = {
    "distortion": "mistral-large-latest",
    "validation": "mistral-large-latest",
    "generation": "mistral-large-latest",
}

# Pipeline configurations
DEFAULT_TIMEOUT = 3.0
DEFAULT_MIU = 0.6
DEFAULT_K_VALUES = [1, 5, 10]
DEFAULT_TEMPERATURE = 0.2

# Distortion configurations
MIU_LEVELS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
}

MIU_DESCRIPTIONS = {
    0.3: "Low distortion - minor word changes",
    0.6: "Medium distortion - significant rephrasing",
    0.9: "High distortion - complete paraphrase",
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
