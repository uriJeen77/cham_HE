"""
Prompt templates and helpers for VQAv2 pipeline stages.
"""

from __future__ import annotations

from typing import Dict, List

VALIDATION_PROMPT_TEMPLATE = (
    "You are validating whether two questions ask for the same information "
    "about a single image.\n\n"
    "Original question:\n{original}\n\n"
    "Distorted question:\n{distorted}\n\n"
    "Answer with 'YES' if they are semantically equivalent, otherwise 'NO'. "
    "Optionally add a brief reason after the YES/NO."
)

VQA_SYSTEM_PROMPT = (
    "You are a visual question answering assistant. "
    "Given the image (referenced by path) and the question, provide a short answer. "
    "If the path is not accessible, still answer based on the described image context."
)


def build_validation_prompt(original: str, distorted: str) -> str:
    return VALIDATION_PROMPT_TEMPLATE.format(original=original, distorted=distorted)


def build_generation_messages(question: str, image_path: str) -> List[Dict[str, str]]:
    user_prompt = f"Image path: {image_path}\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": VQA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
