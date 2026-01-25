"""
VQAv2 dataset loader.

Purpose:
- Load the optimized VQAv2 JSON (list of entries).
- Validate required fields for distortion.
- Provide helpers to prepare inputs for distortion pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REQUIRED_FIELDS = ("question_id", "question")


def load_entries(data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load VQAv2 entries from a JSON file.

    The expected format is a top-level list of objects with fields like:
    question_id, question, image_id, answers, image, etc.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"VQAv2 data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("VQAv2 data must be a JSON list of entries.")

    if limit is not None:
        data = data[:limit]

    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry at index {idx} is not an object.")
        missing = [
            field for field in REQUIRED_FIELDS
            if field not in item or item[field] in (None, "")
        ]
        if missing:
            raise ValueError(
                f"Entry at index {idx} missing required fields: {', '.join(missing)}"
            )
        cleaned.append(item)

    return cleaned


def iter_distortion_inputs(
    entries: Iterable[Dict[str, Any]]
) -> Iterable[Dict[str, Any]]:
    """
    Yield minimal inputs for distortion engines.

    Each yielded item includes:
    - question_id (string)
    - question (string)
    - source (original entry for reference)
    """
    for item in entries:
        yield {
            "question_id": str(item["question_id"]),
            "question": item["question"],
            "source": item,
        }


def load_distortion_inputs(
    data_path: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience helper to load and normalize distortion inputs.
    """
    entries = load_entries(data_path, limit=limit)
    return list(iter_distortion_inputs(entries))
