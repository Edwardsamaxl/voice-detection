"""Diarization segment cache utilities (JSON)."""

import json
import os
from typing import Any


def save_segments(file_path: str, segments: list[dict]) -> None:
    """Save diarization segments to JSON file."""
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path)}")
    if not isinstance(segments, list):
        raise TypeError(f"segments must be list, got {type(segments)}")
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def load_segments(file_path: str) -> list[dict]:
    """Load diarization segments from JSON file."""
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path)}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Segment cache not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in segment cache, got {type(data)}")
    return data
