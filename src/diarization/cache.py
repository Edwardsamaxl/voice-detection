"""Diarization segment cache utilities (JSON)."""

import json
import os
from dataclasses import asdict
from typing import Any

from src.core.types import SpeakerSegment


def save_segments(file_path: str, segments: list[SpeakerSegment]) -> None:
    """Save diarization segments to JSON file."""
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path)}")
    if not isinstance(segments, list):
        raise TypeError(f"segments must be list, got {type(segments)}")
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    normalized = []
    for segment in segments:
        if not isinstance(segment, SpeakerSegment):
            raise ValueError(f"Each segment must be SpeakerSegment, got {type(segment)}")
        record = _segment_to_record(segment)
        normalized.append(record)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def load_segments(file_path: str) -> list[SpeakerSegment]:
    """Load diarization segments from JSON file."""
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path)}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Segment cache not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in segment cache, got {type(data)}")
    segments = []
    for segment in data:
        if not isinstance(segment, dict):
            raise ValueError(f"Each segment must be dict, got {type(segment)}")
        segments.append(_record_to_segment(segment))
    return segments


def _segment_to_record(segment: SpeakerSegment) -> dict[str, Any]:
    record = asdict(segment)
    record["duration"] = round(segment.duration, 3)
    return record


def _record_to_segment(record: dict[str, Any]) -> SpeakerSegment:
    return SpeakerSegment(
        segment_id=str(record.get("segment_id") or ""),
        file=str(record.get("file") or ""),
        start=float(record["start"]),
        end=float(record["end"]),
        local_speaker=str(record.get("local_speaker") or "UNKNOWN"),
        global_speaker=record.get("global_speaker"),
        display_name=record.get("display_name"),
        embedding=record.get("embedding"),
        text=record.get("text"),
        translation=record.get("translation"),
    )
