"""Embedding cache utilities (npz + json) for SpeakerSegment lists."""

import json
import os
from dataclasses import asdict
from glob import glob
from typing import Any

import numpy as np

from config import EMBEDDING_DIM
from src.core.types import SpeakerSegment


def save_embeddings(file_path: str, segments: list[SpeakerSegment]) -> None:
    """Save segments with embeddings to .npz + companion .json.

    The npz file stores embedding arrays, and the json file stores
    segment metadata without embeddings.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path)}")
    if not isinstance(segments, list):
        raise TypeError(f"segments must be list, got {type(segments)}")

    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    meta_list = []
    embs = []
    for seg in segments:
        if not isinstance(seg, SpeakerSegment):
            raise TypeError(f"Each segment must be SpeakerSegment, got {type(seg)}")
        record = _segment_to_record(seg)
        meta = {k: v for k, v in record.items() if k != "embedding"}
        emb = record.get("embedding")
        meta["has_embedding"] = emb is not None
        meta_list.append(meta)
        if emb is not None:
            embs.append(_validate_embedding(emb, seg.segment_id))

    np.savez(
        file_path,
        embeddings=np.stack(embs) if embs else np.empty((0, EMBEDDING_DIM), dtype=np.float32),
    )

    json_path = _json_path(file_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


def load_embeddings(file_path: str) -> list[SpeakerSegment]:
    """Load segments with embeddings from .npz + .json as V2 SpeakerSegments."""
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path)}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding cache not found: {file_path}")

    data = np.load(file_path)
    embs = data.get("embeddings", np.array([]))

    json_path = _json_path(file_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata cache not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    has_embedding_count = sum(1 for meta in meta_list if meta.get("has_embedding", False))
    if has_embedding_count != len(embs):
        raise ValueError(
            f"Embedding row count mismatch: metadata expects {has_embedding_count}, npz has {len(embs)}"
        )

    segments: list[SpeakerSegment] = []
    emb_idx = 0
    for meta in meta_list:
        has_emb = meta.pop("has_embedding", False)
        if has_emb and embs.size > 0 and emb_idx < len(embs):
            meta["embedding"] = _validate_embedding(embs[emb_idx], str(meta.get("segment_id") or "<unknown>"))
            emb_idx += 1
        else:
            meta["embedding"] = None
        segments.append(_record_to_segment(meta))

    return segments


def list_embedding_files(dir_path: str) -> list[str]:
    """List all .npz embedding files in a directory (recursive)."""
    pattern = os.path.join(dir_path, "**", "*.npz")
    return sorted(glob(pattern, recursive=True))


def _json_path(npz_path: str) -> str:
    """Get the companion json path for a npz file."""
    base, _ = os.path.splitext(npz_path)
    return base + ".json"


def _segment_to_record(segment: SpeakerSegment) -> dict[str, Any]:
    record = asdict(segment)
    record["duration"] = segment.duration
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


def _validate_embedding(embedding: Any, segment_id: str) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding for {segment_id} must have shape ({EMBEDDING_DIM},), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Embedding for {segment_id} contains non-finite values")
    return vector
