"""Embedding cache utilities (npz + json) for segments with embeddings."""

import json
import os
from glob import glob
from typing import Any

import numpy as np


def save_embeddings(file_path: str, segments: list[dict]) -> None:
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
        meta = {k: v for k, v in seg.items() if k != "embedding"}
        emb = seg.get("embedding")
        meta["has_embedding"] = emb is not None
        meta_list.append(meta)
        if emb is not None:
            embs.append(np.asarray(emb, dtype=np.float32))

    np.savez(file_path, embeddings=np.stack(embs) if embs else np.array([]))

    json_path = _json_path(file_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


def load_embeddings(file_path: str) -> list[dict]:
    """Load segments with embeddings from .npz + .json."""
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

    segments = []
    emb_idx = 0
    for meta in meta_list:
        seg = dict(meta)
        has_emb = seg.pop("has_embedding", False)
        if has_emb and embs.size > 0 and emb_idx < len(embs):
            seg["embedding"] = embs[emb_idx]
            emb_idx += 1
        else:
            seg["embedding"] = None
        segments.append(seg)

    return segments


def list_embedding_files(dir_path: str) -> list[str]:
    """List all .npz embedding files in a directory (recursive)."""
    pattern = os.path.join(dir_path, "**", "*.npz")
    return sorted(glob(pattern, recursive=True))


def _json_path(npz_path: str) -> str:
    """Get the companion json path for a npz file."""
    base, _ = os.path.splitext(npz_path)
    return base + ".json"
