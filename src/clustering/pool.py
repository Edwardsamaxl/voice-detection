"""Build global embedding pools from cached embedding files."""

from __future__ import annotations

import glob
import os

import numpy as np

from src.core.pool import EmbeddingPool
from src.core.storage import JsonStorage, NpzStorage
from src.core.types import SpeakerSegment


def segment_from_record(record: dict) -> SpeakerSegment:
    """Convert a cached segment record into a V2 SpeakerSegment."""
    embedding = record.get("embedding")
    if isinstance(embedding, list):
        embedding = np.asarray(embedding, dtype=np.float32)
    return SpeakerSegment(
        segment_id=str(record["segment_id"]),
        file=str(record["file"]),
        start=float(record["start"]),
        end=float(record["end"]),
        local_speaker=str(record.get("local_speaker") or "UNKNOWN"),
        global_speaker=record.get("global_speaker"),
        display_name=record.get("display_name"),
        embedding=embedding,
        text=record.get("text"),
        translation=record.get("translation"),
    )


def build_embedding_pool(npz_dir: str, segment_dir: str | None = None) -> EmbeddingPool:
    """Load all cached embedding files and build a V2 embedding pool."""
    if segment_dir is None:
        segment_dir = os.path.join(npz_dir, "../segments")
        if not os.path.isdir(segment_dir):
            segment_dir = npz_dir
    pool = EmbeddingPool()
    npz_storage = NpzStorage(npz_dir)
    json_storage = JsonStorage(segment_dir)
    for npz_path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        data = npz_storage.load(basename)
        embs = data.get("embeddings", np.array([]))
        meta_list = json_storage.load(basename)
        emb_idx = 0
        for meta in meta_list:
            has_emb = meta.get("has_embedding", False)
            if has_emb and embs.size > 0 and emb_idx < len(embs):
                meta = dict(meta)
                meta["embedding"] = np.asarray(embs[emb_idx], dtype=np.float32)
                emb_idx += 1
            seg = segment_from_record(meta)
            if seg.embedding is not None:
                pool.add(seg)
    return pool
