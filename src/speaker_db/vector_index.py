"""FAISS-backed multi-vector speaker index."""

from __future__ import annotations

import numpy as np

from config import TOPK
from src.core.types import SearchResult
from src.embedding.normalize import l2_normalize


class FaissVectorIndex:
    """Small-scale cosine search index using FAISS IndexFlatIP."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self.index = None
        self.vectors = np.empty((0, dim or 0), dtype=np.float32)
        self.labels: list[str] = []

    def build(self, vectors: np.ndarray, spk_ids: list[str]) -> None:
        """Build the FAISS index from normalized vectors and labels."""
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got {vectors.shape}")
        if vectors.shape[0] != len(spk_ids):
            raise ValueError("vectors and spk_ids must have the same length")

        import faiss

        self.dim = int(vectors.shape[1])
        normalized = np.stack([l2_normalize(v).astype(np.float32) for v in vectors])
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(normalized)
        self.vectors = normalized
        self.labels = list(spk_ids)

    def search(self, query_emb: np.ndarray, topk: int = TOPK) -> list[SearchResult]:
        """Search for nearest speaker vectors."""
        if self.index is None or not self.labels:
            return []
        query = l2_normalize(np.asarray(query_emb, dtype=np.float32)).reshape(1, -1)
        scores, indexes = self.index.search(query, min(topk, len(self.labels)))
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indexes[0], strict=False):
            if idx < 0:
                continue
            results.append(SearchResult(speaker=self.labels[int(idx)], score=float(score)))
        return results

    def add(self, new_vector: np.ndarray, spk_id: str) -> None:
        """Add a vector by rebuilding the small flat index."""
        vector = l2_normalize(np.asarray(new_vector, dtype=np.float32)).reshape(1, -1)
        if self.index is None:
            self.build(vector, [spk_id])
            return
        vectors = np.vstack([self.vectors, vector])
        labels = [*self.labels, spk_id]
        self.build(vectors, labels)

    def rebuild(self) -> None:
        """Rebuild the current in-memory index."""
        if self.vectors.size == 0:
            return
        self.build(self.vectors, self.labels)


# Backward-compatible alias
VectorIndex = FaissVectorIndex


def vectors_from_speaker_db(speaker_db: dict, max_emb: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Flatten speaker_db embeddings for vector index construction."""
    vectors = []
    labels = []
    for spk_id, record in speaker_db.items():
        embeddings = record.get("embeddings") or [record["center"]]
        selected = embeddings if max_emb is None else embeddings[:max_emb]
        for emb in selected:
            vectors.append(np.asarray(emb, dtype=np.float32))
            labels.append(spk_id)
    matrix = np.stack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    return matrix, labels
