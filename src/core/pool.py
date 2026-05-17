"""EmbeddingPool: encapsulated collection of SpeakerSegments with lazy matrix caching."""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from config import EMBEDDING_DIM
from .types import SpeakerSegment


class EmbeddingPool:
    """Encapsulates segments, meta-index, and vector matrix.

    Callers interact through a small interface (add, get_by_id, to_matrix,
    apply_labels). Internal consistency (index, dirty flag, matrix cache) is
    managed privately.
    """

    def __init__(self, segments: Iterable[SpeakerSegment] = ()) -> None:
        self._segments: list[SpeakerSegment] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._dirty = False
        for seg in segments:
            self.add(seg)

    # ------------------------------------------------------------------ #
    # Mutation (returns new pool for bulk ops)
    # ------------------------------------------------------------------ #

    def add(self, segment: SpeakerSegment) -> None:
        if segment.segment_id in self._index:
            raise ValueError(f"Duplicate segment_id: {segment.segment_id}")
        self._index[segment.segment_id] = len(self._segments)
        self._segments.append(segment)
        self._dirty = True

    def apply_labels(self, labels: list[int], prefix: str = "SPK_") -> EmbeddingPool:
        """Return a new pool with global_speaker assigned from cluster labels."""
        if len(labels) != len(self._segments):
            raise ValueError(
                f"Label count mismatch: {len(labels)} labels for {len(self._segments)} segments"
            )
        new_segments = [
            seg.with_global_speaker(f"{prefix}{label}")
            for seg, label in zip(self._segments, labels)
        ]
        return EmbeddingPool(new_segments)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_by_id(self, segment_id: str) -> SpeakerSegment | None:
        idx = self._index.get(segment_id)
        return self._segments[idx] if idx is not None else None

    def filter_by_speaker(self, spk_id: str) -> list[SpeakerSegment]:
        return [s for s in self._segments if s.global_speaker == spk_id]

    def to_matrix(self) -> np.ndarray:
        """Lazy-build and cache the (N, EMBEDDING_DIM) embedding matrix.

        The matrix is rebuilt only when the pool has been modified since the
        last call.
        """
        if self._dirty or self._matrix is None:
            embs = [s.embedding for s in self._segments if s.embedding is not None]
            if embs:
                self._matrix = np.stack(embs).astype(np.float32)
            else:
                self._matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            self._dirty = False
        return self._matrix

    def segment_ids(self) -> list[str]:
        return [s.segment_id for s in self._segments]

    def global_speakers(self) -> set[str]:
        return {
            s.global_speaker
            for s in self._segments
            if s.global_speaker is not None
        }

    # ------------------------------------------------------------------ #
    # Sequence protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self) -> Iterator[SpeakerSegment]:
        return iter(self._segments)

    def to_list(self) -> list[SpeakerSegment]:
        return list(self._segments)
