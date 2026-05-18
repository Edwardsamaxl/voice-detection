"""Immutable domain types for the voice-detection pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """A single time-segment of speech with metadata.

    Immutable: use the `with_*` helpers to derive modified copies.
    """

    segment_id: str
    file: str
    start: float
    end: float
    local_speaker: str
    global_speaker: str | None = None
    display_name: str | None = None
    embedding: np.ndarray | None = None
    score: float | None = None
    text: str | None = None
    translation: str | None = None
    sr: int = 16000

    @property
    def duration(self) -> float:
        return self.end - self.start

    def with_embedding(self, emb: np.ndarray | None) -> SpeakerSegment:
        return replace(self, embedding=emb)

    def with_global_speaker(self, spk_id: str) -> SpeakerSegment:
        return replace(self, global_speaker=spk_id)

    def with_display_name(self, name: str) -> SpeakerSegment:
        return replace(self, display_name=name)

    def with_local_speaker(self, speaker: str) -> SpeakerSegment:
        return replace(self, local_speaker=speaker)

    def with_score(self, score: float | None) -> SpeakerSegment:
        return replace(self, score=score)

    def with_sr(self, sr: int) -> SpeakerSegment:
        return replace(self, sr=sr)

    def with_text(self, text: str, translation: str | None = None) -> SpeakerSegment:
        return replace(self, text=text, translation=translation)


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    """Human-readable metadata for a speaker."""

    name: str
    alias: list[str] | None = None
    gender: str | None = None
    notes: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class VectorEntry:
    """A single retained embedding for a speaker."""

    spk_id: str
    embedding: np.ndarray
    duration: float


@dataclass(frozen=True, slots=True)
class SpeakerData:
    """Aggregated speaker metadata: center vector, selected embeddings, and profile.

    V2.3: embeddings and durations kept in SpeakerData for self-contained speaker_db.
    """

    spk_id: str
    center: np.ndarray
    embedding_count: int
    embeddings: list[np.ndarray] | None = None
    durations: list[float] | None = None
    profile: SpeakerProfile | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single result from a vector search."""

    speaker: str
    score: float


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """Outcome of speaker identification."""

    speaker: str | None
    score: float
    confidence: str  # "high", "low", "unknown"
