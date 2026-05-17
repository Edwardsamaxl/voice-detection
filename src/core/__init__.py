"""Core domain models, storage, and pipeline abstractions."""

from .types import (
    IdentificationResult,
    SearchResult,
    SpeakerData,
    SpeakerProfile,
    SpeakerSegment,
    NumpyEncoder,
)
from .pool import EmbeddingPool
from .storage import Storage, JsonStorage, NpzStorage, PickleStorage, MemoryStorage

__all__ = [
    "IdentificationResult",
    "SearchResult",
    "SpeakerData",
    "SpeakerProfile",
    "SpeakerSegment",
    "NumpyEncoder",
    "EmbeddingPool",
    "Storage",
    "JsonStorage",
    "NpzStorage",
    "PickleStorage",
    "MemoryStorage",
]
