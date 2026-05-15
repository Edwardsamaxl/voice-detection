"""Speaker embedding extraction and normalization."""

from .extractor import EmbeddingExtractor
from .normalize import l2_normalize

__all__ = ["EmbeddingExtractor", "l2_normalize"]
