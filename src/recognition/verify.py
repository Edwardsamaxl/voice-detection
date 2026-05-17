"""Speaker recognition verification helpers."""

from __future__ import annotations

from collections import Counter

from config import TOPK_CONSISTENCY_MIN
from src.core.types import SearchResult


def topk_consistency(
    topk_results: list[SearchResult],
    k: int = 5,
    min_consistency: int = TOPK_CONSISTENCY_MIN,
) -> bool:
    """Return True when enough top-k results agree on one speaker."""
    speakers = [r.speaker for r in topk_results[:k] if r.speaker]
    if not speakers:
        return False
    _, count = Counter(speakers).most_common(1)[0]
    return count >= min_consistency
