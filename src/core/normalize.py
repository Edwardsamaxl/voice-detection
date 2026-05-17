"""Embedding normalization utilities."""

import numpy as np


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector.

    Args:
        embedding: 1D numpy array representing a single embedding vector.

    Returns:
        L2-normalized copy. Returns a copy of the original if norm is zero.
    """
    if embedding.size == 0:
        return embedding.copy()

    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding.copy()

    return embedding / norm
