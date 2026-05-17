"""Tests for src.core.pool EmbeddingPool."""

import numpy as np
import pytest

from src.core.pool import EmbeddingPool
from src.core.types import SpeakerSegment


def _seg(
    sid: str,
    emb: np.ndarray | None = None,
    spk: str | None = None,
) -> SpeakerSegment:
    return SpeakerSegment(
        segment_id=sid,
        file="a.wav",
        start=0.0,
        end=1.0,
        local_speaker="A",
        embedding=emb.astype(np.float32) if emb is not None else None,
        global_speaker=spk,
    )


def test_empty_pool_matrix():
    pool = EmbeddingPool()
    assert pool.to_matrix().shape == (0, 192)


def test_add_and_get_by_id():
    pool = EmbeddingPool()
    seg = _seg("s1", np.zeros(192))
    pool.add(seg)
    assert pool.get_by_id("s1") == seg
    assert pool.get_by_id("s2") is None


def test_duplicate_id_raises():
    pool = EmbeddingPool()
    pool.add(_seg("s1", np.zeros(192)))
    with pytest.raises(ValueError):
        pool.add(_seg("s1", np.zeros(192)))


def test_to_matrix_lazy_build():
    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0
    pool = EmbeddingPool([_seg("s1", emb), _seg("s2", emb)])
    mat = pool.to_matrix()
    assert mat.shape == (2, 192)
    assert mat[0, 0] == 1.0


def test_apply_labels():
    emb = np.zeros(192, dtype=np.float32)
    pool = EmbeddingPool([_seg("s1", emb), _seg("s2", emb)])
    new_pool = pool.apply_labels([0, 0])
    assert new_pool.filter_by_speaker("SPK_0")[0].segment_id == "s1"
    assert new_pool.filter_by_speaker("SPK_0")[1].segment_id == "s2"


def test_apply_labels_mismatch_raises():
    pool = EmbeddingPool([_seg("s1", np.zeros(192))])
    with pytest.raises(ValueError):
        pool.apply_labels([0, 1])


def test_filter_by_speaker():
    emb = np.zeros(192, dtype=np.float32)
    pool = EmbeddingPool([
        _seg("s1", emb, "SPK_0"),
        _seg("s2", emb, "SPK_1"),
        _seg("s3", emb, "SPK_0"),
    ])
    assert len(pool.filter_by_speaker("SPK_0")) == 2


def test_global_speakers():
    emb = np.zeros(192, dtype=np.float32)
    pool = EmbeddingPool([_seg("s1", emb, "SPK_0"), _seg("s2", emb, None)])
    assert pool.global_speakers() == {"SPK_0"}


def test_iteration():
    emb = np.zeros(192, dtype=np.float32)
    pool = EmbeddingPool([_seg("s1", emb), _seg("s2", emb)])
    assert len(pool) == 2
    assert [s.segment_id for s in pool] == ["s1", "s2"]
