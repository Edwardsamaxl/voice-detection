"""Compatibility tests for Phase4-10 V2 module boundaries."""

import numpy as np

from src.clustering.cluster import (
    assign_global_speakers,
    assign_global_speakers_from_slr80_filename,
)
from src.clustering.pool import build_embedding_pool
from src.core.pool import EmbeddingPool
from src.core.repository import SpeakerRepository
from src.core.types import SpeakerSegment
from src.speaker_db.vector_index import VectorIndex


def _segment(segment_id: str, emb: np.ndarray) -> SpeakerSegment:
    return SpeakerSegment(
        segment_id=segment_id,
        file="sample.wav",
        start=0.0,
        end=2.0,
        local_speaker="A",
        embedding=emb.astype(np.float32),
    )


def test_pool_to_repository_to_identification():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    pool = EmbeddingPool([_segment("s0", a), _segment("s1", b)])

    clustered = assign_global_speakers(pool, distance_threshold=0.2)
    repo = SpeakerRepository(vector_index=VectorIndex())
    repo.build_from_pool(clustered)
    result = repo.identify(a)

    assert clustered.filter_by_speaker("SPK_0")
    assert result.confidence in {"high", "low"}
    assert result.speaker is None or result.speaker.startswith("SPK_")


def test_legacy_cached_embeddings_without_ids_load_into_pool(tmp_path):
    emb_dir = tmp_path / "embeddings"
    seg_dir = tmp_path / "segments"
    emb_dir.mkdir()
    seg_dir.mkdir()

    emb = np.zeros((1, 192), dtype=np.float32)
    emb[0, 0] = 1.0
    np.savez(emb_dir / "bur_0366_0045318711.npz", embeddings=emb)
    (seg_dir / "bur_0366_0045318711.json").write_text(
        """[
          {
            "segment_id": "",
            "file": "",
            "start": 0.588,
            "end": 4.351,
            "local_speaker": "SPEAKER_00",
            "embedding": null
          }
        ]""",
        encoding="utf-8",
    )

    pool = build_embedding_pool(str(emb_dir), str(seg_dir))
    segment = pool.get_by_id("bur_0366_0045318711_0000")

    assert len(pool) == 1
    assert segment is not None
    assert segment.file == "bur_0366_0045318711"
    assert segment.embedding is not None


def test_slr80_filename_labels_build_expected_speakers():
    a = np.zeros(192, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(192, dtype=np.float32)
    b[1] = 1.0
    pool = EmbeddingPool(
        [
            SpeakerSegment("bur_0366_0001_0000", "bur_0366_0001", 0.0, 2.0, "A", embedding=a),
            SpeakerSegment("bur_0366_0002_0000", "bur_0366_0002", 0.0, 2.0, "A", embedding=a),
            SpeakerSegment("bur_0644_0001_0000", "bur_0644_0001", 0.0, 2.0, "A", embedding=b),
        ]
    )

    labeled = assign_global_speakers_from_slr80_filename(pool)
    repo = SpeakerRepository(vector_index=VectorIndex())
    repo.build_from_pool(labeled)

    assert labeled.global_speakers() == {"BUR_0366", "BUR_0644"}
    assert sorted(repo.all_speakers()) == ["BUR_0366", "BUR_0644"]
