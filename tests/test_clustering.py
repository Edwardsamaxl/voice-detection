"""Tests for clustering module.

Uses synthetic embeddings that mimic the SLR80 train-split characteristics
(e.g. ``bur_0366_5281755035.wav`` naming) so validation stays within the
train domain.
"""

import numpy as np
import pytest

from src.clustering.cluster import (
    agg_clustering,
    assign_global_speakers,
    assign_global_speakers_from_slr80_filename,
    deduplicate_segments,
)
from src.core.pool import EmbeddingPool
from src.core.types import SpeakerSegment


def _seg(
    sid: str,
    emb: np.ndarray | None = None,
    file: str = "bur_0366_5281755035.wav",
    spk: str | None = None,
) -> SpeakerSegment:
    return SpeakerSegment(
        segment_id=sid,
        file=file,
        start=0.0,
        end=1.0,
        local_speaker="A",
        embedding=emb.astype(np.float32) if emb is not None else None,
        global_speaker=spk,
    )


class TestDeduplicateSegments:
    def test_empty_sequence(self):
        assert deduplicate_segments([]) == []

    def test_no_duplicates_all_kept(self):
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        segments = [_seg("s1", emb1), _seg("s2", emb2)]
        result = deduplicate_segments(segments)
        assert len(result) == 2
        assert result[0].segment_id == "s1"
        assert result[1].segment_id == "s2"

    def test_similar_embedding_deduplicated(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_almost = emb * 0.99 + np.array([0.0, 0.01, 0.0], dtype=np.float32)
        segments = [_seg("s1", emb), _seg("s2", emb_almost)]
        result = deduplicate_segments(segments, threshold=0.95)
        assert len(result) == 1
        assert result[0].segment_id == "s1"

    def test_different_embeddings_both_kept(self):
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        segments = [_seg("s1", emb1), _seg("s2", emb2)]
        result = deduplicate_segments(segments, threshold=0.95)
        assert len(result) == 2

    def test_missing_embedding_skipped(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        segments = [_seg("s1", emb), _seg("s2", None)]
        result = deduplicate_segments(segments)
        assert len(result) == 1
        assert result[0].segment_id == "s1"

    def test_returns_new_instances_with_embedding(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        seg = _seg("s1", emb)
        result = deduplicate_segments([seg])
        assert result[0] is not seg
        assert result[0].embedding is not None


class TestAggClustering:
    def test_empty_embeddings(self):
        result = agg_clustering(np.array([]).reshape(0, 192))
        assert result.shape == (0,)
        assert result.dtype == np.int64 or result.dtype == np.int32

    def test_single_vector(self):
        emb = np.ones((1, 192), dtype=np.float32)
        result = agg_clustering(emb)
        assert result.tolist() == [0]

    def test_two_identical_vectors_same_cluster(self):
        emb = np.ones((2, 192), dtype=np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        result = agg_clustering(emb, distance_threshold=0.3)
        assert result[0] == result[1]

    def test_two_dissimilar_vectors_different_clusters(self):
        emb1 = np.zeros((1, 192), dtype=np.float32)
        emb1[0, 0] = 1.0
        emb2 = np.zeros((1, 192), dtype=np.float32)
        emb2[0, 1] = 1.0
        emb = np.vstack([emb1, emb2])
        result = agg_clustering(emb, distance_threshold=0.3)
        assert result[0] != result[1]

    def test_two_stage_merge_reduces_fragments(self):
        # Create 3 clusters where two centroids are close enough to merge
        cluster_a = np.tile([1.0, 0.0] + [0.0] * 190, (3, 1)).astype(np.float32)
        cluster_a = cluster_a / np.linalg.norm(cluster_a, axis=1, keepdims=True)

        cluster_b = np.tile([0.99, 0.01] + [0.0] * 190, (3, 1)).astype(np.float32)
        cluster_b = cluster_b / np.linalg.norm(cluster_b, axis=1, keepdims=True)

        cluster_c = np.tile([0.0, 1.0] + [0.0] * 190, (3, 1)).astype(np.float32)
        cluster_c = cluster_c / np.linalg.norm(cluster_c, axis=1, keepdims=True)

        embeddings = np.vstack([cluster_a, cluster_b, cluster_c])
        labels_one_stage = agg_clustering(embeddings, distance_threshold=0.3, centroid_threshold=None)
        labels_two_stage = agg_clustering(
            embeddings, distance_threshold=0.3, centroid_threshold=0.05
        )
        assert len(np.unique(labels_two_stage)) <= len(np.unique(labels_one_stage))


class TestAssignGlobalSpeakers:
    def test_assigns_spk_labels(self):
        emb1 = np.zeros(192, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(192, dtype=np.float32)
        emb2[1] = 1.0
        pool = EmbeddingPool([_seg("s1", emb1), _seg("s2", emb2)])
        result = assign_global_speakers(pool, distance_threshold=0.3)
        speakers = result.global_speakers()
        assert len(speakers) == 2
        assert all(s.startswith("SPK_") for s in speakers)


class TestAssignGlobalSpeakersFromSlr80Filename:
    def test_extracts_speaker_from_train_filename(self):
        # SLR80 train files look like bur_0366_5281755035.wav
        pool = EmbeddingPool([
            _seg("seg1", np.zeros(192), file="bur_0366_5281755035.wav"),
            _seg("seg2", np.zeros(192), file="bur_0366_5281755036.wav"),
            _seg("seg3", np.zeros(192), file="bur_0421_1234567890.wav"),
        ])
        result = assign_global_speakers_from_slr80_filename(pool)
        ids = {seg.global_speaker for seg in result}
        assert ids == {"BUR_0366", "BUR_0421"}

    def test_missing_pattern_raises(self):
        pool = EmbeddingPool([
            _seg("seg1", np.zeros(192), file="unknown.wav"),
        ])
        with pytest.raises(ValueError, match="Cannot infer SLR80 speaker id"):
            assign_global_speakers_from_slr80_filename(pool)

    def test_custom_prefix(self):
        pool = EmbeddingPool([
            _seg("seg1", np.zeros(192), file="bur_0100_0000000000.wav"),
        ])
        result = assign_global_speakers_from_slr80_filename(pool, prefix="TRAIN_")
        assert result.filter_by_speaker("TRAIN_0100")[0].segment_id == "seg1"
