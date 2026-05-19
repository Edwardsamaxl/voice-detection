"""Tests for speaker DB: FaissVectorIndex and SpeakerRepository."""

import numpy as np
import pytest
from unittest import mock

from config import EMBEDDING_DIM, MAX_EMB
from src.core.pool import EmbeddingPool
from src.core.repository import SpeakerRepository, VectorDb
from src.core.storage import MemoryStorage
from src.core.types import IdentificationResult, SpeakerData, SpeakerProfile, SpeakerSegment, VectorEntry
from src.speaker_db.vector_index import FaissVectorIndex, vectors_from_speaker_db


class TestFaissVectorIndex:
    def test_build_empty(self):
        index = FaissVectorIndex(dim=EMBEDDING_DIM)
        index.build(np.empty((0, EMBEDDING_DIM), dtype=np.float32), [])
        assert index.index is None
        assert index.labels == []

    def test_build_wrong_ndim_raises(self):
        index = FaissVectorIndex()
        with pytest.raises(ValueError, match="vectors must be 2D"):
            index.build(np.ones(EMBEDDING_DIM, dtype=np.float32), ["A"])

    def test_build_length_mismatch_raises(self):
        index = FaissVectorIndex()
        with pytest.raises(ValueError, match="same length"):
            index.build(np.ones((2, EMBEDDING_DIM), dtype=np.float32), ["A"])

    def test_search_empty_index_returns_empty(self):
        index = FaissVectorIndex()
        index.build(np.empty((0, EMBEDDING_DIM), dtype=np.float32), [])
        assert index.search(np.ones(EMBEDDING_DIM, dtype=np.float32)) == []

    def test_add_to_empty_builds(self):
        index = FaissVectorIndex()
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec[0] = 1.0
        index.add(vec, "SPK_0")
        assert index.labels == ["SPK_0"]

    def test_add_rebuilds_index(self):
        index = FaissVectorIndex()
        v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v2[1] = 1.0
        index.add(v1, "SPK_0")
        index.add(v2, "SPK_1")
        assert set(index.labels) == {"SPK_0", "SPK_1"}

    def test_rebuild_noop_when_empty(self):
        index = FaissVectorIndex()
        index.rebuild()
        assert index.index is None


class TestVectorsFromSpeakerDb:
    def test_flatten_embeddings(self):
        speaker_db = {
            "SPK_0": {"embeddings": [np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.5]},
            "SPK_1": {"embeddings": [np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.3]},
        }
        matrix, labels = vectors_from_speaker_db(speaker_db)
        assert matrix.shape == (2, EMBEDDING_DIM)
        assert labels == ["SPK_0", "SPK_1"]

    def test_fallback_to_center(self):
        speaker_db = {
            "SPK_0": {"center": np.ones(EMBEDDING_DIM, dtype=np.float32)},
        }
        matrix, labels = vectors_from_speaker_db(speaker_db)
        assert matrix.shape == (1, EMBEDDING_DIM)
        assert labels == ["SPK_0"]

    def test_max_emb_limits(self):
        speaker_db = {
            "SPK_0": {"embeddings": [np.ones(EMBEDDING_DIM, dtype=np.float32) * i for i in range(5)]},
        }
        matrix, labels = vectors_from_speaker_db(speaker_db, max_emb=3)
        assert matrix.shape == (3, EMBEDDING_DIM)
        assert labels == ["SPK_0"] * 3

    def test_empty_db(self):
        matrix, labels = vectors_from_speaker_db({})
        assert matrix.shape == (0, 0)
        assert labels == []


class TestSpeakerRepositoryBuildFromPool:
    def test_groups_by_global_speaker(self):
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        pool = EmbeddingPool([
            SpeakerSegment("s1", "a.wav", 0.0, 1.0, "A", global_speaker="SPK_0", embedding=emb),
            SpeakerSegment("s2", "a.wav", 1.0, 2.0, "A", global_speaker="SPK_0", embedding=emb),
            SpeakerSegment("s3", "a.wav", 0.0, 1.0, "B", global_speaker="SPK_1", embedding=emb),
        ])
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        repo.build_from_pool(pool)
        assert set(repo.all_speakers()) == {"SPK_0", "SPK_1"}

    def test_skips_missing_global_speaker_or_embedding(self):
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        pool = EmbeddingPool([
            SpeakerSegment("s1", "a.wav", 0.0, 1.0, "A", global_speaker="SPK_0", embedding=emb),
            SpeakerSegment("s2", "a.wav", 0.0, 1.0, "A", global_speaker=None, embedding=emb),
            SpeakerSegment("s3", "a.wav", 0.0, 1.0, "A", global_speaker="SPK_1", embedding=None),
        ])
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        repo.build_from_pool(pool)
        assert repo.all_speakers() == ["SPK_0"]

    def test_max_emb_pruning(self):
        embeddings = [np.ones(EMBEDDING_DIM, dtype=np.float32) * i for i in range(5)]
        pool = EmbeddingPool([
            SpeakerSegment(
                f"s{i}", "a.wav", float(i), float(i + 1), "A",
                global_speaker="SPK_0", embedding=emb,
            )
            for i, emb in enumerate(embeddings)
        ])
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        repo.build_from_pool(pool, max_emb=3)
        spk = repo.get_speaker("SPK_0")
        assert spk is not None
        assert spk.embedding_count == 3


class TestSpeakerRepositoryIdentify:
    def _make_repo(self, speakers: dict[str, np.ndarray]) -> SpeakerRepository:
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        for spk_id, vec in speakers.items():
            repo.add_speaker(spk_id, [vec], [1.0])
        return repo

    def test_no_results_returns_unknown(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        result = repo.identify(np.ones(EMBEDDING_DIM, dtype=np.float32))
        assert result == IdentificationResult(speaker=None, score=0.0, confidence="unknown")

    def test_high_score_returns_high(self):
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec[0] = 1.0
        repo = self._make_repo({"SPK_0": vec})
        result = repo.identify(vec)
        assert result.speaker == "SPK_0"
        assert result.confidence == "high"
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_low_score_returns_low(self):
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec[0] = 1.0
        repo = self._make_repo({"SPK_0": vec})
        query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        query[1] = 1.0
        result = repo.identify(query)
        assert result.confidence == "low"
        assert result.speaker is None

    def test_high_score_with_multiple_speakers(self):
        v0 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v0[0] = 1.0
        v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v1[1] = 1.0
        repo = self._make_repo({"SPK_0": v0, "SPK_1": v1})
        result = repo.identify(v0)
        assert result.speaker == "SPK_0"
        assert result.confidence == "high"


class TestSpeakerRepositoryAddSpeaker:
    def test_empty_embeddings_raises(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        with pytest.raises(ValueError, match="embeddings must not be empty"):
            repo.add_speaker("SPK_0", [], [])

    def test_length_mismatch_raises(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        with pytest.raises(ValueError, match="same length"):
            repo.add_speaker("SPK_0", [np.ones(EMBEDDING_DIM, dtype=np.float32)], [1.0, 2.0])

    def test_adds_and_rebuilds(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        repo.add_speaker("SPK_0", [emb], [1.0])
        assert "SPK_0" in repo.all_speakers()
        spk = repo.get_speaker("SPK_0")
        assert spk is not None
        assert np.isclose(np.linalg.norm(spk.center), 1.0)

    def test_max_emb_pruning(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        embeddings = [
            np.ones(EMBEDDING_DIM, dtype=np.float32) * (1.0 if i % 2 == 0 else -1.0)
            for i in range(25)
        ]
        durations = [1.0] * 25
        repo.add_speaker("SPK_0", embeddings, durations)
        spk = repo.get_speaker("SPK_0")
        assert spk.embedding_count == MAX_EMB


class TestSpeakerRepositoryUpdateSpeaker:
    def test_duration_too_short_returns_false(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        repo.add_speaker("SPK_0", [emb], [1.0])
        assert repo.update_speaker("SPK_0", emb, 0.1) is False

    def test_missing_speaker_returns_false(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        assert repo.update_speaker("SPK_0", emb, 5.0) is False

    def test_dedup_too_similar_returns_false(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        repo.add_speaker("SPK_0", [emb], [1.0])
        assert repo.update_speaker("SPK_0", emb, 5.0) is False

    def test_normal_update_rebuilds(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        repo.add_speaker("SPK_0", [emb], [1.0])
        new_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        new_emb[1] = 1.0
        assert repo.update_speaker("SPK_0", new_emb, 5.0) is True
        spk = repo.get_speaker("SPK_0")
        assert spk.embedding_count == 2


class TestSpeakerRepositoryAssignName:
    def test_assigns_name(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        repo.add_speaker("SPK_0", [emb], [1.0])
        repo.assign_name("SPK_0", "Alice")
        spk = repo.get_speaker("SPK_0")
        assert spk is not None
        assert spk.profile is not None
        assert spk.profile.name == "Alice"

    def test_missing_speaker_raises(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        with pytest.raises(KeyError, match="Speaker not found"):
            repo.assign_name("SPK_0", "Alice")


class TestSpeakerRepositorySaveLoad:
    def test_roundtrip_with_memory_storage(self):
        storage = MemoryStorage()
        repo = SpeakerRepository(
            vector_index=FaissVectorIndex(), storage=storage, vector_storage=storage
        )
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[0] = 1.0
        repo.add_speaker("SPK_0", [emb], [1.0])
        repo.assign_name("SPK_0", "Alice")
        repo.save("db:v1")

        repo2 = SpeakerRepository(
            vector_index=FaissVectorIndex(), storage=storage, vector_storage=storage
        )
        repo2.load("db:v1")
        spk = repo2.get_speaker("SPK_0")
        assert spk is not None
        assert spk.profile is not None
        assert spk.profile.name == "Alice"
        assert np.allclose(spk.center, emb)
        assert spk.embeddings is not None
        assert len(spk.embeddings) == 1
        assert np.allclose(spk.embeddings[0], emb)

    def test_save_without_storage_raises(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        with pytest.raises(RuntimeError, match="No storage configured"):
            repo.save()

    def test_load_without_storage_raises(self):
        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        with pytest.raises(RuntimeError, match="No storage configured"):
            repo.load()
