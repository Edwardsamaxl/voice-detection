"""Tests for embedding cache utilities."""

import os
import tempfile

import numpy as np
import pytest

from config import EMBEDDING_DIM
from src.core.types import SpeakerSegment
from src.embedding.cache import list_embedding_files, load_embeddings, save_embeddings


def _segment(
    segment_id: str,
    local_speaker: str = "A",
    embedding: np.ndarray | None = None,
) -> SpeakerSegment:
    return SpeakerSegment(
        segment_id=segment_id,
        file="a.wav",
        start=0.0,
        end=1.0,
        local_speaker=local_speaker,
        embedding=embedding,
    )


class TestSaveAndLoad:
    def test_roundtrip_basic(self):
        segments = [
            _segment("a_0000", embedding=np.ones(EMBEDDING_DIM) * 0.1),
            _segment("a_0001", local_speaker="B", embedding=np.ones(EMBEDDING_DIM) * 0.2),
            _segment("a_0002", local_speaker="C", embedding=np.ones(EMBEDDING_DIM) * 0.3),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            save_embeddings(path, segments)

            assert os.path.exists(path)
            assert os.path.exists(os.path.join(tmpdir, "test.json"))

            loaded = load_embeddings(path)

        assert len(loaded) == 3
        assert loaded[0].start == 0.0
        assert loaded[0].local_speaker == "A"
        assert loaded[0].embedding is not None
        assert np.allclose(loaded[0].embedding, segments[0].embedding)
        assert loaded[1].local_speaker == "B"
        assert np.allclose(loaded[1].embedding, segments[1].embedding)
        assert loaded[2].local_speaker == "C"
        assert np.allclose(loaded[2].embedding, segments[2].embedding)

    def test_roundtrip_with_none_embedding(self):
        segments = [
            _segment("a_0000", embedding=np.ones(EMBEDDING_DIM)),
            SpeakerSegment(
                segment_id="a_0001",
                file="a.wav",
                start=1.0,
                end=2.0,
                local_speaker="B",
                embedding=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            save_embeddings(path, segments)
            loaded = load_embeddings(path)

        assert loaded[0].embedding is not None
        assert loaded[1].embedding is None

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_embeddings("/nonexistent/file.npz")

    def test_load_missing_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            np.savez(path, embeddings=np.ones((1, EMBEDDING_DIM)))
            with pytest.raises(FileNotFoundError, match="Metadata cache not found"):
                load_embeddings(path)

    def test_rejects_bad_embedding_shape_on_save(self):
        segments = [
            _segment("a_0000", embedding=np.ones(3)),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            with pytest.raises(ValueError, match="must have shape"):
                save_embeddings(path, segments)

    def test_rejects_embedding_metadata_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            np.savez(path, embeddings=np.ones((2, EMBEDDING_DIM), dtype=np.float32))
            with open(os.path.join(tmpdir, "test.json"), "w", encoding="utf-8") as f:
                f.write(
                    '[{"start": 0.0, "end": 1.0, "duration": 1.0, '
                    '"file": "a.wav", "segment_id": "a_0000", '
                    '"local_speaker": "A", "has_embedding": true}]'
                )

            with pytest.raises(ValueError, match="row count mismatch"):
                load_embeddings(path)

    def test_list_embedding_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "sub"))
            open(os.path.join(tmpdir, "a.npz"), "w").close()
            open(os.path.join(tmpdir, "sub", "b.npz"), "w").close()
            open(os.path.join(tmpdir, "c.txt"), "w").close()

            files = list_embedding_files(tmpdir)

        assert len(files) == 2
        assert any("a.npz" in f for f in files)
        assert any("b.npz" in f for f in files)
