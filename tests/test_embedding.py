"""Tests for embedding extraction and normalization."""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
from unittest.mock import MagicMock, patch

from config import EMBEDDING_DIM
from src.embedding.extractor import EmbeddingExtractor
from src.embedding.normalize import l2_normalize


class TestL2Normalize:
    def test_l2_normalize_basic(self):
        emb = np.array([3.0, 4.0])
        result = l2_normalize(emb)
        assert np.isclose(np.linalg.norm(result), 1.0)
        assert np.allclose(result, np.array([0.6, 0.8]))

    def test_l2_normalize_already_normalized(self):
        emb = np.array([1.0, 0.0])
        result = l2_normalize(emb)
        assert np.allclose(result, emb)

    def test_l2_normalize_zero_vector(self):
        emb = np.zeros(5)
        result = l2_normalize(emb)
        assert np.allclose(result, emb)

    def test_l2_normalize_does_not_mutate(self):
        emb = np.array([3.0, 4.0])
        original = emb.copy()
        l2_normalize(emb)
        assert np.allclose(emb, original)


class _FakeTensor:
    """Minimal fake tensor to replace torch Tensor in tests."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def squeeze(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class TestEmbeddingExtractor:
    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_init_loads_model(self, mock_load):
        mock_classifier = MagicMock()
        mock_load.return_value = mock_classifier

        extractor = EmbeddingExtractor(device="cpu")

        mock_load.assert_called_once()
        assert extractor.device == "cpu"
        assert extractor.embedding_dim == EMBEDDING_DIM
        assert extractor.classifier == mock_classifier
        mock_classifier.eval.assert_called_once()

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_init_load_failure(self, mock_load):
        mock_load.side_effect = RuntimeError("model not found")

        with pytest.raises(RuntimeError, match="Failed to load SpeechBrain encoder"):
            EmbeddingExtractor(device="cpu")

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_returns_normalized_embedding(self, mock_load):
        fake_emb = np.ones(EMBEDDING_DIM) * 0.5
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = _FakeTensor(fake_emb)
        mock_load.return_value = mock_classifier

        extractor = EmbeddingExtractor(device="cpu")

        wav = np.ones(16000, dtype=np.float32) * 0.1
        result = extractor.extract(wav, sr=16000)

        assert result.shape == (EMBEDDING_DIM,)
        assert np.isclose(np.linalg.norm(result), 1.0)
        mock_classifier.encode_batch.assert_called_once()

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_int16_input(self, mock_load):
        fake_emb = np.ones(EMBEDDING_DIM) * 0.5
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = _FakeTensor(fake_emb)
        mock_load.return_value = mock_classifier

        extractor = EmbeddingExtractor(device="cpu")

        # Simulate int16 range input
        wav = np.ones(16000, dtype=np.int16) * 1000
        result = extractor.extract(wav, sr=16000)

        assert result.shape == (EMBEDDING_DIM,)
        assert np.isclose(np.linalg.norm(result), 1.0)

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_sr_mismatch_warns(self, mock_load):
        fake_emb = np.ones(EMBEDDING_DIM) * 0.5
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = _FakeTensor(fake_emb)
        mock_load.return_value = mock_classifier

        extractor = EmbeddingExtractor(device="cpu")

        wav = np.ones(16000, dtype=np.float32) * 0.1
        with pytest.warns(UserWarning, match="Sample rate 8000"):
            extractor.extract(wav, sr=8000)

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_invalid_shape(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = EmbeddingExtractor(device="cpu")

        wav = np.ones((2, 16000), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 1D audio array"):
            extractor.extract(wav)

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_empty_array(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = EmbeddingExtractor(device="cpu")

        wav = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="Empty audio array"):
            extractor.extract(wav)

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_for_file(self, mock_load):
        fake_emb = np.ones(EMBEDDING_DIM) * 0.5
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = _FakeTensor(fake_emb)
        mock_load.return_value = mock_classifier

        extractor = EmbeddingExtractor(device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            data = np.random.randn(16000 * 2).astype(np.float32)
            sf.write(wav_path, data, 16000)

            data_store = {
                "test.wav": [
                    {"start": 0.0, "end": 1.0, "local_speaker": "A"},
                    {"start": 1.0, "end": 2.0, "local_speaker": "B"},
                ]
            }
            result = extractor.extract_for_file(data_store, "test.wav", wav_path)

        assert "test.wav" in result
        segments = result["test.wav"]
        assert len(segments) == 2
        assert segments[0]["local_speaker"] == "A"
        assert segments[0]["embedding"] is not None
        assert segments[0]["embedding"].shape == (EMBEDDING_DIM,)
        assert segments[1]["local_speaker"] == "B"
        assert segments[1]["embedding"] is not None
        assert mock_classifier.encode_batch.call_count == 2

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_for_file_with_resample(self, mock_load):
        fake_emb = np.ones(EMBEDDING_DIM) * 0.5
        mock_classifier = MagicMock()
        mock_classifier.encode_batch.return_value = _FakeTensor(fake_emb)
        mock_load.return_value = mock_classifier

        extractor = EmbeddingExtractor(device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            data = np.random.randn(16000 * 2).astype(np.float32)
            sf.write(wav_path, data, 8000)

            data_store = {"test.wav": [{"start": 0.0, "end": 1.0}]}
            result = extractor.extract_for_file(data_store, "test.wav", wav_path)

        segments = result["test.wav"]
        assert len(segments) == 1
        emb = segments[0]["embedding"]
        assert emb is not None
        assert emb.shape == (EMBEDDING_DIM,)
        # ensure resampling happened: 8000Hz * 1s = 8000 samples, resampled to 16000
        call_args = mock_classifier.encode_batch.call_args
        input_tensor = call_args[0][0]
        assert input_tensor.shape[1] == 16000

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_for_file_empty_slice(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = EmbeddingExtractor(device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test.wav")
            data = np.random.randn(16000).astype(np.float32)
            sf.write(wav_path, data, 16000)

            data_store = {"test.wav": [{"start": 2.0, "end": 1.0}]}
            with pytest.warns(UserWarning, match="Empty segment"):
                result = extractor.extract_for_file(data_store, "test.wav", wav_path)

        assert result["test.wav"][0]["embedding"] is None

    @patch.object(EmbeddingExtractor, "_load_classifier")
    def test_extract_for_file_missing_key(self, mock_load):
        mock_load.return_value = MagicMock()
        extractor = EmbeddingExtractor(device="cpu")

        data_store = {"other.wav": [{"start": 0.0, "end": 1.0}]}
        with pytest.raises(KeyError, match="test.wav"):
            extractor.extract_for_file(data_store, "test.wav", "dummy.wav")
