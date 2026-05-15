"""Tests for audio preprocessing."""

import os
import subprocess
import tempfile
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from config import SAMPLE_RATE, TARGET_RMS
from src.audio.preprocess import convert_to_wav, crop_segment, load_wav, rms_normalize


class TestRMSNormalize:
    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        result = rms_normalize(arr)
        assert result.size == 0

    def test_silent_array(self):
        arr = np.zeros(1000, dtype=np.float32)
        result = rms_normalize(arr)
        np.testing.assert_array_equal(result, arr)

    def test_target_rms(self):
        arr = np.random.randn(16000).astype(np.float32)
        result = rms_normalize(arr, target_rms=0.1)
        result_rms = np.sqrt(np.mean(result.astype(np.float64) ** 2))
        assert abs(result_rms - 0.1) < 1e-6

    def test_int16_clipping(self):
        arr = (np.ones(1000) * 30000).astype(np.int16)
        result = rms_normalize(arr, target_rms=40000)
        assert result.dtype == np.int16
        assert np.max(result) <= np.iinfo(np.int16).max


class TestConvertToWav:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            convert_to_wav("nonexistent.mp3", "out.wav")

    def test_output_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")

            duration = 1.0
            sr_in = 44100
            t = np.linspace(0, duration, int(sr_in * duration))
            stereo = np.column_stack(
                (np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t))
            )
            sf.write(input_path, stereo, sr_in)

            convert_to_wav(input_path, output_path)

            assert os.path.exists(output_path)
            data, sr_out = sf.read(output_path, dtype="int16")
            assert sr_out == SAMPLE_RATE
            assert len(data.shape) == 1
            assert data.dtype == np.int16

    def test_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            sf.write(input_path, np.zeros(1000), 16000)

            with mock.patch(
                "subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
            ):
                with pytest.raises(RuntimeError):
                    convert_to_wav(input_path, output_path)


class TestLoadWav:
    def test_load_mono(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            data = np.random.uniform(-0.9, 0.9, 16000).astype(np.float32)
            sf.write(path, data, 16000, subtype="FLOAT")

            loaded, sr = load_wav(path)
            assert sr == 16000
            assert loaded.dtype == np.float32
            np.testing.assert_allclose(loaded, data, atol=1e-6)

    def test_load_stereo_converts_to_mono(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            left = np.random.uniform(-0.9, 0.9, 16000).astype(np.float32)
            right = np.random.uniform(-0.9, 0.9, 16000).astype(np.float32)
            stereo = np.column_stack((left, right))
            sf.write(path, stereo, 16000, subtype="FLOAT")

            loaded, sr = load_wav(path)
            assert sr == 16000
            assert loaded.ndim == 1
            np.testing.assert_allclose(loaded, (left + right) / 2, atol=1e-6)


class TestCropSegment:
    def test_basic_crop(self):
        sr = 16000
        wav = np.random.randn(sr * 2).astype(np.float32)
        result = crop_segment(wav, sr, 0.5, 1.5)
        assert len(result) == sr
        np.testing.assert_array_equal(result, wav[sr // 2 : sr // 2 + sr])

    def test_empty_when_start_gte_end(self):
        wav = np.ones(16000, dtype=np.float32)
        result = crop_segment(wav, 16000, 2.0, 1.0)
        assert result.size == 0

    def test_clamps_to_bounds(self):
        wav = np.ones(16000, dtype=np.float32)
        result = crop_segment(wav, 16000, -1.0, 3.0)
        assert len(result) == 16000
