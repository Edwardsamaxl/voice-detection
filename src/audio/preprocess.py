"""Audio preprocessing utilities."""

import os
import subprocess
import warnings

import numpy as np

from config import SAMPLE_FMT, SAMPLE_RATE, CHANNELS, TARGET_RMS


def convert_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int = SAMPLE_RATE,
    sample_fmt: str = SAMPLE_FMT,
    channels: int = CHANNELS,
) -> str:
    """Convert any audio file to standardized wav format using ffmpeg."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fmt_map = {
        "s16": "pcm_s16le",
        "s32": "pcm_s32le",
        "f32": "pcm_f32le",
    }
    ffmpeg_fmt = fmt_map.get(sample_fmt, "pcm_s16le")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", os.path.abspath(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", ffmpeg_fmt,
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "unknown error"
        raise RuntimeError(f"ffmpeg conversion failed: {stderr_msg}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in PATH.")

    return output_path


def load_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a wav file as a float32 mono numpy array.

    Returns:
        (audio_array, sample_rate)
    """
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def crop_segment(
    wav_array: np.ndarray, sr: int, start: float, end: float
) -> np.ndarray:
    """Crop a segment from a wav array by time range.

    Args:
        wav_array: Full audio array.
        sr: Sample rate.
        start: Start time in seconds.
        end: End time in seconds.

    Returns:
        Cropped 1D numpy array. Empty array if start >= end.
    """
    start_sample = max(0, int(start * sr))
    end_sample = min(len(wav_array), int(end * sr))
    if start_sample >= end_sample:
        return np.array([], dtype=np.float32)
    return wav_array[start_sample:end_sample]


def rms_normalize(wav_array: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Normalize audio array to target RMS level."""
    if wav_array.size == 0:
        warnings.warn("Empty audio array, returning as-is")
        return wav_array

    current_rms = np.sqrt(np.mean(wav_array.astype(np.float64) ** 2))
    if current_rms == 0:
        warnings.warn("Silent audio array, returning as-is")
        return wav_array

    scale = target_rms / current_rms
    normalized = wav_array * scale

    if np.issubdtype(wav_array.dtype, np.integer):
        max_val = np.iinfo(wav_array.dtype).max
        min_val = np.iinfo(wav_array.dtype).min
        normalized = np.clip(normalized, min_val, max_val)

    return normalized.astype(wav_array.dtype)
