"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN."""

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import EMBEDDING_DIM, MODELS_DIR, SAMPLE_RATE


class EmbeddingExtractor:
    """Extract speaker embeddings from audio waveforms using ECAPA-TDNN."""

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | None = None,
        device: str | None = None,
    ):
        """Initialize the ECAPA-TDNN encoder.

        Args:
            model_source: HuggingFace model identifier.
            savedir: Directory to cache downloaded models.
                Defaults to MODELS_DIR / "speechbrain".
            device: "cpu", "cuda", or None for auto-detection.
        """
        if savedir is None:
            savedir = str(Path(MODELS_DIR) / "speechbrain")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = EMBEDDING_DIM

        try:
            self.classifier = self._load_classifier(model_source, savedir)
            self.classifier.eval()
        except ImportError as e:
            raise RuntimeError(
                f"SpeechBrain is not installed. Please install it: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SpeechBrain encoder: {e}")

    def _load_classifier(self, model_source: str, savedir: str) -> Any:
        """Load the SpeechBrain encoder. Extracted for testability."""
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        return EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

    def extract(self, wav_array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extract embedding from a single audio segment.

        Args:
            wav_array: 1D numpy array of audio samples (float32 or int16).
            sr: Sample rate of the audio. Should match the model's expected rate.

        Returns:
            L2-normalized embedding vector.
        """
        if wav_array.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {wav_array.shape}")

        if wav_array.size == 0:
            raise ValueError("Empty audio array")

        if sr != SAMPLE_RATE:
            warnings.warn(
                f"Sample rate {sr} does not match expected {SAMPLE_RATE}. "
                "Consider resampling for best results."
            )

        if wav_array.dtype != np.float32:
            wav_array = wav_array.astype(np.float32)

        # Normalize to [-1, 1] if integer input was passed
        if np.max(np.abs(wav_array)) > 1.0:
            wav_array = wav_array / 32768.0

        waveform = torch.from_numpy(wav_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.classifier.encode_batch(waveform)

        emb = embedding.squeeze().cpu().numpy()

        if emb.shape[0] != self.embedding_dim:
            warnings.warn(
                f"Expected embedding dim {self.embedding_dim}, got {emb.shape[0]}"
            )

        from .normalize import l2_normalize

        return l2_normalize(emb)

    def _resample(
        self, wav_array: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample a 1D audio array using polyphase filtering."""
        if orig_sr == target_sr:
            return wav_array
        import math
        from scipy import signal

        g = math.gcd(orig_sr, target_sr)
        return signal.resample_poly(wav_array, up=target_sr // g, down=orig_sr // g)

    def extract_segments(self, wav_path: str, segments: list[dict]) -> list[dict]:
        """Extract embeddings for a list of segments from a single wav file.

        Args:
            wav_path: Path to the wav file.
            segments: List of segment dicts with 'start' and 'end' keys.

        Returns:
            New list of segments with 'embedding' field populated.
        """
        from src.audio.preprocess import load_wav, crop_segment, rms_normalize

        wav, sr = load_wav(wav_path)

        new_segments: list[dict] = []
        for seg in segments:
            new_seg = dict(seg)
            start = new_seg.get("start", 0.0)
            end = new_seg.get("end", 0.0)
            audio_slice = crop_segment(wav, sr, start, end)
            audio_slice = rms_normalize(audio_slice)

            if audio_slice.size == 0:
                warnings.warn(f"Empty segment {start}-{end}, skipping embedding")
                new_seg["embedding"] = None
            else:
                if sr != SAMPLE_RATE:
                    audio_slice = self._resample(audio_slice, sr, SAMPLE_RATE)
                new_seg["embedding"] = self.extract(audio_slice, sr=SAMPLE_RATE)
            new_segments.append(new_seg)

        return new_segments

    def extract_for_file(
        self,
        data_store: dict[str, list[dict]],
        file_key: str,
        wav_path: str,
    ) -> dict[str, list[dict]]:
        """Extract embeddings for all segments of a single file in data_store.

        Deprecated: Use extract_segments() for new code.

        Args:
            data_store: Unified data store mapping file keys to segment lists.
            file_key: The key in data_store to process (e.g., "file1.wav").
            wav_path: Path to the corresponding wav file.

        Returns:
            New data_store with embeddings populated for the given file_key.
        """
        if file_key not in data_store:
            raise KeyError(f"File key '{file_key}' not found in data_store")

        new_store = dict(data_store)
        new_store[file_key] = self.extract_segments(wav_path, data_store[file_key])
        return new_store
