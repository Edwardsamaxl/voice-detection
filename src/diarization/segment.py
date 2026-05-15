"""Speaker diarization using pyannote.audio."""

import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation


def run_diarization(
    wav_path: str,
    model_name: str = "pyannote/speaker-diarization-3.1",
    token: Optional[str] = None,
    device: Optional[str] = None,
) -> Annotation:
    """Run speaker diarization on a wav file.

    Args:
        wav_path: Path to 16kHz/16bit/mono wav file.
        model_name: HuggingFace model identifier.
        token: HuggingFace token for gated models.
        device: 'cuda', 'cpu', or None for auto.

    Returns:
        pyannote Annotation with speaker labels.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    cache_dir = str(Path(__file__).resolve().parents[2] / "models" / "pyannote")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        pipeline = Pipeline.from_pretrained(
            model_name, token=token, cache_dir=cache_dir
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load diarization model '{model_name}'. "
            f"Ensure you have accepted the model license on HuggingFace "
            f"and provided a valid token if required. Original error: {e}"
        )

    if pipeline is None:
        raise RuntimeError(f"Model returned None: {model_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    # Load audio via torchaudio to bypass broken torchcodec on Windows
    waveform, sample_rate = torchaudio.load(wav_path)
    return pipeline({"waveform": waveform, "sample_rate": sample_rate})
