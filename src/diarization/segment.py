"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_PIPELINE_CACHE: dict[tuple[str, Optional[str], Optional[str]], Any] = {}

def _patch_hf_hub_download() -> None:
    """Patch huggingface_hub.hf_hub_download to accept use_auth_token.

    pyannote.audio 3.3.2 passes use_auth_token to hf_hub_download, but
    huggingface_hub 1.x only accepts token. We patch both the module-level
    attribute and any already-imported references inside pyannote.audio.
    """
    try:
        import huggingface_hub

        # Prevent double-patching which would cause infinite recursion.
        if getattr(huggingface_hub.hf_hub_download, "_vd_patched", False):
            return

        _orig = huggingface_hub.hf_hub_download

        def _patched(*args: Any, **kwargs: Any) -> Any:
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _orig(*args, **kwargs)

        _patched._vd_patched = True
        huggingface_hub.hf_hub_download = _patched

        # If pyannote.audio was already imported, its internal references
        # point to the old function. Update all of them.
        try:
            import sys

            for mod_name, mod in sys.modules.items():
                if mod_name.startswith("pyannote") and mod is not None:
                    if hasattr(mod, "hf_hub_download"):
                        mod.hf_hub_download = _patched
        except Exception:
            pass
    except Exception:
        pass


def _load_pipeline_class() -> Any:
    """Import pyannote lazily so tests can collect without model dependencies."""
    _patch_hf_hub_download()
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import pyannote.audio. Use a clean Windows CPU venv and "
            "install with constraints-win-cpu.txt. Original error: "
            f"{exc}"
        ) from exc
    return Pipeline


def _load_audio(wav_path: str) -> tuple[Any, int]:
    """Load audio with torchaudio and return waveform/sample_rate for pyannote."""
    try:
        import torchaudio
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import torchaudio. Ensure torch and torchaudio are "
            "installed as a matching pair. Original error: "
            f"{exc}"
        ) from exc

    try:
        return torchaudio.load(wav_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load audio with torchaudio: {exc}") from exc


def _select_device(device: Optional[str]) -> Any:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import torch: {exc}") from exc

    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(selected)


def run_diarization(
    wav_path: str,
    model_name: str = "pyannote/speaker-diarization-3.1",
    token: Optional[str] = None,
    device: Optional[str] = None,
    pipeline=None,
):
    """Run speaker diarization on a wav file.

    Args:
        wav_path: Path to a wav file, preferably standardized to 16kHz mono.
        model_name: HuggingFace model identifier.
        token: HuggingFace token for gated models.
        device: "cuda", "cpu", or None for auto.
        pipeline: Reuse an already-loaded pyannote Pipeline instance.

    Returns:
        pyannote.core.Annotation with speaker labels.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    if pipeline is None:
        cache_key = (model_name, token, device)
        pipeline = _PIPELINE_CACHE.get(cache_key)

        if pipeline is None:
            cache_dir = str(Path(__file__).resolve().parents[2] / "models" / "pyannote")
            os.makedirs(cache_dir, exist_ok=True)

            Pipeline = _load_pipeline_class()
            try:
                pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=token,
                    cache_dir=cache_dir,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load diarization model '{model_name}'. "
                    "Ensure you accepted the HuggingFace license and provided HF_TOKEN "
                    f"when required. Original error: {exc}"
                ) from exc

            if pipeline is None:
                raise RuntimeError(f"Model returned None: {model_name}")

            pipeline.to(_select_device(device))
            _PIPELINE_CACHE[cache_key] = pipeline

    waveform, sample_rate = _load_audio(wav_path)
    return pipeline({"waveform": waveform, "sample_rate": sample_rate})
