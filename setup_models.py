"""Download and prepare all required models into the local models/ directory."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is on path so config can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "pyannote": {
        "pipeline": "pyannote/speaker-diarization-3.1",
        "segmentation": "pyannote/segmentation-3.0",
        "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
    },
    "speechbrain": {
        "encoder": "speechbrain/spkrec-ecapa-voxceleb",
    },
    "whisper": {
        "base": "base",
    },
}

RETRIES = 3
DELAY_SECONDS = 5


def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(text)
    print("=" * 60)


def _already_present(path: Path, min_files: int = 1) -> bool:
    """Heuristic: directory exists and contains at least min_files."""
    if not path.exists():
        return False
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files) >= min_files


def setup_pyannote(cache_dir: Path) -> None:
    """Download pyannote pipeline and its sub-models."""
    from pyannote.audio import Pipeline

    cache_dir.mkdir(parents=True, exist_ok=True)

    for name, model_id in MODELS["pyannote"].items():
        target = cache_dir / name.replace("/", "--")
        if _already_present(target, min_files=1):
            print(f"[pyannote] {name}: already cached at {target}")
            continue

        print(f"[pyannote] Downloading {model_id} ...")
        for attempt in range(1, RETRIES + 1):
            try:
                # cache_dir here is passed to huggingface_hub
                _ = Pipeline.from_pretrained(model_id, cache_dir=str(cache_dir))
                print(f"[pyannote] {name}: done")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[pyannote] {name}: attempt {attempt}/{RETRIES} failed: {exc}")
                if attempt == RETRIES:
                    raise RuntimeError(
                        f"Failed to download {model_id} after {RETRIES} attempts"
                    ) from exc
                time.sleep(DELAY_SECONDS)


def setup_speechbrain(savedir: Path) -> None:
    """Download SpeechBrain ECAPA encoder."""
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    savedir.mkdir(parents=True, exist_ok=True)

    model_id = MODELS["speechbrain"]["encoder"]
    if _already_present(savedir, min_files=3):
        print(f"[speechbrain] encoder: already cached at {savedir}")
        return

    print(f"[speechbrain] Downloading {model_id} ...")
    for attempt in range(1, RETRIES + 1):
        try:
            _ = EncoderClassifier.from_hparams(
                source=model_id,
                savedir=str(savedir),
                local_strategy=LocalStrategy.COPY,
            )
            print("[speechbrain] encoder: done")
            break
        except Exception as exc:  # noqa: BLE001
            print(
                f"[speechbrain] encoder: attempt {attempt}/{RETRIES} failed: {exc}"
            )
            if attempt == RETRIES:
                raise RuntimeError(
                    f"Failed to download {model_id} after {RETRIES} attempts"
                ) from exc
            time.sleep(DELAY_SECONDS)


def setup_whisper(model_dir: Path) -> None:
    """Download OpenAI Whisper model weights."""
    import whisper

    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODELS["whisper"]["base"]
    # Whisper downloads to ~/.cache/whisper/ by default; we cannot easily
    # override that, but we can download once and symlink/copy if desired.
    # For simplicity we just trigger the download and let the user know.
    cache_root = Path.home() / ".cache" / "whisper"
    expected = cache_root / f"{model_name}.pt"

    if expected.exists():
        print(f"[whisper] {model_name}: already cached at {expected}")
        return

    print(f"[whisper] Downloading {model_name} ...")
    for attempt in range(1, RETRIES + 1):
        try:
            _ = whisper.load_model(model_name)
            print("[whisper] base: done")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[whisper] base: attempt {attempt}/{RETRIES} failed: {exc}")
            if attempt == RETRIES:
                raise RuntimeError(
                    f"Failed to download whisper {model_name} after {RETRIES} attempts"
                ) from exc
            time.sleep(DELAY_SECONDS)


def main() -> int:
    """Entry point."""
    _banner("Voice-Detection Model Setup")

    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Keep directory in git even though contents are ignored
    gitkeep = models_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    errors: list[str] = []

    # 1. pyannote
    try:
        setup_pyannote(models_dir / "pyannote")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pyannote: {exc}")

    # 2. speechbrain
    try:
        setup_speechbrain(models_dir / "speechbrain")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"speechbrain: {exc}")

    # 3. whisper
    try:
        setup_whisper(models_dir / "whisper")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"whisper: {exc}")

    _banner("Summary")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("All models are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
