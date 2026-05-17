"""Environment health checks for the voice-detection project."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _parse_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    core = version.split("+", 1)[0]
    parts = core.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    py = sys.version_info
    if py.major == 3 and py.minor == 11:
        _ok(f"Python {py.major}.{py.minor}.{py.micro}")
    else:
        msg = f"Python 3.11 is expected, found {py.major}.{py.minor}.{py.micro}"
        failures.append(msg)
        _fail(msg)

    in_venv = sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))
    if in_venv:
        _ok(f"Virtual environment: {sys.prefix}")
    else:
        msg = "Not running inside a virtual environment"
        warnings.append(msg)
        _warn(msg)

    if shutil.which("ffmpeg"):
        _ok("ffmpeg is on PATH")
    else:
        msg = "ffmpeg is not on PATH"
        failures.append(msg)
        _fail(msg)

    versions = {
        "torch": _version("torch"),
        "torchaudio": _version("torchaudio"),
        "numpy": _version("numpy"),
        "pyannote.audio": _version("pyannote.audio"),
        "pyannote.core": _version("pyannote.core"),
        "pyannote.metrics": _version("pyannote.metrics"),
        "speechbrain": _version("speechbrain"),
        "scikit-learn": _version("scikit-learn"),
        "faiss-cpu": _version("faiss-cpu"),
        "openai-whisper": _version("openai-whisper"),
    }

    for package, version in versions.items():
        if version:
            _ok(f"{package} {version}")
        else:
            msg = f"{package} is not installed"
            failures.append(msg)
            _fail(msg)

    torch_mm = _parse_major_minor(versions["torch"])
    torchaudio_mm = _parse_major_minor(versions["torchaudio"])
    if torch_mm and torchaudio_mm and torch_mm == torchaudio_mm:
        _ok("torch and torchaudio major/minor versions match")
    else:
        msg = (
            "torch and torchaudio must be installed as a matching pair "
            f"(found torch={versions['torch']}, torchaudio={versions['torchaudio']})"
        )
        failures.append(msg)
        _fail(msg)

    numpy_mm = _parse_major_minor(versions["numpy"])
    if numpy_mm and numpy_mm[0] < 2:
        _ok("numpy is pinned below 2.0")
    else:
        msg = f"numpy must be <2.0 for this Windows CPU lock (found {versions['numpy']})"
        failures.append(msg)
        _fail(msg)

    pyannote_core_mm = _parse_major_minor(versions["pyannote.core"])
    pyannote_metrics_mm = _parse_major_minor(versions["pyannote.metrics"])
    if pyannote_core_mm and pyannote_core_mm[0] < 6:
        _ok("pyannote.core is constrained below 6")
    else:
        msg = f"pyannote.core must be <6 (found {versions['pyannote.core']})"
        failures.append(msg)
        _fail(msg)

    if pyannote_metrics_mm and pyannote_metrics_mm[0] < 4:
        _ok("pyannote.metrics is constrained below 4")
    else:
        msg = f"pyannote.metrics must be <4 (found {versions['pyannote.metrics']})"
        failures.append(msg)
        _fail(msg)

    if os.environ.get("HF_TOKEN"):
        _ok("HF_TOKEN is set")
    else:
        msg = "HF_TOKEN is not set; real pyannote model loading may fail"
        warnings.append(msg)
        _warn(msg)

    MODELS_DIR.mkdir(exist_ok=True)
    _ok(f"models directory exists: {MODELS_DIR}")

    print()
    print(f"Doctor finished with {len(failures)} failure(s), {len(warnings)} warning(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
