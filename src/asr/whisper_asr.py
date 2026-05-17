"""Whisper ASR backend."""

from __future__ import annotations

from config import MODELS_DIR
from src.core.types import SpeakerSegment


def transcribe(
    wav_path: str,
    language: str = "my",
    model_name: str = "base",
    download_root: str | None = None,
) -> list[SpeakerSegment]:
    """Transcribe audio with Whisper and return segments as V2 SpeakerSegments.

    Each Whisper segment is mapped to a SpeakerSegment with:
    - segment_id: whisper_<index>
    - file: wav_path
    - local_speaker: "UNKNOWN" (to be filled by alignment)
    - text: transcribed text
    """
    try:
        import whisper
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import openai-whisper: {exc}") from exc

    model = whisper.load_model(
        model_name,
        download_root=download_root or f"{MODELS_DIR}/whisper",
    )
    raw = model.transcribe(wav_path, language=language)

    segments: list[SpeakerSegment] = []
    for i, seg in enumerate(raw.get("segments", [])):
        segments.append(
            SpeakerSegment(
                segment_id=f"whisper_{i:04d}",
                file=wav_path,
                start=round(float(seg.get("start", 0.0)), 3),
                end=round(float(seg.get("end", 0.0)), 3),
                local_speaker="UNKNOWN",
                text=seg.get("text", "").strip(),
            )
        )
    return segments
