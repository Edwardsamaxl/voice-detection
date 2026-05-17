"""UniASR ASR backend via ModelScope pipeline."""

from __future__ import annotations

import os

from config import UNIASR_MODEL_DIR
from src.core.types import SpeakerSegment


def _get_audio_duration(wav_path: str) -> float:
    """Return audio duration in seconds."""
    import soundfile as sf

    info = sf.info(wav_path)
    return info.duration


def transcribe_uniasr(
    wav_path: str,
    model_dir: str | None = None,
) -> list[SpeakerSegment]:
    """Transcribe audio with UniASR and return segments as SpeakerSegments.

    Uses ModelScope's ``pipeline`` with ``Tasks.auto_speech_recognition``.
    If the model output contains per-token timestamps they are mapped to
    multiple segments; otherwise a single segment covering the whole file
    is returned.

    Args:
        wav_path: Path to the input WAV file.
        model_dir: Local directory containing the UniASR model.  Defaults to
            ``config.UNIASR_MODEL_DIR``.

    Returns:
        List of ``SpeakerSegment`` with *local_speaker* set to ``"UNKNOWN"``
        (to be filled by the alignment stage).
    """
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import modelscope: {exc}") from exc

    model_dir = model_dir or UNIASR_MODEL_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"UniASR model directory not found: {model_dir}")

    inference = pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_dir,
    )
    result = inference(wav_path)

    # ModelScope FunASR pipelines return a dict like:
    #   {"text": "...", "text_punc": "...",
    #    "timestamp": [[s1, e1, tok1], [s2, e2, tok2], ...]}
    # or a list of such dicts.
    if isinstance(result, list):
        result = result[0] if result else {}

    raw_text = result.get("text", "") if isinstance(result, dict) else ""
    timestamps = result.get("timestamp") if isinstance(result, dict) else None

    segments: list[SpeakerSegment] = []

    if timestamps and isinstance(timestamps, list) and len(timestamps) > 0:
        for i, ts in enumerate(timestamps):
            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start_ms, end_ms = ts[0], ts[1]
            else:
                continue
            start = round(start_ms / 1000.0, 3)
            end = round(end_ms / 1000.0, 3)
            token_text = str(ts[2]) if len(ts) > 2 else ""
            segments.append(
                SpeakerSegment(
                    segment_id=f"uniasr_{i:04d}",
                    file=wav_path,
                    start=start,
                    end=end,
                    local_speaker="UNKNOWN",
                    text=token_text,
                )
            )
    else:
        duration = _get_audio_duration(wav_path)
        segments.append(
            SpeakerSegment(
                segment_id="uniasr_0000",
                file=wav_path,
                start=0.0,
                end=round(duration, 3),
                local_speaker="UNKNOWN",
                text=str(raw_text).strip(),
            )
        )

    return segments
