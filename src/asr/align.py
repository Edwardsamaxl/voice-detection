"""Align ASR segments to diarization speakers."""

from __future__ import annotations

from config import MIN_ASR_SEGMENT
from src.core.types import SpeakerSegment


def align_segments(
    whisper_segments: list[SpeakerSegment],
    diarization_annotation,
) -> list[SpeakerSegment]:
    """Assign a diarization speaker label to each Whisper segment.

    Returns new SpeakerSegment instances with local_speaker set to the
    dominant speaker from the diarization annotation, or "UNKNOWN" / "IGNORE".
    """
    from pyannote.core import Segment

    aligned: list[SpeakerSegment] = []
    for seg in whisper_segments:
        start, end = seg.start, seg.end
        if end - start < MIN_ASR_SEGMENT:
            speaker = "IGNORE"
        else:
            cropped = diarization_annotation.crop(Segment(start, end))
            speaker = "UNKNOWN" if len(cropped) == 0 else str(cropped.argmax())

        aligned.append(seg.with_local_speaker(speaker))
    return aligned
