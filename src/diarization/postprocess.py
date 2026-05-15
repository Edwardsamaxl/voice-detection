"""Post-processing utilities for diarization output."""

from pyannote.core import Annotation, Segment


def rename_labels(
    annotation: Annotation,
    label_mapping: dict[str, str] | None = None,
) -> Annotation:
    """Rename speaker labels in a pyannote Annotation.

    Args:
        annotation: Original pyannote Annotation.
        label_mapping: Dict mapping old labels to new labels.
            If None, returns annotation unchanged.

    Returns:
        New Annotation with renamed labels.
    """
    if not label_mapping:
        return annotation
    return annotation.rename_labels(label_mapping, generator="iterable")


def merge_short_segments(
    annotation: Annotation,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.2,
) -> Annotation:
    """Merge short segments and gaps in a diarization annotation.

    Gaps shorter than min_duration_off between same-speaker segments are filled.
    Segments shorter than min_duration_on are dropped (not merged across speakers).
    """
    if not annotation:
        return annotation

    items = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        items.append((segment.start, segment.end, speaker))

    if not items:
        return annotation

    items.sort(key=lambda x: x[0])

    # Step 1: merge same-speaker gaps
    merged = [items[0]]
    for start, end, speaker in items[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        gap = start - prev_end
        if speaker == prev_speaker and gap <= min_duration_off:
            merged[-1] = (prev_start, max(prev_end, end), prev_speaker)
        else:
            merged.append((start, end, speaker))

    # Step 2: drop short segments
    filtered = [
        (start, end, speaker)
        for start, end, speaker in merged
        if (end - start) >= min_duration_on
    ]

    result = Annotation()
    for start, end, speaker in filtered:
        result[Segment(start, end)] = speaker

    return result


def annotation_to_segments(
    annotation: Annotation,
    min_duration: float = 1.0,
) -> list[dict]:
    """Convert a diarization Annotation to data_store segment format.

    Each segment dict includes placeholder fields for downstream pipeline
    stages: global_speaker, display_name, embedding, text, translation.
    """
    segments = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        duration = segment.end - segment.start
        if duration >= min_duration:
            segments.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "duration": round(duration, 3),
                    "local_speaker": speaker,
                    "global_speaker": None,
                    "display_name": None,
                    "embedding": None,
                    "text": None,
                    "translation": None,
                }
            )

    segments.sort(key=lambda x: x["start"])
    return segments
