"""High-level pipeline orchestration.

The concrete model backends live in src.audio, src.diarization, src.embedding,
src.clustering, src.speaker_db, src.asr, src.translation, and src.recognition.
This module wires those stages without owning their implementation details.
"""

from __future__ import annotations

import os
from dataclasses import replace

from config import CENTROID_THRESHOLD, MIN_CLUSTER_SIZE, UPDATE_MIN_DURATION
from src.clustering.cluster import (
    assign_global_speakers,
    assign_global_speakers_from_slr80_filename,
)
from src.clustering.pool import build_embedding_pool
from src.core.repository import SpeakerRepository
from src.core.types import SpeakerSegment
from src.speaker_db.vector_index import FaissVectorIndex


def build_pipeline(
    embedding_dir: str,
    label_strategy: str = "cluster",
    repo: SpeakerRepository | None = None,
) -> SpeakerRepository:
    """Build a speaker repository from cached embeddings."""
    pool = build_embedding_pool(embedding_dir)
    if label_strategy == "cluster":
        clustered_pool = assign_global_speakers(
            pool, centroid_threshold=CENTROID_THRESHOLD, min_cluster_size=MIN_CLUSTER_SIZE
        )
    elif label_strategy == "slr80_filename":
        clustered_pool = assign_global_speakers_from_slr80_filename(pool)
    else:
        raise ValueError(f"Unknown label_strategy: {label_strategy}")
    target_repo = repo or SpeakerRepository(vector_index=FaissVectorIndex())
    target_repo.build_from_pool(clustered_pool)
    return target_repo


def recognize_pipeline(
    wav_path: str,
    repo: SpeakerRepository,
    extractor,
    translator=None,
    asr_model: str = "base",
    hf_token: str | None = None,
    device: str | None = None,
    asr_backend: str = "uniasr",
) -> list[SpeakerSegment]:
    """Run full recognition pipeline on a new audio file.

    Stages:
      1. Diarization → speaker time segments
      2. Embedding extraction + speaker identification
      3. ASR (Whisper or UniASR) → transcribed segments
      4. Align ASR segments to diarization speakers
      5. Map text into diarization segments and translate

    Args:
        asr_backend: ``"whisper"`` or ``"uniasr"``.

    Returns:
        List of SpeakerSegment with global_speaker, text, and translation.
    """
    from config import MIN_DURATION, MIN_DURATION_OFF, MIN_DURATION_ON
    from src.asr.align import align_segments
    from src.diarization.postprocess import annotation_to_segments, merge_short_segments
    from src.diarization.segment import run_diarization

    annotation = run_diarization(wav_path, token=hf_token, device=device)
    merged = merge_short_segments(
        annotation,
        min_duration_on=MIN_DURATION_ON,
        min_duration_off=MIN_DURATION_OFF,
    )
    diar_segments = annotation_to_segments(merged, min_duration=MIN_DURATION)
    if not diar_segments:
        return []

    basename = os.path.splitext(os.path.basename(wav_path))[0]
    diar_segments = [
        replace(seg, segment_id=f"{basename}_{i:04d}", file=basename)
        for i, seg in enumerate(diar_segments)
    ]

    diar_segments = extractor.extract_segments(wav_path, diar_segments)

    identified_segments: list[SpeakerSegment] = []
    for seg in diar_segments:
        if seg.local_speaker == "IGNORE":
            continue
        if seg.embedding is not None:
            result = repo.identify(seg.embedding)
            if result.speaker:
                seg = seg.with_global_speaker(result.speaker)
                if result.confidence == "high":
                    repo.update_speaker(result.speaker, seg.embedding, seg.duration)
            elif result.confidence != "high":
                if seg.duration < UPDATE_MIN_DURATION:
                    seg = seg.with_global_speaker("UNKNOWN")
                else:
                    new_spk_id = _next_spk_id(repo)
                    repo.add_speaker(
                        spk_id=new_spk_id,
                        embeddings=[seg.embedding],
                        durations=[seg.duration],
                    )
                    seg = seg.with_global_speaker(new_spk_id)
        identified_segments.append(seg)

    if asr_backend == "uniasr":
        from src.asr.uniasr_asr import transcribe_uniasr

        asr_segments = transcribe_uniasr(wav_path)
    else:
        from src.asr.whisper_asr import transcribe

        asr_segments = transcribe(wav_path, language="my", model_name=asr_model)
    aligned_asr = align_segments(asr_segments, merged)

    result_segments: list[SpeakerSegment] = []
    for diar_seg in identified_segments:
        texts = [
            ws.text.strip()
            for ws in aligned_asr
            if ws.end > diar_seg.start
            and ws.start < diar_seg.end
            and ws.local_speaker == diar_seg.local_speaker
            and ws.text
        ]
        text = " ".join(texts).strip()
        translation = None
        if text and translator is not None:
            try:
                translation = translator.translate(text)
            except Exception:
                pass
        seg = diar_seg.with_text(text, translation)
        if diar_seg.global_speaker:
            speaker = repo.get_speaker(diar_seg.global_speaker)
            if speaker and speaker.profile:
                seg = seg.with_display_name(speaker.profile.name)
        result_segments.append(seg)

    return result_segments


def _next_spk_id(repo: SpeakerRepository) -> str:
    """Generate the next sequential SPK_ id from existing speakers."""
    max_id = -1
    for spk_id in repo.all_speakers():
        if spk_id.startswith("SPK_"):
            try:
                num = int(spk_id.split("_", 1)[1])
                if num > max_id:
                    max_id = num
            except ValueError:
                continue
    return f"SPK_{max_id + 1}"
