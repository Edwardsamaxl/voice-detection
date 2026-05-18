"""High-level pipeline orchestration.

The concrete model backends live in src.audio, src.diarization, src.embedding,
src.clustering, src.speaker_db, src.asr, src.translation, and src.recognition.
This module wires those stages without owning their implementation details.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import replace

from config import CENTROID_THRESHOLD, UPDATE_MIN_DURATION
from src.clustering.cluster import (
    assign_global_speakers,
    assign_global_speakers_from_slr80_filename,
)
from src.clustering.pool import build_embedding_pool
from src.core.repository import SpeakerRepository
from src.core.storage import JsonStorage, NpzStorage, PickleStorage
from src.core.types import SpeakerSegment
from src.speaker_db.vector_index import FaissVectorIndex


_GROUND_TRUTH_CSVS = [
    os.path.join("data", "raw", "burmese_asr", "speech_asr_slr80_my_trainsets.csv"),
    os.path.join("data", "raw", "burmese_asr", "speech_asr_slr80_my_devsets.csv"),
    os.path.join("data", "raw", "burmese_asr", "speech_asr_slr80_my_testsets.csv"),
]


def _load_ground_truth_text(audio_path: str) -> str | None:
    """Look up reference text from SLR80 CSV files by basename."""
    basename = os.path.basename(audio_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for csv_name in _GROUND_TRUTH_CSVS:
        csv_path = os.path.join(base_dir, csv_name)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if os.path.basename(row.get("Audio:FILE", "")) == basename:
                    return row.get("Text:LABEL")
    return None


def _levenshtein_distance(a: str, b: str) -> int:
    """Wagner-Fischer algorithm for edit distance on Unicode code points."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def _compute_cer(hypothesis: str | None, reference: str | None) -> float | None:
    """Compute Character Error Rate. Returns 0.0~1.0+ or None if inputs missing."""
    if not hypothesis or not reference:
        return None
    dist = _levenshtein_distance(hypothesis, reference)
    return dist / len(reference)


def build_pipeline(
    embedding_dir: str,
    label_strategy: str = "cluster",
    repo: SpeakerRepository | None = None,
) -> SpeakerRepository:
    """Build a speaker repository from cached embeddings."""
    pool = build_embedding_pool(embedding_dir)
    if label_strategy == "cluster":
        clustered_pool = assign_global_speakers(
            pool, centroid_threshold=CENTROID_THRESHOLD
        )
    elif label_strategy == "slr80_filename":
        clustered_pool = assign_global_speakers_from_slr80_filename(pool)
    else:
        raise ValueError(f"Unknown label_strategy: {label_strategy}")
    target_repo = repo or SpeakerRepository(vector_index=FaissVectorIndex())
    target_repo.build_from_pool(clustered_pool)
    return target_repo


def save_repository(
    repo: SpeakerRepository,
    vector_index: FaissVectorIndex,
    output_dir: str,
) -> None:
    """Persist speaker repository and FAISS vector index."""
    os.makedirs(output_dir, exist_ok=True)
    repo.save("speaker_db:main", "vector_db:main")
    pickle_storage = PickleStorage(output_dir)
    pickle_storage.save(
        "vector_index:main",
        {
            "index": vector_index.index,
            "vectors": vector_index.vectors,
            "labels": vector_index.labels,
        },
    )


def load_repository(repo_dir: str) -> tuple[SpeakerRepository, FaissVectorIndex]:
    """Load speaker repository and restore FAISS vector index if available."""
    vector_index = FaissVectorIndex()
    repo = SpeakerRepository(
        vector_index=vector_index,
        storage=JsonStorage(repo_dir),
        vector_storage=NpzStorage(repo_dir),
    )
    repo.load("speaker_db:main", "vector_db:main")

    pickle_storage = PickleStorage(repo_dir)
    if pickle_storage.exists("vector_index:main"):
        data = pickle_storage.load("vector_index:main")
        if isinstance(data, dict) and "index" in data:
            vector_index.index = data["index"]
            vector_index.vectors = data["vectors"]
            vector_index.labels = data["labels"]
            if vector_index.vectors.size > 0:
                vector_index.dim = int(vector_index.vectors.shape[1])
        else:
            # Legacy format compatibility (vectors + labels only)
            vector_index.build(data["vectors"], data["labels"])
    else:
        repo.rebuild()

    return repo, vector_index


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
        replace(seg, segment_id=f"{basename}_{i:04d}", file=basename, sr=16000)
        for i, seg in enumerate(diar_segments)
    ]

    diar_segments = extractor.extract_segments(wav_path, diar_segments)

    identified_segments: list[SpeakerSegment] = []
    for seg in diar_segments:
        if seg.local_speaker == "IGNORE":
            continue
        if seg.embedding is not None:
            result = repo.identify(seg.embedding)
            seg = seg.with_score(result.score)
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


def run_recognition(
    input_path: str,
    repo_dir: str,
    extractor,
    translator=None,
    asr_model: str = "base",
    hf_token: str | None = None,
    device: str | None = None,
    asr_backend: str = "uniasr",
) -> list[dict]:
    """End-to-end recognition with temp wav conversion, CER, and JSON assembly.

    Returns:
        List of dicts ready for JSON serialization.
    """
    from src.audio.preprocess import convert_to_wav

    wav_path = input_path
    temp_wav = None
    if not input_path.lower().endswith(".wav"):
        temp_wav = tempfile.mktemp(suffix=".wav")
        convert_to_wav(input_path, temp_wav)
        wav_path = temp_wav

    try:
        repo, _vector_index = load_repository(repo_dir)

        segments = recognize_pipeline(
            wav_path=wav_path,
            repo=repo,
            extractor=extractor,
            translator=translator,
            asr_model=asr_model,
            hf_token=hf_token,
            device=device,
            asr_backend=asr_backend,
        )

        full_text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        ground_truth = _load_ground_truth_text(input_path)
        cer = _compute_cer(full_text, ground_truth)
        precision = round(max(0.0, 1.0 - cer), 4) if cer is not None else None

        result = [
            {
                "segment_id": seg.segment_id,
                "file": seg.file,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "duration": round(seg.duration, 3),
                "sr": seg.sr,
                "local_speaker": seg.local_speaker,
                "global_speaker": seg.global_speaker or seg.local_speaker or "UNKNOWN",
                "display_name": seg.display_name,
                "score": round(seg.score, 4) if seg.score is not None else None,
                "text": seg.text,
                "translation": seg.translation,
                "precision": precision,
                "embedding": seg.embedding.tolist() if seg.embedding is not None else None,
            }
            for seg in segments
        ]
        return result
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)


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
