"""CLI entry point for voice-detection pipeline.

Subcommands:
    build       Build speaker repository from cached embeddings.
    recognize   Run end-to-end recognition on a new audio file.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR

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


def cmd_build(args: argparse.Namespace) -> None:
    """Build speaker repository from embeddings."""
    from src.core.repository import SpeakerRepository
    from src.core.storage import JsonStorage, NpzStorage
    from src.speaker_db.vector_index import FaissVectorIndex
    from pipeline import build_pipeline, save_repository

    embedding_dir = args.embedding_dir or os.path.join(PROCESSED_DIR, "burmese_asr", "embeddings")
    output_dir = args.output_dir or os.path.join(PROCESSED_DIR, "speaker_db")

    vector_index = FaissVectorIndex()
    repo = SpeakerRepository(
        vector_index=vector_index,
        storage=JsonStorage(output_dir),
        vector_storage=NpzStorage(output_dir),
    )
    repo = build_pipeline(embedding_dir, label_strategy=args.label_strategy, repo=repo)

    save_repository(repo, vector_index, output_dir)

    print(f"Built repository with {len(repo.all_speakers())} speakers at {output_dir}")


def cmd_recognize(args: argparse.Namespace) -> None:
    """Recognize speakers in a new audio file."""
    from src.diarization.segment import _load_pipeline_class
    from src.embedding.extractor import EmbeddingExtractor
    from src.translation.translator import Translator
    from pipeline import run_recognition

    # pyannote must be imported before SpeechBrain on this Windows CPU stack,
    # otherwise SpeechBrain lazy imports can trip pyannote/torchvision loading.
    _load_pipeline_class()

    repo_dir = args.repo_dir or os.path.join(PROCESSED_DIR, "speaker_db")

    extractor = EmbeddingExtractor(device=args.device)
    translator = Translator(device=args.device) if not args.no_translate else None

    result = run_recognition(
        input_path=args.input,
        repo_dir=repo_dir,
        extractor=extractor,
        translator=translator,
        asr_model=args.asr_model,
        hf_token=args.hf_token,
        device=args.device,
        asr_backend=args.asr_backend,
    )

    full_text = " ".join(seg.get("text", "").strip() for seg in result if seg.get("text")).strip()
    ground_truth = _load_ground_truth_text(args.input)
    cer = _compute_cer(full_text, ground_truth)
    if cer is not None:
        precision = round(max(0.0, 1.0 - cer), 4)
        for seg in result:
            seg["precision"] = precision

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Result saved to {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build speaker repository from embeddings")
    build_parser.add_argument("--embedding_dir", default=None, help="Directory with .npz embedding files")
    build_parser.add_argument("--output_dir", default=None, help="Output directory for speaker_db")
    build_parser.add_argument(
        "--label_strategy",
        default="cluster",
        choices=["cluster", "slr80_filename"],
        help="How to assign global speaker labels before building the repository",
    )

    rec_parser = subparsers.add_parser("recognize", help="Recognize speakers in a new audio file")
    rec_parser.add_argument("input", help="Input audio file path")
    rec_parser.add_argument("--repo_dir", default=None, help="Speaker repository directory")
    rec_parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    rec_parser.add_argument("--device", default=None, help="Device for models (cpu/cuda)")
    rec_parser.add_argument("--asr_backend", default="uniasr", choices=["uniasr", "whisper"], help="ASR backend")
    rec_parser.add_argument("--asr_model", default="base", help="Whisper model name (only used when asr_backend=whisper)")
    rec_parser.add_argument("--hf_token", default=None, help="HuggingFace token for gated models")
    rec_parser.add_argument("--no_translate", action="store_true", help="Skip translation")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "recognize":
        cmd_recognize(args)


if __name__ == "__main__":
    main()
