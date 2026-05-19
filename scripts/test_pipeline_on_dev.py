"""Smoke-test the full pipeline on a random sample of dev-set audios.

Loads the existing speaker_db, runs recognition on N random dev files,
writes JSON outputs, and **never saves the repository** so no speakers
are permanently added or updated.
"""

import argparse
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import load_repository, recognize_pipeline
from src.diarization.segment import _load_pipeline_class
from src.embedding.extractor import EmbeddingExtractor
from src.translation.translator import Translator


def pick_random_dev_files(dev_dir: str, n: int = 5) -> list[str]:
    files = [
        os.path.join(dev_dir, f)
        for f in os.listdir(dev_dir)
        if f.lower().endswith(".wav")
    ]
    if len(files) <= n:
        return files
    return random.sample(files, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test pipeline on dev samples")
    parser.add_argument("--dev_dir", default="data/raw/burmese_asr/dev", help="Dev audio directory")
    parser.add_argument("--repo_dir", default="data/processed/speaker_db", help="Speaker repository directory")
    parser.add_argument("--output_dir", default="data/processed/test_output", help="Where to write JSON results")
    parser.add_argument("--n", type=int, default=5, help="Number of random dev files to test")
    parser.add_argument("--device", default="cpu", help="Device for models")
    parser.add_argument("--asr_backend", default="uniasr", choices=["uniasr", "whisper"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_translate", action="store_true", help="Skip translation")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # pyannote must be imported before SpeechBrain on this Windows CPU stack
    _load_pipeline_class()

    # Load repository once
    repo, vector_index = load_repository(args.repo_dir)
    original_speaker_count = len(repo.all_speakers())
    print(f"Loaded repository with {original_speaker_count} speakers from {args.repo_dir}")

    extractor = EmbeddingExtractor(device=args.device)
    translator = Translator(device=args.device) if not args.no_translate else None

    files = pick_random_dev_files(args.dev_dir, args.n)
    print(f"Selected {len(files)} files: {[os.path.basename(f) for f in files]}")

    for wav_path in files:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\n>>> Processing {basename} ...")

        # Note: recognize_pipeline mutates the in-memory repo (adds/updates speakers),
        # but we never call save() so disk state is untouched.
        segments = recognize_pipeline(
            wav_path=wav_path,
            repo=repo,
            extractor=extractor,
            translator=translator,
            asr_backend=args.asr_backend,
            source_name=wav_path,
        )

        result = [
            {
                "segment_id": seg.segment_id,
                "file": seg.file,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "duration": round(seg.duration, 3),
                "sr": seg.sr,
                "local_speaker": seg.local_speaker,
                "global_speaker": seg.global_speaker or "UNKNOWN",
                "display_name": seg.display_name,
                "score": round(seg.score, 4) if seg.score is not None else None,
                "text": seg.text,
                "translation": seg.translation,
                "precision": None,
                "embedding": seg.embedding.tolist() if seg.embedding is not None else None,
            }
            for seg in segments
        ]

        out_path = os.path.join(args.output_dir, f"{basename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"    Segments: {len(result)}")
        for seg in result:
            text_preview = (seg.get('text') or '')[:60]
            safe_text = text_preview.encode(sys.stdout.encoding or 'utf-8', 'replace').decode(sys.stdout.encoding or 'utf-8')
            print(f"    [{seg['start']:.1f}-{seg['end']:.1f}] {seg['global_speaker']}: {safe_text}")
        print(f"    Written: {out_path}")

    current_speaker_count = len(repo.all_speakers())
    print(f"\n=== Summary ===")
    print(f"Original speakers in repo: {original_speaker_count}")
    print(f"Speakers in repo after test (in-memory only): {current_speaker_count}")
    print(f"Repository was NOT saved — disk state unchanged.")
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
