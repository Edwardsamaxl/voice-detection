"""End-to-end pipeline evaluation on dev set.

Evaluates both speaker identification accuracy and ASR precision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import load_repository, recognize_pipeline
from src.asr.text_normalize import normalize_for_cer
from src.diarization.segment import _load_pipeline_class
from src.embedding.extractor import EmbeddingExtractor
from src.translation.translator import Translator

_SLR80_SPEAKER_RE = re.compile(r"\bbur_(\d+)_")


def load_ground_truth(csv_path: str) -> dict[str, str]:
    """Return {basename: text} from SLR80 CSV."""
    mapping: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[os.path.basename(row["Audio:FILE"])] = row.get("Text:LABEL", "")
    return mapping


def extract_true_speaker(basename: str) -> str | None:
    m = _SLR80_SPEAKER_RE.search(basename)
    return m.group(1) if m else None


def levenshtein(a: str, b: str) -> int:
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


def compute_cer(hyp: str, ref: str) -> float:
    """Compute CER after stripping spaces (Burmese does not use spaces for word boundaries)."""
    ref_norm = normalize_for_cer(ref, remove_spaces=True)
    hyp_norm = normalize_for_cer(hyp, remove_spaces=True)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    if not hyp_norm:
        return 1.0
    return levenshtein(hyp_norm, ref_norm) / len(ref_norm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full pipeline on dev set")
    parser.add_argument("--dev_dir", default="data/raw/burmese_asr/dev")
    parser.add_argument("--repo_dir", default="data/processed/speaker_db_full")
    parser.add_argument("--csv", default="data/raw/burmese_asr/speech_asr_slr80_my_devsets.csv")
    parser.add_argument("--output_dir", default="data/output/eval_results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--asr_backend", default="uniasr", choices=["uniasr", "whisper"])
    parser.add_argument("--limit", type=int, default=None, help="Max files to evaluate")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--no_translate", action="store_true", help="Skip translation")
    args = parser.parse_args()

    _load_pipeline_class()

    repo, _vector_index = load_repository(args.repo_dir)
    print(f"Loaded repository with {len(repo.all_speakers())} speakers")

    extractor = EmbeddingExtractor(device=args.device)
    translator = Translator(device=args.device) if not args.no_translate else None
    gt_map = load_ground_truth(args.csv)

    wav_files = sorted(
        os.path.join(args.dev_dir, f)
        for f in os.listdir(args.dev_dir)
        if f.lower().endswith(".wav")
    )
    if args.limit:
        wav_files = wav_files[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    per_file: list[dict] = []
    total_segments = 0
    correct_speaker_segments = 0
    total_char_err = 0.0
    total_ref_chars = 0

    for idx, wav_path in enumerate(wav_files, 1):
        basename = os.path.basename(wav_path)
        true_speaker = extract_true_speaker(basename)
        ref_text = gt_map.get(basename, "")

        print(f"[{idx}/{len(wav_files)}] {basename} ...", flush=True)
        t0 = time.perf_counter()
        segments = recognize_pipeline(
            wav_path=wav_path,
            repo=repo,
            extractor=extractor,
            translator=translator,
            asr_backend=args.asr_backend,
            hf_token=args.hf_token,
            source_name=wav_path,
        )
        elapsed = time.perf_counter() - t0

        # Speaker accuracy per segment
        file_correct = 0
        for seg in segments:
            pred = seg.global_speaker or ""
            pred_id = pred.replace("BUR_", "") if pred.startswith("BUR_") else pred
            if pred_id == true_speaker:
                file_correct += 1
                correct_speaker_segments += 1

        # ASR precision per file
        full_hyp = " ".join(s.text or "" for s in segments).strip()
        cer = compute_cer(full_hyp, ref_text)
        precision = round(max(0.0, 1.0 - cer), 4)
        total_char_err += cer * len(ref_text) if ref_text else 0
        total_ref_chars += len(ref_text) if ref_text else 0
        total_segments += len(segments)

        # Write per-file segment JSON (same format as test_pipeline_on_dev.py)
        segment_result = [
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
                "precision": precision,
                "embedding": seg.embedding.tolist() if seg.embedding is not None else None,
            }
            for seg in segments
        ]
        file_json_path = os.path.join(args.output_dir, f"{os.path.splitext(basename)[0]}.json")
        with open(file_json_path, "w", encoding="utf-8") as f:
            json.dump(segment_result, f, ensure_ascii=False, indent=2)

        per_file.append(
            {
                "file": basename,
                "true_speaker": true_speaker,
                "segments_count": len(segments),
                "correct_speaker_segments": file_correct,
                "speaker_accuracy": round(file_correct / len(segments), 4) if segments else 0.0,
                "cer": round(cer, 4),
                "precision": precision,
                "elapsed_seconds": round(elapsed, 2),
                "reference": ref_text,
                "hypothesis": full_hyp,
            }
        )
        print(
            f"  segs={len(segments)} speaker_acc={per_file[-1]['speaker_accuracy']} "
            f"precision={precision} CER={cer:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

    avg_cer = total_char_err / max(total_ref_chars, 1)
    summary = {
        "files_processed": len(per_file),
        "total_segments": total_segments,
        "overall_speaker_accuracy": round(correct_speaker_segments / max(total_segments, 1), 4),
        "overall_cer": round(avg_cer, 4),
        "overall_precision": round(max(0.0, 1.0 - avg_cer), 4),
        "per_file": per_file,
    }

    out_path = os.path.join(args.output_dir, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Files processed : {summary['files_processed']}")
    print(f"Total segments  : {summary['total_segments']}")
    print(f"Speaker accuracy: {summary['overall_speaker_accuracy']}")
    print(f"Overall CER     : {summary['overall_cer']}")
    print(f"Overall precision: {summary['overall_precision']}")
    print(f"Report saved to : {out_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
