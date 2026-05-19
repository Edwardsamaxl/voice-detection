"""Evaluate an XLS-R CTC checkpoint on SLR80 Burmese ASR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.text_normalize import (  # noqa: E402
    edit_distance,
    normalize_burmese_asr_text,
    normalize_for_cer,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "burmese_asr"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "asr_xlsr_ctc" / "best"


@dataclass
class EvalResult:
    audio: str
    ref: str
    hyp: str
    cer: float
    precision: float
    cer_with_spaces: float
    ref_chars: int
    seconds: float


def load_rows(data_dir: Path, split: str, limit: int | None = None) -> list[dict[str, str]]:
    csv_path = data_dir / f"speech_asr_slr80_my_{split}sets.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit is not None else rows


def read_audio_16k_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != 16000:
        gcd = math.gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // gcd, sr // gcd).astype(np.float32)
    return mono.astype(np.float32, copy=False)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def batch_items(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    rows = load_rows(data_dir, args.split, args.limit)
    results: list[EvalResult] = []
    total_err = 0
    total_chars = 0
    total_err_with_spaces = 0
    total_chars_with_spaces = 0
    start_all = time.perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(batch_items(rows, args.batch_size), start=1):
            batch_start = time.perf_counter()
            audios = [read_audio_16k_mono(data_dir / row["Audio:FILE"]) for row in batch]
            inputs = processor(
                audios,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyps = processor.batch_decode(pred_ids)
            seconds = time.perf_counter() - batch_start

            for row, hyp in zip(batch, hyps):
                ref = normalize_burmese_asr_text(row["Text:LABEL"])
                hyp = normalize_burmese_asr_text(hyp)
                ref_metric = list(normalize_for_cer(ref, remove_spaces=True))
                hyp_metric = list(normalize_for_cer(hyp, remove_spaces=True))
                ref_with_spaces = list(normalize_for_cer(ref, remove_spaces=False))
                hyp_with_spaces = list(normalize_for_cer(hyp, remove_spaces=False))

                err = edit_distance(ref_metric, hyp_metric)
                err_with_spaces = edit_distance(ref_with_spaces, hyp_with_spaces)
                total_err += err
                total_chars += len(ref_metric)
                total_err_with_spaces += err_with_spaces
                total_chars_with_spaces += len(ref_with_spaces)

                cer = err / len(ref_metric) if ref_metric else (0.0 if not hyp_metric else 1.0)
                results.append(
                    EvalResult(
                        audio=row["Audio:FILE"],
                        ref=ref,
                        hyp=hyp,
                        cer=cer,
                        precision=round(1.0 - cer, 4),
                        cer_with_spaces=(
                            err_with_spaces / len(ref_with_spaces)
                            if ref_with_spaces
                            else (0.0 if not hyp_with_spaces else 1.0)
                        ),
                        ref_chars=len(ref_metric),
                        seconds=seconds / max(len(batch), 1),
                    )
                )

            if batch_index == 1 or batch_index % args.progress_every == 0:
                print(
                    f"[{args.split}] {len(results)}/{len(rows)} "
                    f"CER={total_err / max(total_chars, 1):.4f}",
                    flush=True,
                )

    summary = {
        "split": args.split,
        "model_dir": str(model_dir),
        "count": len(results),
        "cer": total_err / max(total_chars, 1),
        "precision": round(1.0 - total_err / max(total_chars, 1), 4),
        "cer_with_spaces": total_err_with_spaces / max(total_chars_with_spaces, 1),
        "total_char_err": total_err,
        "total_chars": total_chars,
        "elapsed_seconds": time.perf_counter() - start_all,
        "examples": [asdict(item) for item in results[: args.examples]],
    }
    output_path = output_dir / f"{args.split}_eval.json"
    output_path.write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(item) for item in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    evaluate(parse_args())
