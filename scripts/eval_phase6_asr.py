"""Evaluate Phase 6 Burmese ASR on SLR80 CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = (
    PROJECT_ROOT
    / ".codex_realtest"
    / "modelscope"
    / "iic"
    / "speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "burmese_asr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".codex_realtest" / "phase6_asr_eval"


@dataclass
class SampleResult:
    audio: str
    ref: str
    hyp: str
    wer: float
    cer: float
    ref_words: int
    ref_chars: int
    seconds: float


def normalize_text(text: str) -> str:
    """Normalize whitespace only; keep Myanmar characters intact."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text.strip())


def normalize_for_metric(text: str, drop_terminal_dollar: bool = False) -> str:
    text = normalize_text(text)
    if drop_terminal_dollar:
        text = re.sub(r"(?:\s+)?ဒေါ်လာ\s*$", "", text).strip()
        text = re.sub(r"(?:\s+)?လာ\s*$", "", text).strip()
    return text


def ensure_16k_mono(wav_path: Path, data_dir: Path, cache_dir: Path) -> Path:
    rel = wav_path.relative_to(data_dir)
    out_path = cache_dir / rel
    if out_path.exists():
        return out_path

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != 16000:
        gcd = int(np.gcd(sr, 16000))
        mono = resample_poly(mono, 16000 // gcd, sr // gcd).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mono, 16000, subtype="PCM_16")
    return out_path


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r_token in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h_token in enumerate(hyp, start=1):
            cost = 0 if r_token == h_token else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def word_tokens(text: str) -> list[str]:
    text = normalize_text(text)
    return text.split() if text else []


def char_tokens(text: str) -> list[str]:
    return [ch for ch in normalize_text(text).replace(" ", "")]


def error_rate(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def extract_text(result: Any) -> str:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)


def load_rows(data_dir: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    csv_path = data_dir / f"speech_asr_slr80_my_{split}sets.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit is not None else rows


def select_rows(
    rows: list[dict[str, str]],
    sample_size: int | None,
    seed: int,
) -> list[tuple[int, dict[str, str]]]:
    indexed = list(enumerate(rows))
    if sample_size is None or sample_size >= len(indexed):
        return indexed
    rng = random.Random(seed)
    sampled = rng.sample(indexed, sample_size)
    return sorted(sampled, key=lambda item: item[0])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.ignore_predictor_length:
        from funasr.models.uniasr.beam_search import BeamSearchScama

        original_forward = BeamSearchScama.forward

        def forward_without_predictor_length(self, *f_args, **f_kwargs):
            f_kwargs["maxlen"] = None
            f_kwargs["minlen"] = 0
            f_kwargs["maxlenratio"] = 0.0
            f_kwargs["minlenratio"] = 0.0
            return original_forward(self, *f_args, **f_kwargs)

        BeamSearchScama.forward = forward_without_predictor_length

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav16_cache = output_dir / "wav16k"

    print(f"loading model: {model_path}", flush=True)
    asr = pipeline(
        task=Tasks.auto_speech_recognition,
        model=str(model_path),
        device=args.device,
    )

    all_rows = load_rows(data_dir, args.split, args.limit)
    rows = select_rows(all_rows, args.sample_size, args.seed)
    results: list[SampleResult] = []
    total_word_err = total_words = 0
    total_char_err = total_chars = 0
    start_all = time.perf_counter()

    csv_rows: list[dict[str, Any]] = []

    for idx, (source_row, row) in enumerate(rows, start=1):
        rel_audio = row["Audio:FILE"]
        wav_path = data_dir / rel_audio
        infer_path = ensure_16k_mono(wav_path, data_dir, wav16_cache) if args.prepare_16k else wav_path
        ref = normalize_for_metric(
            row["Text:LABEL"],
            drop_terminal_dollar=args.drop_terminal_dollar,
        )

        start = time.perf_counter()
        param_dict = {
            "decoding_model": args.decoding_model,
            "token_num_relax": args.token_num_relax,
            "beam_size": args.beam_size,
            "penalty": args.penalty,
        }
        raw = asr(str(infer_path), param_dict=param_dict)
        seconds = time.perf_counter() - start
        hyp = normalize_for_metric(
            extract_text(raw),
            drop_terminal_dollar=args.drop_terminal_dollar,
        )

        ref_w = word_tokens(ref)
        hyp_w = word_tokens(hyp)
        ref_c = char_tokens(ref)
        hyp_c = char_tokens(hyp)
        word_err = edit_distance(ref_w, hyp_w)
        char_err = edit_distance(ref_c, hyp_c)

        total_word_err += word_err
        total_words += len(ref_w)
        total_char_err += char_err
        total_chars += len(ref_c)

        sample = SampleResult(
            audio=rel_audio,
            ref=ref,
            hyp=hyp,
            wer=word_err / len(ref_w) if ref_w else (0.0 if not hyp_w else 1.0),
            cer=char_err / len(ref_c) if ref_c else (0.0 if not hyp_c else 1.0),
            ref_words=len(ref_w),
            ref_chars=len(ref_c),
            seconds=seconds,
        )
        results.append(sample)
        csv_rows.append(
            {
                "source_row": source_row,
                "audio": sample.audio,
                "wer": sample.wer,
                "cer": sample.cer,
                "ref_words": sample.ref_words,
                "ref_chars": sample.ref_chars,
                "seconds": sample.seconds,
                "ref": sample.ref,
                "hyp": sample.hyp,
            }
        )

        if idx == 1 or idx % args.progress_every == 0 or idx == len(rows):
            print(
                f"[{args.split}] {idx}/{len(rows)} "
                f"WER={total_word_err / max(total_words, 1):.4f} "
                f"CER={total_char_err / max(total_chars, 1):.4f} "
                f"last={seconds:.2f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - start_all
    summary = {
        "split": args.split,
        "model": str(model_path),
        "decoding_model": args.decoding_model,
        "prepare_16k": args.prepare_16k,
        "drop_terminal_dollar": args.drop_terminal_dollar,
        "token_num_relax": args.token_num_relax,
        "beam_size": args.beam_size,
        "penalty": args.penalty,
        "ignore_predictor_length": args.ignore_predictor_length,
        "source_count": len(all_rows),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "count": len(results),
        "wer": total_word_err / max(total_words, 1),
        "cer": total_char_err / max(total_chars, 1),
        "total_word_err": total_word_err,
        "total_words": total_words,
        "total_char_err": total_char_err,
        "total_chars": total_chars,
        "elapsed_seconds": elapsed,
        "avg_seconds_per_file": elapsed / max(len(results), 1),
        "examples": [asdict(item) for item in results[: args.examples]],
    }

    sample_tag = f"sample{args.sample_size}_seed{args.seed}" if args.sample_size else "all"
    prep_tag = "16k" if args.prepare_16k else "raw"
    norm_tag = "nodollar" if args.drop_terminal_dollar else "plain"
    length_tag = "freeLen" if args.ignore_predictor_length else f"relax{args.token_num_relax}"
    decode_tag = f"{length_tag}_beam{args.beam_size}_pen{args.penalty:g}"
    suffix = (
        f"{args.split}_{args.decoding_model}_{prep_tag}_{norm_tag}_"
        f"{decode_tag}_{sample_tag}_{len(results)}"
    )
    (output_dir / f"{suffix}.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [asdict(item) for item in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / f"{suffix}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_row",
                "audio",
                "wer",
                "cer",
                "ref_words",
                "ref_chars",
                "seconds",
                "ref",
                "hyp",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decoding_model", choices=["fast", "normal", "offline"], default="offline")
    parser.add_argument("--prepare_16k", action="store_true")
    parser.add_argument("--drop_terminal_dollar", action="store_true")
    parser.add_argument("--token_num_relax", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--ignore_predictor_length", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
