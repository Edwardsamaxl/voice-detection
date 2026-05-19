"""Train a minimal XLS-R CTC baseline on SLR80 Burmese ASR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.text_normalize import (  # noqa: E402
    WORD_DELIMITER,
    char_error_rate,
    edit_distance,
    normalize_burmese_asr_text,
    normalize_for_cer,
    text_to_ctc_tokens,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "burmese_asr"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "asr_xlsr_ctc"
DEFAULT_BASE_MODEL = "facebook/wav2vec2-xls-r-300m"


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


def build_vocab(rows: list[dict[str, str]], vocab_path: Path) -> dict[str, int]:
    chars: set[str] = set()
    for row in rows:
        chars.update(text_to_ctc_tokens(row["Text:LABEL"]))

    chars.discard(" ")
    vocab_tokens = sorted(ch for ch in chars if ch)
    if WORD_DELIMITER not in vocab_tokens:
        vocab_tokens.append(WORD_DELIMITER)

    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab


def make_processor(vocab_path: Path) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token=None,
        eos_token=None,
        word_delimiter_token=WORD_DELIMITER,
        do_lower_case=False,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


class Slr80CtcDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, str]], data_dir: Path, processor: Wav2Vec2Processor):
        self.rows = rows
        self.data_dir = data_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = self.data_dir / row["Audio:FILE"]
        audio = read_audio_16k_mono(audio_path)
        input_values = self.processor(audio, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(text_to_ctc_tokens(row["Text:LABEL"])).input_ids
        return {
            "input_values": input_values,
            "labels": labels,
            "audio": row["Audio:FILE"],
            "ref": normalize_burmese_asr_text(row["Text:LABEL"]),
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch


def make_compute_metrics(
    processor: Wav2Vec2Processor,
    eval_rows: list[dict[str, str]],
    examples_dir: Path,
    max_examples: int,
):
    call_count = {"value": 0}

    def compute_metrics(pred: Any) -> dict[str, float]:
        call_count["value"] += 1
        pred_ids = np.argmax(pred.predictions, axis=-1)

        hyp_texts = processor.batch_decode(pred_ids)
        ref_texts = [
            normalize_burmese_asr_text(row["Text:LABEL"])
            for row in eval_rows[: len(hyp_texts)]
        ]

        total_no_space_err = 0
        total_no_space_chars = 0
        total_with_space_err = 0
        total_with_space_chars = 0
        examples: list[dict[str, Any]] = []
        for idx, (ref, hyp) in enumerate(zip(ref_texts, hyp_texts)):
            cer = char_error_rate(ref, hyp, remove_spaces=True)
            cer_with_space = char_error_rate(ref, hyp, remove_spaces=False)
            ref_no_space = list(normalize_for_cer(ref, remove_spaces=True))
            hyp_no_space = list(normalize_for_cer(hyp, remove_spaces=True))
            ref_with_space = list(normalize_for_cer(ref, remove_spaces=False))
            hyp_with_space = list(normalize_for_cer(hyp, remove_spaces=False))
            total_no_space_err += edit_distance(ref_no_space, hyp_no_space)
            total_no_space_chars += len(ref_no_space)
            total_with_space_err += edit_distance(ref_with_space, hyp_with_space)
            total_with_space_chars += len(ref_with_space)
            if idx < max_examples:
                examples.append(
                    {
                        "audio": eval_rows[idx]["Audio:FILE"] if idx < len(eval_rows) else None,
                        "ref": normalize_burmese_asr_text(ref),
                        "hyp": normalize_burmese_asr_text(hyp),
                        "cer": cer,
                        "ref_metric": normalize_for_cer(ref, remove_spaces=True),
                        "hyp_metric": normalize_for_cer(hyp, remove_spaces=True),
                    }
                )

        metrics = {
            "cer": total_no_space_err / max(total_no_space_chars, 1),
            "cer_with_spaces": total_with_space_err / max(total_with_space_chars, 1),
        }
        examples_dir.mkdir(parents=True, exist_ok=True)
        (examples_dir / f"eval_examples_{call_count['value']:03d}.json").write_text(
            json.dumps({"metrics": metrics, "examples": examples}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metrics

    return compute_metrics


def resolve_fp16(value: str) -> bool:
    if value == "auto":
        return torch.cuda.is_available()
    return value == "true"


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.output_dir)
    checkpoint_dir = model_dir / "checkpoints"
    best_dir = model_dir / "best"
    vocab_path = model_dir / "vocab.json"
    examples_dir = model_dir / "eval_examples"

    train_rows = load_rows(data_dir, "train", args.train_limit)
    eval_limit = args.train_limit if args.eval_split == "train" and args.eval_limit is None else args.eval_limit
    if args.eval_split == "dev" and eval_limit is None:
        eval_limit = args.dev_limit
    dev_rows = load_rows(data_dir, args.eval_split, eval_limit)
    if not train_rows:
        raise ValueError("No training rows found.")
    if not dev_rows:
        raise ValueError("No dev rows found.")

    # Build vocab from the FULL train set so rare characters are never missed
    # even when --train_limit is used for faster experimentation.
    full_train_rows = load_rows(data_dir, "train")
    build_vocab(full_train_rows, vocab_path)
    processor = make_processor(vocab_path)

    train_dataset = Slr80CtcDataset(train_rows, data_dir, processor)
    dev_dataset = Slr80CtcDataset(dev_rows, data_dir, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # Prevent CTC blank collapse: initialize blank bias to strongly negative
    # so the model must predict real characters initially.
    blank_id = processor.tokenizer.pad_token_id
    with torch.no_grad():
        model.lm_head.bias[blank_id] = args.blank_bias_init
    print(f"Initialized blank token (id={blank_id}) bias to {args.blank_bias_init}", flush=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        fp16=resolve_fp16(args.fp16),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        compute_metrics=make_compute_metrics(processor, dev_rows, examples_dir, args.examples),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=processor.feature_extractor,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    processor.save_pretrained(str(model_dir))

    metrics = trainer.evaluate()
    summary = {
        "base_model": args.base_model,
        "train_count": len(train_rows),
        "eval_split": args.eval_split,
        "dev_count": len(dev_rows),
        "train_metrics": train_result.metrics,
        "eval_metrics": metrics,
        "best_model_dir": str(best_dir),
        "vocab_path": str(vocab_path),
    }
    (model_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)
    parser.add_argument("--eval_split", choices=["train", "dev"], default="dev")
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--fp16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--freeze_feature_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--blank_bias_init", type=float, default=-10.0, help="Initial bias for CTC blank token (negative discourages blank)")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    train(parse_args())
