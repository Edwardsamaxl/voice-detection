"""Text normalization helpers for Burmese ASR experiments."""

from __future__ import annotations

import re
import unicodedata


DATASET_TERMINAL_MARKER = "ဒေါ်လာ"
TERMINAL_LA_MARKER = "လာ"
WORD_DELIMITER = "|"


def normalize_burmese_asr_text(text: str) -> str:
    """Normalize SLR80 Burmese transcripts without altering Myanmar marks."""
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized.endswith(DATASET_TERMINAL_MARKER):
        normalized = normalized[: -len(DATASET_TERMINAL_MARKER)].strip()

    if normalized == TERMINAL_LA_MARKER:
        return ""
    if normalized.endswith(f" {TERMINAL_LA_MARKER}"):
        normalized = normalized[: -len(TERMINAL_LA_MARKER)].strip()

    return normalized


def normalize_for_cer(text: str, remove_spaces: bool = True) -> str:
    """Normalize text for CER, optionally dropping spaces as the main metric."""
    normalized = normalize_burmese_asr_text(text)
    if remove_spaces:
        normalized = normalized.replace(" ", "")
    return normalized


def text_to_ctc_tokens(text: str) -> str:
    """Convert normalized training text to CTC tokenizer text."""
    return normalize_burmese_asr_text(text).replace(" ", WORD_DELIMITER)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Compute Levenshtein edit distance for token lists."""
    prev = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def char_error_rate(ref: str, hyp: str, remove_spaces: bool = True) -> float:
    """Compute character error rate after Burmese ASR normalization."""
    ref_chars = list(normalize_for_cer(ref, remove_spaces=remove_spaces))
    hyp_chars = list(normalize_for_cer(hyp, remove_spaces=remove_spaces))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)
