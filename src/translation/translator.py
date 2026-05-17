"""NLLB-200 translation backend (mya_Mymr -> zho_Hans)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from config import MODELS_DIR, TRANSLATION_MODEL, TRANSLATION_SRC_LANG, TRANSLATION_TGT_LANG

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_DEVICE: str | None = None


def _resolve_device(preferred: str | None = None) -> str:
    """Return 'cuda' if available, else 'cpu'."""
    global _DEFAULT_DEVICE  # noqa: PLW0603
    if preferred is not None:
        return preferred
    if _DEFAULT_DEVICE is None:
        try:
            import torch

            _DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _DEFAULT_DEVICE = "cpu"
    return _DEFAULT_DEVICE


def _get_cache_dir() -> str | None:
    """Return project-local translation cache when it has model files.

    If the project cache is empty, let transformers use its standard
    HuggingFace cache. This avoids duplicating large translation models when
    they already exist in the user's local HF cache.
    """
    cache = os.path.join(MODELS_DIR, "translation")
    if os.path.isdir(cache) and any(os.scandir(cache)):
        return cache
    return None


class Translator:
    """Offline Burmese -> Chinese translator using Meta's NLLB-200.

    The model is loaded lazily on first translation to avoid heavy import
    overhead when the module is merely imported.
    """

    def __init__(
        self,
        model_name: str = TRANSLATION_MODEL,
        src_lang: str = TRANSLATION_SRC_LANG,
        tgt_lang: str = TRANSLATION_TGT_LANG,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = _resolve_device(device)
        self.cache_dir = cache_dir or _get_cache_dir()

        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSeq2SeqLM | None = None
        self._tgt_token_id: int | None = None

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Lazy-load tokenizer and model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers library is required for translation. "
                "Install it: pip install transformers sentencepiece sacremoses"
            ) from exc

        logger.info("Loading translation model %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self._model.eval()

        # Resolve target-language token id (compat across transformers versions)
        tgt_token_id = self._resolve_tgt_token_id(self.tgt_lang)
        if tgt_token_id is None:
            raise RuntimeError(
                f"Target language '{self.tgt_lang}' not supported by "
                f"tokenizer for model {self.model_name}"
            )
        self._tgt_token_id = tgt_token_id
        logger.info("Translation model ready on %s", self.device)

    def _resolve_tgt_token_id(self, lang: str) -> int | None:
        """Return the token id for a language code, or None if unknown."""
        tok = self._tokenizer
        assert tok is not None

        # Newer transformers
        try:
            return int(tok.convert_tokens_to_ids(lang))
        except (TypeError, ValueError):
            pass

        # Older transformers
        mapping = getattr(tok, "lang_code_to_id", None)
        if mapping is not None and lang in mapping:
            return int(mapping[lang])

        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def translate(
        self,
        text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> str:
        """Translate a single string.

        Args:
            text: Source text in Burmese (or the configured src_lang).
            source_lang: Optional override of the source language code.
            target_lang: Optional override of the target language code.

        Returns:
            Translated Chinese text.
        """
        if not text or not text.strip():
            return text

        self._load()
        results = self.translate_batch([text], source_lang, target_lang)
        return results[0]

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[str]:
        """Translate multiple strings in one forward pass.

        Args:
            texts: List of source strings.
            source_lang: Optional override of the source language code.
            target_lang: Optional override of the target language code.

        Returns:
            List of translated strings (same order as input).
        """
        if not texts:
            return []

        self._load()
        assert self._tokenizer is not None and self._model is not None

        src = source_lang or self.src_lang
        tgt = target_lang or self.tgt_lang

        # Resolve target token id for this call (may differ if target_lang overridden)
        tgt_token_id = self._tgt_token_id if tgt == self.tgt_lang else self._resolve_tgt_token_id(tgt)
        if tgt_token_id is None:
            raise RuntimeError(f"Unsupported target language: {tgt}")

        self._tokenizer.src_lang = src
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=256,
            )

        decoded: list[str] = self._tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return decoded


def translate_text(
    text: str,
    source_lang: str = TRANSLATION_SRC_LANG,
    target_lang: str = TRANSLATION_TGT_LANG,
) -> str:
    """Translate one string with the default translator singleton."""
    return Translator(src_lang=source_lang, tgt_lang=target_lang).translate(text)
