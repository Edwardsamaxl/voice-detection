"""Tests for the NLLB-based translation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.translation.translator import Translator, translate_text


class TestTranslatorInterface:
    """Unit tests using mocked transformers to avoid downloading models."""

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_translate_single(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 256_047
        mock_tokenizer.batch_decode.return_value = ["你好"]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_model.generate.return_value = mock_output
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        translator = Translator(device="cpu")
        result = translator.translate("မင်္ဂလာပါ")

        assert result == "你好"
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args.kwargs
        assert call_kwargs["forced_bos_token_id"] == 256_047

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_translate_batch(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 256_047
        mock_tokenizer.batch_decode.return_value = ["你好", "再见"]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        translator = Translator(device="cpu")
        results = translator.translate_batch(["hello1", "hello2"])

        assert results == ["你好", "再见"]
        mock_tokenizer_cls.from_pretrained.assert_called_once()

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_empty_string_passthrough(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 256_047
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        translator = Translator(device="cpu")
        assert translator.translate("") == ""
        assert translator.translate("   ") == "   "
        mock_model.generate.assert_not_called()

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_custom_lang_codes(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 123
        mock_tokenizer.batch_decode.return_value = ["translated"]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        translator = Translator(device="cpu")
        result = translator.translate(
            "text", source_lang="eng_Latn", target_lang="fra_Latn"
        )

        assert result == "translated"
        assert mock_tokenizer.src_lang == "eng_Latn"

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_translate_text_top_level(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 256_047
        mock_tokenizer.batch_decode.return_value = ["你好"]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        result = translate_text("မင်္ဂလာပါ")
        assert result == "你好"

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_model_loaded_only_once(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 256_047
        mock_tokenizer.batch_decode.return_value = ["a", "b"]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        translator = Translator(device="cpu")
        translator.translate("x")
        translator.translate("y")

        mock_tokenizer_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()


class TestTranslatorErrors:
    def test_transformers_missing_raises(self):
        with patch.dict("sys.modules", {"transformers": None}):
            translator = Translator(device="cpu")
            with pytest.raises(RuntimeError, match="transformers library is required"):
                translator.translate("test")

    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_unsupported_target_lang_raises(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()

        def _resolve_lang(lang: str):
            return 256_047 if lang == "zho_Hans" else None

        mock_tokenizer.convert_tokens_to_ids.side_effect = _resolve_lang
        mock_tokenizer.lang_code_to_id = {}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        translator = Translator(device="cpu")
        with pytest.raises(RuntimeError, match="Unsupported target language"):
            translator.translate("test", target_lang="xxx_Xxx")
