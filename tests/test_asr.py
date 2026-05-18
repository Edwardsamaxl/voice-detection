"""Tests for ASR and alignment modules."""

import os
import tempfile
from unittest import mock

import pytest
from pyannote.core import Annotation, Segment

from src.asr.align import align_segments
from src.asr.text_normalize import normalize_burmese_asr_text, normalize_for_cer, text_to_ctc_tokens
from src.asr.whisper_asr import transcribe
from src.core.types import SpeakerSegment


class TestTranscribe:
    def test_import_error(self):
        with mock.patch("builtins.__import__", side_effect=ImportError("no whisper")):
            with pytest.raises(RuntimeError, match="Failed to import openai-whisper"):
                transcribe("dummy.wav")

    def test_successful_transcription(self):
        mock_model = mock.Mock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " hello world "},
                {"start": 2.0, "end": 3.5, "text": "test"},
            ]
        }

        with mock.patch("whisper.load_model", return_value=mock_model):
            result = transcribe("audio.wav", language="my", model_name="base")

        assert len(result) == 2
        assert result[0].segment_id == "whisper_0000"
        assert result[0].file == "audio.wav"
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[0].text == "hello world"
        assert result[0].local_speaker == "UNKNOWN"

        assert result[1].segment_id == "whisper_0001"
        assert result[1].text == "test"

    def test_empty_segments(self):
        mock_model = mock.Mock()
        mock_model.transcribe.return_value = {"segments": []}

        with mock.patch("whisper.load_model", return_value=mock_model):
            result = transcribe("audio.wav")

        assert result == []

    def test_missing_text_defaults_to_empty(self):
        mock_model = mock.Mock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0}]
        }

        with mock.patch("whisper.load_model", return_value=mock_model):
            result = transcribe("audio.wav")

        assert result[0].text == ""

    def test_download_root_override(self):
        mock_model = mock.Mock()
        mock_model.transcribe.return_value = {"segments": []}

        with mock.patch("whisper.load_model", return_value=mock_model) as mock_load:
            transcribe("audio.wav", download_root="/custom/root")

        mock_load.assert_called_once_with("base", download_root="/custom/root")


class TestBurmeseAsrNormalize:
    def test_removes_terminal_dataset_marker(self):
        assert normalize_burmese_asr_text("မင်္ဂလာပါ ဒေါ်လာ") == "မင်္ဂလာပါ"

    def test_removes_only_isolated_terminal_la(self):
        assert normalize_burmese_asr_text("မင်္ဂ လာ") == "မင်္ဂ"
        assert normalize_burmese_asr_text("မင်္ဂလာ") == "မင်္ဂလာ"

    def test_cer_normalization_drops_spaces(self):
        assert normalize_for_cer("မင်္ဂ လာ") == "မင်္ဂ"

    def test_ctc_training_text_uses_word_delimiter(self):
        assert text_to_ctc_tokens("မင်္ဂလာ ပါ") == "မင်္ဂလာ|ပါ"


class TestAlignSegments:
    def test_empty_whisper_segments(self):
        ann = Annotation()
        ann[Segment(0.0, 2.0)] = "SPEAKER_00"
        result = align_segments([], ann)
        assert result == []

    def test_whisper_segment_too_short_marked_ignore(self):
        ann = Annotation()
        ann[Segment(0.0, 2.0)] = "SPEAKER_00"

        whisper_segments = [
            SpeakerSegment("w1", "a.wav", 0.0, 0.2, "UNKNOWN"),
        ]
        result = align_segments(whisper_segments, ann)
        assert result[0].local_speaker == "IGNORE"

    def test_dominant_speaker_assigned(self):
        ann = Annotation()
        ann[Segment(0.0, 3.0)] = "SPEAKER_00"
        ann[Segment(3.0, 5.0)] = "SPEAKER_01"

        whisper_segments = [
            SpeakerSegment("w1", "a.wav", 0.0, 2.5, "UNKNOWN"),
            SpeakerSegment("w2", "a.wav", 3.0, 4.5, "UNKNOWN"),
        ]
        result = align_segments(whisper_segments, ann)
        assert result[0].local_speaker == "SPEAKER_00"
        assert result[1].local_speaker == "SPEAKER_01"

    def test_unknown_when_no_overlap(self):
        ann = Annotation()
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"

        whisper_segments = [
            SpeakerSegment("w1", "a.wav", 5.0, 7.0, "UNKNOWN"),
        ]
        result = align_segments(whisper_segments, ann)
        assert result[0].local_speaker == "UNKNOWN"

    def test_returns_new_instances(self):
        ann = Annotation()
        ann[Segment(0.0, 2.0)] = "SPEAKER_00"

        whisper_segments = [
            SpeakerSegment("w1", "a.wav", 0.0, 2.0, "UNKNOWN"),
        ]
        result = align_segments(whisper_segments, ann)
        assert result[0] is not whisper_segments[0]
        assert whisper_segments[0].local_speaker == "UNKNOWN"
