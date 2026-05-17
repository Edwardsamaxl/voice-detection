"""Tests for diarization module."""

import json
import os
import tempfile
from unittest import mock

import pytest
from pyannote.core import Annotation, Segment

from dataclasses import FrozenInstanceError

from src.core.types import SpeakerSegment
from src.diarization.postprocess import annotation_to_segments, merge_short_segments, rename_labels
from src.diarization.segment import run_diarization


class TestRunDiarization:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_diarization("nonexistent.wav")

    def test_model_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVE")
            tmp_path = f.name

        try:
            mock_pipeline_cls = mock.Mock()
            mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("mock error")
            with mock.patch("src.diarization.segment._load_pipeline_class", return_value=mock_pipeline_cls):
                with pytest.raises(RuntimeError, match="Failed to load"):
                    run_diarization(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_pipeline_none(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVE")
            tmp_path = f.name

        try:
            mock_pipeline_cls = mock.Mock()
            mock_pipeline_cls.from_pretrained.return_value = None
            with mock.patch("src.diarization.segment._load_pipeline_class", return_value=mock_pipeline_cls):
                with pytest.raises(RuntimeError, match="returned None"):
                    run_diarization(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_successful_run(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVE")
            tmp_path = f.name

        mock_annotation = Annotation()
        mock_annotation[Segment(0.0, 2.0)] = "A"

        mock_pipeline = mock.Mock()
        mock_pipeline.return_value = mock_annotation

        try:
            mock_pipeline_cls = mock.Mock()
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
            with mock.patch("src.diarization.segment._load_pipeline_class", return_value=mock_pipeline_cls), \
                mock.patch("src.diarization.segment._select_device", return_value="cpu"), \
                mock.patch("src.diarization.segment._load_audio", return_value=("waveform", 16000)):
                result = run_diarization(tmp_path, device="cpu")
                assert isinstance(result, Annotation)
                assert len(list(result.itertracks())) == 1
        finally:
            os.unlink(tmp_path)


class TestMergeShortSegments:
    def test_empty_annotation(self):
        ann = Annotation()
        result = merge_short_segments(ann)
        assert len(list(result.itertracks())) == 0

    def test_merge_same_speaker_gap(self):
        ann = Annotation()
        ann[Segment(0.0, 1.0)] = "A"
        ann[Segment(1.1, 2.0)] = "A"
        result = merge_short_segments(ann, min_duration_off=0.2)
        tracks = list(result.itertracks(yield_label=True))
        assert len(tracks) == 1
        assert tracks[0][0] == Segment(0.0, 2.0)

    def test_keep_large_gap(self):
        ann = Annotation()
        ann[Segment(0.0, 1.0)] = "A"
        ann[Segment(2.0, 3.0)] = "A"
        result = merge_short_segments(ann, min_duration_off=0.2)
        tracks = list(result.itertracks(yield_label=True))
        assert len(tracks) == 2

    def test_drop_short_segment(self):
        ann = Annotation()
        ann[Segment(0.0, 2.0)] = "A"
        ann[Segment(2.0, 2.1)] = "B"
        result = merge_short_segments(ann, min_duration_on=0.3)
        tracks = list(result.itertracks(yield_label=True))
        assert len(tracks) == 1
        assert tracks[0][2] == "A"
        assert tracks[0][0] == Segment(0.0, 2.0)


class TestRenameLabels:
    def test_no_mapping_returns_unchanged(self):
        ann = Annotation()
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"
        result = rename_labels(ann)
        tracks = list(result.itertracks(yield_label=True))
        assert tracks[0][2] == "SPEAKER_00"

    def test_mapping_renames_labels(self):
        ann = Annotation()
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"
        ann[Segment(1.0, 2.0)] = "SPEAKER_01"
        result = rename_labels(ann, {"SPEAKER_00": "A", "SPEAKER_01": "B"})
        labels = {label for _, _, label in result.itertracks(yield_label=True)}
        assert labels == {"A", "B"}


class TestAnnotationToSegments:
    def test_empty_annotation(self):
        ann = Annotation()
        result = annotation_to_segments(ann)
        assert result == []

    def test_filter_short(self):
        ann = Annotation()
        ann[Segment(0.0, 0.5)] = "A"
        ann[Segment(1.0, 3.0)] = "B"
        result = annotation_to_segments(ann, min_duration=1.0)
        assert len(result) == 1
        assert isinstance(result[0], SpeakerSegment)
        assert result[0].local_speaker == "B"
        assert result[0].duration == 2.0
        assert result[0].global_speaker is None
        assert result[0].embedding is None

    def test_sorted_output(self):
        ann = Annotation()
        ann[Segment(5.0, 7.0)] = "A"
        ann[Segment(1.0, 3.0)] = "B"
        result = annotation_to_segments(ann, min_duration=1.0)
        assert result[0].start == 1.0
        assert result[1].start == 5.0

    def test_speaker_segments_are_immutable(self):
        ann = Annotation()
        ann[Segment(0.0, 2.0)] = "A"
        result = annotation_to_segments(ann)
        assert len(result) == 1
        seg = result[0]
        assert isinstance(seg, SpeakerSegment)
        assert seg.segment_id == ""
        assert seg.file == ""
        assert seg.global_speaker is None
        assert seg.display_name is None
        assert seg.embedding is None
        assert seg.text is None
        assert seg.translation is None
        with pytest.raises(FrozenInstanceError):
            seg.local_speaker = "B"
