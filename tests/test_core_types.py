"""Tests for src.core.types immutable dataclasses."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.core.types import (
    IdentificationResult,
    SearchResult,
    SpeakerData,
    SpeakerProfile,
    SpeakerSegment,
)


def test_speaker_segment_duration():
    seg = SpeakerSegment(segment_id="s1", file="a.wav", start=1.0, end=3.5, local_speaker="A")
    assert seg.duration == 2.5


def test_speaker_segment_immutable():
    seg = SpeakerSegment(segment_id="s1", file="a.wav", start=0.0, end=1.0, local_speaker="A")
    with pytest.raises(FrozenInstanceError):
        seg.start = 2.0


def test_with_embedding_returns_new_instance():
    seg = SpeakerSegment(segment_id="s1", file="a.wav", start=0.0, end=1.0, local_speaker="A")
    emb = np.array([1.0, 0.0], dtype=np.float32)
    new_seg = seg.with_embedding(emb)
    assert new_seg is not seg
    assert new_seg.embedding is emb
    assert seg.embedding is None


def test_with_global_speaker():
    seg = SpeakerSegment(segment_id="s1", file="a.wav", start=0.0, end=1.0, local_speaker="A")
    new_seg = seg.with_global_speaker("SPK_0")
    assert new_seg.global_speaker == "SPK_0"
    assert seg.global_speaker is None


def test_with_text_and_translation():
    seg = SpeakerSegment(segment_id="s1", file="a.wav", start=0.0, end=1.0, local_speaker="A")
    new_seg = seg.with_text("hello", "你好")
    assert new_seg.text == "hello"
    assert new_seg.translation == "你好"


def test_speaker_data_creation():
    center = np.array([1.0, 0.0], dtype=np.float32)
    data = SpeakerData(spk_id="SPK_0", center=center, embeddings=[center])
    assert data.spk_id == "SPK_0"


def test_identification_result():
    r = IdentificationResult(speaker="SPK_0", score=0.85, confidence="high")
    assert r.confidence == "high"


def test_search_result():
    r = SearchResult(speaker="SPK_0", score=0.9)
    assert r.speaker == "SPK_0"
