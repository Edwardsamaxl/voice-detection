"""Tests for high-level pipeline orchestration."""

import numpy as np
import pytest

import pipeline
from src.core.pool import EmbeddingPool
from src.core.repository import SpeakerRepository
from src.core.storage import MemoryStorage
from src.core.types import IdentificationResult, SpeakerSegment
from src.speaker_db.vector_index import FaissVectorIndex


def test_build_pipeline_accepts_preconfigured_repo(monkeypatch):
    repo = SpeakerRepository(
        vector_index=FaissVectorIndex(),
        storage=MemoryStorage(),
    )
    monkeypatch.setattr(pipeline, "build_embedding_pool", lambda _path: EmbeddingPool())

    result = pipeline.build_pipeline("unused", repo=repo)

    assert result is repo
    assert result.all_speakers() == []


def test_faiss_index_empty_build_searches_empty():
    index = FaissVectorIndex()

    index.build(np.empty((0, 192), dtype=np.float32), [])

    assert index.search(np.ones(192, dtype=np.float32)) == []


def test_recognize_pipeline_updates_known_speaker_and_uses_aligned_speaker(monkeypatch):
    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0
    diar_segment = SpeakerSegment(
        segment_id="",
        file="",
        start=0.0,
        end=5.0,
        local_speaker="A",
    )

    class FakeExtractor:
        def extract_segments(self, _wav_path, segments):
            return [seg.with_embedding(emb) for seg in segments]

    class FakeRepo:
        def __init__(self):
            self.updated = []

        def identify(self, query_emb):
            assert np.allclose(query_emb, emb)
            return IdentificationResult("SPK_0", 0.95, "high")

        def update_speaker(self, spk_id, new_emb, duration):
            self.updated.append((spk_id, new_emb, duration))
            return True

        def get_speaker(self, _spk_id):
            return None

    monkeypatch.setattr(
        "src.diarization.segment.run_diarization",
        lambda *_args, **_kwargs: "annotation",
    )
    monkeypatch.setattr(
        "src.diarization.postprocess.merge_short_segments",
        lambda annotation, **_kwargs: annotation,
    )
    monkeypatch.setattr(
        "src.diarization.postprocess.annotation_to_segments",
        lambda *_args, **_kwargs: [diar_segment],
    )
    monkeypatch.setattr(
        "src.asr.whisper_asr.transcribe",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "src.asr.align.align_segments",
        lambda *_args, **_kwargs: [
            SpeakerSegment("w0", "sample", 0.5, 1.5, "B", text="wrong"),
            SpeakerSegment("w1", "sample", 1.5, 2.5, "A", text="right"),
            SpeakerSegment("w2", "sample", 2.5, 2.8, "IGNORE", text="ignored"),
        ],
    )

    repo = FakeRepo()
    result = pipeline.recognize_pipeline(
        wav_path="sample.wav",
        repo=repo,
        extractor=FakeExtractor(),
        translator=None,
        asr_backend="whisper",
    )

    assert len(result) == 1
    assert result[0].global_speaker == "SPK_0"
    assert result[0].text == "right"
    assert repo.updated == [("SPK_0", emb, 5.0)]


class TestBuildPipeline:
    def test_slr80_filename_strategy(self, monkeypatch):
        emb = np.zeros(192, dtype=np.float32)
        pool = EmbeddingPool([
            SpeakerSegment(
                segment_id="seg1",
                file="bur_0366_5281755035.wav",
                start=0.0,
                end=1.0,
                local_speaker="A",
                embedding=emb,
            ),
        ])
        monkeypatch.setattr(pipeline, "build_embedding_pool", lambda _path: pool)

        repo = pipeline.build_pipeline("unused", label_strategy="slr80_filename")
        assert repo.all_speakers() == ["BUR_0366"]

    def test_unknown_strategy_raises(self, monkeypatch):
        monkeypatch.setattr(pipeline, "build_embedding_pool", lambda _path: EmbeddingPool())
        with pytest.raises(ValueError, match="Unknown label_strategy"):
            pipeline.build_pipeline("unused", label_strategy="invalid")


class TestNextSpkId:
    def test_sequential_id(self):
        class FakeRepo:
            def all_speakers(self):
                return ["SPK_0", "SPK_2"]

        assert pipeline._next_spk_id(FakeRepo()) == "SPK_3"

    def test_no_existing_spk(self):
        class FakeRepo:
            def all_speakers(self):
                return ["BUR_0366"]

        assert pipeline._next_spk_id(FakeRepo()) == "SPK_0"

    def test_ignores_non_numeric_suffix(self):
        class FakeRepo:
            def all_speakers(self):
                return ["SPK_abc", "SPK_1"]

        assert pipeline._next_spk_id(FakeRepo()) == "SPK_2"


class TestRecognizePipelineNewSpeaker:
    def test_adds_new_speaker_when_low_confidence_and_long_duration(self, monkeypatch):
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        diar_segment = SpeakerSegment(
            segment_id="",
            file="",
            start=0.0,
            end=5.0,
            local_speaker="A",
        )

        class FakeExtractor:
            def extract_segments(self, _wav_path, segments):
                return [seg.with_embedding(emb) for seg in segments]

        class FakeRepo:
            def __init__(self):
                self.added = []
                self._speakers = {}

            def identify(self, _query_emb):
                return IdentificationResult(None, 0.3, "low")

            def add_speaker(self, spk_id, embeddings, durations):
                self.added.append((spk_id, embeddings, durations))

            def get_speaker(self, _spk_id):
                return None

            def all_speakers(self):
                return list(self._speakers.keys())

            def update_speaker(self, _spk_id, _new_emb, _duration):
                return True

        monkeypatch.setattr("src.diarization.segment.run_diarization", lambda *_args, **_kw: "annotation")
        monkeypatch.setattr(
            "src.diarization.postprocess.merge_short_segments",
            lambda annotation, **_kw: annotation,
        )
        monkeypatch.setattr(
            "src.diarization.postprocess.annotation_to_segments",
            lambda *_args, **_kw: [diar_segment],
        )
        monkeypatch.setattr("src.asr.whisper_asr.transcribe", lambda *_args, **_kw: [])
        monkeypatch.setattr("src.asr.align.align_segments", lambda *_args, **_kw: [])

        repo = FakeRepo()
        result = pipeline.recognize_pipeline(
            wav_path="sample.wav", repo=repo, extractor=FakeExtractor(), translator=None,
            asr_backend="whisper",
        )
        assert len(result) == 1
        assert result[0].global_speaker == "SPK_0"
        assert len(repo.added) == 1
        assert repo.added[0][0] == "SPK_0"

    def test_unknown_when_low_confidence_and_short_duration(self, monkeypatch):
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        diar_segment = SpeakerSegment(
            segment_id="",
            file="",
            start=0.0,
            end=1.0,
            local_speaker="A",
        )

        class FakeExtractor:
            def extract_segments(self, _wav_path, segments):
                return [seg.with_embedding(emb) for seg in segments]

        class FakeRepo:
            def identify(self, _query_emb):
                return IdentificationResult(None, 0.3, "low")

            def get_speaker(self, _spk_id):
                return None

            def all_speakers(self):
                return []

            def update_speaker(self, _spk_id, _new_emb, _duration):
                return True

            def add_speaker(self, spk_id, embeddings, durations, profile=None):
                pass

        monkeypatch.setattr("src.diarization.segment.run_diarization", lambda *_args, **_kw: "annotation")
        monkeypatch.setattr(
            "src.diarization.postprocess.merge_short_segments",
            lambda annotation, **_kw: annotation,
        )
        monkeypatch.setattr(
            "src.diarization.postprocess.annotation_to_segments",
            lambda *_args, **_kw: [diar_segment],
        )
        monkeypatch.setattr("src.asr.whisper_asr.transcribe", lambda *_args, **_kw: [])
        monkeypatch.setattr("src.asr.align.align_segments", lambda *_args, **_kw: [])

        repo = FakeRepo()
        result = pipeline.recognize_pipeline(
            wav_path="sample.wav", repo=repo, extractor=FakeExtractor(), translator=None,
            asr_backend="whisper",
        )
        assert result[0].global_speaker == "SPK_0"

    def test_skips_ignore_segments(self, monkeypatch):
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        seg_keep = SpeakerSegment("", "", 0.0, 2.0, "A")
        seg_ignore = SpeakerSegment("", "", 2.0, 3.0, "IGNORE")

        class FakeExtractor:
            def extract_segments(self, _wav_path, segments):
                return [seg.with_embedding(emb) for seg in segments]

        class FakeRepo:
            def identify(self, _query_emb):
                return IdentificationResult("SPK_0", 0.95, "high")

            def update_speaker(self, _spk_id, _new_emb, _duration):
                return True

            def get_speaker(self, _spk_id):
                return None

            def all_speakers(self):
                return []

            def add_speaker(self, _spk_id, _embeddings, _durations):
                pass

        monkeypatch.setattr("src.diarization.segment.run_diarization", lambda *_args, **_kw: "annotation")
        monkeypatch.setattr(
            "src.diarization.postprocess.merge_short_segments",
            lambda annotation, **_kw: annotation,
        )
        monkeypatch.setattr(
            "src.diarization.postprocess.annotation_to_segments",
            lambda *_args, **_kw: [seg_keep, seg_ignore],
        )
        monkeypatch.setattr("src.asr.whisper_asr.transcribe", lambda *_args, **_kw: [])
        monkeypatch.setattr("src.asr.align.align_segments", lambda *_args, **_kw: [])

        repo = FakeRepo()
        result = pipeline.recognize_pipeline(
            wav_path="sample.wav", repo=repo, extractor=FakeExtractor(), translator=None,
            asr_backend="whisper",
        )
        assert len(result) == 1
        assert result[0].local_speaker == "A"

    def test_empty_diarization_returns_empty(self, monkeypatch):
        monkeypatch.setattr("src.diarization.segment.run_diarization", lambda *_args, **_kw: "annotation")
        monkeypatch.setattr(
            "src.diarization.postprocess.merge_short_segments",
            lambda annotation, **_kw: annotation,
        )
        monkeypatch.setattr(
            "src.diarization.postprocess.annotation_to_segments",
            lambda *_args, **_kw: [],
        )

        repo = SpeakerRepository(vector_index=FaissVectorIndex())
        result = pipeline.recognize_pipeline(
            wav_path="sample.wav", repo=repo, extractor=object(), translator=None,
            asr_backend="whisper",
        )
        assert result == []

    def test_translator_integration(self, monkeypatch):
        emb = np.zeros(192, dtype=np.float32)
        emb[0] = 1.0
        diar_segment = SpeakerSegment(
            segment_id="",
            file="",
            start=0.0,
            end=2.0,
            local_speaker="A",
        )

        class FakeExtractor:
            def extract_segments(self, _wav_path, segments):
                return [seg.with_embedding(emb) for seg in segments]

        class FakeRepo:
            def identify(self, _query_emb):
                return IdentificationResult("SPK_0", 0.95, "high")

            def update_speaker(self, _spk_id, _new_emb, _duration):
                return True

            def get_speaker(self, _spk_id):
                return None

            def all_speakers(self):
                return []

            def add_speaker(self, _spk_id, _embeddings, _durations):
                pass

        class FakeTranslator:
            def translate(self, text):
                return f"translated:{text}"

        monkeypatch.setattr("src.diarization.segment.run_diarization", lambda *_args, **_kw: "annotation")
        monkeypatch.setattr(
            "src.diarization.postprocess.merge_short_segments",
            lambda annotation, **_kw: annotation,
        )
        monkeypatch.setattr(
            "src.diarization.postprocess.annotation_to_segments",
            lambda *_args, **_kw: [diar_segment],
        )
        monkeypatch.setattr(
            "src.asr.whisper_asr.transcribe",
            lambda *_args, **_kw: [SpeakerSegment("w0", "sample", 0.0, 2.0, "A", text="hello")],
        )
        monkeypatch.setattr(
            "src.asr.align.align_segments",
            lambda *_args, **_kw: [SpeakerSegment("w0", "sample", 0.0, 2.0, "A", text="hello")],
        )

        repo = FakeRepo()
        result = pipeline.recognize_pipeline(
            wav_path="sample.wav",
            repo=repo,
            extractor=FakeExtractor(),
            translator=FakeTranslator(),
            asr_backend="whisper",
        )
        assert len(result) == 1
        assert result[0].translation == "translated:hello"
