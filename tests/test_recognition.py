"""Tests for recognition / verification module."""

import pytest

from src.core.types import SearchResult
from src.recognition.verify import topk_consistency


class TestTopkConsistency:
    def test_empty_results_returns_false(self):
        assert topk_consistency([]) is False

    def test_all_same_speaker_passes(self):
        results = [SearchResult("SPK_0", 0.9)] * 5
        assert topk_consistency(results, k=5, min_consistency=4) is True

    def test_majority_agreement_passes(self):
        results = [
            SearchResult("SPK_0", 0.9),
            SearchResult("SPK_0", 0.85),
            SearchResult("SPK_0", 0.8),
            SearchResult("SPK_0", 0.75),
            SearchResult("SPK_1", 0.7),
        ]
        assert topk_consistency(results, k=5, min_consistency=4) is True

    def test_not_enough_agreement_fails(self):
        results = [
            SearchResult("SPK_0", 0.9),
            SearchResult("SPK_0", 0.85),
            SearchResult("SPK_1", 0.8),
            SearchResult("SPK_1", 0.75),
            SearchResult("SPK_2", 0.7),
        ]
        assert topk_consistency(results, k=5, min_consistency=4) is False

    def test_k_smaller_than_results(self):
        results = [
            SearchResult("SPK_0", 0.9),
            SearchResult("SPK_0", 0.85),
            SearchResult("SPK_1", 0.8),
            SearchResult("SPK_1", 0.75),
        ]
        assert topk_consistency(results, k=2, min_consistency=2) is True

    def test_results_with_empty_speaker_filtered(self):
        results = [
            SearchResult("SPK_0", 0.9),
            SearchResult("", 0.85),
            SearchResult("SPK_0", 0.8),
            SearchResult("SPK_0", 0.75),
            SearchResult("SPK_1", 0.7),
        ]
        assert topk_consistency(results, k=5, min_consistency=3) is True

    def test_results_all_empty_speakers(self):
        results = [SearchResult("", 0.9)] * 5
        assert topk_consistency(results, k=5, min_consistency=1) is False

    def test_default_parameters(self):
        results = [SearchResult("SPK_0", 0.9)] * 5
        assert topk_consistency(results) is True
