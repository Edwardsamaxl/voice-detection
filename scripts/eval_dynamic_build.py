"""Simulate dynamic speaker repository building in a streaming scenario.

Speakers arrive one by one. For each speaker:
- First half of embeddings = "arrival" (query before in library)
- Second half = "follow-up" (test recognition after adding)

Metrics:
- First-encounter accuracy
- Follow-up accuracy
- False acceptance / false rejection
- Speaker count drift
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.repository import SpeakerRepository, VectorIndex
from src.core.types import IdentificationResult, NumpyEncoder
from src.speaker_db.vector_index import FaissVectorIndex


EMB_DIR = Path("data/processed/burmese_asr/train/embeddings")
OUTPUT_DIR = Path(".codex_realtest")
OUTPUT_PATH = OUTPUT_DIR / "dynamic_build_result.json"
SEED = 42


def load_embeddings_by_speaker() -> dict[str, list[np.ndarray]]:
    """Load all .npz embeddings grouped by speaker_id."""
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for npz_path in sorted(EMB_DIR.glob("*.npz")):
        stem = npz_path.stem  # e.g. bur_0366_0045318711
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        spk_id = parts[1]
        data = np.load(npz_path)
        emb = data["embeddings"].astype(np.float32)
        if emb.ndim == 2:
            for row in emb:
                groups[spk_id].append(row)
        elif emb.ndim == 1:
            groups[spk_id].append(emb)
    return dict(groups)


def split_embeddings(embs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """50/50 split: first half arrival, second half follow-up."""
    n = len(embs)
    mid = n // 2
    return embs[:mid], embs[mid:]


def run_dynamic_build() -> dict[str, Any]:
    """Run the streaming simulation and return results dict."""
    speaker_embs = load_embeddings_by_speaker()
    speaker_ids = list(speaker_embs.keys())
    rng = random.Random(SEED)
    rng.shuffle(speaker_ids)

    repo = SpeakerRepository(vector_index=FaissVectorIndex(dim=192))

    records: list[dict[str, Any]] = []
    arrival_total = 0
    arrival_correct_unknown = 0

    follow_up_total = 0
    follow_up_correct = 0
    false_acceptance = 0
    false_rejection = 0

    for spk_id in speaker_ids:
        embs = speaker_embs[spk_id]
        arrival_embs, follow_up_embs = split_embeddings(embs)

        # ---- Arrival phase: speaker NOT in library yet ----
        # Test only the FIRST arrival embedding as "first encounter".
        # If UNKNOWN, add ALL arrival embeddings at once so the speaker
        # has enough vectors for consistency check in follow-up.
        if arrival_embs:
            arrival_total += 1
            first_emb = arrival_embs[0]
            try:
                result = repo.identify(first_emb)
            except Exception as exc:
                records.append({
                    "phase": "arrival",
                    "expected": spk_id,
                    "predicted": None,
                    "status": "ERROR",
                    "score": 0.0,
                    "error": str(exc),
                })
                # Still add remaining embeddings so follow-up can work
                if len(arrival_embs) > 1:
                    repo.add_speaker(spk_id, arrival_embs[1:], [1.0] * (len(arrival_embs) - 1))
                continue

            is_unknown = result.speaker is None
            if is_unknown:
                arrival_correct_unknown += 1
                repo.add_speaker(spk_id, arrival_embs, [1.0] * len(arrival_embs))

            records.append({
                "phase": "arrival",
                "expected": spk_id,
                "predicted": result.speaker,
                "status": "UNKNOWN" if is_unknown else "KNOWN",
                "score": result.score,
            })

        # ---- Follow-up phase: speaker now in library ----
        for emb in follow_up_embs:
            follow_up_total += 1
            try:
                result = repo.identify(emb)
            except Exception as exc:
                records.append({
                    "phase": "follow_up",
                    "expected": spk_id,
                    "predicted": None,
                    "status": "ERROR",
                    "score": 0.0,
                    "error": str(exc),
                })
                continue

            predicted = result.speaker
            if predicted == spk_id:
                follow_up_correct += 1
                status = "CORRECT"
            elif predicted is None:
                false_rejection += 1
                status = "FALSE_REJECTION"
            else:
                false_acceptance += 1
                status = "FALSE_ACCEPTANCE"

            records.append({
                "phase": "follow_up",
                "expected": spk_id,
                "predicted": predicted,
                "status": status,
                "score": result.score,
            })

    summary = {
        "first_encounter_accuracy": (
            arrival_correct_unknown / arrival_total if arrival_total else 0.0
        ),
        "follow_up_accuracy": (
            follow_up_correct / follow_up_total if follow_up_total else 0.0
        ),
        "false_acceptance_count": false_acceptance,
        "false_rejection_count": false_rejection,
        "follow_up_total": follow_up_total,
        "arrival_total": arrival_total,
        "final_speaker_count": len(repo.all_speakers()),
        "expected_speaker_count": len(speaker_ids),
        "speaker_count_drift": len(repo.all_speakers()) - len(speaker_ids),
    }

    return {
        "summary": summary,
        "records": records,
        "speaker_arrival_order": speaker_ids,
    }


def main() -> None:
    results = run_dynamic_build()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    summary = results["summary"]
    print("Dynamic Build Evaluation Complete")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  First-encounter accuracy : {summary['first_encounter_accuracy']:.4f}")
    print(f"  Follow-up accuracy       : {summary['follow_up_accuracy']:.4f}")
    print(f"  False acceptance         : {summary['false_acceptance_count']}")
    print(f"  False rejection          : {summary['false_rejection_count']}")
    print(f"  Final speaker count      : {summary['final_speaker_count']} (expected {summary['expected_speaker_count']})")


if __name__ == "__main__":
    main()
