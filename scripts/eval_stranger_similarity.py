"""Evaluate similarity distribution between in-library and out-of-library speakers.

Usage:
    python scripts/eval_stranger_similarity.py \
        --embedding_dir data/processed/burmese_asr/train/embeddings \
        --in_speakers 0366,3260,5189,5903,5932 \
        --out_speakers 2446,4409,9762 \
        --output results_group1.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MAX_EMB, TOPK
from src.embedding.normalize import l2_normalize


def load_embeddings(embedding_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load all embeddings from .npz files, grouped by speaker_id."""
    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    npz_files = sorted(embedding_dir.glob("*.npz"))
    for npz_path in npz_files:
        match = re.match(r"bur_(\d+)_", npz_path.stem)
        if not match:
            continue
        spk_id = match.group(1)
        data = np.load(npz_path)
        emb = data["embeddings"]
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        for row in emb:
            speaker_embeddings.setdefault(spk_id, []).append(row.astype(np.float32))
    return speaker_embeddings


def build_library(
    all_embeddings: dict[str, list[np.ndarray]], in_speakers: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Build small library from in_speakers.

    Returns:
        centers: {spk_id: center_vector}
        selected_matrix: (N, dim) all selected embeddings for FAISS-like search
        selected_labels: list of spk_id for each row in selected_matrix
    """
    centers: dict[str, np.ndarray] = {}
    selected_vectors: list[np.ndarray] = []
    selected_labels: list[str] = []

    for spk_id in in_speakers:
        embs = all_embeddings.get(spk_id, [])
        if not embs:
            continue
        matrix = np.stack(embs)
        # Simple center: mean then l2 normalize
        center = l2_normalize(matrix.mean(axis=0)).astype(np.float32)
        centers[spk_id] = center

        # Select up to MAX_EMB closest to center
        if len(embs) <= MAX_EMB:
            selected = embs
        else:
            sims = np.dot(matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10), center)
            indices = np.argsort(sims)[-MAX_EMB:]
            selected = [embs[i] for i in indices]

        for vec in selected:
            norm = np.linalg.norm(vec)
            if norm > 0:
                selected_vectors.append((vec / norm).astype(np.float32))
            else:
                selected_vectors.append(vec.astype(np.float32))
            selected_labels.append(spk_id)

    if not selected_vectors:
        return centers, np.empty((0, 0), dtype=np.float32), []

    selected_matrix = np.stack(selected_vectors).astype(np.float32)
    return centers, selected_matrix, selected_labels


def evaluate_query(
    query_emb: np.ndarray,
    centers: dict[str, np.ndarray],
    selected_matrix: np.ndarray,
    selected_labels: list[str],
) -> dict:
    """Run a single query against the library.

    Returns metrics including center max score, FAISS-like top-k, consistency.
    """
    query = l2_normalize(query_emb).astype(np.float32)

    # Stage 1: center coarse ranking
    center_scores = {
        spk_id: float(np.dot(query, center))
        for spk_id, center in centers.items()
    }
    best_center_spk = max(center_scores, key=center_scores.get) if center_scores else None
    best_center_score = center_scores[best_center_spk] if best_center_spk else 0.0

    # Stage 2: global FAISS-like top-k search (IndexFlatIP, vectors are normalized)
    faiss_results: list[dict] = []
    if selected_matrix.size > 0:
        sims = np.dot(selected_matrix, query)
        topk = min(TOPK, len(sims))
        top_indices = np.argsort(sims)[-topk:][::-1]
        for idx in top_indices:
            faiss_results.append({
                "speaker": selected_labels[int(idx)],
                "score": float(sims[int(idx)]),
            })

    # Consistency check
    consistency_spk = None
    consistency_count = 0
    consistency_pass = False
    faiss_avg_score = 0.0
    if faiss_results:
        speakers = [r["speaker"] for r in faiss_results]
        if speakers:
            consistency_spk, consistency_count = Counter(speakers).most_common(1)[0]
            consistency_pass = consistency_count >= 4  # TOPK_CONSISTENCY_MIN

        # Average score of top-k results (all of them, not just consensus speaker)
        faiss_avg_score = float(np.mean([r["score"] for r in faiss_results]))

    # Average score of consensus speaker vectors in top-k
    consensus_avg_score = 0.0
    if faiss_results and consistency_spk:
        consensus_scores = [r["score"] for r in faiss_results if r["speaker"] == consistency_spk]
        if consensus_scores:
            consensus_avg_score = float(np.mean(consensus_scores))

    return {
        "center_max_score": best_center_score,
        "center_best_spk": best_center_spk,
        "faiss_topk": faiss_results,
        "faiss_avg_score": faiss_avg_score,
        "faiss_consensus_spk": consistency_spk,
        "faiss_consensus_count": consistency_count,
        "faiss_consistency_pass": consistency_pass,
        "faiss_consensus_avg_score": consensus_avg_score,
    }


def evaluate_group(
    all_embeddings: dict[str, list[np.ndarray]],
    in_speakers: list[str],
    out_speakers: list[str],
) -> dict:
    """Run evaluation for one group."""
    centers, selected_matrix, selected_labels = build_library(all_embeddings, in_speakers)

    results = {
        "in_library": [],
        "out_of_library": [],
        "config": {
            "in_speakers": in_speakers,
            "out_speakers": out_speakers,
            "library_vectors": len(selected_labels),
            "library_speakers": len(centers),
        },
    }

    # In-library: use all embeddings from in_speakers as queries
    for spk_id in in_speakers:
        embs = all_embeddings.get(spk_id, [])
        for i, emb in enumerate(embs):
            metrics = evaluate_query(emb, centers, selected_matrix, selected_labels)
            metrics["true_speaker"] = spk_id
            metrics["query_index"] = i
            results["in_library"].append(metrics)

    # Out-of-library: use all embeddings from out_speakers as queries
    for spk_id in out_speakers:
        embs = all_embeddings.get(spk_id, [])
        for i, emb in enumerate(embs):
            metrics = evaluate_query(emb, centers, selected_matrix, selected_labels)
            metrics["true_speaker"] = spk_id
            metrics["query_index"] = i
            results["out_of_library"].append(metrics)

    # Summary stats
    def summarize(items: list[dict]) -> dict:
        if not items:
            return {}
        center_scores = [x["center_max_score"] for x in items]
        faiss_avgs = [x["faiss_avg_score"] for x in items]
        consensus_avgs = [x["faiss_consensus_avg_score"] for x in items if x["faiss_consensus_avg_score"] > 0]
        consistency_pass_rate = sum(x["faiss_consistency_pass"] for x in items) / len(items)
        return {
            "count": len(items),
            "center_max_mean": float(np.mean(center_scores)),
            "center_max_std": float(np.std(center_scores)),
            "center_max_min": float(np.min(center_scores)),
            "center_max_max": float(np.max(center_scores)),
            "center_max_median": float(np.median(center_scores)),
            "faiss_avg_mean": float(np.mean(faiss_avgs)),
            "faiss_avg_std": float(np.std(faiss_avgs)),
            "faiss_avg_min": float(np.min(faiss_avgs)),
            "faiss_avg_max": float(np.max(faiss_avgs)),
            "faiss_avg_median": float(np.median(faiss_avgs)),
            "consensus_avg_mean": float(np.mean(consensus_avgs)) if consensus_avgs else 0.0,
            "consistency_pass_rate": consistency_pass_rate,
        }

    results["summary"] = {
        "in_library": summarize(results["in_library"]),
        "out_of_library": summarize(results["out_of_library"]),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate stranger similarity distribution.")
    parser.add_argument("--embedding_dir", required=True, help="Directory with .npz embeddings.")
    parser.add_argument("--in_speakers", required=True, help="Comma-separated speaker IDs in library.")
    parser.add_argument("--out_speakers", required=True, help="Comma-separated speaker IDs as strangers.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    in_speakers = [s.strip() for s in args.in_speakers.split(",")]
    out_speakers = [s.strip() for s in args.out_speakers.split(",")]

    print(f"Loading embeddings from {embedding_dir}...")
    all_embeddings = load_embeddings(embedding_dir)
    print(f"Loaded {len(all_embeddings)} speakers.")

    print(f"Building library with {in_speakers}...")
    print(f"Testing strangers {out_speakers}...")

    results = evaluate_group(all_embeddings, in_speakers, out_speakers)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
