"""Clustering and global speaker ID assignment."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import DEDUP_THRESHOLD, DISTANCE_THRESHOLD
from src.core.pool import EmbeddingPool
from src.core.types import SpeakerSegment

_SLR80_SPEAKER_PATTERN = re.compile(r"\bbur_(\d+)_")


def deduplicate_segments(
    segments: Sequence[SpeakerSegment],
    threshold: float = DEDUP_THRESHOLD,
) -> list[SpeakerSegment]:
    """Remove near-duplicate normalized embeddings from a segment sequence."""
    kept: list[SpeakerSegment] = []
    kept_vectors: list[np.ndarray] = []

    for segment in segments:
        emb = segment.embedding
        if emb is None:
            continue
        vector = np.asarray(emb, dtype=np.float32)
        if kept_vectors:
            sims = np.dot(np.stack(kept_vectors), vector)
            if float(np.max(sims)) > threshold:
                continue
        kept.append(segment.with_embedding(vector))
        kept_vectors.append(vector)
    return kept


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalized vectors."""
    return float(1 - np.dot(a, b))


def _compute_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[int, tuple[np.ndarray, int]]:
    """Return {label: (centroid, count)} for each cluster."""
    result: dict[int, tuple[np.ndarray, int]] = {}
    for label in np.unique(labels):
        mask = labels == label
        cluster_embs = embeddings[mask]
        centroid = np.mean(cluster_embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        result[int(label)] = (centroid, cluster_embs.shape[0])
    return result


def _merge_by_centroid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroid_threshold: float,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """Merge clusters whose centroids are within cosine distance threshold."""
    centroids = _compute_centroids(embeddings, labels)

    valid_labels = [
        lbl for lbl, (_, count) in centroids.items() if count >= min_cluster_size
    ]
    noise_labels = [
        lbl for lbl, (_, count) in centroids.items() if count < min_cluster_size
    ]

    if not valid_labels:
        unique = np.unique(labels)
        mapping = {int(u): i for i, u in enumerate(unique)}
        return np.array([mapping[int(l)] for l in labels], dtype=int)

    merged_map: dict[int, int] = {}
    next_id = 0

    for i, lbl_a in enumerate(valid_labels):
        if lbl_a in merged_map:
            continue
        merged_map[lbl_a] = next_id
        centroid_a = centroids[lbl_a][0]

        for lbl_b in valid_labels[i + 1 :]:
            if lbl_b in merged_map:
                continue
            dist = _cosine_distance(centroid_a, centroids[lbl_b][0])
            if dist < centroid_threshold:
                merged_map[lbl_b] = next_id
        next_id += 1

    final_labels = np.full_like(labels, -1)
    for old_lbl, new_lbl in merged_map.items():
        final_labels[labels == old_lbl] = new_lbl

    for noise_lbl in noise_labels:
        noise_mask = labels == noise_lbl
        if not np.any(noise_mask):
            continue
        noise_embs = embeddings[noise_mask]
        noise_centroid = np.mean(noise_embs, axis=0)
        norm = np.linalg.norm(noise_centroid)
        if norm > 0:
            noise_centroid = noise_centroid / norm

        best_lbl = valid_labels[0]
        best_dist = float("inf")
        for vl in valid_labels:
            d = _cosine_distance(noise_centroid, centroids[vl][0])
            if d < best_dist:
                best_dist = d
                best_lbl = vl
        final_labels[noise_mask] = merged_map[best_lbl]

    return final_labels


def agg_clustering(
    embeddings: np.ndarray,
    distance_threshold: float = DISTANCE_THRESHOLD,
    centroid_threshold: float | None = None,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """Cluster embeddings with cosine-distance agglomerative clustering.

    When *centroid_threshold* is provided, a second merge stage is applied:
    clusters whose centroids are within *centroid_threshold* cosine distance
    are merged.  This yields far fewer fragments than a single threshold.
    """
    if embeddings.size == 0:
        return np.array([], dtype=int)
    if embeddings.shape[0] == 1:
        return np.array([0], dtype=int)

    kwargs = {
        "n_clusters": None,
        "distance_threshold": distance_threshold,
        "linkage": "average",
    }
    try:
        model = AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="cosine", **kwargs)
    labels = model.fit_predict(embeddings)

    if centroid_threshold is not None and centroid_threshold > 0:
        labels = _merge_by_centroid(
            embeddings, labels, centroid_threshold, min_cluster_size
        )

    return labels


def assign_global_speakers(
    pool: EmbeddingPool,
    distance_threshold: float = DISTANCE_THRESHOLD,
    centroid_threshold: float | None = None,
    min_cluster_size: int = 1,
) -> EmbeddingPool:
    """Assign SPK_<label> IDs to all segments in a V2 embedding pool."""
    labels = agg_clustering(
        pool.to_matrix(),
        distance_threshold=distance_threshold,
        centroid_threshold=centroid_threshold,
        min_cluster_size=min_cluster_size,
    )
    return pool.apply_labels([int(label) for label in labels])


def assign_global_speakers_from_slr80_filename(
    pool: EmbeddingPool,
    prefix: str = "BUR_",
) -> EmbeddingPool:
    """Assign global speakers from SLR80 file names.

    SLR80 utterances are named like ``bur_0366_5281755035.wav`` where the
    middle numeric field is the speaker id. For this dataset, using that label
    is the correct validation baseline; unsupervised clustering fragments the
    same 20 real speakers into hundreds of artificial clusters.
    """
    labeled_segments = []
    for seg in pool:
        source = f"{seg.segment_id} {seg.file}"
        match = _SLR80_SPEAKER_PATTERN.search(source)
        if match is None:
            raise ValueError(
                "Cannot infer SLR80 speaker id from "
                f"segment_id={seg.segment_id!r}, file={seg.file!r}"
            )
        labeled_segments.append(seg.with_global_speaker(f"{prefix}{match.group(1)}"))
    return EmbeddingPool(labeled_segments)
