"""SpeakerRepository: unified seam for speaker DB, vector index, and profiles."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from config import (
    MAX_EMB,
    TOPK,
    TOPK_CONSISTENCY_MIN,
    T_HIGH,
    UPDATE_MIN_DURATION,
    UPDATE_DEDUP_THRESHOLD,
)
from .normalize import l2_normalize
from .pool import EmbeddingPool
from .storage import Storage
from .types import IdentificationResult, SpeakerData, SpeakerProfile


class VectorIndex(Protocol):
    """Adapter seam for vector search backends (FAISS, etc.)."""

    def build(self, vectors: np.ndarray, labels: list[str]) -> None: ...
    def search(self, query: np.ndarray, topk: int = 5) -> list[Any]: ...
    def add(self, vector: np.ndarray, label: str) -> None: ...
    def rebuild(self) -> None: ...


class SpeakerRepository:
    """Encapsulates speaker_db, vector_db, and speaker_profile.

    This is the single seam through which the rest of the codebase interacts
    with speaker identity data.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        storage: Storage | None = None,
    ) -> None:
        self._speakers: dict[str, SpeakerData] = {}
        self._vector_index = vector_index
        self._storage = storage

    # ------------------------------------------------------------------ #
    # Build from pool (Phase 4-5 entry point)
    # ------------------------------------------------------------------ #

    def build_from_pool(self, pool: EmbeddingPool, max_emb: int = MAX_EMB) -> None:
        """Aggregate pool segments into speakers and rebuild the vector index."""
        groups: dict[str, list] = {}
        for seg in pool:
            spk_id = seg.global_speaker
            if spk_id is None or seg.embedding is None:
                continue
            groups.setdefault(spk_id, []).append(seg)

        speakers: dict[str, SpeakerData] = {}
        all_vectors: list[np.ndarray] = []
        all_labels: list[str] = []

        for spk_id, segments in groups.items():
            embeddings = [seg.embedding for seg in segments]
            durations = [seg.duration for seg in segments]

            matrix = np.stack(embeddings)
            weights = np.asarray(durations, dtype=np.float32)
            center = l2_normalize(np.average(matrix, axis=0, weights=weights)).astype(np.float32)

            # Select embeddings closest to weighted center
            if len(embeddings) <= max_emb:
                selected_embs = embeddings
                selected_durations = durations
            else:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized_matrix = matrix / norms
                center_norm = np.linalg.norm(center)
                if center_norm == 0:
                    center_norm = 1.0
                normalized_center = center / center_norm
                sims = np.dot(normalized_matrix, normalized_center)
                topk_indices = np.argsort(sims)[-max_emb:]
                selected_embs = [embeddings[i] for i in topk_indices]
                selected_durations = [durations[i] for i in topk_indices]

            speakers[spk_id] = SpeakerData(
                spk_id=spk_id,
                center=center,
                embeddings=selected_embs,
                durations=selected_durations,
                profile=None,
            )

            for emb in selected_embs:
                all_vectors.append(l2_normalize(emb))
                all_labels.append(spk_id)

        self._speakers = speakers

        if all_vectors:
            vectors_matrix = np.stack(all_vectors).astype(np.float32)
            self._vector_index.build(vectors_matrix, all_labels)
        else:
            self._vector_index.build(
                np.empty((0, 0), dtype=np.float32), []
            )

    # ------------------------------------------------------------------ #
    # Identification (Phase 8)
    # ------------------------------------------------------------------ #

    def identify(self, query_emb: np.ndarray) -> IdentificationResult:
        """Dual-threshold identification: FAISS TopK + consistency check."""
        from src.recognition.verify import topk_consistency

        results = self._vector_index.search(query_emb, topk=TOPK)
        if not results:
            return IdentificationResult(
                speaker=None, score=0.0, confidence="unknown"
            )

        best = results[0]
        best_score = float(best.score if hasattr(best, "score") else best["score"])
        best_speaker = str(best.speaker if hasattr(best, "speaker") else best["speaker"])

        consistent = topk_consistency(
            results,
            k=TOPK,
            min_consistency=TOPK_CONSISTENCY_MIN,
        )

        if best_score >= T_HIGH and consistent:
            return IdentificationResult(
                speaker=best_speaker, score=best_score, confidence="high"
            )

        return IdentificationResult(
            speaker=None, score=best_score, confidence="low"
        )

    # ------------------------------------------------------------------ #
    # Mutation (Phase 9)
    # ------------------------------------------------------------------ #

    def add_speaker(
        self,
        spk_id: str,
        embeddings: list[np.ndarray],
        durations: list[float],
        profile: SpeakerProfile | None = None,
    ) -> None:
        """Add a brand-new speaker to the repository.

        - Compute weighted center using durations as weights.
        - If embeddings count exceeds MAX_EMB, keep the MAX_EMB closest to center.
        - L2-normalize center and each retained embedding.
        - Create SpeakerData and rebuild the vector index.
        """
        if not embeddings:
            raise ValueError("embeddings must not be empty")
        if len(embeddings) != len(durations):
            raise ValueError("embeddings and durations must have the same length")

        matrix = np.stack(embeddings)
        weights = np.asarray(durations, dtype=np.float32)
        center = l2_normalize(np.average(matrix, axis=0, weights=weights)).astype(np.float32)

        if len(embeddings) <= MAX_EMB:
            selected_embs = embeddings
            selected_durations = durations
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_matrix = matrix / norms
            center_norm = np.linalg.norm(center)
            if center_norm == 0:
                center_norm = 1.0
            normalized_center = center / center_norm
            sims = np.dot(normalized_matrix, normalized_center)
            topk_indices = np.argsort(sims)[-MAX_EMB:]
            selected_embs = [embeddings[i] for i in topk_indices]
            selected_durations = [durations[i] for i in topk_indices]

        self._speakers[spk_id] = SpeakerData(
            spk_id=spk_id,
            center=center,
            embeddings=selected_embs,
            durations=selected_durations,
            profile=profile,
        )
        self._rebuild_index()

    def update_speaker(self, spk_id: str, new_emb: np.ndarray, duration: float) -> bool:
        """Conditionally add a new embedding to a speaker and rebuild index."""
        if duration < UPDATE_MIN_DURATION:
            return False

        speaker = self._speakers.get(spk_id)
        if speaker is None:
            return False

        # Deduplication
        center = speaker.center
        sim = float(np.dot(l2_normalize(new_emb), center))
        if sim > UPDATE_DEDUP_THRESHOLD:
            return False

        new_embeddings = speaker.embeddings + [new_emb.astype(np.float32)]
        new_durations = (speaker.durations or [1.0] * len(speaker.embeddings)) + [duration]

        # Recalculate weighted center
        matrix = np.stack(new_embeddings)
        weights = np.asarray(new_durations, dtype=np.float32)
        new_center = l2_normalize(np.average(matrix, axis=0, weights=weights)).astype(np.float32)

        # Prune embeddings: keep MAX_EMB closest to the new center
        if len(new_embeddings) > MAX_EMB:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_matrix = matrix / norms
            center_norm = np.linalg.norm(new_center)
            if center_norm == 0:
                center_norm = 1.0
            normalized_center = new_center / center_norm
            sims = np.dot(normalized_matrix, normalized_center)
            topk_indices = np.argsort(sims)[-MAX_EMB:]
            new_embeddings = [new_embeddings[i] for i in topk_indices]
            new_durations = [new_durations[i] for i in topk_indices]

        self._speakers[spk_id] = SpeakerData(
            spk_id=spk_id,
            center=new_center,
            embeddings=new_embeddings,
            durations=new_durations,
            profile=speaker.profile,
        )

        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        """Rebuild vector index from current speakers."""
        all_vectors: list[np.ndarray] = []
        all_labels: list[str] = []

        for spk_id, speaker in self._speakers.items():
            for emb in speaker.embeddings:
                all_vectors.append(l2_normalize(emb))
                all_labels.append(spk_id)

        if all_vectors:
            vectors_matrix = np.stack(all_vectors).astype(np.float32)
            self._vector_index.build(vectors_matrix, all_labels)
        else:
            self._vector_index.build(
                np.empty((0, 0), dtype=np.float32), []
            )

    def rebuild(self) -> None:
        """Public method to rebuild the vector index."""
        self._rebuild_index()

    def assign_name(self, spk_id: str, name: str) -> None:
        speaker = self._speakers.get(spk_id)
        if speaker is None:
            raise KeyError(f"Speaker not found: {spk_id}")
        profile = SpeakerProfile(name=name)
        self._speakers[spk_id] = SpeakerData(
            spk_id=speaker.spk_id,
            center=speaker.center,
            embeddings=speaker.embeddings,
            durations=speaker.durations,
            profile=profile,
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_speaker(self, spk_id: str) -> SpeakerData | None:
        return self._speakers.get(spk_id)

    def all_speakers(self) -> list[str]:
        return list(self._speakers.keys())

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, key: str = "speaker_db:main") -> None:
        if self._storage is None:
            raise RuntimeError("No storage configured")

        serializable: dict[str, dict] = {}
        for spk_id, speaker in self._speakers.items():
            profile_data: dict[str, Any] = {}
            if speaker.profile is not None:
                profile_data = {
                    "name": speaker.profile.name,
                    "alias": speaker.profile.alias,
                    "gender": speaker.profile.gender,
                    "notes": speaker.profile.notes,
                    "created_at": speaker.profile.created_at,
                }
            serializable[spk_id] = {
                "spk_id": speaker.spk_id,
                "center": speaker.center.tolist(),
                "embeddings": [emb.tolist() for emb in speaker.embeddings],
                "durations": speaker.durations,
                "profile": profile_data,
            }

        self._storage.save(key, serializable)

    def load(self, key: str = "speaker_db:main") -> None:
        if self._storage is None:
            raise RuntimeError("No storage configured")

        raw = self._storage.load(key)
        speakers: dict[str, SpeakerData] = {}

        for spk_id, data in raw.items():
            profile_data = data.get("profile", {})
            profile = None
            if profile_data and profile_data.get("name"):
                profile = SpeakerProfile(
                    name=profile_data.get("name", "UNKNOWN"),
                    alias=profile_data.get("alias"),
                    gender=profile_data.get("gender"),
                    notes=profile_data.get("notes"),
                    created_at=profile_data.get("created_at"),
                )

            embeddings = [np.asarray(emb, dtype=np.float32) for emb in data["embeddings"]]
            durations = data.get("durations")
            if durations is None:
                durations = [1.0] * len(embeddings)

            speakers[spk_id] = SpeakerData(
                spk_id=data["spk_id"],
                center=np.asarray(data["center"], dtype=np.float32),
                embeddings=embeddings,
                durations=durations,
                profile=profile,
            )

        self._speakers = speakers
