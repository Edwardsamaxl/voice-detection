"""SpeakerRepository: unified seam for speaker DB, vector index, and profiles."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from config import (
    MAX_EMB,
    TOPK,
    TOPK_CONSISTENCY_MIN,
    T_HIGH,
    T_LOW,
    UPDATE_MIN_DURATION,
    UPDATE_DEDUP_THRESHOLD,
)
from .normalize import l2_normalize
from .pool import EmbeddingPool
from .storage import Storage
from .types import IdentificationResult, SpeakerData, SpeakerProfile, VectorEntry


class VectorIndex(Protocol):
    """Adapter seam for vector search backends (FAISS, etc.)."""

    def build(self, vectors: np.ndarray, labels: list[str]) -> None: ...
    def search(self, query: np.ndarray, topk: int = 5) -> list[Any]: ...
    def add(self, vector: np.ndarray, label: str) -> None: ...
    def rebuild(self) -> None: ...


class VectorDb:
    """Independent vector storage: retains selected embeddings per speaker."""

    def __init__(self) -> None:
        self._entries: dict[str, list[VectorEntry]] = {}

    def get_entries(self, spk_id: str) -> list[VectorEntry]:
        return list(self._entries.get(spk_id, []))

    def set_entries(self, spk_id: str, entries: list[VectorEntry]) -> None:
        self._entries[spk_id] = list(entries)

    def all_vectors(self) -> tuple[np.ndarray, list[str]]:
        all_vectors: list[np.ndarray] = []
        all_labels: list[str] = []
        for spk_id, entries in self._entries.items():
            for entry in entries:
                all_vectors.append(entry.embedding)
                all_labels.append(spk_id)
        if not all_vectors:
            return np.empty((0, 0), dtype=np.float32), []
        return np.stack(all_vectors).astype(np.float32), all_labels

    def remove_speaker(self, spk_id: str) -> None:
        self._entries.pop(spk_id, None)

    def to_serializable(self) -> dict[str, Any]:
        """Convert to a dict suitable for NpzStorage."""
        spk_ids: list[str] = []
        embeddings: list[np.ndarray] = []
        durations: list[float] = []
        for spk_id, entries in self._entries.items():
            for entry in entries:
                spk_ids.append(spk_id)
                embeddings.append(entry.embedding)
                durations.append(entry.duration)
        if not spk_ids:
            return {
                "spk_ids": np.array([], dtype=str),
                "embeddings": np.empty((0, 0), dtype=np.float32),
                "durations": np.array([], dtype=np.float32),
            }
        return {
            "spk_ids": np.array(spk_ids, dtype=str),
            "embeddings": np.stack(embeddings).astype(np.float32),
            "durations": np.array(durations, dtype=np.float32),
        }

    @classmethod
    def from_serializable(cls, data: dict[str, np.ndarray]) -> VectorDb:
        db = cls()
        spk_ids = data.get("spk_ids", np.array([], dtype=str))
        embeddings = data.get("embeddings", np.empty((0, 0), dtype=np.float32))
        durations = data.get("durations", np.array([], dtype=np.float32))
        if embeddings.size == 0:
            return db
        for i in range(len(spk_ids)):
            spk_id = str(spk_ids[i])
            emb = embeddings[i].astype(np.float32)
            dur = float(durations[i]) if i < len(durations) else 1.0
            entry = VectorEntry(spk_id=spk_id, embedding=emb, duration=dur)
            db._entries.setdefault(spk_id, []).append(entry)
        return db


class SpeakerRepository:
    """Encapsulates speaker_db, vector_db, and speaker_profile.

    This is the single seam through which the rest of the codebase interacts
    with speaker identity data.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        storage: Storage | None = None,
        vector_storage: Storage | None = None,
    ) -> None:
        self._speakers: dict[str, SpeakerData] = {}
        self._vector_db = VectorDb()
        self._vector_index = vector_index
        self._storage = storage
        self._vector_storage = vector_storage

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
        vector_db = VectorDb()

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
                embedding_count=len(selected_embs),
                profile=None,
            )

            entries = [
                VectorEntry(spk_id=spk_id, embedding=l2_normalize(emb), duration=dur)
                for emb, dur in zip(selected_embs, selected_durations)
            ]
            vector_db.set_entries(spk_id, entries)

        self._speakers = speakers
        self._vector_db = vector_db

        vectors_matrix, labels = vector_db.all_vectors()
        if vectors_matrix.size > 0:
            self._vector_index.build(vectors_matrix, labels)
        else:
            self._vector_index.build(np.empty((0, 0), dtype=np.float32), [])

    # ------------------------------------------------------------------ #
    # Identification (Phase 8)
    # ------------------------------------------------------------------ #

    def identify(self, query_emb: np.ndarray) -> IdentificationResult:
        """Two-stage identification with global FAISS + topk consistency."""
        if not self._speakers:
            return IdentificationResult(speaker=None, score=0.0, confidence="unknown")

        query = l2_normalize(query_emb)

        # Stage 1: center coarse ranking
        center_scores: list[tuple[str, float]] = []
        for spk_id, speaker in self._speakers.items():
            score = float(np.dot(query, speaker.center))
            center_scores.append((spk_id, score))
        center_scores.sort(key=lambda x: x[1], reverse=True)
        n_candidates = min(5, len(center_scores))
        candidates = [spk_id for spk_id, _ in center_scores[:n_candidates]]

        # Quick rejection: if best center score is too low
        best_center_score = center_scores[0][1] if center_scores else 0.0
        if best_center_score < T_LOW:
            return IdentificationResult(speaker=None, score=best_center_score, confidence="low")

        # Stage 2: global FAISS top-k search
        topk_results = self._vector_index.search(query, TOPK)
        if not topk_results:
            return IdentificationResult(speaker=None, score=best_center_score, confidence="low")

        # Determine majority speaker from topk_results
        speakers = [r.speaker for r in topk_results if r.speaker]
        from collections import Counter

        majority_spk, _ = Counter(speakers).most_common(1)[0]

        # Final score: average similarity of majority speaker vectors in top-k
        majority_scores = [r.score for r in topk_results if r.speaker == majority_spk]
        avg_score = float(np.mean(majority_scores)) if majority_scores else 0.0

        if majority_spk in candidates and avg_score >= T_HIGH:
            return IdentificationResult(
                speaker=majority_spk, score=avg_score, confidence="high"
            )

        return IdentificationResult(
            speaker=None, score=avg_score, confidence="low"
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
            embedding_count=len(selected_embs),
            profile=profile,
        )

        entries = [
            VectorEntry(spk_id=spk_id, embedding=l2_normalize(emb), duration=dur)
            for emb, dur in zip(selected_embs, selected_durations)
        ]
        self._vector_db.set_entries(spk_id, entries)
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

        existing = self._vector_db.get_entries(spk_id)
        new_embeddings = [e.embedding for e in existing] + [new_emb.astype(np.float32)]
        new_durations = [e.duration for e in existing] + [duration]

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
            embedding_count=len(new_embeddings),
            profile=speaker.profile,
        )

        entries = [
            VectorEntry(spk_id=spk_id, embedding=l2_normalize(emb), duration=dur)
            for emb, dur in zip(new_embeddings, new_durations)
        ]
        self._vector_db.set_entries(spk_id, entries)
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        """Rebuild vector index from current vector_db."""
        vectors_matrix, labels = self._vector_db.all_vectors()
        if vectors_matrix.size > 0:
            self._vector_index.build(vectors_matrix, labels)
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
            embedding_count=speaker.embedding_count,
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

    def save(
        self,
        speaker_key: str = "speaker_db:main",
        vector_key: str = "vector_db:main",
    ) -> None:
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
                "embedding_count": speaker.embedding_count,
                "profile": profile_data,
            }

        self._storage.save(speaker_key, serializable)

        if self._vector_storage is not None:
            self._vector_storage.save(vector_key, self._vector_db.to_serializable())

    def load(
        self,
        speaker_key: str = "speaker_db:main",
        vector_key: str = "vector_db:main",
    ) -> None:
        if self._storage is None:
            raise RuntimeError("No storage configured")

        raw = self._storage.load(speaker_key)
        speakers: dict[str, SpeakerData] = {}
        vector_db = VectorDb()

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

            center = np.asarray(data["center"], dtype=np.float32)

            # V2.2 backward compatibility: old format stored embeddings in speaker_db
            if "embeddings" in data:
                embeddings = [np.asarray(emb, dtype=np.float32) for emb in data["embeddings"]]
                durations = data.get("durations")
                if durations is None:
                    durations = [1.0] * len(embeddings)
                count = len(embeddings)
                entries = [
                    VectorEntry(spk_id=spk_id, embedding=l2_normalize(emb), duration=dur)
                    for emb, dur in zip(embeddings, durations)
                ]
                vector_db.set_entries(spk_id, entries)
            else:
                count = data.get("embedding_count", 0)

            speakers[spk_id] = SpeakerData(
                spk_id=data["spk_id"],
                center=center,
                embedding_count=count,
                profile=profile,
            )

        self._speakers = speakers

        if self._vector_storage is not None and self._vector_storage.exists(vector_key):
            vec_data = self._vector_storage.load(vector_key)
            vector_db = VectorDb.from_serializable(vec_data)

        self._vector_db = vector_db
        self._rebuild_index()
