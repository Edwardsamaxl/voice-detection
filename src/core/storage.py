"""Unified storage seam with multiple adapters."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Protocol

import numpy as np

from .types import NumpyEncoder


class Storage(Protocol):
    """Abstract seam for persisting and loading arbitrary data."""

    def save(self, key: str, data: Any) -> None: ...
    def load(self, key: str) -> Any: ...
    def exists(self, key: str) -> bool: ...


class JsonStorage:
    """File-based JSON storage for metadata, segments, and profiles."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe_key = key.replace(":", os.sep)
        return os.path.join(self.base_dir, f"{safe_key}.json")

    def save(self, key: str, data: Any) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or self.base_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    def load(self, key: str) -> Any:
        path = self._path(key)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class NpzStorage:
    """File-based numpy storage for embedding matrices and arrays."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe_key = key.replace(":", os.sep)
        return os.path.join(self.base_dir, f"{safe_key}.npz")

    def save(self, key: str, data: np.ndarray | dict[str, np.ndarray]) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or self.base_dir, exist_ok=True)
        if isinstance(data, np.ndarray):
            np.savez(path, data=data)
        else:
            np.savez(path, **data)

    def load(self, key: str) -> dict[str, np.ndarray]:
        path = self._path(key)
        return dict(np.load(path))

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class PickleStorage:
    """File-based pickle storage for complex Python objects (e.g. FAISS indices)."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe_key = key.replace(":", os.sep)
        return os.path.join(self.base_dir, f"{safe_key}.pkl")

    def save(self, key: str, data: Any) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or self.base_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, key: str) -> Any:
        path = self._path(key)
        with open(path, "rb") as f:
            return pickle.load(f)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class MemoryStorage:
    """In-memory storage for testing."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> None:
        self._data[key] = data

    def load(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"Key not found in memory storage: {key}")
        return self._data[key]

    def exists(self, key: str) -> bool:
        return key in self._data
