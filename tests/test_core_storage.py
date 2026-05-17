"""Tests for src.core.storage adapters."""

import json

import numpy as np
import pytest

from src.core.storage import JsonStorage, MemoryStorage, NpzStorage, PickleStorage
from src.core.types import NumpyEncoder


def test_numpy_encoder():
    arr = np.array([1.0, 2.0])
    s = json.dumps({"data": arr}, cls=NumpyEncoder)
    assert "1.0" in s


def test_json_storage_roundtrip(tmp_path):
    storage = JsonStorage(str(tmp_path))
    storage.save("test:key", {"value": 42})
    assert storage.exists("test:key")
    assert storage.load("test:key") == {"value": 42}


def test_json_storage_nested_key(tmp_path):
    storage = JsonStorage(str(tmp_path))
    storage.save("speaker_db:main", {"speakers": ["SPK_0"]})
    assert (tmp_path / "speaker_db" / "main.json").exists()


def test_npz_storage_roundtrip(tmp_path):
    storage = NpzStorage(str(tmp_path))
    arr = np.array([1.0, 2.0, 3.0])
    storage.save("embeddings:test", arr)
    loaded = storage.load("embeddings:test")
    assert np.allclose(loaded["data"], arr)


def test_npz_storage_dict(tmp_path):
    storage = NpzStorage(str(tmp_path))
    data = {"a": np.array([1.0]), "b": np.array([2.0])}
    storage.save("test", data)
    loaded = storage.load("test")
    assert np.allclose(loaded["a"], [1.0])


def test_pickle_storage_roundtrip(tmp_path):
    storage = PickleStorage(str(tmp_path))
    storage.save("obj:key", {"complex": [1, 2, 3]})
    assert storage.load("obj:key") == {"complex": [1, 2, 3]}


def test_memory_storage_roundtrip():
    storage = MemoryStorage()
    storage.save("key", "value")
    assert storage.exists("key")
    assert storage.load("key") == "value"


def test_memory_storage_missing_raises():
    storage = MemoryStorage()
    with pytest.raises(KeyError):
        storage.load("missing")
