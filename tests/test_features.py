import numpy as np
import time
from pathlib import Path
from mlfs.core.features import (
    extract,
    extract_all,
    ExtensionRegistry,
    FEATURE_NAMES,
    NUM_FEATURES,
    F_SIZE, F_EXTENSION_ID, F_NAME_LENGTH, F_DEPTH, F_MODIFIED_DELTA, F_IS_HIDDEN,
)


def test_feature_vector_shape(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec.shape == (NUM_FEATURES,)
    assert vec.dtype == np.float64


def test_file_size(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"x" * 1234)
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec[F_SIZE] == 1234.0


def test_name_length(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("pass")
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec[F_NAME_LENGTH] == len("hello.py")


def test_depth(tmp_path):
    subdir = tmp_path / "a" / "b"
    subdir.mkdir(parents=True)
    f = subdir / "deep.txt"
    f.write_text("deep")
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec[F_DEPTH] == 2.0


def test_hidden_file(tmp_path):
    f = tmp_path / ".hidden"
    f.write_text("secret")
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec[F_IS_HIDDEN] == 1.0


def test_not_hidden(tmp_path):
    f = tmp_path / "visible.txt"
    f.write_text("visible")
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec[F_IS_HIDDEN] == 0.0


def test_extension_registry_consistency():
    registry = ExtensionRegistry()
    id1 = registry.encode(".py")
    id2 = registry.encode(".py")
    id3 = registry.encode(".txt")
    assert id1 == id2
    assert id1 != id3


def test_extension_case_insensitive():
    registry = ExtensionRegistry()
    assert registry.encode(".PY") == registry.encode(".py")


def test_extract_all_shape(tmp_path):
    for i in range(5):
        (tmp_path / f"file_{i}.txt").write_text("x" * i)
    matrix, registry = extract_all(
        [p for p in tmp_path.iterdir() if p.is_file()],
        root=tmp_path,
    )
    assert matrix.shape == (5, NUM_FEATURES)


def test_modified_delta_is_positive(tmp_path):
    f = tmp_path / "recent.txt"
    f.write_text("new")
    registry = ExtensionRegistry()
    vec = extract(f, root=tmp_path, registry=registry)
    assert vec[F_MODIFIED_DELTA] >= 0.0
