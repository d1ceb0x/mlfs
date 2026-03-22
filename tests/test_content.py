import numpy as np
import pytest
from pathlib import Path
from mlfs.core.content import (
    sample,
    sample_all,
    _shannon_entropy,
    _is_text,
    NUM_CONTENT_FEATURES,
    CONTENT_FEATURE_NAMES,
)


# --- entropy tests ---

def test_entropy_empty():
    assert _shannon_entropy(b"") == 0.0


def test_entropy_single_byte():
    # All same byte = zero entropy
    assert _shannon_entropy(b"\x00" * 100) == 0.0


def test_entropy_max():
    # All 256 byte values equally = ~8.0
    data = bytes(range(256))
    h = _shannon_entropy(data)
    assert 7.9 < h <= 8.0


def test_entropy_text_is_moderate():
    # Normal English text should be mid-range
    data = b"the quick brown fox jumps over the lazy dog " * 10
    h = _shannon_entropy(data)
    assert 3.0 < h < 6.0


def test_entropy_random_is_high():
    # Random bytes should be high entropy
    rng = np.random.default_rng(42)
    data = bytes(rng.integers(0, 256, size=512, dtype=np.uint8))
    h = _shannon_entropy(data)
    assert h > 7.0


# --- is_text tests ---

def test_is_text_with_plain_text():
    assert _is_text(b"hello world\nthis is a text file\n") is True


def test_is_text_with_python_code():
    assert _is_text(b"def foo():\n    return 42\n") is True


def test_is_text_with_binary():
    assert _is_text(bytes(range(256))) is False


def test_is_text_empty():
    assert _is_text(b"") is False


# --- sample tests ---

def test_sample_shape(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world\n" * 20)
    vec = sample(f)
    assert vec.shape == (NUM_CONTENT_FEATURES,)
    assert vec.dtype == np.float64


def test_sample_text_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 42\n" * 10)
    vec = sample(f)
    assert vec[0] == 1.0          # is_text
    assert vec[1] > 0.0           # entropy > 0
    assert vec[2] > 0.0           # line_count > 0
    assert vec[3] > 0.0           # token_count > 0


def test_sample_binary_file(tmp_path):
    f = tmp_path / "binary.bin"
    f.write_bytes(bytes(range(256)) * 2)
    vec = sample(f)
    assert vec[0] == 0.0          # is_text = False
    assert vec[2] == 0.0          # line_count = 0 for binary
    assert vec[3] == 0.0          # token_count = 0 for binary


def test_sample_missing_file(tmp_path):
    vec = sample(tmp_path / "nonexistent.txt")
    assert vec.shape == (NUM_CONTENT_FEATURES,)
    assert np.all(vec == 0.0)


def test_sample_all_shape(tmp_path):
    files = []
    for i in range(5):
        f = tmp_path / f"file_{i}.txt"
        f.write_text(f"content {i}\n" * 10)
        files.append(f)

    matrix = sample_all(files)
    assert matrix.shape == (5, NUM_CONTENT_FEATURES)


def test_high_entropy_file(tmp_path):
    # Simulate a suspicious file with random bytes but .txt extension
    rng = np.random.default_rng(0)
    f = tmp_path / "suspicious.txt"
    f.write_bytes(bytes(rng.integers(0, 256, size=512, dtype=np.uint8)))
    vec = sample(f)
    # Should NOT be detected as text, and entropy should be high
    assert vec[0] == 0.0          # is_text = False
    assert vec[1] > 7.0           # high entropy — suspicious!
