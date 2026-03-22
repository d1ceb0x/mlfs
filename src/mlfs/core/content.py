"""
content.py — lightweight content sampling for files.

Rule: never read full files. Sample only the first SAMPLE_BYTES bytes.

Features extracted:
    [0] is_text         — 1.0 if file appears to be UTF-8 text, else 0.0
    [1] entropy         — Shannon entropy of sampled bytes (0.0 - 8.0)
    [2] line_count      — newline count in sample (text files only, else 0.0)
    [3] token_count     — whitespace-split word count in sample (text only, else 0.0)
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np

# How many bytes to read from each file — never more than this
SAMPLE_BYTES = 512

CONTENT_FEATURE_NAMES = [
    "is_text",
    "entropy",
    "line_count",
    "token_count",
]

NUM_CONTENT_FEATURES = len(CONTENT_FEATURE_NAMES)


def _shannon_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy of a byte sequence.

    H = -Σ p(x) * log2(p(x))

    Returns a value in [0.0, 8.0].
    8.0 = perfectly random (encrypted/compressed).
    ~4.5 = typical source code.
    ~3.0 = typical English text.
    0.0 = all bytes identical.
    """
    if not data:
        return 0.0

    # Count occurrences of each byte value (0-255)
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)

    # Compute probabilities — only for bytes that actually appear
    total = len(data)
    probs = counts[counts > 0] / total

    # Shannon entropy formula
    return float(-np.sum(probs * np.log2(probs)))


def _is_text(data: bytes) -> bool:
    """
    Heuristic: attempt UTF-8 decode. If it succeeds, treat as text.
    Also rejects if >30% of bytes are non-printable (catches latin-1 binaries).
    """
    if not data:
        return False
    try:
        decoded = data.decode("utf-8")
        # Extra check: reject if too many non-printable characters
        non_printable = sum(1 for c in decoded if not c.isprintable() and c not in "\n\r\t")
        return (non_printable / len(decoded)) < 0.30
    except (UnicodeDecodeError, ZeroDivisionError):
        return False


def sample(path: Path) -> np.ndarray:
    """
    Sample up to SAMPLE_BYTES from a file and extract content features.

    Returns:
        np.ndarray of shape (NUM_CONTENT_FEATURES,) with dtype float64
    """
    try:
        with open(path, "rb") as f:
            data = f.read(SAMPLE_BYTES)
    except (PermissionError, OSError, FileNotFoundError):
        # Return zeros if we can't read the file
        return np.zeros(NUM_CONTENT_FEATURES, dtype=np.float64)

    is_text = _is_text(data)
    entropy = _shannon_entropy(data)

    if is_text:
        try:
            text = data.decode("utf-8", errors="ignore")
            line_count = float(text.count("\n"))
            token_count = float(len(text.split()))
        except Exception:
            line_count = 0.0
            token_count = 0.0
    else:
        line_count = 0.0
        token_count = 0.0

    return np.array(
        [float(is_text), entropy, line_count, token_count],
        dtype=np.float64,
    )


def sample_all(paths: list[Path]) -> np.ndarray:
    """
    Sample content features for a list of files.

    Returns:
        np.ndarray of shape (n_files, NUM_CONTENT_FEATURES)
    """
    vectors = [sample(p) for p in paths]
    if not vectors:
        return np.empty((0, NUM_CONTENT_FEATURES), dtype=np.float64)
    return np.vstack(vectors)
