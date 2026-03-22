"""
features.py — extract a fixed-length feature vector from a file path.

Feature vector layout (6 features):
    [0] file_size       — size in bytes
    [1] extension_id    — file extension encoded as integer
    [2] name_length     — length of filename (no path)
    [3] depth           — directory depth relative to scan root
    [4] modified_delta  — seconds since last modified
    [5] is_hidden       — 1 if filename starts with '.', else 0
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Feature index constants — use these instead of magic numbers
F_SIZE = 0
F_EXTENSION_ID = 1
F_NAME_LENGTH = 2
F_DEPTH = 3
F_MODIFIED_DELTA = 4
F_IS_HIDDEN = 5

FEATURE_NAMES = [
    "file_size",
    "extension_id",
    "name_length",
    "depth",
    "modified_delta",
    "is_hidden",
]

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class ExtensionRegistry:
    """
    Maps file extensions to integer IDs.
    Unknown extensions get assigned a new ID on first encounter.
    """

    _registry: dict[str, int] = field(default_factory=dict)
    _counter: int = 0

    def encode(self, ext: str) -> int:
        ext = ext.lower()
        if ext not in self._registry:
            self._registry[ext] = self._counter
            self._counter += 1
        return self._registry[ext]

    @property
    def mapping(self) -> dict[str, int]:
        return dict(self._registry)


def extract(
    path: Path,
    root: Path,
    registry: ExtensionRegistry,
    now: float | None = None,
) -> np.ndarray:
    """
    Extract a feature vector for a single file.

    Args:
        path:     absolute path to the file
        root:     the scan root (used to compute depth)
        registry: shared ExtensionRegistry across all files in a scan
        now:      current timestamp (injectable for testing)

    Returns:
        np.ndarray of shape (NUM_FEATURES,) with dtype float64
    """
    if now is None:
        now = time.time()

    stat = path.stat()

    # [0] file size in bytes
    file_size = float(stat.st_size)

    # [1] extension encoded as int
    extension_id = float(registry.encode(path.suffix))

    # [2] length of just the filename
    name_length = float(len(path.name))

    # [3] depth relative to root (root itself = 0)
    try:
        depth = float(len(path.relative_to(root).parts) - 1)
    except ValueError:
        depth = 0.0

    # [4] seconds since last modified
    modified_delta = float(now - stat.st_mtime)

    # [5] hidden file flag
    is_hidden = 1.0 if path.name.startswith(".") else 0.0

    return np.array(
        [file_size, extension_id, name_length, depth, modified_delta, is_hidden],
        dtype=np.float64,
    )


def extract_all(
    paths: list[Path],
    root: Path,
    registry: ExtensionRegistry | None = None,
) -> tuple[np.ndarray, ExtensionRegistry]:
    """
    Extract feature vectors for a list of files.

    Returns:
        matrix: np.ndarray of shape (n_files, NUM_FEATURES)
        registry: the ExtensionRegistry used (reusable for future scans)
    """
    if registry is None:
        registry = ExtensionRegistry()

    now = time.time()
    vectors = []

    for path in paths:
        try:
            vec = extract(path, root, registry, now=now)
            vectors.append(vec)
        except (PermissionError, FileNotFoundError, OSError):
            # Skip files we can't stat
            continue

    if not vectors:
        return np.empty((0, NUM_FEATURES), dtype=np.float64), registry

    return np.vstack(vectors), registry
