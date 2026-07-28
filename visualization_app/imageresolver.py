"""Re-root a CSV's stored image_path onto this machine's local image dirs.

The CSVs were produced on a compute node where images lived under paths like
``/home/pmonserrat/datasets/imagenet/extracted_data/n04357314/xxx.JPEG`` - paths
that don't exist here. Given the dataset name and the stored path, this module
finds the corresponding local file (see config.DATASET_IMAGE_ROOTS).

Resolution order for a given (dataset, stored_path):
  1. If the stored path exists as-is, use it (handy if you ever run the app on
     the same machine that produced the CSV).
  2. Re-root: keep the last N path components (config.DATASET_PATH_DEPTH) and
     join them onto the local dataset root.
  3. Fallback: a lazily-built basename index of the whole dataset root, so a
     file still resolves even if its class-folder differs from the stored path.
"""

import os
from functools import lru_cache

import config


def _alias(dataset):
    """Normalise dataset names to the keys used in config."""
    if not dataset:
        return None
    d = dataset.strip().lower()
    if d in ("places", "places-365", "places_365"):
        return "places365"
    if d in ("big-5", "big_5", "big5_dataset"):
        return "big5"
    if d in ("imagenet-1k", "imagenet1k"):
        return "imagenet"
    return d


@lru_cache(maxsize=8)
def _basename_index(root):
    """Map basename -> absolute path for every file under `root` (built once
    per root). Used only as a last-resort fallback, so the one-time walk cost
    is acceptable."""
    index = {}
    if not root or not os.path.isdir(root):
        return index
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            # First occurrence wins; basenames are unique enough in practice
            # for these datasets.
            index.setdefault(name, os.path.join(dirpath, name))
    return index


def resolve(dataset, stored_path):
    """Return an absolute local path to the image, or None if not found."""
    if not stored_path:
        return None

    # 1. As-is.
    if os.path.isfile(stored_path):
        return stored_path

    ds = _alias(dataset)
    root = config.DATASET_IMAGE_ROOTS.get(ds)
    if not root:
        return None

    norm = stored_path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]

    # 2. Re-root using the last N components.
    depth = config.DATASET_PATH_DEPTH.get(ds, 1)
    if parts:
        tail = os.path.join(*parts[-depth:]) if depth <= len(parts) else os.path.join(*parts)
        candidate = os.path.join(root, tail)
        if os.path.isfile(candidate):
            return candidate

    # 3. Basename fallback.
    hit = _basename_index(root).get(parts[-1] if parts else "")
    if hit and os.path.isfile(hit):
        return hit

    return None
