"""Configuration for the prediction-visualization app.

Everything the app needs to know about *where things live on THIS machine*
lives here. The user-editable values are read from ``config.json`` next to this
file (see the keys below); this module just supplies defaults and normalises
them into the shape the rest of the code expects.

Two kinds of location matter:

  1. CSV_SEARCH_DIRS - directories scanned (recursively) for the prediction
     ``*predictions*.csv`` files produced by ``scripts/run_vlm_pipeline.py`` and
     ``scripts/evaluate_taxonomy_labeling.py``. The file picker lists everything
     found here. (config.json key: ``csv_search_dirs``, relative paths are
     resolved against this folder.)

  2. DATASET_IMAGE_ROOTS - where the actual images live locally. The
     ``image_path`` stored in a CSV was written on the compute node
     (e.g. ``/home/pmonserrat/datasets/...``) and does NOT exist on this
     machine, so we re-root it: for imagenet/places we keep the last two path
     components (``<class_id>/<file>``) and join them onto the local root; for
     big5 we match on the bare filename. See ``imageresolver.py``.
     (config.json key: ``dataset_roots``.)
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# ---- defaults (used when config.json is missing a key) --------------------
HOST = "127.0.0.1"
PORT = 8070

DATASET_IMAGE_ROOTS = {
    "imagenet": "/home/pau/Documentos/Universidad/Master/TFM/ImageNet/extracted_data",
    "places365": "/home/pau/Documentos/Universidad/Master/TFM/Places/val_formatted",
    "big5": "/home/pau/Documentos/Universidad/Master/TFM/BIG_5_Dataset",
    # COCO not downloaded yet - set its path in config.json when available.
}

CSV_SEARCH_DIRS = [
    os.path.join(_REPO, "results"),
    "/home/pau/Documentos/Universidad/Master/TFM",
]

# How many trailing path components to keep from the CSV's image_path when
# re-rooting onto the local root. imagenet/places store <class_id>/<file>
# (2 components); big5/coco are flat (1 component = just the filename).
DATASET_PATH_DEPTH = {
    "imagenet": 2,
    "places365": 2,
    "big5": 1,
    "coco": 1,
}

# `places` is the alias used in config.json's dataset_roots; internally we key
# everything on `places365` (matching the pipeline's dataset name).
_ROOT_KEY_ALIASES = {"places": "places365"}


def _apply_config_json():
    """Overlay user settings from config.json (if present) onto the defaults."""
    global HOST, PORT
    path = os.path.join(_HERE, "config.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (ValueError, OSError) as exc:
        print(f"[config] warning: could not read config.json ({exc}); using defaults")
        return

    HOST = cfg.get("host", HOST)
    PORT = int(cfg.get("port", PORT))

    for key, root in (cfg.get("dataset_roots") or {}).items():
        if not root:
            continue  # empty string = "not available yet" (e.g. coco)
        DATASET_IMAGE_ROOTS[_ROOT_KEY_ALIASES.get(key, key)] = root

    dirs = cfg.get("csv_search_dirs")
    if dirs:
        resolved = []
        for d in dirs:
            resolved.append(d if os.path.isabs(d) else os.path.normpath(os.path.join(_HERE, d)))
        # keep it a module-level list without rebinding the name the rest of
        # the code imported.
        CSV_SEARCH_DIRS[:] = resolved


_apply_config_json()
