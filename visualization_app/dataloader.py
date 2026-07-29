"""Discover and parse the prediction CSVs.

Handles the two CSV schemas produced by the project:

  * VLM pipeline  (scripts/run_vlm_pipeline.py) - one row per image, with a
    `dataset` column and JSON-encoded `objects` / `gt_targets` cells.
  * Taxonomy calibration (scripts/evaluate_taxonomy_labeling.py) - one row per
    (image, class_name), simpler flat columns, no `dataset` column.

The parser is schema-agnostic: it returns every column as-is and only adds a
couple of derived, UI-friendly fields. The frontend renders whatever columns
are present.
"""

import csv
import json
import os
import re

import config

# CSVs can contain very long JSON cells (the `objects` field); lift the limit.
csv.field_size_limit(10 ** 7)

# axis columns we know how to render as gt-vs-pred badges, in display order.
AXIS_PAIRS = [
    ("nature", "gt_nature", "pred_nature"),
    ("biotic", "gt_biotic", "pred_biotic"),
    ("material", "gt_material", "pred_material"),
    # VLM-pipeline image-level nature (COCO/BIG-5).
    ("image nature", "image_gt_nature", "image_pred_nature"),
]

_DATASET_TOKENS = ["imagenet", "places365", "places", "coco", "big5"]


def discover_csvs():
    """Return a list of {path, name, dir, mtime, kind, dataset} for every
    prediction CSV found under config.CSV_SEARCH_DIRS."""
    seen = set()
    out = []
    for base in config.CSV_SEARCH_DIRS:
        if not base or not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                if not name.endswith(".csv"):
                    continue
                # Only prediction CSVs, not arbitrary spreadsheets.
                if "predictions" not in name and "prediction" not in name:
                    continue
                full = os.path.abspath(os.path.join(dirpath, name))
                if full in seen:
                    continue
                seen.add(full)
                out.append({
                    "path": full,
                    "name": name,
                    "dir": dirpath,
                    "mtime": os.path.getmtime(full),
                    "kind": _kind_from_name(name),
                    "dataset": _dataset_from_name(name),
                })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _kind_from_name(name):
    if name.startswith("evaluate_taxonomy") or "taxonomy_calibration" in name:
        return "taxonomy_calibration"
    return "vlm_pipeline"


def _dataset_from_name(name):
    low = name.lower()
    for tok in _DATASET_TOKENS:
        if re.search(r"[_\-]" + tok + r"[_\-]", low) or low.endswith(tok + "_predictions.csv"):
            return "places365" if tok == "places" else tok
    return None


def _maybe_json(value):
    """Parse a cell that looks like JSON (the objects / gt_targets fields)."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def load_csv(path):
    """Parse one prediction CSV.

    Returns a dict:
        {
          "columns": [...],          # header, in file order
          "kind": "vlm_pipeline"|"taxonomy_calibration",
          "rows": [ {col: value, ...,
                     "_dataset": str, "_image_name": str,
                     "_objects": <parsed|None>, "_gt_targets": <parsed|None>},
                    ... ],
        }
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    kind = _kind_from_name(os.path.basename(path))
    file_dataset = _dataset_from_name(os.path.basename(path))

    for r in rows:
        # dataset: explicit column (VLM pipeline) wins, else infer from filename,
        # else from the image_path.
        ds = (r.get("dataset") or "").strip() or file_dataset
        if not ds:
            ds = _dataset_from_path(r.get("image_path", ""))
        r["_dataset"] = ds
        r["_image_name"] = os.path.basename((r.get("image_path") or "").replace("\\", "/"))
        # Pre-parse the JSON-encoded cells so the frontend gets structured data.
        r["_objects"] = _maybe_json(r.get("objects"))
        r["_gt_targets"] = _maybe_json(r.get("gt_targets"))

    return {"columns": columns, "kind": kind, "rows": rows}


def _dataset_from_path(image_path):
    low = (image_path or "").lower()
    if "imagenet" in low:
        return "imagenet"
    if "places" in low:
        return "places365"
    if "coco" in low or "val2017" in low:
        return "coco"
    if "big_5" in low or "big5" in low:
        return "big5"
    return None
