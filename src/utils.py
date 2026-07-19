"""
utils.py

Shared helper for the two top-level evaluation scripts (evaluate_taxonomy_
labeling.py and run_vlm_pipeline.py) to persist results into one JSON file
per script, structured as:

    {
      "<dataset>": {
        "dataset_class_stats": {
          "<config_key>": { ...distinct-class nature/biotic/material counts... },
          ...
        },
        "<model_label>": { ...metrics..., "evaluated_at": "<ISO timestamp>" },
        ...
      },
      ...
    }

Each call merges into whatever is already on disk (rather than overwriting
the whole file), so results for different datasets/models accumulate across
runs. Rerunning the same (dataset, model) pair overwrites just that entry
with the newest results.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def format_duration(seconds):
    """Format a duration in seconds as "D-HH:MM:SS" (SLURM-style elapsed-time
    format), e.g. 3725.4 -> "0-01:02:05". Returns None unchanged (so callers
    storing an unavailable timing, e.g. an artifact without an inference
    footer, keep null rather than a misleading "0-00:00:00")."""
    if seconds is None:
        return None
    total_seconds = int(round(seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"


class BatchProgress:
    """Shared --verbose progress logger for the two top-level scripts' batched
    VLM-inference loops (run_vlm_pipeline.py's run_inference,
    evaluate_taxonomy_labeling.py's main batch loop). Previously each script
    hand-rolled its own version; run_vlm_pipeline.py's just printed "N/M images
    processed" with no timing, while evaluate_taxonomy_labeling.py printed a
    per-batch ETA but computed it off an off-by-one batch count
    (`num_batches - batch_idx + 1`, which over-counts the remaining batches by
    2). This one fixes that and is used by both, so progress output stays
    identical in format and doesn't drift between scripts again.

    ETA is a RUNNING AVERAGE over every batch completed so far (elapsed /
    done), not just the last batch's duration — a single batch can be an
    outlier (a retry, a slow image download), so the running average gives a
    more stable estimate of time remaining.
    """

    def __init__(self, num_batches, label="batch", verbose=True):
        self.num_batches = num_batches
        self.label = label
        self.verbose = verbose
        self._t0 = time.time()
        self._last = self._t0

    def tick(self, batch_idx, n_done=None, n_total=None, extra=None):
        """Call once after finishing batch `batch_idx` (0-indexed). No-op when
        verbose=False, so callers don't need to guard every call site."""
        now = time.time()
        batch_seconds = now - self._last
        self._last = now
        if not self.verbose:
            return
        done = batch_idx + 1
        elapsed = now - self._t0
        avg = elapsed / done
        remaining = avg * (self.num_batches - done)
        msg = (f"[INFO] {self.label} {done}/{self.num_batches} done in {batch_seconds:.1f}s "
               f"(avg {avg:.1f}s/batch)")
        if n_done is not None and n_total is not None:
            msg += f" | {n_done}/{n_total} items"
        msg += f" | elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)


def update_results_store(path, dataset, model, metrics):
    """Merge `metrics` into results_store[dataset][model] at `path`, creating
    or updating the file on disk, and return the full updated store."""
    path = Path(path)
    store = {}
    if path.exists():
        with open(path) as f:
            store = json.load(f)

    entry = dict(metrics)
    entry["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    store.setdefault(dataset, {})[model] = entry

    with open(path, "w") as f:
        json.dump(store, f, indent=4)

    return store


def compute_class_stats(targets):
    """Distinct-class nature/biotic/material breakdown over a flat list of GT
    target dicts (each carrying class_name/synset_id/gt_nature/gt_biotic/
    gt_material, per src/loaders/dataset_loader.py's target shape).

    De-duplicates by synset_id (falling back to class_name when no synset is
    attached, e.g. BIG-5's holistic "scene" targets) so a class recurring
    across many images/targets is only counted once — this reports the
    dataset's TARGET CLASS composition, not per-instance/per-image counts.
    Biotic and material are only meaningful for nature-positive classes (a
    non-nature class carries gt_biotic=gt_material=None), so those two
    breakdowns are counted over the nature subset only.
    """
    seen = {}
    for t in targets:
        if t.get("gt_nature") is None:
            continue
        key = t.get("synset_id") or t.get("class_name")
        if key is None or key in seen:
            continue
        seen[key] = t

    classes = list(seen.values())
    nature_classes = [c for c in classes if c["gt_nature"]]

    return {
        "total_classes": len(classes),
        "nature": sum(1 for c in nature_classes if c["gt_nature"]),
        "no_nature": len(classes) - len(nature_classes),
        "biotic": sum(1 for c in nature_classes if c.get("gt_biotic") is True),
        "abiotic": sum(1 for c in nature_classes if c.get("gt_biotic") is False),
        "material": sum(1 for c in nature_classes if c.get("gt_material") is True),
        "immaterial": sum(1 for c in nature_classes if c.get("gt_material") is False),
    }


def update_dataset_class_stats(path, dataset, config_key, stats):
    """Store `stats` (from compute_class_stats) at
    results_store[dataset]["dataset_class_stats"][config_key], sibling to the
    per-model entries under that dataset. Since sampling is deterministic
    (fixed seed=42, same --max_samples -> same subset), this is written once
    per distinct sampling configuration and left untouched by reruns of the
    same configuration; DIFFERENT configurations (e.g. 1000 vs the full
    50000-image dataset) accumulate side by side under their own config_key
    rather than overwriting one another."""
    path = Path(path)
    store = {}
    if path.exists():
        with open(path) as f:
            store = json.load(f)

    store.setdefault(dataset, {}).setdefault("dataset_class_stats", {})[config_key] = stats

    with open(path, "w") as f:
        json.dump(store, f, indent=4)

    return store
