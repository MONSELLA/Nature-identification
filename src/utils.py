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

import contextlib
import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def crosses_decile(prev_done, done, total, step=0.1):
    """Whether progress from `prev_done` to `done` (out of `total`) just
    crossed into a new `step`-sized fraction of the total, or reached
    completion. Shared by every batched --verbose progress print in the
    pipeline (CLIP text/image encoding, BatchProgress.tick, per-image
    scoring) so all of them log roughly every `step` fraction of the total
    (10% by default) instead of once per batch regardless of batch size —
    unthrottled per-batch prints are unusable once batch counts get large
    (e.g. one line per 77-text CLIP batch across an 80k-text candidate-vocab
    encode: ~1000 lines for one encode_text call). Always True at/past
    completion so the final state is never silently skipped.
    """
    if total <= 0 or done >= total:
        return True
    return int(prev_done / total / step) != int(done / total / step)


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
    """Shared --verbose progress logger for the batched VLM-inference and
    CLIP-encoding loops (run_vlm_pipeline.py's run_inference, the grounding
    pipeline, the CLIP encoders). Previously each loop hand-rolled its own
    version, with inconsistent output and at least one off-by-one ETA
    (`num_batches - batch_idx + 1`, over-counting the remaining batches by 2).
    One implementation means progress output stays identical in format and
    cannot drift between call sites again.

    ETA is a RUNNING AVERAGE over every batch completed so far (elapsed /
    done), not just the last batch's duration — a single batch can be an
    outlier (a retry, a slow image download), so the running average gives a
    more stable estimate of time remaining.
    """

    def __init__(self, num_batches, label="batch", verbose=True, report_every=0.1):
        self.num_batches = num_batches
        self.label = label
        self.verbose = verbose
        self.report_every = report_every
        self._t0 = time.time()
        self._last = self._t0

    def tick(self, batch_idx, n_done=None, n_total=None, extra=None):
        """Call once after finishing batch `batch_idx` (0-indexed). No-op when
        verbose=False, so callers don't need to guard every call site.

        Only actually PRINTS every `report_every` fraction of num_batches
        (default 10%, always including the final batch) — see
        utils.crosses_decile. Timing bookkeeping (elapsed/avg/ETA) still
        accumulates every tick regardless, so the numbers in whichever line
        DOES print are correct running averages, not just stats for the one
        reported batch.

        `report_every=0` (or None) disables the throttle entirely and prints
        EVERY batch. Used by the VLM inference loop, where one line per batch
        is what you actually want: batches are minutes long and number only in
        the hundreds, so per-batch timing/ETA is useful rather than spam —
        unlike the CLIP-encode loops crosses_decile's default was sized for,
        which can run ~1000 tiny batches inside a single call.
        """
        now = time.time()
        batch_seconds = now - self._last
        self._last = now
        if not self.verbose:
            return
        done = batch_idx + 1
        if self.report_every and not crosses_decile(batch_idx, done, self.num_batches, self.report_every):
            return
        elapsed = now - self._t0
        avg = elapsed / done
        # Clamp at 0 rather than let a caller's UNDER-counted num_batches (or
        # more batches genuinely running than were estimated up front) drive
        # this negative. Confirmed a real production bug: `ground_records`
        # falls back to num_batches=1 whenever its own caller doesn't know the
        # true total (n_total=None, which is EVERY real run — --max_samples,
        # the only source of a total there, is a smoke-test-only flag), so
        # `done` sails past that "1" from the second batch on, and
        # `format_duration` renders the resulting negative seconds as a
        # nonsensical "-1-23:56:48" (Python's divmod floors a negative
        # dividend toward -inf, so -3372 seconds becomes days=-1 plus a large
        # positive remainder, not "-0-00:56:12" as you'd naively expect). "n/a"
        # here is an honest signal the original estimate was wrong, which a
        # more negative number dressed up as a duration is not.
        remaining = max(0.0, avg * (self.num_batches - done))
        remaining_str = format_duration(remaining) if done <= self.num_batches else "n/a"
        msg = (f"[INFO] {self.label} {done}/{self.num_batches} done in {batch_seconds:.1f}s "
               f"(avg {avg:.1f}s/batch)")
        if n_done is not None and n_total is not None:
            msg += f" | {n_done}/{n_total} items"
        msg += f" | elapsed {format_duration(elapsed)} | ETA {remaining_str}"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)


@contextlib.contextmanager
def _locked_results_store(path):
    """Exclusive advisory lock (flock) on `<path>.lock`, held across an ENTIRE
    read-modify-write cycle on the shared results-store JSON at `path`.

    Without this, update_results_store/update_dataset_image_stats were a
    textbook unlocked read-modify-write: json.load the whole file, mutate the
    in-memory dict, json.dump the whole file back. Every model sharing one
    dataset's --output_file (the SLURM array launcher runs several models
    against the same run_name/output_file — see the "not locked" warning
    already in scripts/run_vlm_pipeline.sbatch's own header comment) calls
    this at the end of its own multi-hour run. If process B's json.load
    happens before process A's json.dump lands, B's own later json.dump
    silently overwrites the whole file with a copy that never saw A's entry
    at all — not a crash, not a warning, just a vanished model result. This
    is exactly what happened in production (2026-08-05): three ImageNet runs
    (mistral/qwen/gemma) each completed successfully end-to-end, screenshots
    and all, but only mistral's entry survived in the results JSON.

    Blocking (LOCK_EX, no _NB/timeout) is deliberate and different from
    _ArtifactLock in run_vlm_pipeline.py: this critical section is a small
    JSON file (read + merge + write, milliseconds), not an hours-long
    inference run, so a second process should simply wait its short turn
    rather than fail outright — unlike racing a multi-hour VLM run, queuing
    briefly here costs nothing meaningful."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def update_results_store(path, dataset, model, metrics):
    """Merge `metrics` into results_store[dataset][model] at `path`, creating
    or updating the file on disk, and return the full updated store.

    The read, merge, and write all happen under ONE held lock (see
    _locked_results_store) so this whole function is atomic with respect to
    any other process — including update_dataset_image_stats below — doing
    the same against the same path."""
    path = Path(path)
    with _locked_results_store(path):
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


def compute_image_stats(records):
    """Per-IMAGE nature/biotic/material GT composition of a run's dataset,
    over the artifact's image records (each carrying a "targets" list, per
    src/loaders/dataset_loader.py's target shape).

    REPLACES the earlier distinct-CLASS breakdown (compute_class_stats), which
    de-duplicated targets by synset_id. That was meaningless for BIG-5:
    load_big5 gives every image the SAME holistic "scene" target with no
    synset, so all 500 images collapsed to a single "class" and the stats read
    `total_classes: 1`. BIG-5 has no class vocabulary at all — its GT is an
    image-level annotation — so counting images is the only breakdown that
    means the same thing across all four datasets, and it matches what the
    per-axis support numbers in the console summary already report.

    Counting rules, mirroring how the axes are actually scored:
      - nature       : image-level OR (nature if ANY target is nature), the
                       same rule as run_vlm_pipeline.image_gt_nature.
      - biotic/material: counted over the NATURE-POSITIVE images only, since a
                       non-nature image carries no biotic/material GT.
      - BIG-5's gt_biotic/gt_material are LISTS (usually one element, both
                       [True, False] on a coder disagreement — see load_big5).
                       EVERY element is counted as its own instance, exactly
                       as the scoring loop does, so a disagreement image adds
                       one to `biotic` AND one to `abiotic`. On the other
                       datasets these are plain bools and count once.
      - An image whose GT nature is missing entirely is skipped (unknown, not
                       "no nature").
    """
    def _as_list(v):
        # BIG-5 stores these as lists; imagenet/coco/places as a plain bool.
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple)) else [v]

    stats = {"total_images": 0, "nature": 0, "no_nature": 0,
             "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0}

    for rec in records:
        targets = rec.get("targets") or []
        nat_vals = [t.get("gt_nature") for t in targets if t.get("gt_nature") is not None]
        if not nat_vals:
            continue  # no usable nature GT on this image at all
        stats["total_images"] += 1
        is_nature = any(nat_vals)
        if not is_nature:
            stats["no_nature"] += 1
            continue
        stats["nature"] += 1
        for t in targets:
            for b in _as_list(t.get("gt_biotic")):
                stats["biotic" if b else "abiotic"] += 1
            for m in _as_list(t.get("gt_material")):
                stats["material" if m else "immaterial"] += 1

    return stats


def update_dataset_image_stats(path, dataset, config_key, stats):
    """Store `stats` (from compute_image_stats) at
    results_store[dataset]["dataset_image_stats"][config_key], sibling to the
    per-model entries under that dataset. Since sampling is deterministic
    (fixed seed=42, same --max_samples -> same subset), this is written once
    per distinct sampling configuration and left untouched by reruns of the
    same configuration; DIFFERENT configurations (e.g. 1000 vs the full
    50000-image dataset) accumulate side by side under their own config_key
    rather than overwriting one another.

    Writes under "dataset_image_stats" — the old "dataset_class_stats" key is
    deliberately NOT reused, so an existing results store keeps its old
    (class-based) numbers under the old key instead of silently mixing two
    different definitions under one name.

    Same lock as update_results_store (see _locked_results_store) and for the
    identical reason: this writes the SAME shared per-dataset JSON file, so an
    unlocked read-modify-write here could just as easily clobber a concurrent
    model's entry — or have ITS OWN update clobbered by one."""
    path = Path(path)
    with _locked_results_store(path):
        store = {}
        if path.exists():
            with open(path) as f:
                store = json.load(f)

        store.setdefault(dataset, {}).setdefault("dataset_image_stats", {})[config_key] = stats

        with open(path, "w") as f:
            json.dump(store, f, indent=4)

    return store

    return store
