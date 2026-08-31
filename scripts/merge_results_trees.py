#!/usr/bin/env python3
"""
merge_results_trees.py

Consolidate one results tree into another — e.g. fold
results/vlm_pipeline/ablation_no_caption/ into
results/vlm_pipeline/baseline_no_caption/ so every no-caption run lives under
one root.

WHY THIS ISN'T `cp -r`. Each dataset directory holds a MERGED results JSON
keyed `dataset -> model` (src.utils.update_results_store). Two trees have two
such files, and copying one over the other would discard every model entry in
the destination rather than adding to it. The artifacts and prediction CSVs
ARE plain per-model files and are copied as-is; only the JSONs need a real
key-wise merge.

NON-DESTRUCTIVE. The source tree is never modified or deleted — this copies.
Delete the source yourself once you have checked the result. A destination
file is never silently overwritten either: an existing artifact/CSV with the
same name is reported as a collision and skipped unless --overwrite is given,
because the same filename in two trees means the same model was run twice and
which copy is authoritative is a judgement call, not a default.

  python scripts/merge_results_trees.py \\
      --src results/vlm_pipeline/ablation_no_caption \\
      --dst results/vlm_pipeline/baseline_no_caption
  python scripts/merge_results_trees.py --src ... --dst ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Keys that are properties of the DATASET rather than of any one model
# (compute_image_stats/update_dataset_image_stats write these). They describe
# the same images either way, so the destination's copy is kept and the
# source's is only used to fill a gap.
DATASET_LEVEL_KEYS = ("dataset_image_stats", "dataset_class_stats")


def merge_results_json(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    """Fold src's `dataset -> model` entries into dst. Returns (added, kept)."""
    with open(src) as f:
        src_store = json.load(f)
    dst_store = {}
    if dst.exists():
        with open(dst) as f:
            dst_store = json.load(f)

    added = collided = 0
    for dataset, models in src_store.items():
        bucket = dst_store.setdefault(dataset, {})
        for model, entry in models.items():
            if model in DATASET_LEVEL_KEYS:
                bucket.setdefault(model, entry)   # dataset-level: don't clobber
                continue
            if model in bucket:
                # Same model scored in both trees. Keeping the destination's is
                # arbitrary, so say so rather than deciding silently.
                print(f"      COLLISION (kept destination's): {dataset} -> {model}")
                collided += 1
                continue
            bucket[model] = entry
            added += 1

    if not dry_run:
        # Write via a temp file in the same directory, then atomically replace —
        # a crash mid-write leaves the original results JSON untouched.
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(dst_store, f, indent=4)
        os.replace(tmp, dst)
    return added, collided


def copy_files(src_dir: Path, dst_dir: Path, overwrite: bool, dry_run: bool) -> tuple[int, int]:
    """Copy every file in src_dir into dst_dir. Returns (copied, skipped)."""
    if not src_dir.is_dir():
        return 0, 0
    copied = skipped = 0
    for f in sorted(src_dir.iterdir()):
        # .lock files are per-run advisory locks, and .bak-<timestamp> files are
        # phase_infer's "never silently clobber an existing artifact" backups —
        # neither is a result, and carrying them into the consolidated tree would
        # just make it harder to see what is actually there.
        if not f.is_file() or f.name.endswith(".lock") or ".jsonl.bak-" in f.name:
            continue
        target = dst_dir / f.name
        if target.exists() and not overwrite:
            print(f"      COLLISION (skipped): {target.relative_to(dst_dir.parent.parent)}")
            skipped += 1
            continue
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        copied += 1
    return copied, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Tree to merge FROM (never modified).")
    ap.add_argument("--dst", required=True, help="Tree to merge INTO.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite colliding artifacts/CSVs instead of skipping them.")
    ap.add_argument("--dry_run", action="store_true", help="Report, write nothing.")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"--src {src} is not a directory")
    dst.mkdir(parents=True, exist_ok=True)

    totals = dict(files=0, skipped=0, entries=0, collided=0)
    for ds_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        print(f"\n{dataset}:")
        out_ds = dst / dataset

        for sub in ("responses", "predictions"):
            n, s = copy_files(ds_dir / sub, out_ds / sub, args.overwrite, args.dry_run)
            if n or s:
                print(f"   {sub}: {n} copied{f', {s} skipped' if s else ''}")
            totals["files"] += n
            totals["skipped"] += s

        # Results JSONs. The two trees name them differently (each --output_file
        # was chosen per run), so the destination's own name is the target when
        # it exists; otherwise the source's name is carried over.
        src_jsons = [p for p in ds_dir.glob("*.json") if not p.name.endswith(".lock")]
        dst_jsons = [p for p in out_ds.glob("*.json") if not p.name.endswith(".lock")] \
            if out_ds.is_dir() else []
        for sj in src_jsons:
            target = dst_jsons[0] if dst_jsons else (out_ds / sj.name)
            if not args.dry_run:
                out_ds.mkdir(parents=True, exist_ok=True)
            a, c = merge_results_json(sj, target, args.dry_run)
            print(f"   {sj.name} -> {target.name}: {a} model entr{'y' if a == 1 else 'ies'} added"
                  + (f", {c} collision(s)" if c else ""))
            totals["entries"] += a
            totals["collided"] += c

    print(f"\n{'DRY RUN — nothing written' if args.dry_run else 'Merged'}: "
          f"{totals['files']} file(s), {totals['entries']} results entr(y/ies)"
          + (f", {totals['skipped']} file collision(s) skipped" if totals["skipped"] else "")
          + (f", {totals['collided']} entry collision(s) kept from destination" if totals["collided"] else ""))
    print(f"Source tree left untouched: {src}")


if __name__ == "__main__":
    main()
