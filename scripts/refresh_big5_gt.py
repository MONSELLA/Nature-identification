#!/usr/bin/env python3
"""
refresh_big5_gt.py

Patches a COMPLETED BIG-5 VLM-pipeline responses artifact (the .jsonl written
by `run_vlm_pipeline.py --stage infer`, e.g. `vlm_responses_<model>.jsonl`)
with FRESH ground-truth annotations, IN PLACE, WITHOUT re-running inference.

WHY THIS EXISTS: `--stage score` reads GT purely from each record's own
"targets" field, which `phase_infer` baked into the artifact at inference
time from whichever `--*_gt_csv` files were passed to `--stage infer` (see
`src.loaders.dataset_loader.load_big5`). Scoring never re-reads the GT CSVs
itself. So when the underlying annotations change AFTER inference already
ran (e.g. a revised Weibo/Twitter majority-vote CSV), `--stage score` alone
keeps scoring against the STALE targets baked into the artifact — pointing
it at the new CSVs does nothing, because it never looks at them.

This script closes that gap as a one-time patch: reload the (new) GT CSVs
through the EXACT SAME loader inference itself uses (`load_dataset` ->
`load_big5`/`big5_sources`), match each freshly-loaded GT instance to an
already-inferred image record by `image_path` (stable across a GT change —
`load_big5` derives it purely from `platform_id`/slot index/images_dir via
glob, never from the annotation VALUES), and overwrite ONLY that record's
"targets" field in place. `--stage score` can then be re-run against the
patched artifact completely unmodified. Predictions themselves (caption,
extracted objects, per-object labels, hybrid finals, groundings, etc.) are
NEVER touched — this script writes to exactly one field per record.

DURABILITY / CORRUPTION SAFETY. Mirrors run_vlm_pipeline.py's own artifact
safety conventions (a SEPARATE, self-contained implementation here rather
than importing that module — importing it would drag in vLLM/torch/CLIP
just to patch a text field; the lock-file NAMING convention, `<path>.lock`,
is kept identical on purpose so this still correctly excludes a concurrent
`--stage infer`/`--stage score` against the same artifact):
  - Holds an exclusive advisory lock on `<responses_file>.lock` (same
    convention as phase_infer/phase_score's own `_ArtifactLock`) for the
    WHOLE operation, so a concurrent infer/score run against this exact
    artifact can't race the patch.
  - Refuses to patch an artifact that doesn't already end in a footer line
    (i.e. `--stage infer` may still be running/appending to it) — patching a
    file that isn't finished yet is exactly the kind of concurrent-writer
    corruption the lock above also guards against.
  - Streams the patched output to a TEMP FILE IN THE SAME DIRECTORY, fsyncs
    it to stable storage, THEN atomically `os.replace()`s it over the
    original (POSIX-atomic on one filesystem) — the same pattern
    run_vlm_pipeline.py's `_prepend_header` uses for its own in-place
    rewrite. A kill/crash mid-write leaves the ORIGINAL artifact completely
    untouched (only the `.gtrefresh.tmp` sidecar is ever partial, and it is
    unlinked on any failure) — there is no window where the file at
    `--responses_file` itself is half-patched.
  - Refuses (by default) to patch at all if any image record's `image_path`
    has no match in the freshly-loaded GT: silently leaving those records'
    `targets` stale while claiming a full refresh would be worse than
    failing loudly. `--allow_unmatched` opts into proceeding anyway
    (unmatched records simply keep their OLD targets).
  - A line that fails to parse as JSON is passed through byte-for-byte
    unchanged (this script's job is refreshing GT, not repairing whatever
    corruption may already be in the file).

USAGE:
    python scripts/refresh_big5_gt.py \\
        --responses_file results/.../vlm_responses_<model>.jsonl \\
        --weibo_ch0_gt_csv /path/to/new_weiboch6B0_majority.csv \\
        --weibo_ch1_gt_csv /path/to/new_weiboch6B1_majority.csv \\
        --big_5_weibo_images_dir /path/to/big_5/weibo

    Add --dry_run first to see patched/unmatched counts without writing
    anything. Only pass the GT CSV(s)/images dir(s) relevant to the
    artifact's own dataset (read off its header) — e.g. a `big5_twitter`
    artifact only needs --twitter_en_gt_csv/--twitter_es_gt_csv/
    --big_5_twitter_images_dir; passing Weibo ones alongside is harmless
    (simply unused) since the dataset name alone decides which sources
    `load_dataset` builds.
"""

import argparse
import errno
import fcntl
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from src.loaders.dataset_loader import BIG5_DATASETS, load_dataset


# =============================================================================
# Artifact locking — deliberately duplicated from run_vlm_pipeline.py's
# _ArtifactLock/ArtifactLockHeld (NOT imported — see module docstring) rather
# than reimplemented from scratch, so the lock-file path convention and
# failure semantics stay identical to what phase_infer/phase_score use. If
# that class's behavior ever changes there, mirror the change here too.
# =============================================================================
class ArtifactLockHeld(RuntimeError):
    """Another process already holds the lock on this exact artifact."""


class _ArtifactLock:
    def __init__(self, artifact_path):
        self.lock_path = Path(str(artifact_path) + ".lock")
        self._fh = None

    def acquire(self):
        self._fh = open(self.lock_path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.ENOLCK, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS):
                print(f"⚠️ [lock] {self.lock_path.parent} does not support flock "
                      f"({e.strerror}) — continuing WITHOUT an artifact lock. Make "
                      f"sure no other infer/score/refresh job runs against this same "
                      f"artifact concurrently.")
                return self
            self._fh.close()
            self._fh = None
            raise ArtifactLockHeld(
                f"Another process already holds the lock on {self.lock_path} — i.e. "
                f"is already running against this exact --responses_file "
                f"({self.lock_path.with_suffix('')}). Refusing to patch GT while a "
                f"concurrent infer/score/refresh could be reading or writing the same "
                f"file. Check `squeue`/`ps aux | grep run_vlm_pipeline` for a "
                f"duplicate job, let it finish or kill it, then retry.")
        self._fh.write(f"pid={os.getpid()} time={datetime.now().isoformat()}\n")
        self._fh.flush()
        return self

    def release(self):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


# =============================================================================
# GT reload
# =============================================================================
def _load_new_targets_by_path(dataset_name, args):
    """Reload GT via the SAME loader phase_infer uses, keyed by image_path
    (see module docstring — image_path is derived from platform_id/slot/
    images_dir, never from the annotation values, so it stays stable across
    a GT-content-only change)."""
    instances = load_dataset(
        dataset_name, taxonomy_graph=None,
        twitter_en_gt=args.twitter_en_gt_csv, twitter_es_gt=args.twitter_es_gt_csv,
        twitter_images_dir=args.big_5_twitter_images_dir,
        weibo_ch0_gt=args.weibo_ch0_gt_csv, weibo_ch1_gt=args.weibo_ch1_gt_csv,
        weibo_images_dir=args.big_5_weibo_images_dir,
    )
    if not instances:
        raise ValueError(
            "No BIG-5 instances loaded from the given GT CSV(s)/images dir(s) — check "
            "the paths, and that they cover the platform(s) this artifact's own "
            "dataset name selects (see --responses_file's header 'dataset' field).")
    by_path = {}
    dupes = 0
    for inst in instances:
        p = inst["image_path"]
        if p in by_path:
            dupes += 1
        by_path[p] = inst["targets"]
    if dupes:
        print(f"⚠️  [refresh-gt] {dupes} image_path(s) appeared in more than one GT "
              f"source/row — kept the LAST one loaded (shouldn't normally happen; the "
              f"per-platform sources are supposed to be disjoint image sets).")
    return by_path


# =============================================================================
# Patching
# =============================================================================
def _peek_header(path):
    """Read just the header record (must be the file's first non-blank line —
    matches the convention every writer/reader in this pipeline relies on)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") != "header":
                raise ValueError(
                    f"{path}'s first non-blank line is not a header record "
                    f"(record_type={obj.get('record_type')!r}) — this doesn't look "
                    f"like a valid pipeline artifact.")
            return obj
    raise ValueError(f"{path} is empty or has no header line.")


def _stream_patch(in_path, out_f, new_targets_by_path, stats):
    """Copy `in_path` to `out_f` line by line, patching `targets` on every
    image record whose `image_path` matches `new_targets_by_path`. Header/
    footer/corrupted/non-matching lines are passed through unchanged (a
    corrupted line is written back byte-for-byte, not re-serialized, since it
    couldn't be parsed in the first place). Returns (saw_header, saw_footer).
    Mutates `stats` (a dict of counters) as it goes."""
    saw_header = saw_footer = False
    with open(in_path) as src:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                stats["corrupted_passthrough"] += 1
                out_f.write(line if line.endswith("\n") else line + "\n")
                continue
            rt = obj.get("record_type")
            if rt == "header":
                saw_header = True
            elif rt == "footer":
                saw_footer = True
            elif rt == "image":
                path = obj.get("image_path")
                if path in new_targets_by_path:
                    obj["targets"] = new_targets_by_path[path]
                    stats["patched"] += 1
                else:
                    stats["unmatched"] += 1
                    stats["unmatched_paths"].append(path)
            out_f.write(json.dumps(obj) + "\n")
    return saw_header, saw_footer


def _new_stats():
    return {"patched": 0, "unmatched": 0, "unmatched_paths": [], "corrupted_passthrough": 0}


def _print_unmatched(stats):
    shown = stats["unmatched_paths"][:20]
    more = f" (+{stats['unmatched'] - len(shown)} more)" if stats["unmatched"] > len(shown) else ""
    print(f"   unmatched image_path(s): {shown}{more}")


def _patch_artifact(path, new_targets_by_path, allow_unmatched):
    """The actual in-place patch, via temp-file + fsync + atomic rename (see
    module docstring). Raises (leaving `path` untouched) on any problem;
    never leaves a partial file at `path` itself."""
    tmp = path.with_name(path.name + ".gtrefresh.tmp")
    stats = _new_stats()
    try:
        with open(tmp, "w") as out:
            saw_header, saw_footer = _stream_patch(path, out, new_targets_by_path, stats)
            if not saw_header:
                raise ValueError(f"{path} has no header line — refusing to patch what "
                                  f"doesn't look like a valid pipeline artifact.")
            if not saw_footer:
                raise ValueError(
                    f"{path} has no footer line — this looks like an INCOMPLETE "
                    f"--stage infer run (still running, or killed mid-write). Refusing "
                    f"to patch a file that might still be actively written to; finish "
                    f"or clean up that run first.")
            if stats["unmatched"] and not allow_unmatched:
                raise ValueError(
                    f"{stats['unmatched']} image record(s) had no match in the "
                    f"freshly-loaded GT (first few: {stats['unmatched_paths'][:10]}). "
                    f"Refusing to patch — leaving their 'targets' silently STALE while "
                    f"claiming a full GT refresh would be worse than failing loudly. "
                    f"Re-check the --*_gt_csv / --big_5_*_images_dir paths match what "
                    f"--stage infer originally used, or pass --allow_unmatched to "
                    f"proceed anyway (unmatched records simply keep their OLD "
                    f"targets, unchanged).")
            # Durability: force the fully-patched temp file to stable storage
            # BEFORE the atomic rename — an os.replace() that "succeeds" over
            # data still sitting in the page cache is not actually durable
            # (same reasoning as run_vlm_pipeline.py's _durable_write/footer
            # fsync — never claim success before the bytes are committed).
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a stray half-written .gtrefresh.tmp behind. `path`
        # itself was never opened for writing above, so it is guaranteed
        # untouched at this point regardless of what failed.
        Path(tmp).unlink(missing_ok=True)
        raise
    print(f"✅ [refresh-gt] patched {stats['patched']} record(s), "
          f"{stats['unmatched']} unmatched"
          + (f", {stats['corrupted_passthrough']} corrupted line(s) passed through unchanged"
             if stats["corrupted_passthrough"] else "")
          + f" -> {path}")


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses_file", type=str, required=True,
                   help="The .jsonl artifact to patch IN PLACE (written by a previous "
                        "`--stage infer`/`--stage all` run). Must already have a footer "
                        "line (i.e. that run finished).")
    p.add_argument("--twitter_en_gt_csv", type=str, default=None,
                   help="Refreshed BIG-5 Twitter ENGLISH majority-vote GT CSV — same "
                        "schema run_vlm_pipeline.py's --stage infer expects.")
    p.add_argument("--twitter_es_gt_csv", type=str, default=None,
                   help="Refreshed BIG-5 Twitter SPANISH majority-vote GT CSV.")
    p.add_argument("--weibo_ch0_gt_csv", type=str, default=None,
                   help="Refreshed BIG-5 Weibo majority-vote GT CSV, split ch6B0.")
    p.add_argument("--weibo_ch1_gt_csv", type=str, default=None,
                   help="Refreshed BIG-5 Weibo majority-vote GT CSV, split ch6B1.")
    p.add_argument("--big_5_twitter_images_dir", type=str, default="./big_5_images",
                   help="Same Twitter images folder --stage infer was run with — GT "
                        "rows are matched to already-inferred records by the local "
                        "image path this loader derives from it, so it must be the "
                        "SAME folder (or at least yield the same paths).")
    p.add_argument("--big_5_weibo_images_dir", type=str, default="./big_5_weibo_images",
                   help="Same Weibo images folder --stage infer was run with.")
    p.add_argument("--allow_unmatched", action="store_true",
                   help="Proceed even if some already-inferred image records have no "
                        "match in the freshly-loaded GT (those records simply keep "
                        "their OLD targets). Default is to refuse and print every "
                        "unmatched path, since a silent partial refresh is easy to "
                        "mistake for a complete one.")
    p.add_argument("--dry_run", action="store_true",
                   help="Load the new GT and report patched/unmatched counts without "
                        "writing anything.")
    return p


def main():
    args = build_parser().parse_args()
    responses_path = Path(args.responses_file)
    if not responses_path.exists():
        raise SystemExit(f"{responses_path} does not exist.")

    lock = _ArtifactLock(responses_path)
    lock.acquire()
    try:
        header = _peek_header(responses_path)
        dataset = header.get("dataset")
        if dataset not in BIG5_DATASETS:
            raise ValueError(
                f"{responses_path}'s header says dataset={dataset!r}, but this script "
                f"only refreshes GT for BIG-5 datasets {BIG5_DATASETS} — ImageNet/"
                f"COCO/Places365 GT doesn't come from these CSVs at all.")

        print(f"🔄 [refresh-gt] dataset={dataset!r}, target artifact={responses_path}")
        new_targets = _load_new_targets_by_path(dataset, args)
        print(f"🔄 [refresh-gt] loaded {len(new_targets)} fresh GT-annotated image(s) "
              f"from the new CSV(s)")

        if args.dry_run:
            stats = _new_stats()
            _stream_patch(responses_path, io.StringIO(), new_targets, stats)
            print(f"🔎 [refresh-gt] DRY RUN — would patch {stats['patched']}, leave "
                  f"{stats['unmatched']} unmatched. Nothing written.")
            if stats["unmatched"]:
                _print_unmatched(stats)
            return

        _patch_artifact(responses_path, new_targets, allow_unmatched=args.allow_unmatched)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
