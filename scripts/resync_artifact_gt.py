#!/usr/bin/env python
"""
scripts/resync_artifact_gt.py

Re-sync the GROUND-TRUTH labels stored inside an existing VLM artifact
(`vlm_responses_<model>.jsonl`) with the CURRENT taxonomy Excel file.

WHY THIS EXISTS
---------------
GT is resolved from the Excel ONCE, during `--stage infer`, and written into
each image record's `targets` list (`gt_nature`/`gt_biotic`/`gt_material`).
`--stage score` never reloads the dataset — it reads `rec["targets"]` straight
off the artifact (see `run_vlm_pipeline.phase_score`). So when the Excel
annotations are corrected, re-running `--stage score` alone changes NOTHING:
the stale GT is baked into the `.jsonl`.

This script closes that gap WITHOUT paying for VLM inference again. It re-runs
exactly the same lookup the loaders use (`dataset_loader.get_gt_from_graph`,
i.e. `TaxonomyGraph.resolve_labels`) for every target synset and rewrites the
three GT fields in place. Nothing model-produced is touched: captions,
objects, object_labels, object_finals, groundings and every header field are
copied through byte-for-byte.

It is DELIBERATELY GENERAL — it re-resolves every target rather than special-
casing whichever classes were just corrected, so it stays valid for any future
annotation change and doubles as a "is this artifact's GT current?" audit
(run with no flags: it reports drift and writes nothing).

BIG-5 SAFETY
------------
BIG-5 targets carry no `synset_id` (their GT comes from the annotation CSVs,
and `gt_biotic`/`gt_material` are LISTS to encode coder disagreement — see
`dataset_loader.load_big5`). Any target without a `synset_id` is left
completely untouched, so pointing this at a BIG-5 artifact is a no-op rather
than a corruption.

USAGE
-----
    # audit only (default): report what WOULD change, write nothing
    python scripts/resync_artifact_gt.py results/.../responses/*.jsonl

    # actually rewrite, keeping a .bak of each file
    python scripts/resync_artifact_gt.py results/.../responses/*.jsonl --apply

    # a whole directory, no backups (they are large; the artifacts are
    # reproducible from the Excel + this script)
    python scripts/resync_artifact_gt.py --dir results/.../responses --apply --no-backup

After `--apply`, re-run `--stage score` for each model to regenerate the
results JSON / predictions CSV / W&B numbers.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loaders.excel_loader import TaxonomyGraph  # noqa: E402
from src.vlm_pipeline import resolve_hybrid_label  # noqa: E402

GT_FIELDS = ("gt_nature", "gt_biotic", "gt_material")
FINAL_FIELDS = ("final_nature", "final_biotic", "final_material")


def writer_active(path):
    """True if a phase_infer job currently holds the artifact's `.lock`.

    phase_infer holds an exclusive flock on `<artifact>.lock` for its whole
    write loop, and phase_score probes the same lock before scoring. This tool
    REWRITES the artifact, so racing a live writer is strictly worse than
    racing a reader: the writer appends while we replace the file underneath
    it, and its remaining records are lost on os.replace. Probe the same lock
    and refuse.

    A filesystem that doesn't support flock reports "not locked" here rather
    than blocking every run on that mount (same rationale as _ArtifactLock)."""
    lock_path = path + ".lock"
    if not os.path.exists(lock_path):
        return False
    try:
        fh = open(lock_path, "a")
    except OSError:
        return False
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        fh.close()


def gt_for_synset(synset_id, graph, max_hops):
    """Same resolution `dataset_loader.get_gt_from_graph` performs, but with an
    explicit max_hops so the artifact's own header value is honoured rather
    than resolve_labels' default."""
    labels = graph.resolve_labels(synset_id, max_hops=max_hops)
    if not labels:
        return None
    node_attrs = graph.graph.nodes.get(labels["resolved_from_node"], {})
    mat_val = node_attrs.get("tangibility")
    is_nature = labels["is_nature"]
    if is_nature:
        life = labels.get("life_category")
        gt_biotic = (life == "biotic") if life else None
        gt_material = (mat_val == "material") if mat_val else True
    else:
        gt_biotic = gt_material = None
    return {"gt_nature": is_nature, "gt_biotic": gt_biotic, "gt_material": gt_material}


def process_file(path, graph, apply_changes, backup, verbose, predictions):
    changed_targets = Counter()   # (synset, field, old -> new) -> count
    changed_preds = Counter()     # (field, old -> new, via synset) -> count
    pred_phrases = Counter()
    nature_flips = 0
    n_records = n_targets = n_skipped = n_unresolved = n_objects = 0
    changed_records = 0

    out_lines = []
    max_hops = 3
    cache = {}
    saw_footer = False

    # Announce BEFORE the work, not after: a 250 MB artifact takes ~10 s and a
    # whole directory a couple of minutes, and a log that prints nothing until
    # the very end is indistinguishable from a hung job.
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n{path}  ({size_mb:.0f} MB)", flush=True)

    if writer_active(path):
        print("  !! SKIPPED: an inference job currently holds this artifact's .lock — it is "
              "still\n     being written. Let that job finish, then re-run this tool.", flush=True)
        return 0, True

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                out_lines.append(line)
                continue
            rec = json.loads(line)

            if rec.get("record_type") == "footer":
                saw_footer = True
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                continue

            if rec.get("record_type") == "header":
                # The header carries the max_hops the run was produced with;
                # honour it so re-resolution matches the original run exactly.
                max_hops = rec.get("max_hops", 3)
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                continue

            n_records += 1
            if n_records % 20000 == 0:
                print(f"    ... {n_records} records", flush=True)
            rec_changed = False
            for tgt in rec.get("targets") or []:
                n_targets += 1
                syn = tgt.get("synset_id")
                if not syn:
                    # BIG-5 target (list-valued GT, no synset) — never touch.
                    n_skipped += 1
                    continue
                if syn not in cache:
                    cache[syn] = gt_for_synset(syn, graph, max_hops)
                new = cache[syn]
                if new is None:
                    # Class no longer resolves at all. Refuse to invent a
                    # label or silently drop the target — report and leave.
                    n_unresolved += 1
                    continue
                for f in GT_FIELDS:
                    if tgt.get(f) != new[f]:
                        changed_targets[(syn, f, str(tgt.get(f)), str(new[f]))] += 1
                        tgt[f] = new[f]
                        rec_changed = True
            # ---- PREDICTION side -------------------------------------------
            # The hybrid label of a MAPPED-nature object takes nature+biotic
            # from the taxonomy (only material comes from the VLM), so a
            # corrected annotation leaves `object_finals` stale in exactly the
            # same way it left `targets` stale — and `--stage score` does not
            # recompute it either (it skips any record that already has
            # object_finals). resolve_hybrid_label is fully deterministic given
            # the STORED per-object VLM label plus the graph, so this is a pure
            # re-derivation: no model, no image, nothing regenerated.
            if predictions and rec.get("object_finals") is not None:
                for i, (obj, lab, old) in enumerate(zip(rec.get("objects") or [],
                                                        rec.get("object_labels") or [],
                                                        rec["object_finals"])):
                    n_objects += 1
                    new = resolve_hybrid_label(obj, lab, graph, max_hops=max_hops)
                    if any(old.get(f) != new.get(f) for f in FINAL_FIELDS):
                        for f in FINAL_FIELDS:
                            if old.get(f) != new.get(f):
                                changed_preds[(f, str(old.get(f)), str(new.get(f)),
                                               str(new.get("mapped_synset")))] += 1
                                if f == "final_nature":
                                    nature_flips += 1
                        pred_phrases[obj] += 1
                        rec["object_finals"][i] = new
                        rec_changed = True

            if rec_changed:
                changed_records += 1
            out_lines.append(json.dumps(rec, ensure_ascii=False))

    n_changes = sum(changed_targets.values()) + sum(changed_preds.values())
    print(f"  records={n_records}  targets={n_targets}  "
          f"changed_targets={n_changes}  changed_records={changed_records}"
          + (f"  skipped_no_synset={n_skipped}" if n_skipped else "")
          + (f"  UNRESOLVED={n_unresolved}" if n_unresolved else ""), flush=True)
    if changed_targets and verbose:
        print("  GT (targets):")
        for (syn, field, old, new), k in sorted(changed_targets.items(),
                                                key=lambda kv: -kv[1]):
            print(f"    {syn:<24} {field:<12} {old:>5} -> {new:<5}  ({k} images)")
    if changed_preds:
        print(f"  PREDICTIONS (object_finals): {sum(changed_preds.values())} of {n_objects} objects")
        if verbose:
            for (field, old, new, syn), k in sorted(changed_preds.items(), key=lambda kv: -kv[1]):
                print(f"    {syn:<24} {field:<14} {old:>5} -> {new:<5}  (x{k})")
            print(f"    phrases: {', '.join(f'{p} x{k}' for p, k in pred_phrases.most_common(8))}")
    if not saw_footer:
        print(f"  !! INCOMPLETE ARTIFACT: no footer record — this run never finished "
              f"({n_records} images).\n"
              f"     Re-scoring it produces metrics over that subset only, which are NOT "
              f"comparable\n     with the other models. Finish the inference run "
              f"(--resume) before scoring it.", flush=True)
    if nature_flips:
        print(f"  !! WARNING: {nature_flips} objects changed final_NATURE. Grounding only ever ran\n"
              f"     on final_nature==True entities, so any object_groundings/object_instances in\n"
              f"     this artifact no longer cover the same entity set — the grounding stage must\n"
              f"     be re-run for this artifact, a re-score alone is NOT sufficient.")

    if n_changes and apply_changes:
        d = os.path.dirname(os.path.abspath(path))
        if backup:
            bak = path + ".bak"
            if not os.path.exists(bak):
                os.replace(path, bak)
                with open(bak, "r", encoding="utf-8") as _s, \
                     open(path, "w", encoding="utf-8") as _d:
                    _d.write(_s.read())
        # atomic replace: write a sibling temp file, fsync, then rename
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write("\n".join(out_lines) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        print(f"  -> REWRITTEN{' (backup: ' + path + '.bak)' if backup else ''}", flush=True)
    elif n_changes:
        print("  -> dry run, nothing written (pass --apply)")

    return n_changes, saw_footer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="*", help="one or more vlm_responses_*.jsonl")
    ap.add_argument("--dir", help="directory to scan for vlm_responses_*.jsonl")
    ap.add_argument("--excel_path", default="data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx")
    ap.add_argument("--sheet_name", default="data corrected")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite the files (default: audit only)")
    ap.add_argument("--no-backup", dest="backup", action="store_false",
                    help="skip writing a .bak beside each rewritten artifact")
    ap.add_argument("--quiet", action="store_true", help="omit the per-class change table")
    ap.add_argument("--gt-only", dest="predictions", action="store_false",
                    help="re-sync only targets[] (GT), leaving object_finals stale. "
                         "NOT recommended: it corrects GT while leaving the mapping-derived "
                         "half of the predictions on the old annotations.")
    args = ap.parse_args()

    paths = list(args.artifacts)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "vlm_responses_*.jsonl")))
    paths = [p for p in dict.fromkeys(paths) if not p.endswith((".lock", ".bak", ".tmp"))]
    if not paths:
        ap.error("no artifacts given (pass paths or --dir)")

    sheet = args.sheet_name if not str(args.sheet_name).isdigit() else int(args.sheet_name)
    graph = TaxonomyGraph()
    graph.load_excel(args.excel_path, sheet_name=sheet)
    print(f"taxonomy: {args.excel_path} [{sheet}]  |  {len(paths)} artifact(s)"
          f"  |  mode: {'APPLY' if args.apply else 'audit (dry run)'}")

    results = [process_file(p, graph, args.apply, args.backup, not args.quiet, args.predictions)
               for p in paths]
    total = sum(n for n, _ in results)
    incomplete = [p for p, (_, ok) in zip(paths, results) if not ok]
    print(f"\nTOTAL changed targets: {total}")
    if total and not args.apply:
        print("Re-run with --apply to write, then re-run `--stage score` per model.")
    if incomplete:
        print(f"\n!! {len(incomplete)} artifact(s) are incomplete or locked — check the warnings above:")
        for p in incomplete:
            print(f"     {p}")
    if args.predictions:
        print("NOTE: prediction changes reflect the CURRENT taxonomy, which may differ from the\n"
              "      one a run was made with for reasons beyond the latest edit (rows appended to\n"
              "      the sheet after a run resolve phrases that were unmapped at infer time).\n"
              "      The per-synset table above shows every cause; check it before --apply.")


if __name__ == "__main__":
    main()
