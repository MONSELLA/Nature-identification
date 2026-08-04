#!/usr/bin/env python3
"""
build_summary_table.py — one flat CSV of every (dataset, model) result found
under a results tree, ready to build thesis tables from directly.

Each row is HONESTLY LABELED with whether the underlying artifact actually
finished (has a footer) and whether its record count matches what the
results-store JSON claims — so a partial/incomplete run (like the mistral
ImageNet 35,694-vs-50,000 gap) shows up as a visible flag in the table
itself, rather than silently looking identical to a finished run. Building
tables off unflagged numbers is exactly what caused the last few rounds of
confusion, so this script refuses to hide that distinction.

Reads results-store JSON files (the ones update_results_store/phase_score
write — see src/utils.py) directly; does NOT re-run any scoring. If a
(dataset, model) you expect is missing entirely from the output, it's
because no results JSON has an entry for it yet — run --stage score for it
first (audit_artifacts.py tells you which artifacts are ready to score).

Usage:
    python scripts/build_summary_table.py --results_dir /home/pmonserrat/code/results
    python scripts/build_summary_table.py --results_dir results --out summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

RESPONSES_SUBDIR = "responses"


def _scan_artifact_counts(path):
    """Minimal, defensive .jsonl scan — same shape as audit_artifacts.py's
    _scan_artifact, kept separate/standalone here so this script has zero
    dependency on that one (both are meant to be copy-pasteable one-file
    tools that run with nothing but the stdlib)."""
    n_images = 0
    has_footer = False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rt = obj.get("record_type")
                if rt == "footer":
                    has_footer = True
                elif rt != "header":
                    n_images += 1
    except OSError:
        return None, False
    return n_images, has_footer


def _get(d, *keys, default=None):
    """Nested-dict lookup that returns `default` on any missing key/None
    intermediate, instead of raising — different datasets populate different
    metric blocks (e.g. clipmatch/hierarchical are ImageNet+Places only, see
    CLAUDE.md's ClipMatch scoping), so most rows will legitimately have gaps
    for those columns, not most rows failing an assert."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


FIELDS = [
    "dataset", "model", "evaluated_at",
    "n_images_summary", "n_images_artifact", "artifact_has_footer", "complete",
    "nature_acc", "nature_precision", "nature_recall", "nature_f1",
    "biotic_acc", "biotic_precision", "biotic_recall", "biotic_f1",
    "material_acc", "material_precision", "material_recall", "material_f1",
    "clipscore", "f_clipscore", "object_clipscore",
    "clipmatch_top1", "hierarchical_hp", "hierarchical_hr", "hierarchical_hf1",
    "hierarchical_wup", "execution_time_total",
    "results_json_path", "artifact_path",
]


def build_rows(results_dir):
    root = Path(results_dir)
    # Every results-store JSON sits directly in a <run_name> dir, a sibling
    # of that dir's responses/ and predictions/ subfolders (see
    # run_vlm_pipeline.py's phase_score: out_path = out_dir / output_file).
    # Excluding responses/ and predictions/ themselves avoids picking up
    # anything unrelated that happens to end in .json under those subdirs.
    results_jsons = [p for p in root.rglob("*.json")
                     if RESPONSES_SUBDIR not in p.parts and "predictions" not in p.parts]

    rows = []
    for rj_path in sorted(results_jsons):
        try:
            store = json.loads(rj_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: couldn't read {rj_path}: {e}", file=sys.stderr)
            continue
        run_dir = rj_path.parent
        for dataset, models in store.items():
            for model, entry in models.items():
                if model == "dataset_image_stats" or not isinstance(entry, dict):
                    continue  # sibling metadata key, not a model result

                # Cross-check against the artifact this entry claims to
                # summarize. model_name (preferred) or the raw "model"
                # string, "/" -> "_", matches _resolve_responses_file's own
                # _model_slug so this points at the SAME file phase_score read.
                model_slug = (entry.get("model_name") or model).replace("/", "_")
                artifact_path = run_dir / RESPONSES_SUBDIR / f"vlm_responses_{model_slug}.jsonl"
                artifact_n, artifact_footer = (None, None)
                if artifact_path.exists():
                    artifact_n, artifact_footer = _scan_artifact_counts(artifact_path)

                n_summary = entry.get("n_images")
                complete = bool(artifact_footer) if artifact_footer is not None else None
                mismatch = (artifact_n is not None and n_summary is not None
                           and artifact_n != n_summary)

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "evaluated_at": entry.get("evaluated_at"),
                    "n_images_summary": n_summary,
                    "n_images_artifact": artifact_n,
                    "artifact_has_footer": artifact_footer,
                    "complete": ("N/A (artifact not found)" if artifact_n is None
                                else ("MISMATCH" if mismatch else ("YES" if complete else "NO"))),
                    "nature_acc": _get(entry, "nature", "accuracy"),
                    "nature_precision": _get(entry, "nature", "precision"),
                    "nature_recall": _get(entry, "nature", "recall"),
                    "nature_f1": _get(entry, "nature", "f1"),
                    "biotic_acc": _get(entry, "biotic_matched", "accuracy"),
                    "biotic_precision": _get(entry, "biotic_matched", "precision"),
                    "biotic_recall": _get(entry, "biotic_matched", "recall"),
                    "biotic_f1": _get(entry, "biotic_matched", "f1"),
                    "material_acc": _get(entry, "material_matched", "accuracy"),
                    "material_precision": _get(entry, "material_matched", "precision"),
                    "material_recall": _get(entry, "material_matched", "recall"),
                    "material_f1": _get(entry, "material_matched", "f1"),
                    "clipscore": _get(entry, "reference_free", "clipscore"),
                    "f_clipscore": _get(entry, "reference_free", "f_clipscore"),
                    "object_clipscore": _get(entry, "reference_free", "object_clipscore"),
                    "clipmatch_top1": _get(entry, "clipmatch", "top1_accuracy"),
                    "hierarchical_hp": _get(entry, "hierarchical", "hp"),
                    "hierarchical_hr": _get(entry, "hierarchical", "hr"),
                    "hierarchical_hf1": _get(entry, "hierarchical", "hf1"),
                    "hierarchical_wup": _get(entry, "hierarchical", "wup"),
                    "execution_time_total": _get(entry, "execution_time", "total"),
                    "results_json_path": str(rj_path),
                    "artifact_path": str(artifact_path) if artifact_path.exists() else None,
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="summary_table.csv",
                    help="Output CSV path (default: summary_table.csv in the current dir).")
    args = ap.parse_args()

    rows = build_rows(args.results_dir)
    if not rows:
        print(f"No results-store JSON entries found under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_incomplete = sum(1 for r in rows if r["complete"] != "YES")
    print(f"Wrote {len(rows)} rows to {args.out}")
    if n_incomplete:
        print(f"⚠️  {n_incomplete}/{len(rows)} row(s) are NOT flagged complete — "
              f"check the 'complete' column before using them in a table:")
        for r in rows:
            if r["complete"] != "YES":
                print(f"   {r['dataset']:<14} {r['model']:<45} complete={r['complete']} "
                      f"(summary says n_images={r['n_images_summary']}, "
                      f"artifact has {r['n_images_artifact']})")


if __name__ == "__main__":
    main()
