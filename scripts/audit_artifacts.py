#!/usr/bin/env python3
"""
audit_artifacts.py — inventory every VLM-pipeline artifact under a results
tree and report how many samples each one actually holds.

Answers exactly the question "how many images are really stored for model X
on dataset Y" without trusting any single number in isolation: the response
artifact's own record count, whether it has a header/footer, the matching
predictions CSV's row count, and (if the results-store JSON has an entry for
that dataset/model) the summary's own recorded "n_images" — all four are
independently derived, so a mismatch between them is itself the diagnostic.

Layout this script assumes (see run_vlm_pipeline.py's _resolve_responses_file
and phase_score, RESPONSES_SUBDIR/PREDICTIONS_SUBDIR):
    <results_dir>/<run_name>/<output_file>                      (results JSON)
    <results_dir>/<run_name>/responses/vlm_responses_<model>.jsonl
    <results_dir>/<run_name>/predictions/<stem>_<dataset>_<model>_predictions.csv

Usage:
    python scripts/audit_artifacts.py --results_dir /home/pmonserrat/code/results
    python scripts/audit_artifacts.py --results_dir results --json   # machine-readable
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def _scan_artifact(path):
    """Read a responses .jsonl WITHOUT importing run_vlm_pipeline.py (this
    script has to run with nothing but the stdlib — no vllm/numpy needed just
    to count lines). Mirrors _read_artifact's/_already_inferred_paths' own
    line-classification logic closely enough for an audit, but never raises
    on a malformed file — a truncated tail line or a missing header is
    exactly the kind of thing this script exists to SURFACE, not choke on."""
    n_images = 0
    has_header = False
    has_footer = False
    dataset = model = model_name = None
    n_bad_lines = 0
    n_nulls = 0
    # Read as BYTES, not text: NUL-byte corruption (storage-level data loss —
    # see scripts/diagnose_artifact.py) is invisible to a line count, because
    # NULs contain no newlines. A 170 MB artifact whose interior was zeroed
    # reports a few hundred "lines" and looks merely incomplete, while
    # actually being a destroyed full-size file — a completely different
    # problem with a completely different fix (storage/quota, not the
    # pipeline). Counting NULs here is what tells those two apart across
    # every artifact at once.
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        return {"error": str(e)}
    n_nulls = raw.count(b"\x00")
    for bline in raw.split(b"\n"):
        bline = bline.strip()
        if not bline:
            continue
        try:
            obj = json.loads(bline)
        except (json.JSONDecodeError, UnicodeDecodeError):
            n_bad_lines += 1
            continue
        if not isinstance(obj, dict):
            n_bad_lines += 1
            continue
        rt = obj.get("record_type")
        if rt == "header":
            has_header = True
            dataset = obj.get("dataset")
            model = obj.get("model")
            model_name = obj.get("model_name")
        elif rt == "footer":
            has_footer = True
        else:
            n_images += 1
    # Mirrors _model_slug's own preference (model_name over the full
    # family/name "model" string) so the slug used here to find a matching
    # predictions CSV agrees with the one phase_score actually wrote it with.
    model_slug = (model_name or model or "").replace("/", "_")
    size_bytes = len(raw)
    # Estimate how many records this file's SIZE could hold, from the average
    # size of the records that DID survive. When NULs are present, the gap
    # between that estimate and n_images is the amount of finished work the
    # storage lost — the number that actually matters when deciding whether
    # to re-run.
    implied_records = None
    if n_nulls and n_images:
        bytes_per_rec = (size_bytes - n_nulls) / n_images
        if bytes_per_rec > 0:
            implied_records = int(size_bytes / bytes_per_rec)
    return {
        "n_images": n_images,
        "has_header": has_header,
        "has_footer": has_footer,
        "dataset": dataset,
        "model": model,
        "model_slug": model_slug,
        "n_bad_lines": n_bad_lines,
        "size_bytes": size_bytes,
        "n_nulls": n_nulls,
        "null_fraction": (n_nulls / size_bytes) if size_bytes else 0.0,
        "implied_records": implied_records,
    }


def _count_csv_rows(path):
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header row
            return sum(1 for _ in reader)
    except OSError:
        return None


def _load_results_json_counts(path):
    """dataset -> model -> n_images, read straight from a results-store JSON
    (the same file update_results_store writes). `dataset_image_stats` is a
    sibling key under each dataset, not a model, so it's skipped here."""
    out = {}
    try:
        store = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return out
    for dataset, models in store.items():
        for model, entry in models.items():
            if model == "dataset_image_stats" or not isinstance(entry, dict):
                continue
            out.setdefault(dataset, {})[model] = entry.get("n_images")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True,
                    help="Root results directory (the --results_dir the pipeline itself used).")
    ap.add_argument("--json", action="store_true",
                    help="Print one JSON object instead of the human-readable table.")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    rows = []
    for jsonl_path in sorted(root.rglob("responses/*.jsonl")):
        info = _scan_artifact(jsonl_path)
        run_dir = jsonl_path.parent.parent  # .../<run_name>/responses/<file> -> .../<run_name>

        # Find a matching predictions CSV: <stem>_<dataset>_<model_slug>_predictions.csv
        # under this run's predictions/ dir. The FILENAME is the primary slug
        # source (_resolve_responses_file names the artifact
        # vlm_responses_<slug>.jsonl the same way regardless of whether the
        # header ever got written), falling back to the header's own
        # model_name/model only if the filename doesn't follow that pattern.
        # Matching by dataset ALONE (this script's first version) is wrong
        # whenever more than one model shares a dataset folder — confirmed by
        # this script's own test: it silently attributed qwen's 50000-row CSV
        # to mistral's 35694-record artifact.
        fname_slug = jsonl_path.stem
        if fname_slug.startswith("vlm_responses_"):
            fname_slug = fname_slug[len("vlm_responses_"):]
        model_slug = fname_slug or info.get("model_slug") or ""
        pred_dir = run_dir / "predictions"
        csv_matches = []
        if pred_dir.is_dir() and info.get("dataset") and model_slug:
            csv_matches = sorted(pred_dir.glob(f"*_{info['dataset']}_{model_slug}_predictions.csv"))
        csv_rows = _count_csv_rows(csv_matches[0]) if csv_matches else None

        # Find any results-store JSON directly in run_dir and pull its
        # recorded n_images for this (dataset, model), if present.
        summary_n_images = None
        for results_json in run_dir.glob("*.json"):
            counts = _load_results_json_counts(results_json)
            m = counts.get(info.get("dataset"), {})
            if info.get("model") in m:
                summary_n_images = m[info["model"]]
                break

        rows.append({
            "run_dir": str(run_dir),
            "artifact": str(jsonl_path),
            "dataset": info.get("dataset"),
            "model": info.get("model"),
            "artifact_n_images": info.get("n_images"),
            "has_header": info.get("has_header"),
            "has_footer": info.get("has_footer"),
            "n_bad_lines": info.get("n_bad_lines"),
            "predictions_csv": str(csv_matches[0]) if csv_matches else None,
            "predictions_csv_rows": csv_rows,
            "results_json_n_images": summary_n_images,
            "size_bytes": info.get("size_bytes"),
            "n_nulls": info.get("n_nulls"),
            "null_fraction": info.get("null_fraction"),
            "implied_records": info.get("implied_records"),
        })

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print(f"No responses/*.jsonl artifacts found under {root}")
        return

    print(f"{'dataset':<14} {'model':<45} {'artifact':>9} {'hdr':>4} {'ftr':>4} "
          f"{'bad':>4} {'csv_rows':>9} {'json_n_images':>14}  flags")
    print("-" * 130)
    n_corrupt = 0
    for r in rows:
        flags = []
        # NUL corruption FIRST and loudest: it changes what every other flag
        # on this row means. "INCOMPLETE(no footer)" on a NUL-corrupted file
        # doesn't mean inference stopped early — it means inference very
        # likely FINISHED and the storage ate the result, including the
        # footer. Reporting them the other way round sends you off re-running
        # jobs when the real problem is the filesystem.
        if r.get("n_nulls"):
            n_corrupt += 1
            implied = r.get("implied_records")
            est = (f", size implies ~{implied:,} records were written"
                   if implied else "")
            flags.append(f"⛔ NUL-CORRUPTED: {r['null_fraction']:.0%} of "
                         f"{r['size_bytes'] / 1e6:.0f}MB is zero bytes{est}")
        if not r["has_header"]:
            flags.append("NO-HEADER")
        if not r["has_footer"]:
            flags.append("INCOMPLETE(no footer)")
        if r["n_bad_lines"]:
            flags.append(f"{r['n_bad_lines']} unparseable line(s)")
        if (r["results_json_n_images"] is not None
                and r["artifact_n_images"] is not None
                and r["results_json_n_images"] != r["artifact_n_images"]):
            flags.append(f"MISMATCH: results JSON says {r['results_json_n_images']} "
                         f"but artifact has {r['artifact_n_images']}")
        if r["predictions_csv_rows"] is not None and r["predictions_csv_rows"] != r["artifact_n_images"]:
            flags.append(f"csv rows ({r['predictions_csv_rows']}) != artifact records")
        print(f"{str(r['dataset']):<14} {str(r['model']):<45} "
              f"{str(r['artifact_n_images']):>9} "
              f"{'Y' if r['has_header'] else 'N':>4} "
              f"{'Y' if r['has_footer'] else 'N':>4} "
              f"{r['n_bad_lines'] or 0:>4} "
              f"{str(r['predictions_csv_rows']):>9} "
              f"{str(r['results_json_n_images']):>14}  "
              f"{', '.join(flags)}")

    if n_corrupt:
        print()
        print(f"⛔ {n_corrupt} of {len(rows)} artifact(s) contain NUL bytes — "
              f"STORAGE-LEVEL data loss, not a pipeline bug.")
        print("   These files kept their full size on disk while their contents were")
        print("   replaced by zeros, which is what happens when written data never")
        print("   reaches the disk (node crash before writeback, NFS/cache fault, or")
        print("   a quota/disk-full condition hit at flush time). No amount of file")
        print("   locking in this codebase can prevent or undo it.")
        print("   NEXT: check your quota (`quota -s`, `df -h` on the results")
        print("   filesystem) and report the affected paths to your cluster admins")
        print("   BEFORE re-running — a re-run onto unhealthy storage loses the")
        print("   compute again. See scripts/diagnose_artifact.py for per-file detail.")


if __name__ == "__main__":
    main()
