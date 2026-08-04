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
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
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
    except OSError as e:
        return {"error": str(e)}
    # Mirrors _model_slug's own preference (model_name over the full
    # family/name "model" string) so the slug used here to find a matching
    # predictions CSV agrees with the one phase_score actually wrote it with.
    model_slug = (model_name or model or "").replace("/", "_")
    return {
        "n_images": n_images,
        "has_header": has_header,
        "has_footer": has_footer,
        "dataset": dataset,
        "model": model,
        "model_slug": model_slug,
        "n_bad_lines": n_bad_lines,
        "size_bytes": Path(path).stat().st_size,
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
    for r in rows:
        flags = []
        if not r["has_header"]:
            flags.append("NO-HEADER")
        if not r["has_footer"]:
            flags.append("INCOMPLETE(no footer)")
        if r["n_bad_lines"]:
            flags.append(f"{r['n_bad_lines']} truncated line(s)")
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


if __name__ == "__main__":
    main()
