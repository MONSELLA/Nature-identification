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
    python scripts/audit_artifacts.py --results_dir /home/pmonserrat/code/results/vlm_pipeline/baseline/imagenet/responses
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
    """(row_count, nul_byte_count) for a predictions CSV.

    The NUL check matters as much here as it does for the .jsonl artifacts —
    arguably more, since the predictions CSV is the file that's actually read
    for tables and qualitative review, and it can outlive a damaged artifact
    (confirmed: qwen/imagenet's 50,000-row CSV survived while its artifact
    was down to 28,960 records). The same storage fault that zeroed an
    artifact can zero a CSV, and a NUL-filled CSV would otherwise be reported
    here as simply "unreadable" with no hint as to why.

    Both passes STREAM rather than reading the file whole: a 50,000-image
    predictions CSV carries per-object JSON blobs and mask RLEs and can run
    to hundreds of MB, which is not something an audit should pull into
    memory. Rows are counted with csv.reader rather than by counting
    newlines, because quoted fields (captions, JSON) legitimately contain
    embedded newlines — counting '\\n' would badly overcount.

    csv.reader raises _csv.Error("line contains NUL") on a corrupted file,
    which the previous version did not catch (it caught only OSError), so a
    single NUL-damaged CSV would have crashed the whole audit run."""
    p = Path(path)
    n_nulls = 0
    try:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                n_nulls += chunk.count(b"\x00")
    except OSError:
        return None, None
    n_rows = None
    try:
        with open(p, newline="", errors="replace") as f:
            reader = csv.reader(f)
            next(reader, None)  # header row
            n_rows = sum(1 for _ in reader)
    except (OSError, csv.Error):
        n_rows = None  # corrupted beyond parsing; n_nulls explains why
    return n_rows, n_nulls


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
        csv_rows, csv_nulls = (_count_csv_rows(csv_matches[0]) if csv_matches
                               else (None, None))

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
            "predictions_csv_nulls": csv_nulls,
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
        # The predictions CSV is the file actually used for tables and
        # qualitative review, so its own integrity is reported separately from
        # the artifact's — it can survive a damaged artifact, and can be
        # damaged while the artifact survives.
        if r.get("predictions_csv_nulls"):
            flags.append(f"⛔ PREDICTIONS CSV NUL-CORRUPTED "
                         f"({r['predictions_csv_nulls']:,} NUL bytes)")
        elif r["predictions_csv_rows"] is not None and r["predictions_csv_rows"] != r["artifact_n_images"]:
            # NOT necessarily bad: a CSV with MORE rows than the artifact means
            # scoring ran when the artifact was still intact and the artifact
            # was damaged afterwards — i.e. the CSV is the surviving copy of a
            # completed run, which is exactly the case worth keeping.
            more = r["predictions_csv_rows"] > (r["artifact_n_images"] or 0)
            flags.append(
                f"csv has {r['predictions_csv_rows']} rows vs artifact's "
                f"{r['artifact_n_images']}"
                + (" — CSV PRESERVES a completed run the artifact lost" if more else ""))
        print(f"{str(r['dataset']):<14} {str(r['model']):<45} "
              f"{str(r['artifact_n_images']):>9} "
              f"{'Y' if r['has_header'] else 'N':>4} "
              f"{'Y' if r['has_footer'] else 'N':>4} "
              f"{r['n_bad_lines'] or 0:>4} "
              f"{str(r['predictions_csv_rows']):>9} "
              f"{str(r['results_json_n_images']):>14}  "
              f"{', '.join(flags)}")

    # --- Predictions-CSV inventory ------------------------------------------
    # Printed as its own section because it answers a DIFFERENT question from
    # the table above. The table is about artifacts (can this be resumed /
    # re-scored?); this is about the predictions CSV, which is what actually
    # gets read for results tables and qualitative review. A CSV can be intact
    # while its artifact is destroyed — in which case that run's per-image
    # results are NOT lost, whatever the artifact column says.
    intact = [r for r in rows
              if r.get("predictions_csv_rows") and not r.get("predictions_csv_nulls")]
    damaged = [r for r in rows if r.get("predictions_csv_nulls")]
    absent = [r for r in rows if not r.get("predictions_csv_rows")
              and not r.get("predictions_csv_nulls")]
    print()
    print("=" * 70)
    print("PREDICTIONS CSVs — the files used for results tables / qualitative review")
    print("=" * 70)
    print(f"INTACT ({len(intact)}):")
    for r in sorted(intact, key=lambda r: (str(r["dataset"]), str(r["model"]))):
        note = ""
        if r["artifact_n_images"] and r["predictions_csv_rows"] > r["artifact_n_images"]:
            note = "   <- survives a damaged artifact"
        print(f"   {str(r['dataset']):<14} {str(r['model']):<48} "
              f"{r['predictions_csv_rows']:>7,} rows{note}")
    if damaged:
        print(f"\nNUL-CORRUPTED ({len(damaged)}):")
        for r in damaged:
            print(f"   {str(r['dataset']):<14} {str(r['model']):<48} "
                  f"{r['predictions_csv_nulls']:,} NUL bytes")
    if absent:
        print(f"\nNO CSV ({len(absent)}) — scoring never completed for these:")
        for r in absent:
            print(f"   {str(r['dataset']):<14} {str(r['model']):<48}")

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
