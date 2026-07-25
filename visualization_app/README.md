# BIG-5 Prediction Viewer

A small, self-contained web app for browsing the prediction CSVs produced by
`scripts/run_vlm_pipeline.py` and `scripts/evaluate_taxonomy_labeling.py`. It
shows each image next to **all** the information in its CSV row, lets you page
through images quickly, and filters by name / GT / prediction / mismatches.

Everything lives inside this folder and uses only the Python **standard
library** (no Flask / pandas install needed).

## Run

```bash
conda activate cv
python visualization_app/server.py
```

Then open <http://127.0.0.1:8070> in a browser. Stop with `Ctrl+C`.

## What it shows

- The image (resolved to its local copy — see *Configuration*).
- **Verdict cards** for each taxonomy axis (nature / biotic / material, plus
  image-level nature for COCO/BIG-5): ground truth → prediction, flagged as
  ✓ match or ✗ mismatch. Values like `True`/`yes`/`biotic`/`material` are
  normalised so the two schemas compare correctly.
- The **caption** and the model's **reasoning** (when present).
- The **extracted objects** table (VLM pipeline): object text, whether it was
  WordNet-mapped or VLM-labelled, and its per-axis labels.
- The **ground-truth targets** and per-target matched objects.
- A **Fields** table with every remaining column (F-CLIPScore, ClipMatch,
  hierarchical metrics, etc.).

Both CSV schemas are auto-detected:
- **VLM pipeline** — one row per image, with JSON `objects` / `gt_targets`.
- **Taxonomy calibration** — one row per (image, class), flat columns.

## Navigation & filters

- **Prev / Next** buttons, **← / →** arrow keys, or the **jump-to-#** box.
- Click any item in the **Results** list on the left.
- **Search** matches image name, class name, caption, reasoning, and predicted
  class.
- **Dropdown filters** are generated automatically from low-cardinality columns
  (dataset, gt/pred axes, parse_failed, …).
- **Only gt ≠ pred mismatches** and **Only parse failures** toggles for quick
  error spot-checking.

## Configuration — `config.py`

- `CSV_SEARCH_DIRS` — directories scanned (recursively) for `*predictions*.csv`.
  Defaults to the repo `results/` tree and the TFM root.
- `DATASET_IMAGE_ROOTS` — where the images live **on this machine**. The
  `image_path` stored in a CSV was written on the compute node and doesn't exist
  here, so it is re-rooted: for imagenet/places the last two path components
  (`<class_id>/<file>`) are joined onto the local root; for big5 the bare
  filename is matched. COCO is not downloaded yet — add its path here when it is.
- `DATASET_PATH_DEPTH` — how many trailing path components to keep per dataset.
- `HOST` / `PORT` — server bind address (default `127.0.0.1:8070`).

## Files

| file | purpose |
|------|---------|
| `server.py` | stdlib HTTP server + JSON/image API |
| `config.py` | paths, ports (edit this) |
| `dataloader.py` | discover + parse the CSVs, detect schema/dataset |
| `imageresolver.py` | re-root a CSV `image_path` onto the local image dirs |
| `static/index.html`, `static/style.css`, `static/app.js` | the UI |
