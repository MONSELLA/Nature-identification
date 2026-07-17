"""
results_store.py

Shared helper for the two top-level evaluation scripts (evaluate_taxonomy_
labeling.py and run_vlm_pipeline.py) to persist results into one JSON file
per script, structured as:

    {
      "<dataset>": {
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
from datetime import datetime, timezone
from pathlib import Path


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
