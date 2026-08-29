#!/usr/bin/env python3
"""
scripts/combine_grounding_metrics.py

Merge every per-model `*_results.json` written by `score_grounding_gt.py`
into ONE combined JSON, so a cross-model comparison (e.g. rebuilding a table
like the paper's COCO Table 45, but for the BIG-5 grounding GT) never needs to
open several files. Reads `results/grounding_eval/metrics/` by default (see
scripts/score_grounding_gt.py's own docstring for what that directory holds
and scripts/score_grounding_gt.py --out for how a single model's file is
produced).

    python scripts/combine_grounding_metrics.py

Nothing here recomputes anything — it is a pure JSON merge over files
`score_grounding_gt.py` already wrote. Re-run it after adding or re-scoring
any model; it always reads the CURRENT contents of the metrics directory, so
it never goes stale on its own the way a hand-copied summary would.

OUTPUT SHAPE: the top-level JSON object IS the metrics — `{model_name:
{...that model's full results dict...}, ...}` — nothing else. No wrapper key,
no file-provenance bookkeeping, so the file holds solely the metrics, per Pau's
explicit ask; run this script with `-v`/watch its console output if you want
to know which source file fed which key.

MODEL NAME: recovered from the filename (`big5_grounding_<name>_results.json`
-> `<name>`), then passed through `MODEL_DISPLAY_NAMES` for a table-ready
label — e.g. the RFT adapter names `rft_selfdistill`/
`rft_distill_gemma26b_a4b` become `self-distill`/`teacher-distill` (per
CLAUDE.md's fine-tuning section: the LoRA adapter's own training data source
is what distinguishes them, not the base model, which is `google/gemma-4-12B-it`
for both). A `<name>` with no entry in the table is kept AS WRITTEN and
flagged in the console output, so an unrecognized model is never silently
mis-keyed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PREFIX, _SUFFIX = "big5_grounding_", "_results.json"

# Raw <name> (from the filename `score_grounding_gt.py --out` was given) ->
# the display label used as this file's top-level key. Extend this table
# rather than renaming files on disk when a new model needs a nicer label.
MODEL_DISPLAY_NAMES = {
    "google_gemma-4-E4B-it": "Gemma-4-E4B",
    "google_gemma-4-12B-it": "Gemma-4-12B",
    "google_gemma-4-26B-A4B-it": "Gemma-4-26B-A4B",
    "rft_selfdistill": "self-distill",
    "rft_distill_gemma26b_a4b": "teacher-distill",
}


def model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith(_PREFIX) and base.endswith(_SUFFIX):
        return base[len(_PREFIX):-len(_SUFFIX)]
    return base[:-len(".json")] if base.endswith(".json") else base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_dir",
                    default=os.path.join(os.path.dirname(APP_DIR), "results",
                                         "grounding_eval", "metrics"),
                    help="directory holding the per-model *_results.json files")
    ap.add_argument("--out",
                    default=None,
                    help="output path (default: <metrics_dir>/"
                         "all_models_results.json)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.metrics_dir, "all_models_results.json")
    paths = sorted(p for p in glob.glob(os.path.join(args.metrics_dir, "*_results.json"))
                   if os.path.abspath(p) != os.path.abspath(out))
    if not paths:
        raise SystemExit(f"No *_results.json files found under {args.metrics_dir}")

    combined = {}
    for p in paths:
        raw_name = model_name_from_path(p)
        display_name = MODEL_DISPLAY_NAMES.get(raw_name, raw_name)
        if raw_name not in MODEL_DISPLAY_NAMES:
            print(f"  NOTE: {raw_name!r} has no entry in MODEL_DISPLAY_NAMES — "
                  f"keeping it as-is. Add it to the table if it needs a "
                  f"table-ready label.")
        if display_name in combined:
            raise SystemExit(f"Two files resolve to the same key {display_name!r}: "
                             f"{p} and an earlier one — check MODEL_DISPLAY_NAMES "
                             f"for a collision.")
        with open(p, encoding="utf-8") as fh:
            combined[display_name] = json.load(fh)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, default=str)
    print(f"combined {len(combined)} model(s) -> {out}")
    for name, p in zip(combined, paths):
        print(f"  - {name}  <-  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
