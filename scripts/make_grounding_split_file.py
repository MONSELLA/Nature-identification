#!/usr/bin/env python3
"""
scripts/make_grounding_split_file.py

Write the 340 hand-drawn BIG-5 grounding-annotation image basenames as a
`--split_file` for `run_vlm_pipeline.py --stage infer` — one filename per
line, matched by basename (`run_vlm_pipeline._filter_to_split`).

WHY THIS EXISTS. Every model evaluated against the grounding GT so far
already had a FULL-DATASET responses artifact (produced once for the normal
VLM-pipeline benchmark), so `subset_artifact_for_gt.py` could filter it down
to the 340 annotated images AFTER inference. A freshly fine-tuned model (e.g.
a LoRA adapter) has no such artifact yet — evaluating it means running
inference for the first time, and running it over the full ~6663-image BIG-5
platforms just to keep 340 would be pure waste. This script lets inference
itself be restricted to exactly those 340 images from the start via
`--split_file`, which is strictly cheaper than infer-then-subset.

Reuses `subset_artifact_for_gt.gt_image_ids` rather than re-reading the
`processed/{nature,no_nature}/*.json` filenames a second way, so the two
scripts can never disagree about which 340 images that is.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subset_artifact_for_gt import gt_image_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt_dir", required=True,
                    help="grounding GT dir holding nature/ and no_nature/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = gt_image_ids(args.gt_dir)
    if not ids:
        sys.exit(f"No GT records under {args.gt_dir}/{{nature,no_nature}}")

    with open(args.out, "w", encoding="utf-8") as fh:
        for name in sorted(ids):
            fh.write(name + "\n")
    print(f"wrote {len(ids)} image basenames -> {args.out}")


if __name__ == "__main__":
    main()
