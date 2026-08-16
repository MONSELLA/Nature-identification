#!/usr/bin/env python3
"""
scripts/subset_artifact_for_gt.py

Cut one or more VLM artifacts down to just the images that the hand-drawn
grounding GT actually covers, and report whether those records are already
grounded.

WHY THIS EXISTS. The BIG-5 artifacts hold every image of their platform
(measured: 1122 twitter + 5541 weibo = 6663 records), while the manual
grounding GT covers 340 of them. Running SAM3 over all 6663 to evaluate 340
would spend roughly 20x the GPU time for no extra measurement — and BIG-5
images are precisely the raw social-media resolutions that make the vision
encoder OOM-prone (recap v18/v19), so the wasted work is also the riskiest
work.

WHY A SEPARATE FILE RATHER THAN `--in_place` ON THE ORIGINAL. The project
convention is ONE artifact per run, enriched in place by grounding
(CLAUDE.md). Grounding only 340 of 6663 records IN PLACE would leave the
production artifact partially grounded — every ungrounded record
indistinguishable from "SAM3 found nothing" for any later analysis, and the
dataset-wide nature relevance scores silently computed over a 5% subset. So
the subset is written to its own file, the production artifacts are never
touched, and a full BIG-5 grounding run (when it happens) remains the single
source of truth for those.

`--max_samples` on run_grounding_pipeline.py cannot do this job: it takes the
first N records in artifact order, which is not the annotated set.

Prints a machine-readable `SUBSET_GROUNDED=<n>/<total>` line for the calling
job script to branch on.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def gt_image_ids(gt_dir):
    """Every image the GT covers, both splits, keyed as a bare file name."""
    ids = set()
    for split in ("nature", "no_nature"):
        for path in glob.glob(os.path.join(gt_dir, split, "*.json")):
            # The processed filename IS "<image_id>.json", so the stem is the
            # image name — no need to open and parse 340 files.
            ids.add(os.path.basename(path)[:-len(".json")])
    return ids


def _is_grounded(rec):
    """Has the GROUNDING stage run over this record?

    Tested by KEY PRESENCE, never truthiness. An image with no nature entities
    to ground — every one of the 170 no-nature images, and any nature image
    whose entities all failed the mask threshold — ends up with
    `object_groundings: []`, which is falsy but means "grounding ran and there
    was correctly nothing to record". Treating that as ungrounded would
    re-run SAM3 over those images on every single invocation and never reach a
    fixed point.

    `nature_relevance_score_coverage_ratio` is checked first because
    src/grounding_pipeline.py writes it for EVERY record it processes,
    unconditionally, whether or not anything grounded.
    """
    return ("nature_relevance_score_coverage_ratio" in rec
            or "object_groundings" in rec)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True, nargs="+",
                    help="source artifact(s); records are merged by image name")
    ap.add_argument("--gt_dir", required=True,
                    help="grounding GT dir holding nature/ and no_nature/")
    ap.add_argument("--out", required=True, help="subset artifact to write")
    args = ap.parse_args()

    wanted = gt_image_ids(args.gt_dir)
    if not wanted:
        sys.exit(f"No GT records under {args.gt_dir}/{{nature,no_nature}}")

    header = None
    kept = {}
    total_records = 0
    for path in args.artifact:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "image_path" not in rec:
                    # Artifact header (run config). When several artifacts are
                    # merged (e.g. big5_twitter + big5_weibo), keep the first
                    # one but OVERWRITE its "dataset" field to say so plainly —
                    # otherwise the subset silently inherits "big5_twitter"
                    # even though it mixes both platforms. The only thing that
                    # field drives downstream is run_grounding_pipeline.py's
                    # --instance_grounding auto-detect (on iff dataset=="coco"),
                    # so this is a cosmetic fix, not a behavior change — but
                    # worth it so the grounding banner and any saved header
                    # don't mislabel a mixed-platform run as pure twitter.
                    if header is None:
                        header = dict(rec)
                        if len(args.artifact) > 1:
                            header["dataset"] = "big5_gt_subset"
                    continue
                total_records += 1
                name = os.path.basename(rec["image_path"])
                if name in wanted:
                    kept[name] = rec

    grounded = sum(1 for r in kept.values() if _is_grounded(r))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        if header is not None:
            fh.write(json.dumps(header) + "\n")
        for name in sorted(kept):
            fh.write(json.dumps(kept[name]) + "\n")

    missing = sorted(wanted - set(kept))
    print(f"  scanned {total_records} artifact records across "
          f"{len(args.artifact)} file(s)")
    print(f"  kept {len(kept)} of {len(wanted)} GT images -> {args.out}")
    if missing:
        # Not fatal: the scorer counts an absent image's GT regions as false
        # negatives, which is the honest treatment. But it is worth surfacing,
        # because a large number here usually means the wrong artifacts were
        # passed rather than a genuinely incomplete inference run.
        print(f"  WARNING: {len(missing)} GT image(s) absent from the "
              f"artifact(s): {missing[:5]}")
    print(f"SUBSET_GROUNDED={grounded}/{len(kept)}")


if __name__ == "__main__":
    main()
