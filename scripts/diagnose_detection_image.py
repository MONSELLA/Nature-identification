#!/usr/bin/env python3
"""
scripts/diagnose_detection_image.py

Why did (or didn't) a specific image's predicted box match a specific GT box?

Prints, for ONE image in a scored predictions CSV: every evaluated GT box,
every predicted INSTANCE box, and the full IoU matrix between them — plus the
buckets the scorer actually put each prediction in (TP / FP / excluded /
nature-on-non-nature) and the per-instance confidence scores.

THE DISTINCTION THIS EXISTS TO SETTLE: the prediction viewer's "SAM3
GROUNDINGS" panel draws `object_groundings` — the SEMANTIC, concept-level mask
(one blob per entity, binarized at --mask_threshold). The detection evaluation
does NOT use those. It uses `object_instances` — SAM3's INSTANCE head, gated by
--instance_score_threshold, each instance's mask reduced to its tight bbox by
grounding_pipeline.mask_to_bbox. A semantic overlay can look like a perfect fit
over a GT box while the instance box that was actually scored is somewhere
else, is much larger, or was never produced at all. When a match looks like it
"obviously should have happened", check the numbers here before suspecting the
matcher.

Usage:
    python scripts/diagnose_detection_image.py <predictions.csv> <image_substring>

e.g.
    python scripts/diagnose_detection_image.py \\
        results/.../vlm_pipeline_coco_results_coco_google_gemma-4-12B-it_predictions.csv \\
        000000223747
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.detection_metrics import box_iou, DEFAULT_IOU_THRESHOLD


def _fmt_box(b):
    return "[" + ", ".join(f"{v:7.1f}" for v in b) + "]"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    csv_path, needle = sys.argv[1], sys.argv[2]

    # The predictions CSV carries big JSON blobs per row (masks, groundings).
    csv.field_size_limit(sys.maxsize)

    row = None
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if needle in r["image_path"]:
                row = r
                break
    if row is None:
        print(f"No row whose image_path contains {needle!r}")
        sys.exit(1)

    print(f"IMAGE: {row['image_path']}")
    print(f"caption: {(row.get('caption') or '')[:200]}")
    print()

    def _j(key):
        raw = row.get(key)
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    matches = _j("detection_matches")
    fps = _j("detection_false_positives")
    excluded = _j("detection_excluded_predictions")
    missed = _j("detection_missed_gt")
    nonnat = _j("detection_nature_on_non_nature")

    print("--- SCORER'S OWN VERDICT (counts from the scored run) ---")
    for k in ("detection_n_gt_boxes_image", "detection_n_pred_boxes_image",
              "detection_tp_image", "detection_fp_image", "detection_fn_image",
              "detection_excluded_pred_image", "detection_crowd_suppressed_image"):
        print(f"  {k:<38} {row.get(k)}")
    print()

    print(f"--- MATCHED PAIRS (TP): {len(matches)} ---")
    for m in matches:
        print(f"  GT {m['gt_class']:<16} {_fmt_box(m['gt_bbox'])}")
        print(f"  -> {m['pred_object']:<16} {_fmt_box(m['pred_bbox'])} "
              f"IoU {m['iou']:.3f} score {m.get('pred_score', 0):.3f} "
              f"exact={m['exact_match']} hF1={m['hf1']:.3f}")
    print()

    print(f"--- MISSED GT (FN): {len(missed)} ---")
    for m in missed:
        print(f"  {m['gt_class']:<16} {_fmt_box(m['gt_bbox'])}")
    print()

    print(f"--- FALSE POSITIVES: {len(fps)} ---")
    for p in fps:
        print(f"  {p['pred_object']:<16} {_fmt_box(p['pred_bbox'])} "
              f"score {p.get('pred_score', 0):.3f} (names evaluated class "
              f"{p.get('named_class')!r})")
    print()

    print(f"--- EXCLUDED PREDICTIONS: {len(excluded)} ---")
    for p in excluded:
        print(f"  {p['pred_object']:<16} {_fmt_box(p['pred_bbox'])} "
              f"score {p.get('pred_score', 0):.3f}")
    print()

    if nonnat:
        print(f"--- NATURE BOX ON NON-NATURE GT: {len(nonnat)} ---")
        for p in nonnat:
            print(f"  {p['pred_object']:<16} {_fmt_box(p['pred_bbox'])} "
                  f"-> GT {p['gt_class']} {_fmt_box(p['gt_bbox'])} IoU {p['iou']:.3f}")
        print()

    # THE ACTUAL ANSWER: every GT box against every predicted box, so a
    # "why didn't these two pair up" question becomes a number you can read.
    gt_boxes = ([(m["gt_class"], m["gt_bbox"]) for m in matches]
                + [(m["gt_class"], m["gt_bbox"]) for m in missed])
    pred_boxes = ([(m["pred_object"], m["pred_bbox"]) for m in matches]
                  + [(p["pred_object"], p["pred_bbox"]) for p in fps]
                  + [(p["pred_object"], p["pred_bbox"]) for p in excluded])

    if not gt_boxes or not pred_boxes:
        print("(no GT and/or predicted boxes to cross-compare)")
        return

    print(f"--- FULL IoU MATRIX  ({len(gt_boxes)} GT x {len(pred_boxes)} pred) ---")
    print(f"    threshold for a match: IoU >= {DEFAULT_IOU_THRESHOLD} "
          f"(or whatever --detection_iou_threshold the run used)")
    print()
    header = " " * 26 + "".join(f"{n[:11]:>13}" for n, _ in pred_boxes)
    print(header)
    for gname, gbox in gt_boxes:
        cells = ""
        for pname, pbox in pred_boxes:
            iou = box_iou(gbox, pbox)
            mark = "*" if iou >= DEFAULT_IOU_THRESHOLD else " "
            cells += f"{iou:12.3f}{mark}"
        print(f"  GT {gname[:20]:<22}{cells}")
    print()
    print("  * = clears the IoU threshold. A pair with no '*' could never have")
    print("    matched no matter what the assignment algorithm did.")


if __name__ == "__main__":
    main()
