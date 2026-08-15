"""
src/evaluation/detection_metrics.py

Mask-IoU detection evaluation for COCO (COCO's own `segm` task, never `bbox`)
— the metric half of the Grounding pipeline's COCO story (the pixel half is
src/grounding_pipeline.py's instance grounding, and the wiring is
scripts/run_vlm_pipeline.py's `--stage score`).

WHAT THIS EVALUATES
    The VLM pipeline extracts entity PHRASES ("a cow", "grass"); SAM3's
    SEMANTIC head turns each nature-labeled phrase into one concept-level MASK.
    COCO ships per-instance segmentation with a class label for every
    annotation. This module answers the two questions that pairing makes
    possible, and keeps them SEPARATE:

      1. LOCALIZATION — did the pipeline's mask land where COCO says an
         object is?  TP / FP / FN -> precision, recall, F1.
      2. LABEL CORRECTNESS — for the regions that DID match a GT object, was
         the entity named correctly?  Scored two ways: exact match (the strict
         view) and hierarchically (hP/hR/hF1 + Wu-Palmer, the graded view that
         gives "bull" partial credit against a GT of "cow" instead of flat
         zero).

    Reporting them separately is the point. A single fused number would let a
    model that localizes badly but names well look identical to one that does
    the reverse, and those are different failures with different fixes.

GRANULARITY IS ENTITY (CONCEPT) LEVEL, set by the caller
    (`scripts/run_vlm_pipeline.py`'s `score_image_entities`, this module's only
    production consumer). Each prediction is SAM3's `semantic_seg` mask for one
    extracted entity — already one region per concept — and each GT entry is
    every annotated instance of one class merged into ONE region. The functions
    below are granularity-agnostic (they take regions and counters), but the
    numbers they produce are about CONCEPTS, not individual objects, because
    that is what this pipeline predicts. An instance-level caller existed and
    was removed: it punished SAM3's undeduplicated duplicate queries and COCO's
    non-exhaustive annotation of crowded scenes, neither a model error.

NO AVERAGE PRECISION — deliberate, and the reason is not "we didn't get to it".
    AP ranks predictions by confidence and integrates the resulting PR curve,
    so it needs a per-prediction confidence. `semantic_seg` is a dense logit
    map: there is no scalar score attached to a concept's mask. An earlier
    version scored the instance head instead and ranked by each entity's
    strongest instance, which was an engineering convenience rather than a
    measurement — "the strongest instance that happened to clear the score
    gate", not a calibrated confidence for the region actually being matched.
    Manufacturing a replacement (mean sigmoid inside the mask, pixel count,
    ...) would be strictly worse than reporting nothing. `sweep_summary` is
    the headline instead, and needs no confidence at all: it varies the IoU
    THRESHOLD at a fixed prediction set, which measures MASK TIGHTNESS.

MATCHING IS ON MASKS, NEVER BOXES. What SAM3 actually produces IS a mask, and
    COCO ships per-instance `segmentation` for every annotation, so reducing
    both sides to boxes first discards real signal: a box around a curled-up
    dog is mostly the sofa behind it, and two masks that overlap poorly can
    share a nearly identical box (`mask_iou_matrix`, via pycocotools' own
    `iou`, is what `score_image_entities` actually calls; `box_iou`/
    `iou_matrix` below remain as general-purpose box-geometry primitives —
    used by the pure-geometry test suite and by crowd suppression's box
    variant — not as an alternative matching mode). Two crossing diagonal
    strokes have box IoU 1.000 and mask IoU 0.000 (verified in the test suite)
    — that gap is why there is no box-matching option in this pipeline at all.

MATCHING IS ALSO CLASS-AGNOSTIC — deliberately, and this is the non-obvious
    part. Standard detection evaluation matches a prediction to a GT instance
    only if their CLASSES already agree. That is precisely the wrong protocol
    here: if a predicted "bull" could never be paired with a GT "cow", it
    would be counted as a false positive AND the cow as a false negative, and
    the hierarchical label metrics — the whole reason for evaluating this way
    — would never see the pair at all. So assignment uses geometric overlap
    ONLY, and the class comparison happens afterwards, on the pairs the
    geometry produced. Consequence to keep in mind when reading the numbers:
    precision/recall here measure LOCALIZATION alone, not "detected the right
    thing" — the label side of that is exactly what the accompanying label
    metrics report.

    Assignment is one-to-one (Hungarian, maximizing total IoU over pairs that
    clear `iou_threshold`), not greedy, so one large predicted instance cannot
    claim several GT instances and no GT instance can be double-counted.

UNMATCHED PREDICTIONS — the curated-vocabulary rule.
    COCO annotates 80 curated classes, not everything visible. A predicted
    mask around a real tree in a real photograph is a CORRECT detection of
    something COCO simply never labeled; scoring it as a false positive would
    punish the pipeline for being right. But a predicted "dog" that matches no
    GT dog instance IS a genuine error and must count. So an unmatched
    prediction is a FALSE POSITIVE only when its own entity phrase resolves to
    a class in the evaluated GT vocabulary, and is EXCLUDED (counted,
    reported, but not charged against precision) otherwise.
    `classify_unmatched_prediction` implements exactly that test; the excluded
    count is always reported alongside precision so the size of that
    exemption is visible rather than hidden.

CROWD REGIONS.
    COCO's `iscrowd=1` annotations cover a GROUP with one loose region.
    Following COCO's own protocol they are neither required to be detected
    (not a false negative when missed) nor counted against precision when a
    prediction lands inside one (not a false positive) — see
    `crowd_suppressed` (box path) / `mask_ioa` (mask path, the one this
    project actually uses).

NO TRUE NEGATIVES.
    Detection has no "correctly rejected background region" to count: the
    space of masks that could have been predicted and weren't is unbounded.
    TN is therefore undefined and no accuracy is reported here — only
    precision, recall and F1, all of which are well-defined without it.
    This is a deliberate departure from the axis metrics elsewhere in the
    project, which DO report accuracy because their negative class is a real,
    finite thing.

Pure math + WordNet only: no torch, no model loading, no I/O (mask_iou_matrix
excepted, which needs pycocotools for its C-level RLE math but still no torch/
model loading). Everything here is unit-testable on plain lists of boxes/RLEs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Standard COCO-style operating point for "the box is on the right object".
DEFAULT_IOU_THRESHOLD = 0.5

# COCO's own IoU ladder: 0.50, 0.55, ..., 0.95. Used by the strictness sweep
# (sweep_summary); the name is kept for continuity with COCO's convention even
# though this project no longer computes AP over it.
COCO_AP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

# Fraction of a PREDICTED box that must fall inside a crowd region before the
# prediction is treated as explained by that crowd and exempted from the false
# positive count. Intersection-over-prediction-AREA, not IoU — a crowd box is
# typically far larger than any single prediction, so IoU would stay tiny even
# when the prediction sits entirely inside it.
DEFAULT_CROWD_IOA_THRESHOLD = 0.5

# COCO's OWN object-size stratification, taken verbatim from pycocotools'
# `Params(iouType="segm").areaRng` — small < 32^2 px, medium < 96^2 px, large
# beyond. Not this project's invention: AP^small/AP^medium/AP^large is a
# standard part of the COCO detection AND segmentation protocols (Lin et al.
# 2014), reported by every method on the COCO leaderboard, and cocoeval applies
# these exact cut-offs to the `segm` task using each annotation's MASK area.
#
# WHY SIZE STRATIFICATION AT ALL: a pooled score is dominated by whatever size
# happens to be most common, and small objects are systematically the hardest —
# a few pixels of boundary error destroys the IoU of a 20x20 region while
# barely moving a 300x300 one. Splitting by size separates "the model misses
# small things" from "the model's masks are loose", which the pooled number
# cannot.
COCO_AREA_RANGES = (
    ("small", 0, 32 ** 2),
    ("medium", 32 ** 2, 96 ** 2),
    ("large", 96 ** 2, float("inf")),
)
AREA_BUCKETS = tuple(name for name, _, _ in COCO_AREA_RANGES)


def area_bucket(area: float) -> str:
    """COCO size bucket ("small"/"medium"/"large") for a region of `area` px.

    Boundaries follow cocoeval exactly: a region is in a bucket when
    `lo <= area < hi`, so 1024 px is the first "medium" and 9216 px the first
    "large". Anything at or below 0 is reported "small" (a degenerate region
    cannot be anything else).
    """
    for name, lo, hi in COCO_AREA_RANGES:
        if lo <= area < hi:
            return name
    return AREA_BUCKETS[-1]


# =============================================================================
# Box geometry
# =============================================================================
def box_area(box: Sequence[float]) -> float:
    """Area of an [x1, y1, x2, y2] box; 0.0 for a degenerate/inverted box."""
    w = max(0.0, float(box[2]) - float(box[0]))
    h = max(0.0, float(box[3]) - float(box[1]))
    return w * h


def box_intersection(a: Sequence[float], b: Sequence[float]) -> float:
    """Area of the overlap between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes, in [0, 1]."""
    inter = box_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0.0 else 0.0


def iou_matrix(gt_boxes: Sequence[Sequence[float]],
               pred_boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """Dense (n_gt, n_pred) IoU matrix. Empty on either side gives an
    appropriately-shaped empty array rather than raising, so callers don't
    need to special-case images with no GT or no prediction."""
    mat = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            mat[i, j] = box_iou(g, p)
    return mat


def mask_iou_matrix(gt_rles: Sequence[Dict[str, Any]],
                    pred_rles: Sequence[Dict[str, Any]],
                    gt_iscrowd: Optional[Sequence[bool]] = None) -> np.ndarray:
    """Dense (n_gt, n_pred) MASK IoU matrix, via pycocotools' own `maskUtils.iou`.

    This is the segmentation counterpart of `iou_matrix`, and the one that
    actually fits this project: the pipeline's predictions ARE masks (SAM3's
    instance head), and COCO ships per-instance `segmentation` for every
    annotation, so reducing both sides to boxes before comparing them discards
    real signal. A box around a curled-up dog is mostly the sofa behind it; a
    box around a diagonal giraffe is mostly sky. Two masks that overlap poorly
    can still have near-identical boxes, and box IoU cannot tell them apart.
    COCO's own evaluation has always had both an `iouType="bbox"` and an
    `iouType="segm"` task for exactly this reason — this function is the latter.

    Computed with pycocotools' C implementation rather than decoding to dense
    arrays and intersecting in NumPy: it operates directly on the
    run-length encoding, so it stays fast at COCO scale and never materializes
    a full HxW boolean array per pair.

    `gt_iscrowd` is passed through to pycocotools, which switches a crowd
    column from IoU to intersection-over-DETECTION-area — the same asymmetry
    `box_ioa` implements for the box path, and the reason a crowd region
    dwarfing a single prediction still suppresses it.
    """
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=float)
    mask_utils = _mask_utils()

    def _native(rle):
        counts = rle["counts"]
        return {"size": list(rle["size"]),
                "counts": counts.encode("ascii") if isinstance(counts, str) else counts}

    gt_native = [_native(r) for r in gt_rles]
    pred_native = [_native(r) for r in pred_rles]
    crowd = [int(bool(c)) for c in (gt_iscrowd or [False] * len(gt_rles))]
    # maskUtils.iou returns (n_detections, n_gt); this module's convention
    # everywhere else is (n_gt, n_pred), so transpose.
    ious = mask_utils.iou(pred_native, gt_native, crowd)
    return np.asarray(ious, dtype=float).reshape(len(pred_rles), len(gt_rles)).T


def _mask_utils():
    """Import pycocotools' mask module lazily, mirroring
    grounding_pipeline._mask_utils, so this module stays importable (and its
    pure-geometry helpers unit-testable) without the C extension present."""
    from pycocotools import mask as mask_utils
    return mask_utils


def merge_rles(rles: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Union several RLE masks into one, via pycocotools' own `merge`.

    Used to collapse an ENTITY's many instance masks into a single
    concept-level mask (and, on the GT side, a CLASS's many annotated
    instances into one region) for the entity-level evaluation. Overlapping
    pixels are counted once — the whole reason to union rather than sum.
    Returns None for an empty input.
    """
    if not rles:
        return None
    mask_utils = _mask_utils()
    native = []
    for rle in rles:
        counts = rle["counts"]
        native.append({"size": list(rle["size"]),
                       "counts": counts.encode("ascii") if isinstance(counts, str) else counts})
    merged = mask_utils.merge(native, intersect=False)
    counts = merged["counts"]
    return {"size": [int(merged["size"][0]), int(merged["size"][1])],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def rle_area(rle: Optional[Dict[str, Any]]) -> int:
    """Pixel count of an RLE mask, straight from pycocotools (no decode)."""
    if rle is None:
        return 0
    mask_utils = _mask_utils()
    counts = rle["counts"]
    native = {"size": list(rle["size"]),
              "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    return int(mask_utils.area(native))


def mask_ioa(pred_rle: Dict[str, Any], region_rle: Dict[str, Any]) -> float:
    """Intersection over the PREDICTION's own mask area — the mask counterpart
    of `box_ioa`, used for crowd suppression on the segmentation path."""
    mask_utils = _mask_utils()

    def _native(rle):
        counts = rle["counts"]
        return {"size": list(rle["size"]),
                "counts": counts.encode("ascii") if isinstance(counts, str) else counts}

    # iscrowd=1 makes pycocotools normalize by the DETECTION's area rather
    # than the union — exactly the "how much of this prediction sits inside
    # that region" question, computed without decoding either mask.
    val = mask_utils.iou([_native(pred_rle)], [_native(region_rle)], [1])
    return float(np.asarray(val).reshape(-1)[0]) if np.asarray(val).size else 0.0


def box_ioa(pred: Sequence[float], region: Sequence[float]) -> float:
    """Intersection over the PREDICTION's own area — "how much of this
    prediction sits inside that region". Used for crowd suppression, where IoU
    is the wrong measure (see DEFAULT_CROWD_IOA_THRESHOLD)."""
    area = box_area(pred)
    if area <= 0.0:
        return 0.0
    return float(box_intersection(pred, region) / area)


# =============================================================================
# One-to-one assignment
# =============================================================================
def match_boxes(
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ious: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Class-agnostic one-to-one box assignment maximizing total IoU.

    Returns `(matches, unmatched_gt, unmatched_pred)` where each match is
    `(gt_index, pred_index, iou)` and every match's IoU is `>= iou_threshold`.

    `ious` optionally supplies an already-computed (n_gt, n_pred) matrix. The
    strictness sweep re-matches the same regions at ten IoU thresholds, and
    recomputing the matrix ten times per image would be pure waste — the
    geometry doesn't change, only the cut-off applied to it.

    Hungarian assignment rather than the greedy highest-IoU-first loop most
    detection code uses: greedy can strand a GT instance whose only above-threshold
    partner was taken by an earlier, marginally better pairing, which
    understates recall on crowded images (exactly what COCO has a lot of).
    Falls back to greedy automatically if SciPy isn't importable, so this
    module keeps working in a minimal environment — the fallback is noted in
    the caller's summary, never silently swapped in without a trace.
    """
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    if ious is None:
        ious = iou_matrix(gt_boxes, pred_boxes)
    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment MINIMIZES, so negate. Pairs below threshold are
        # filtered afterwards rather than being made infeasible up front — the
        # assignment is over the full matrix, then trimmed.
        rows, cols = linear_sum_assignment(-ious)
        candidate_pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        candidate_pairs = _greedy_pairs(ious)

    matches = [(int(i), int(j), float(ious[i, j]))
               for i, j in candidate_pairs if ious[i, j] >= iou_threshold]
    matched_gt = {i for i, _, _ in matches}
    matched_pred = {j for _, j, _ in matches}
    return (matches,
            [i for i in range(n_gt) if i not in matched_gt],
            [j for j in range(n_pred) if j not in matched_pred])


def _greedy_pairs(ious: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy fallback for `match_boxes` when SciPy is unavailable: repeatedly
    take the globally highest remaining IoU and retire both its row and its
    column."""
    pairs: List[Tuple[int, int]] = []
    remaining = ious.copy()
    while remaining.size and remaining.max() > 0.0:
        i, j = np.unravel_index(int(remaining.argmax()), remaining.shape)
        pairs.append((int(i), int(j)))
        remaining[i, :] = -1.0
        remaining[:, j] = -1.0
    return pairs


def crowd_suppressed(
    pred_box: Sequence[float],
    crowd_boxes: Sequence[Sequence[float]],
    ioa_threshold: float = DEFAULT_CROWD_IOA_THRESHOLD,
) -> bool:
    """True if this prediction falls far enough inside some crowd region to be
    exempt from the false-positive count (COCO's own convention)."""
    return any(box_ioa(pred_box, c) >= ioa_threshold for c in crowd_boxes)


# =============================================================================
# The curated-vocabulary false-positive rule
# =============================================================================
def classify_unmatched_prediction(
    pred_label_terms: Sequence[str],
    eval_vocab_terms: Dict[str, str],
) -> Optional[str]:
    """Decide whether an unmatched prediction is a FALSE POSITIVE or EXCLUDED.

    `pred_label_terms` is the set of normalized surface forms for the predicted
    entity (its phrase, its head noun, its WordNet lemmas — whatever the caller
    built). `eval_vocab_terms` maps each normalized surface form of every class
    IN THE EVALUATED GT VOCABULARY to that class's name.

    Returns the matched class name when the prediction names an evaluated
    class (-> caller counts a false positive), or None when it names something
    outside the vocabulary (-> caller counts it as excluded).

    Note the vocabulary tested against is the EVALUATED one, which on this
    project is the nature-mapped subset of COCO's 80 rather than all 80. That
    distinction matters and is not a shortcut: since only nature-labeled
    entities are grounded, GT instances of non-nature classes are not evaluation
    targets at all, so a prediction naming one of them could never find a
    partner no matter how well localized it was. Charging it as a false
    positive would penalize the pipeline for an instance the protocol removed from
    play — the same reasoning that exempts the tree.
    """
    for term in pred_label_terms:
        if term in eval_vocab_terms:
            return eval_vocab_terms[term]
    return None


# =============================================================================
# Aggregation
# =============================================================================
def detection_summary(counts: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the running detection counters into the reported numbers.

    `counts` carries: tp, fp, fn, excluded_pred, crowd_suppressed,
    n_gt_instances, n_pred_instances.

    Precision's denominator is TP + FP, which by construction EXCLUDES the
    curated-vocabulary exemptions — `excluded_predictions` is reported next to
    it so the exemption's size is always visible. There is no accuracy key:
    see the module docstring on why TN is undefined here. There is no AP key
    either: see the module docstring on why the semantic head this pipeline
    scores supports no confidence ranking.
    """
    tp, fp, fn = counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        "iou_threshold": counts.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "excluded_predictions": counts.get("excluded_pred", 0),
        "crowd_suppressed_predictions": counts.get("crowd_suppressed", 0),
        "n_gt_instances": counts.get("n_gt_instances", 0),
        "n_pred_instances": counts.get("n_pred_instances", 0),
    }
    return summary


def sweep_summary(sweep_counts: Dict[float, Dict[str, int]]) -> Dict[str, Any]:
    """Precision/recall/F1 as a FUNCTION of the IoU threshold.

    `sweep_counts` is `{iou_threshold: {"tp","fp","fn", ...}}`, pooled over the
    dataset — each threshold's counters come from its own independent matching
    pass, because a pair clearing IoU 0.50 need not clear 0.75.

    THIS IS NOT AP, and the distinction matters when reporting:

      * P/R/F1 at threshold t is ONE OPERATING POINT — every confirmed
        entity mask is included, and the only thing varying across the sweep
        is how much overlap counts as a hit. The
        SHAPE of that curve is a direct readout of MASK TIGHTNESS: an F1 that
        holds from 0.50 to 0.75 means the masks genuinely trace their objects;
        one that collapses means they are loose blobs that only just cleared
        the permissive threshold.
      * AP would instead sweep the CONFIDENCE cutoff at fixed overlap and
        integrate the resulting PR curve, measuring whether the confidence
        RANKING is informative. This project does NOT report AP — see the
        module docstring on why the semantic head supports no such ranking.

    Reports each rung plus three headline numbers mirroring COCO's own naming:
    `@0.50` (permissive), `@0.75` (strict), and `@[.50:.95]` (the mean across
    the ladder, so a single number still reflects the whole curve).
    """
    per_threshold: Dict[float, Dict[str, Any]] = {}
    for t, c in sorted(sweep_counts.items()):
        tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_threshold[t] = {"tp": tp, "fp": fp, "fn": fn,
                            "precision": precision, "recall": recall, "f1": f1,
                            "excluded_predictions": c.get("excluded_pred", 0),
                            "crowd_suppressed_predictions": c.get("crowd_suppressed", 0)}
    if not per_threshold:
        return {}

    def _at(t: float, key: str) -> float:
        return per_threshold.get(t, {}).get(key, 0.0)

    out: Dict[str, Any] = {"per_iou": per_threshold}
    for key in ("precision", "recall", "f1"):
        out[f"{key}_50"] = _at(0.5, key)
        out[f"{key}_75"] = _at(0.75, key)
        out[f"{key}_50_95"] = float(np.mean([v[key] for v in per_threshold.values()]))
    return out


def size_summary(size_counts: Dict[float, Dict[str, Dict[str, int]]]) -> Dict[str, Any]:
    """Precision/recall/F1 split by COCO OBJECT-SIZE bucket, per IoU threshold.

    `size_counts` is `{iou_threshold: {bucket: {"tp","fp","fn"}}}`, pooled over
    the dataset. Buckets are COCO's own small/medium/large (`COCO_AREA_RANGES`),
    the same cut-offs cocoeval applies to the `segm` task.

    HOW A REGION IS ASSIGNED A BUCKET, mirroring cocoeval.evaluateImg:
      * a TRUE POSITIVE and a FALSE NEGATIVE take the bucket of their GT region
        (`g['area']` in cocoeval — GT outside the range is ignored there);
      * a FALSE POSITIVE takes the bucket of its OWN predicted region
        (`d['area']` in cocoeval — a detection outside the range is ignored).

    This yields a clean property cocoeval itself does not have: every TP, FN and
    FP lands in exactly one bucket, so the three buckets PARTITION the pooled
    counts and sum back to the `sweep_summary` totals. cocoeval re-runs matching
    per area range with ignore flags, which can in principle form different
    pairs per range; here matching is done once and the results partitioned,
    which is deterministic and reconciles exactly.

    Returned per bucket: `precision`/`recall`/`f1` at `@0.50`, `@0.75` and the
    `@[.50:.95]` mean (the same three headline points as `sweep_summary`), the
    full `per_iou` table, and `n_gt` — the bucket's GT support at IoU 0.50,
    which is what makes a weak bucket readable as "genuinely bad" versus
    "almost no support".
    """
    if not size_counts:
        return {}

    out: Dict[str, Any] = {}
    for bucket in AREA_BUCKETS:
        per_threshold: Dict[float, Dict[str, Any]] = {}
        for t, buckets in sorted(size_counts.items()):
            c = (buckets or {}).get(bucket, {})
            tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_threshold[t] = {"tp": tp, "fp": fp, "fn": fn,
                                "precision": precision, "recall": recall, "f1": f1}
        entry: Dict[str, Any] = {"per_iou": per_threshold}
        for key in ("precision", "recall", "f1"):
            entry[f"{key}_50"] = per_threshold.get(0.5, {}).get(key, 0.0)
            entry[f"{key}_75"] = per_threshold.get(0.75, {}).get(key, 0.0)
            entry[f"{key}_50_95"] = float(np.mean([v[key] for v in per_threshold.values()])) \
                if per_threshold else 0.0
        base = per_threshold.get(0.5, {})
        entry["n_gt"] = base.get("tp", 0) + base.get("fn", 0)
        out[bucket] = entry
    return out


def label_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the per-matched-pair label scores into the reported numbers.

    Each record is `{"exact_match": bool, "hp","hr","hf1","wup": float,
    "resolved": bool}` — one per TRUE POSITIVE pair (label quality is only
    defined where a prediction actually landed on a GT object).

    Two views of the hierarchical numbers, mirroring the convention the
    ImageNet/Places hP/hR reporting already uses:
      * pooled       — a prediction whose phrase resolved to no WordNet node
                       scores 0.0 and stays in the average (failures as error).
      * `_resolved`  — those pairs are dropped instead, isolating "when the
                       phrase resolved at all, how close was it".
    Both are reported as mean ± population std, again matching the existing
    hierarchical reporting, because a mean alone can't tell a tight cluster
    from a bimodal split.
    """
    if not records:
        return {"support": 0}

    def _stats(vals: List[float], prefix: str) -> Dict[str, float]:
        if not vals:
            return {f"{prefix}": 0.0, f"{prefix}_std": 0.0}
        arr = np.asarray(vals, dtype=float)
        return {f"{prefix}": float(arr.mean()), f"{prefix}_std": float(arr.std())}

    out: Dict[str, Any] = {
        "support": len(records),
        "exact_match_accuracy": float(np.mean([bool(r["exact_match"]) for r in records])),
        "resolution_failure_rate": float(np.mean([not r["resolved"] for r in records])),
    }
    for key in ("hp", "hr", "hf1", "wup"):
        out.update(_stats([float(r[key]) for r in records], key))

    resolved = [r for r in records if r["resolved"]]
    out["support_resolved"] = len(resolved)
    for key in ("hp", "hr", "hf1", "wup"):
        out.update(_stats([float(r[key]) for r in resolved], f"{key}_resolved"))
    return out


def axis_agreement_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-matched-pair biotic/material axis agreement into
    per-axis accuracy + support, over TRUE POSITIVE detection pairs only.

    Each record is one `score_axis_agreement(...)` dict — see its docstring
    for why NATURE has no entry here (every matched pair is nature-vs-nature
    by construction in this evaluation, so an "agreement rate" for it would
    misreport a tautology as a measurement).

    A pair where either side has no opinion on an axis (`{axis}_agree` is
    None) is dropped from THAT axis's support rather than counted as either
    an agreement or a disagreement — an unmapped GT class or an unresolved
    VLM label is a missing value, not evidence of disagreement.
    """
    out: Dict[str, Any] = {}
    for axis in ("biotic", "material"):
        vals = [r[f"{axis}_agree"] for r in records if r.get(f"{axis}_agree") is not None]
        out[axis] = {"support": len(vals),
                     "accuracy": float(np.mean(vals)) if vals else 0.0}
    return out
