"""
src/evaluation/detection_metrics.py

Mask-IoU detection evaluation for COCO (COCO's own `segm` task, never `bbox`)
— the metric half of the Grounding pipeline's COCO story (the pixel half is
src/grounding_pipeline.py's instance grounding, and the wiring is
scripts/run_vlm_pipeline.py's `--stage score`).

WHAT THIS EVALUATES
    The VLM pipeline extracts entity PHRASES ("a cow", "grass"); SAM3's
    instance head turns each nature-labeled phrase into instance MASKS. COCO
    ships per-instance segmentation with a class label for every annotation.
    This module answers the two questions that pairing makes possible, and
    keeps them SEPARATE:

      1. LOCALIZATION — did the pipeline's mask land where COCO says an
         object is?  TP / FP / FN -> precision, recall, F1, plus AP.
      2. LABEL CORRECTNESS — for the instances that DID match a GT object, was
         the entity named correctly?  Scored two ways: exact match (the strict
         view) and hierarchically (hP/hR/hF1 + Wu-Palmer, the graded view that
         gives "bull" partial credit against a GT of "cow" instead of flat
         zero).

    Reporting them separately is the point. A single fused number would let a
    model that localizes badly but names well look identical to one that does
    the reverse, and those are different failures with different fixes.

MATCHING IS ON MASKS, NEVER BOXES. What SAM3 actually produces IS a mask, and
    COCO ships per-instance `segmentation` for every annotation, so reducing
    both sides to boxes first discards real signal: a box around a curled-up
    dog is mostly the sofa behind it, and two masks that overlap poorly can
    share a nearly identical box (`mask_iou_matrix`, via pycocotools' own
    `iou`, is what `score_image_detection` actually calls; `box_iou`/
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
    precision, recall, F1 and AP, all of which are well-defined without it.
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

# COCO's own AP sweep: 0.50, 0.55, ..., 0.95.
COCO_AP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

# Fraction of a PREDICTED box that must fall inside a crowd region before the
# prediction is treated as explained by that crowd and exempted from the false
# positive count. Intersection-over-prediction-AREA, not IoU — a crowd box is
# typically far larger than any single prediction, so IoU would stay tiny even
# when the prediction sits entirely inside it.
DEFAULT_CROWD_IOA_THRESHOLD = 0.5


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
    AP sweep re-matches the same boxes at ten IoU thresholds, and recomputing
    the matrix ten times per image would be pure waste — the geometry doesn't
    change, only the cut-off applied to it.

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
# Average precision
# =============================================================================
def average_precision(
    scored_predictions: Sequence[Tuple[float, bool]],
    n_gt: int,
) -> float:
    """Class-agnostic AP from `(score, is_true_positive)` pairs pooled over the
    whole dataset, using COCO's 101-point interpolated precision.

    Excluded predictions (the curated-vocabulary rule above) must be left OUT
    of `scored_predictions` by the caller — including them as negatives would
    reimpose exactly the penalty that rule exists to remove.

    Returns 0.0 when there is no GT to recall.
    """
    if n_gt <= 0:
        return 0.0
    if not scored_predictions:
        return 0.0

    order = sorted(range(len(scored_predictions)),
                   key=lambda k: scored_predictions[k][0], reverse=True)
    tp = np.array([1.0 if scored_predictions[k][1] else 0.0 for k in order])
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)

    recalls = tp_cum / float(n_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)

    # Make precision monotonically non-increasing as recall grows (the standard
    # envelope), then sample it at 101 evenly spaced recall levels.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    levels = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recalls, levels, side="left")
    sampled = np.where(idx < len(precisions), precisions[np.minimum(idx, len(precisions) - 1)], 0.0)
    sampled[idx >= len(precisions)] = 0.0
    return float(sampled.mean())


# =============================================================================
# Aggregation
# =============================================================================
def detection_summary(counts: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the running detection counters into the reported numbers.

    `counts` carries: tp, fp, fn, excluded_pred, crowd_suppressed, n_gt_instances,
    n_pred_instances, plus `ap_records` ({iou_threshold: [(score, is_tp), ...]}).

    Precision's denominator is TP + FP, which by construction EXCLUDES the
    curated-vocabulary exemptions — `excluded_predictions` is reported next to
    it so the exemption's size is always visible. There is no accuracy key:
    see the module docstring on why TN is undefined here.
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

    ap_records = counts.get("ap_records") or {}
    n_gt = counts.get("n_gt_instances", 0)
    per_threshold = {t: average_precision(recs, n_gt) for t, recs in sorted(ap_records.items())}
    if per_threshold:
        summary["ap_50"] = per_threshold.get(0.5, 0.0)
        summary["ap_50_95"] = float(np.mean(list(per_threshold.values())))
        summary["ap_per_iou"] = per_threshold
    return summary


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
