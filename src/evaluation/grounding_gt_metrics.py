"""
src/evaluation/grounding_gt_metrics.py

Mask-IoU evaluation of the Grounding pipeline against the HAND-DRAWN BIG-5
grounding annotations — the BIG-5 counterpart of `detection_metrics.py`'s COCO
story, deliberately built on the same primitives so the two results tables read
the same way and can be compared side by side.

WHAT IS SHARED WITH COCO (imported, never reimplemented):
    `mask_iou_matrix`, `merge_rles`, `rle_area`, `match_boxes` (one-to-one
    assignment), `area_bucket`/`COCO_AREA_RANGES` (size stratification),
    `COCO_AP_IOU_THRESHOLDS` (the IoU ladder), and every aggregation function —
    `sweep_summary`, `size_summary`, `label_summary`, `axis_agreement_summary`.
    Label scoring reuses `taxonomy_metrics.resolve_phrase_to_wordnet` with NO
    anchor synset, exactly as COCO does, for the same anti-leakage reason.

WHAT IS NEW HERE, and why COCO does not need it:

  0. MULTI-NAMED GT REGIONS. One annotated mask may carry SEVERAL names, and
     the evaluation treats every one of them as equally correct: exact match
     succeeds if the predicted phrase names ANY of them, and the hierarchical
     score (hP/hR/hF1 + Wu-Palmer) is taken from the BEST-SCORING name. This
     replaced an earlier convention that DUPLICATED a region under a second
     name (a cloud mask copied as `sky`), which harmed recall by inventing a
     GT region no concept-level segmenter can produce twice — see
     convert_grounding_annotations' NAMES section for the full argument.
     A region is still bucketed, counted and reported under its PRIMARY
     (first) name; only the naming credit consults all of them.

  1. VOID REGIONS FOR OVERLAPPING GT (`build_void_rles`, `void_iou_matrix`).
     The annotator drew `cloud` ON TOP OF `sky`, `bird` inside `sky`, `sun`
     inside `sky`. Measured on the shipped annotations: 88 of 414 GT pairs
     overlap (21.3%), pairwise GT IoU reaches 0.747, and 14 objects sit >90%
     inside a larger one. COCO's own protocol has no answer for this because
     COCO instances of one class rarely nest that way. NOTE this is a DIFFERENT
     case from multi-naming above and both are needed: multi-naming is for one
     region that is two things at once, voiding is for two regions of genuinely
     different extent where one nests inside the other.

     The problem is a convention we cannot verify in advance: the annotator's
     `sky` INCLUDES the pixels behind the clouds (it is amodal — the sky really
     is there). Whether SAM3's "sky" mask does the same is unknown. Scoring
     those contested pixels either way would presume the answer:

       * count them as sky   -> a model that excludes clouds is penalized;
       * carve them out      -> a model that includes them is penalized, and on
                                the worst image GT sky drops 705,697 -> 307,435
                                px, capping a perfect sky segmentation at IoU
                                0.44 and FAILING it at threshold 0.50.

     So they are marked VOID instead — excluded from BOTH the numerator and the
     denominator:

         IoU(P, G) = |(P n G) \\ V| / |(P u G) \\ V|

     A predicted sky that covers its clouds is neither rewarded nor penalized
     for those pixels; one that omits them, likewise. Cloud itself keeps every
     pixel and is scored normally. This is not a bespoke invention — void /
     "ignore" regions are exactly how Kirillov et al.'s Panoptic Quality,
     Cityscapes, and COCO's own `iscrowd` handle unscoreable pixels.

     DIRECTION IS CRITICAL. Void for GT X is the union of the SMALLER GT
     regions nested in it. Voiding the contained object instead would erase 8
     GT objects outright (sun, star, gull, Moon, clouds, birds sit ENTIRELY
     inside sky and would be reduced to zero pixels).

     A side benefit that matters for PQ: after voiding, `sky \\ cloud` and
     `cloud` are disjoint, which restores the one-prediction-matches-at-most-
     one-GT property that panoptic-style matching assumes and that a max GT IoU
     of 0.747 otherwise breaks.

  2. NO CURATED-VOCABULARY EXEMPTION (this module has no
     `classify_unmatched_prediction` counterpart, on purpose).
     COCO annotates 80 curated classes, so a correctly-segmented tree is not a
     hallucination and must be exempted — measured at 76% of all predictions on
     the gemma run, which is why COCO's precision must always be quoted next to
     `excluded_predictions`. THESE annotations are EXHAUSTIVE: every nature
     entity in the image was drawn. An unmatched prediction therefore has no
     excuse, and every one is charged as a false positive. Consequence worth
     stating when the two tables appear together: precision here is a STRICTER
     and more meaningful number than COCO's, and the two are not comparable
     as-is.

  3. MANY-TO-ONE ABSORPTION (`absorb_predictions`).
     The GT says `greenery`; the model may emit `tree` + `bush` + `foliage`
     whose UNION covers it. Strict one-to-one charges two of those as false
     positives even when the union is perfect — the mask-side twin of the
     granularity mismatch hP/hR exists to fix on the label side. Reported as a
     SEPARATE table, never merged into the one-to-one numbers, because it is
     strictly more permissive: publishing only the merged view would make a
     model that segments cleanly indistinguishable from one that shotguns
     overlapping masks. The absorbed-group-size distribution is itself the
     measurement of how often the model decomposed one annotated concept.

  4. PANOPTIC-STYLE QUALITY (`pq_summary`).
     PQ = sum_TP IoU / (TP + 0.5 FP + 0.5 FN) = SQ x RQ, which separates HOW
     WELL matched masks trace their objects (SQ) from HOW MANY objects were
     found and named at all (RQ). Reported as PQ-STYLE because standard PQ
     assumes a partitioned GT and this GT is not one even after voiding
     (voiding makes the SCORED pixels disjoint, it does not turn the annotation
     into a partition of the image).

NO AVERAGE PRECISION, for the identical reason as COCO: `semantic_seg` is a
dense logit map with no per-region confidence to rank by. The IoU SWEEP is the
headline here too, and needs no confidence.

Pure math + WordNet + pycocotools. No torch, no model loading, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation import detection_metrics
from src.evaluation.detection_metrics import (  # re-exported for the driver
    AREA_BUCKETS, COCO_AP_IOU_THRESHOLDS, area_bucket, mask_iou_matrix,
    match_boxes, merge_rles, rle_area,
)

# Fraction of an absorbed prediction's OWN area that must fall inside the GT
# region before it may join a many-to-one group. Without this, a GT `greenery`
# could swallow a huge predicted `sky` for a sliver of extra intersection.
DEFAULT_MIN_INSIDE = 0.5


# =============================================================================
# RLE helpers
# =============================================================================
def _native(rle: Dict[str, Any]) -> Dict[str, Any]:
    """pycocotools wants `counts` as BYTES; JSON storage keeps it as str.
    Mirrors the same conversion in detection_metrics/grounding_pipeline."""
    counts = rle["counts"]
    return {"size": list(rle["size"]),
            "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def _area(rle: Optional[Dict[str, Any]]) -> int:
    if rle is None:
        return 0
    from pycocotools import mask as mask_utils
    return int(mask_utils.area(_native(rle)))


def _merge(rles: Sequence[Dict[str, Any]], intersect: bool) -> Optional[Dict[str, Any]]:
    """Union (intersect=False) or intersection (intersect=True) of RLE masks,
    computed by pycocotools directly on the run-length encoding — no dense HxW
    array is ever materialized, which matters because these annotations include
    images up to 1080x8050."""
    if not rles:
        return None
    from pycocotools import mask as mask_utils
    merged = mask_utils.merge([_native(r) for r in rles], intersect=intersect)
    counts = merged["counts"]
    return {"size": [int(merged["size"][0]), int(merged["size"][1])],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


# =============================================================================
# Void regions
# =============================================================================
def build_void_rles(gt_rles: Sequence[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
    """Per-GT void region: the union of every STRICTLY SMALLER GT region that
    overlaps it, intersected with the GT itself.

    Strictly smaller (`<`, not `<=`) so two equal-area regions never void each
    other — that would punch holes in both and leave no container, which is not
    the nesting relationship being modelled.

    Intersecting with the GT itself keeps the invariant the IoU algebra in
    `void_iou` depends on: V is always a SUBSET of G.

    Returns None in a slot when that GT has nothing nested inside it (the
    common case — 326 of 377 objects), which lets the IoU path skip the extra
    RLE work entirely.
    """
    areas = [_area(r) for r in gt_rles]
    out: List[Optional[Dict[str, Any]]] = []
    for i, rle in enumerate(gt_rles):
        nested = [gt_rles[j] for j in range(len(gt_rles))
                  if j != i and areas[j] < areas[i]]
        if not nested:
            out.append(None)
            continue
        void = _merge([_merge(nested, intersect=False), rle], intersect=True)
        out.append(void if _area(void) > 0 else None)
    return out


def void_iou(pred_rle: Dict[str, Any], gt_rle: Dict[str, Any],
             void_rle: Optional[Dict[str, Any]] = None) -> float:
    """IoU with void pixels removed from BOTH numerator and denominator.

    Computed entirely with pycocotools set algebra rather than by decoding to
    dense masks. Because `build_void_rles` guarantees V is a subset of G:

        |(P n G) \\ V| = |P n G| - |P n V|      (V c G, so P n G n V = P n V)
        |(P u G) \\ V| = |P u G| - |V|          (V c G c P u G)

    so four cheap RLE areas give the exact void-aware IoU with no HxW
    allocation anywhere.
    """
    inter = _area(_merge([pred_rle, gt_rle], intersect=True))
    union = _area(_merge([pred_rle, gt_rle], intersect=False))
    if void_rle is not None:
        inter -= _area(_merge([pred_rle, void_rle], intersect=True))
        union -= _area(void_rle)
    return float(inter / union) if union > 0 else 0.0


def tie_break_identical_gt(ious, gt_rles, gt_label_sets, pred_labels,
                           epsilon=1e-9):
    """Nudge the IoU matrix so that an EXACT geometric tie between two GT
    regions is resolved in favour of the label-consistent pairing.

    WHY THIS IS NEEDED. Two GT regions with BYTE-IDENTICAL masks have identical
    rows in the IoU matrix, so every assignment permuting them scores exactly
    the same total and the solver's choice among them is arbitrary. A model that
    correctly said "sky" and "cloud" can be cross-paired — its cloud mask
    matched against GT sky — and scored as having named BOTH wrong, for reasons
    that have nothing to do with the model.

    MOSTLY INERT NOW, and deliberately kept anyway. The 13 identical-mask pairs
    that motivated this came from the removed `derive_sky_from_cloud`
    duplication; a single region carrying both names produces no tie at all.
    But nothing prevents an annotator drawing two genuinely coincident regions
    (a `tree` and a `foliage` traced identically), and this is the correct
    handling if they ever do — so it stays, costing one dict pass when no
    duplicates exist.

    `gt_label_sets[i]` is the collection of surface forms GT region i answers
    to — a mask may carry several names, and agreement with ANY of them is what
    makes a pairing label-consistent.

    WHY THIS IS NOT GT LEAKAGE. The tie is EXACT: the geometry carries literally
    zero information distinguishing the two options, and the nudge is
    `epsilon`-sized, so it can never promote a pairing that geometry disfavours
    by any real margin. It cannot change TP/FP/FN counts — the same number of
    regions match either way — only WHICH of two indistinguishable pairings is
    reported, and therefore only the label metrics. When two GT regions occupy
    exactly the same pixels, no evaluation can tell which is which from pixels
    alone, and the model cannot either; scoring the arbitrary permutation as an
    error would measure the annotation convention, not the prediction.

    Applied only to GT regions whose masks are exactly equal. Near-identical
    (but distinct) regions are left completely alone, because there the
    geometry DOES carry information and must be allowed to decide.
    """
    if ious.size == 0 or not gt_label_sets or not pred_labels:
        return ious
    counts = {}
    for i, rle in enumerate(gt_rles):
        counts.setdefault(rle["counts"], []).append(i)
    groups = [idxs for idxs in counts.values() if len(idxs) > 1]
    if not groups:
        return ious
    out = ious.copy()
    for idxs in groups:
        for gi in idxs:
            for pj, plabel in enumerate(pred_labels):
                if any(_labels_agree(g, plabel) for g in (gt_label_sets[gi] or ())):
                    out[gi, pj] += epsilon
    return out


def _labels_agree(gt_label, pred_phrase):
    """Cheap surface-form agreement used ONLY for the exact-tie nudge above.

    Deliberately simple (normalized equality or a trailing-word match) rather
    than the full `phrase_matches_terms` machinery: it only has to separate
    "sky" from "cloud" among regions that are already geometrically
    indistinguishable, and a false negative here just leaves the tie arbitrary,
    which is the pre-existing behaviour.
    """
    g = (gt_label or "").strip().lower()
    p = (pred_phrase or "").strip().lower()
    return bool(g) and (g == p or p.endswith(" " + g))


def void_iou_matrix(gt_rles: Sequence[Dict[str, Any]],
                    pred_rles: Sequence[Dict[str, Any]],
                    void_rles: Optional[Sequence[Optional[Dict[str, Any]]]] = None
                    ) -> np.ndarray:
    """(n_gt, n_pred) void-aware IoU matrix.

    With `void_rles=None` this is plain mask IoU and delegates to the SAME
    pycocotools path COCO uses (`detection_metrics.mask_iou_matrix`), so the
    no-void comparison run is guaranteed to be computing the identical quantity
    COCO does — not a lookalike.
    """
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=float)
    if void_rles is None:
        return mask_iou_matrix(gt_rles, pred_rles)
    out = np.zeros((len(gt_rles), len(pred_rles)), dtype=float)
    for gi, gt in enumerate(gt_rles):
        void = void_rles[gi]
        if void is None:
            # No nested regions: plain IoU, straight from the C implementation.
            out[gi, :] = mask_iou_matrix([gt], pred_rles)[0, :]
            continue
        for pi, pred in enumerate(pred_rles):
            out[gi, pi] = void_iou(pred, gt, void)
    return out


# =============================================================================
# Many-to-one absorption
# =============================================================================
def absorb_predictions(
    gt_rle: Dict[str, Any],
    void_rle: Optional[Dict[str, Any]],
    base_pred_idx: Optional[int],
    candidate_idxs: Sequence[int],
    pred_rles: Sequence[Dict[str, Any]],
    ious: np.ndarray,
    gt_index: int,
    iou_threshold: float,
    min_inside: float = DEFAULT_MIN_INSIDE,
) -> Tuple[List[int], float]:
    """Grow ONE GT region's prediction by absorbing unmatched predicted masks.

    Greedy submodular coverage: repeatedly add whichever remaining candidate
    most increases the union's void-aware IoU against this GT, stopping when
    nothing improves it. Returns `(absorbed_indices, final_iou)`.

    Greedy is the right algorithm here (unlike for one-to-one assignment, where
    it is not): the objective is monotone submodular set coverage, there is no
    competition between GTs for a candidate once the guards below apply, and
    the alternative — searching every subset — is exponential for no gain.

    THREE GUARDS, each closing a specific way this would otherwise inflate the
    score:

      1. Only UNMATCHED predictions are offered (the caller's
         `candidate_idxs`), so a mask already credited to its own GT can never
         be counted a second time here.
      2. A candidate that already clears `iou_threshold` against some OTHER GT
         is refused — it belongs to that one. Without this, GT `sky` would
         absorb a predicted `cloud`, which sits >90% inside it; the group would
         then be scored as if the model had never distinguished them.
      3. A candidate must lie at least `min_inside` INSIDE this GT (measured on
         the candidate's OWN area, the same intersection-over-prediction-area
         asymmetry `detection_metrics.mask_ioa` uses for crowd suppression),
         and must STRICTLY increase IoU. Together these stop a group growing by
         swallowing a large neighbour for a sliver of overlap, or by re-adding
         pixels it already covers.
    """
    current = pred_rles[base_pred_idx] if base_pred_idx is not None else None
    best = void_iou(current, gt_rle, void_rle) if current is not None else 0.0
    absorbed: List[int] = []
    pool = list(candidate_idxs)

    while pool:
        gain_idx, gain_iou, gain_union = None, best, None
        for pi in pool:
            # Guard 2: belongs to a different GT.
            column = ious[:, pi]
            if any(column[gj] >= iou_threshold
                   for gj in range(len(column)) if gj != gt_index):
                continue
            # Guard 3a: must sit mostly inside this GT.
            pred = pred_rles[pi]
            pred_area = _area(pred)
            if pred_area <= 0:
                continue
            inside = _area(_merge([pred, gt_rle], intersect=True))
            if inside < min_inside * pred_area:
                continue
            union = _merge([current, pred], intersect=False) if current is not None else pred
            cand = void_iou(union, gt_rle, void_rle)
            # Guard 3b: must strictly improve.
            if cand > gain_iou:
                gain_idx, gain_iou, gain_union = pi, cand, union
        if gain_idx is None:
            break
        current, best = gain_union, gain_iou
        absorbed.append(gain_idx)
        pool.remove(gain_idx)
    return absorbed, best


def pick_primary_contributor(gt_rle: Dict[str, Any],
                             void_rle: Optional[Dict[str, Any]],
                             member_rles: Sequence[Dict[str, Any]]) -> int:
    """Which member of a many-to-one group should stand in for the group's
    NAME? Returns the index (into `member_rles`) of the mask that accounts for
    the most GT pixels — i.e. the member with the largest void-aware
    intersection with the GT region.

    A many-to-one group can mix a well-named member with junk ("tree" +
    "vague blob"), and scoring the group's naming by its BEST member (as this
    module used to) or its MEAN silently rewards or dilutes on members that
    contributed almost nothing to the actual match. Picking by GEOMETRIC
    CONTRIBUTION instead asks a different, more defensible question: of
    everything the model said about this region, what did it call the part it
    actually found? That is the one name a reader would reasonably attribute
    to "the model's answer for this GT object".

    Ties (equal contribution) resolve to the first such member, deterministically.
    """
    best_idx, best_area = 0, -1
    for i, rle in enumerate(member_rles):
        area = _area(_merge([rle, gt_rle], intersect=True))
        if void_rle is not None:
            # void_rle is a SUBSET of gt_rle (see build_void_rles), so
            # (rle & void) == (rle & gt & void) — no need to re-intersect
            # with gt_rle first, matching the identity void_iou already relies on.
            area -= _area(_merge([rle, void_rle], intersect=True))
        if area > best_area:
            best_idx, best_area = i, area
    return best_idx


# =============================================================================
# False-positive taxonomy
# =============================================================================
# A prediction that overlaps a MATCHED prediction by at least this much is
# treated as a redundant second mask for the same region rather than an
# independent claim about the image.
DEFAULT_DUPLICATE_IOU = 0.5

# Below this IoU against every GT region, a prediction is not "nearly right" —
# it is over content the annotator did not mark as nature at all.
NEAR_MISS_MIN_IOU = 0.10


def classify_false_positive(pred_idx, ious, matched_pred_idxs, pred_pred_iou,
                            iou_threshold, duplicate_iou=DEFAULT_DUPLICATE_IOU):
    """Why is this unmatched prediction a false positive? Three causes, which
    call for completely different fixes:

      "duplicate"     — it overlaps an ALREADY-MATCHED prediction by
                        >= `duplicate_iou`. The model emitted two names for one
                        region ("tree" and "green tree"; "sky" and "blue sky")
                        and SAM3, which applies no NMS, dutifully grounded both.
                        One wins the one-to-one assignment and the other is
                        charged. This is an EXTRACTION-side redundancy problem,
                        not a perception failure — the pixels were right.
      "near_miss"     — its best IoU against some GT region is between
                        `NEAR_MISS_MIN_IOU` and `iou_threshold`. The model found
                        the right thing but the mask is too loose or too tight
                        to clear the bar. A MASK-QUALITY problem.
      "hallucination" — it overlaps no GT region meaningfully at all. Under
                        EXHAUSTIVE annotation this is the real thing: nature
                        claimed where the annotator saw none.

    Without this split, "precision 0.44" is uninterpretable: a model whose FPs
    are all duplicates is doing something quite different from one whose FPs are
    all hallucinations, and only the latter is a perception failure.
    `pred_pred_iou` is the prediction-vs-prediction IoU matrix (computed once
    per image by the caller and reused across the sweep).
    """
    if matched_pred_idxs and pred_pred_iou is not None:
        for mi in matched_pred_idxs:
            if mi != pred_idx and pred_pred_iou[pred_idx, mi] >= duplicate_iou:
                return "duplicate"
    best_gt_iou = float(ious[:, pred_idx].max()) if ious.shape[0] else 0.0
    if best_gt_iou >= NEAR_MISS_MIN_IOU:
        return "near_miss"
    return "hallucination"


def pred_pred_iou_matrix(pred_rles):
    """Symmetric (n_pred, n_pred) IoU matrix among predictions themselves.

    Needed only by the false-positive taxonomy, to spot the case where SAM3
    grounded two differently-named extractions onto the same pixels. Computed
    once per image via the same pycocotools path everything else uses.
    """
    if not pred_rles:
        return np.zeros((0, 0), dtype=float)
    return mask_iou_matrix(pred_rles, pred_rles)


# =============================================================================
# Per-label breakdown
# =============================================================================
def label_breakdown(rows, min_support=1):
    """Per-GT-label recall and mask quality, pooled over the dataset.

    The single pooled recall says how much was found; this says WHAT was found.
    It separates two failure modes a pooled number cannot: a model that misses
    amorphous "stuff" (sky, greenery, water — hard boundaries, easy concepts)
    from one that misses discrete objects (bird, flower — easy boundaries, hard
    concepts). Those point at different parts of the pipeline.

    Each row is `{"gt_label", "found": bool, "iou": float|None,
    "exact_match": bool|None}`. Returns per label: support, recall, mean IoU
    over the found ones, and exact-naming rate over the found ones.

    ONE ROW PER GT NAME, not per GT region: a mask named both `cloud` and `sky`
    contributes a row to each, because it genuinely is both concepts and a
    reader asking "is sky found?" wants that mask counted. Consequence to state
    when quoting this table: the supports SUM TO MORE than the GT region count
    whenever multi-named regions exist, so they are not a partition of the GT
    the way the size buckets are. `exact_match` on such a row is that NAME's
    own surface-form verdict, so `sky` recall 1.00 with an exact-name rate of
    0.20 reads correctly as "the region was always found, but the model
    usually called it something else (here: cloud)".
    """
    by_label = {}
    for r in rows:
        slot = by_label.setdefault(r["gt_label"], {"support": 0, "found": 0,
                                                   "ious": [], "exact": []})
        slot["support"] += 1
        if r["found"]:
            slot["found"] += 1
            slot["ious"].append(r["iou"])
            if r["exact_match"] is not None:
                slot["exact"].append(bool(r["exact_match"]))
    out = {}
    for label, s in by_label.items():
        if s["support"] < min_support:
            continue
        out[label] = {
            "support": s["support"],
            "recall": s["found"] / s["support"],
            "mean_iou_when_found": float(np.mean(s["ious"])) if s["ious"] else 0.0,
            "exact_name_rate_when_found": (float(np.mean(s["exact"]))
                                           if s["exact"] else 0.0),
        }
    # Most-annotated labels first: that is the order in which a reader should
    # care about them, and it keeps the low-support tail from leading.
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["support"]))


# =============================================================================
# Panoptic-style quality
# =============================================================================
def pq_summary(matched_ious: Sequence[float],
               weights: Optional[Sequence[float]],
               n_fp: int, n_fn: int) -> Dict[str, Any]:
    """Panoptic-style quality in its sum-of-IoU form:

        PQ = sum_TP (IoU x w) / (|TP| + 0.5|FP| + 0.5|FN|)   =   SQ x RQ

    `weights` generalizes PQ's usual "the label is correct" requirement, so the
    SAME denominator serves three questions and they stay directly comparable:

        weights=None            -> LOCALIZATION only (class-agnostic), the
                                   counterpart of COCO precision/recall here.
        weights=0/1 exact match -> classic PQ (found it AND named it).
        weights=hF1 per pair    -> HIERARCHICAL PQ, which gives a predicted
                                   "bull" against GT "cow" its ~0.94 of credit
                                   instead of the flat zero classic PQ scores
                                   it. This is the number that carries the
                                   thesis's granularity argument into the
                                   pixel-level evaluation.

    SQ is the mean IoU over matched pairs; RQ is recovered as PQ/SQ so the
    identity PQ = SQ x RQ holds exactly for every weighting.

    Reported as PQ-STYLE, never plain "PQ": standard PQ is defined over a
    partitioned GT, and this annotation is not a partition (see the module
    docstring on void regions).
    """
    tp = len(matched_ious)
    denom = tp + 0.5 * n_fp + 0.5 * n_fn
    if denom <= 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": n_fp, "fn": n_fn}
    if weights is None:
        weights = [1.0] * tp
    numer = float(sum(i * w for i, w in zip(matched_ious, weights)))
    sq = float(np.mean(matched_ious)) if tp else 0.0
    return {"pq": numer / denom,
            "sq": sq,
            "rq": (numer / sq / denom) if sq > 0 else 0.0,
            "tp": tp, "fp": n_fp, "fn": n_fn}


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Mean + POPULATION std + support — the same convention the hierarchical
    reporting already uses project-wide, because a mean alone cannot tell a
    tight cluster from a bimodal split."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "support": 0}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "support": int(arr.size)}


# =============================================================================
# No-nature bucket
# =============================================================================
def no_nature_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the images annotated as containing NO nature at all.

    Their GT is the EMPTY mask set, treated as ABSOLUTE per the annotation
    protocol: every one of these images was majority-vote coded as having no
    nature, so any grounded nature mask is a false positive with no area
    threshold and no benefit of the doubt.

    IoU, PQ and recall are all UNDEFINED here (there is no GT region to overlap
    and nothing to recall), which is exactly why this bucket is reported
    separately and never pooled with the nature images. What it uniquely
    provides is the only unambiguous FALSE-POSITIVE evidence in the whole
    evaluation: on a nature image, an over-eager pipeline that paints
    everything still scores well on recall, but here any mask at all is wrong.
    Without this bucket, precision would be conditioned on "nature is present",
    which is not the deployment condition at 2M images.
    """
    if not rows:
        return {"n_images": 0}
    flagged = [bool(r["n_pred_masks"]) for r in rows]
    return {
        "n_images": len(rows),
        "image_false_positive_rate": float(np.mean(flagged)),
        "n_images_with_false_positive": int(sum(flagged)),
        "n_false_positive_masks": int(sum(r["n_pred_masks"] for r in rows)),
        "mean_false_positive_masks_per_image": float(
            np.mean([r["n_pred_masks"] for r in rows])),
        # NOTE this is the same quantity as the grounding pipeline's own
        # `coverage_ratio` (nature px / total px over the merged mask union) —
        # recomputed here from the stored masks rather than trusted, so the two
        # agreeing is a CHECK. They are reported once, as this key, plus
        # `center_weighted` below which is genuinely different.
        "mean_nature_pixel_fraction": float(
            np.mean([r["nature_px_fraction"] for r in rows])),
        # The pooled mean above is diluted by every correctly-empty image, so
        # it understates how much of an OFFENDING image gets painted as nature.
        # Both are reported: the pooled one is the dataset-level rate, this one
        # says how severe a hallucination is when it happens.
        "mean_nature_pixel_fraction_when_flagged": (
            float(np.mean([r["nature_px_fraction"] for r in rows
                           if r["n_pred_masks"]]))
            if any(r["n_pred_masks"] for r in rows) else 0.0),
        # center_weighted is the one that adds information here: it weights
        # central pixels more, so a hallucination in the middle of the frame
        # scores higher than the same area in a corner.
        "mean_relevance_center_weighted": (
            float(np.mean([r["center_weighted"] for r in rows
                           if r.get("center_weighted") is not None]))
            if any(r.get("center_weighted") is not None for r in rows) else None),
        # Kept only as a consistency check against mean_nature_pixel_fraction
        # above; the two are the same quantity by construction.
        "mean_relevance_coverage_ratio": (
            float(np.mean([r["coverage_ratio"] for r in rows
                           if r.get("coverage_ratio") is not None]))
            if any(r.get("coverage_ratio") is not None for r in rows) else None),
    }
