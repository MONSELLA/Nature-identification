#!/usr/bin/env python3
"""
scripts/score_grounding_gt.py

Score the Grounding pipeline against the HAND-DRAWN BIG-5 grounding
annotations — the BIG-5 counterpart of `run_vlm_pipeline.py --stage score`'s
COCO detection block, built on the same primitives (see
src/evaluation/grounding_gt_metrics.py's docstring for exactly what is shared
and what is new).

INPUTS
  --artifact   the VLM artifact enriched in place by the grounding pipeline
               (`vlm_responses_<model>.jsonl`), read for `object_groundings`
               (SAM3's SEMANTIC masks) and `object_finals` (hybrid labels).
  --gt_dir     `grounding_annotations/processed`, written by
               scripts/convert_grounding_annotations.py.

WHAT IS REPORTED, and why each block is kept separate

  summary["grounding"]            localization at the headline IoU threshold
  summary["grounding_iou_sweep"]  THE HEADLINE — P/R/F1 across COCO's ladder
  summary["grounding_by_size"]    small/medium/large split
  summary["grounding_pq"]         PQ-style: localization / exact / hierarchical
  summary["grounding_labels"]     naming: exact match + hP/hR/hF1 + Wu-Palmer
  summary["grounding_axes"]       biotic/material agreement on matched pairs
  summary["grounding_many_to_one"] the permissive decomposition-tolerant table
  summary["grounding_no_nature"]  false positives where GT is empty

  They are never merged into one number for the same reason the COCO block
  keeps localization and naming apart: a model that localizes badly but names
  well would otherwise look identical to one that does the reverse, and those
  are different failures with different fixes.

FOUR DIFFERENCES FROM THE COCO BLOCK, all forced by the data rather than
chosen for convenience:

  0. MULTI-NAMED GT REGIONS. One annotated mask may carry several names, all
     equally correct for the same pixels (a cloud's pixels are sky pixels too).
     Exact match succeeds if the predicted phrase names ANY of them; the
     hierarchical credit is taken from the BEST-SCORING name. This replaced an
     earlier convention that duplicated the region under a second name, which
     harmed recall by inventing a GT region no concept-level segmenter can
     produce twice — see convert_grounding_annotations' NAMES section. A
     SECOND consequence, on the precision side: when the model extracts BOTH
     names as separate masks over the same pixels, one-to-one assignment can
     only match one, so item 2 below (`second_name_credit`) excuses the other
     rather than charging it as a duplicate false positive.

  1. VOID REGIONS. This GT overlaps itself (cloud drawn on sky, max pairwise
     GT IoU 0.747). Contested pixels are excluded from both sides of the IoU
     rather than scored either way — see the metrics module. `--no_void` turns
     this off for comparison.

  2. NO CURATED-VOCABULARY EXEMPTION, BUT ONE NARROWER ONE. These annotations
     are EXHAUSTIVE, so COCO's "names nothing in the 80 classes" exemption
     (76% of predictions on the gemma run) has no analogue here — nearly every
     unmatched prediction is a false positive. The one exception is
     second-name credit (`second_name_credit`): a multi-named GT region
     (`cloud` + `sky`) can draw two separate, individually correct
     extractions, and one-to-one assignment can only match one of them: the
     other is excused, via the same `excluded_pred` bucket COCO's exemption
     uses, rather than scored as a `duplicate` false positive for correctly
     naming the region's other identity. Precision from this script is still a
     stricter number than COCO's and the two must not be compared directly.

  3. NATURE AXIS IS NOT SCORED on matched pairs, exactly as in the COCO block:
     only `final_nature is True` entities are grounded and every GT object here
     is nature, so agreement would be 1.0 by construction. The real nature-axis
     evidence is the no-nature bucket's false-positive rate.

NO GT LEAKAGE: a predicted phrase is resolved by
`taxonomy_metrics.resolve_phrase_to_wordnet(phrase, anchor_synset_id=None)` —
the anchor is withheld because the only one available would be the GT synset,
and steering sense disambiguation with it would inflate the very hP/hR it
feeds. Identical to the COCO block's own choice.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from functools import lru_cache

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.evaluation import detection_metrics, grounding_gt_metrics as ggm  # noqa: E402
from src.evaluation import taxonomy_metrics  # noqa: E402
from src.vlm_pipeline import _normalize_object  # noqa: E402


@lru_cache(maxsize=None)
def _synset_lemma_terms(synset_id):
    """Normalized WordNet lemma surface forms for a synset, cached.

    Byte-for-byte the same computation as run_vlm_pipeline._synset_lemma_terms
    (which lives in that script and cannot be imported here without pulling in
    the VLM stack). It shares this module's `_normalize_object`, imported from
    src.vlm_pipeline, so the normalization really is identical rather than
    merely similar.
    """
    from nltk.corpus import wordnet as wn
    try:
        return frozenset(_normalize_object(l.name().replace("_", " "))
                         for l in wn.synset(synset_id).lemmas())
    except Exception:
        return frozenset()


# =============================================================================
# Surface-form helpers — deliberate mirrors of run_vlm_pipeline's own
# =============================================================================
# Reimplemented here rather than imported because they live in
# scripts/run_vlm_pipeline.py, whose module-level imports pull in the VLM
# stack; this scorer must run on a machine with no GPU libraries. The
# NORMALIZATION itself is imported (`_normalize_object`, `_synset_lemma_terms`)
# so "the same surface form" means bit-for-bit the same thing in both.
def _primary_label(gt_obj):
    """The PRIMARY (first) name of a GT region — what tables group and display
    by, so a multi-named region reads under one stable heading."""
    names = gt_names(gt_obj)
    return gt_name_display(names[0]) if names else ""


def _all_labels(gt_obj):
    """Every name of a GT region, joined — used wherever a human reads a row and
    needs to see that a mask answers to more than one name."""
    texts = [gt_name_display(n) for n in gt_names(gt_obj)]
    return " / ".join(t for t in texts if t)


def gt_names(gt_obj):
    """Every name one GT region carries, as a list, primary first.

    A mask may be named several things at once — a cloud's pixels are also sky
    pixels, so the region is annotated `cloud` + `sky` rather than drawn twice
    (see convert_grounding_annotations' NAMES section for why the duplication
    it replaced harmed recall). A GT record written before multi-naming has
    only the flat fields, which become a one-element list, so both formats
    score identically.
    """
    names = gt_obj.get("names")
    if isinstance(names, list) and names:
        return names
    return [{"entity_label": gt_obj.get("entity_label"),
             "entity_label_freetext": gt_obj.get("entity_label_freetext"),
             "wordnet_synset_id": gt_obj.get("wordnet_synset_id")}]


def gt_name_text(name):
    """The surface form of one name — free text when that is what the annotator
    used, otherwise the WordNet-derived label. Empty when the name carries no
    written label at all; use `gt_name_display` where a row must never vanish.
    """
    if name.get("unmapped_from_wordnet"):
        return name.get("entity_label_freetext") or name.get("entity_label") or ""
    return name.get("entity_label") or name.get("entity_label_freetext") or ""


def gt_name_display(name):
    """`gt_name_text`, falling back to the bare synset id.

    Only for HUMAN-FACING grouping (the per-label table, the CSV): a name with a
    synset but no written label would otherwise contribute no row at all and
    disappear from the recall table. It must NOT feed `name_match_terms` — a
    synset id is not a surface form any model would ever emit, so treating it as
    one would be a free false negative on exact match.
    """
    return gt_name_text(name) or name.get("wordnet_synset_id") or ""


def name_match_terms(name):
    """Normalized surface forms that count as naming ONE name of a GT region:
    its annotator-written label plus every WordNet lemma of its synset."""
    terms = set()
    label = gt_name_text(name)
    if label:
        terms.add(_normalize_object(label))
    syn = name.get("wordnet_synset_id")
    if syn:
        terms |= _synset_lemma_terms(syn)
    return {t for t in terms if t}


def gt_match_terms(gt_obj):
    """Every surface form that counts as naming this GT region — the UNION over
    all of its names, because they are equally correct answers for the same
    pixels."""
    terms = set()
    for name in gt_names(gt_obj):
        terms |= name_match_terms(name)
    return terms


def phrase_matches_terms(phrase, terms):
    """Whole normalized phrase, or any term as a TRAILING SPAN.

    Identical rule to run_vlm_pipeline.phrase_matches_terms, and the reason it
    is a trailing span rather than "appears anywhere": English noun phrases are
    right-headed, so "huge potted plant" IS a potted plant while "cow shed" is
    NOT a cow. This is what makes the exact-match tier handle the modifier case
    ("huge cow" vs GT "cow") without hand-written special cases.
    """
    norm = _normalize_object(phrase)
    if norm in terms:
        return True
    return any(norm.endswith(" " + term) for term in terms)


def score_label_agreement(phrase, gt_obj, graph):
    """Naming credit for one matched pair — a direct mirror of
    run_vlm_pipeline.score_label_agreement, so the BIG-5 and COCO label tables
    mean the same thing.

    EXACT MATCH is the surface-form test above. HIERARCHICAL resolves the
    phrase to WordNet with NO anchor (anti-leakage) and scores ancestral-closure
    overlap, giving graded credit where exact match scores a flat zero.

    MULTI-NAMED GT, the one place this differs from the COCO mirror. A GT region
    may carry several names, all equally correct for the same pixels (`cloud`
    and `sky`). So:

      * EXACT MATCH succeeds if the phrase names ANY of them — the union of
        every name's surface forms, computed by `gt_match_terms`.
      * HIERARCHICAL is scored against EVERY name and the BEST hF1 wins, with
        Wu-Palmer breaking a tie and name order breaking the rest. Taking the
        best is the only reading consistent with the annotation: the annotator
        asserted the region IS all of these things, so a prediction that hits
        one of them exactly is not wrong just because it missed the other.
        Averaging instead would penalize a perfectly correct answer for the
        existence of a second, equally valid label.

    `matched_gt_name` / `matched_gt_synset` record which name supplied the
    credit, so the CSV can show that a "cloud" prediction was scored against
    the region's `cloud` name rather than its `sky` one.

    NO LEAKAGE IS INTRODUCED. The phrase is still resolved ONCE, anchorless —
    the GT names are only ever the targets it is compared against, never an
    input to the disambiguation.
    """
    exact = phrase_matches_terms(phrase, gt_match_terms(gt_obj))
    names = gt_names(gt_obj)
    pred_synset = taxonomy_metrics.resolve_phrase_to_wordnet(phrase, anchor_synset_id=None)
    candidates = [n for n in names if n.get("wordnet_synset_id")]

    if pred_synset is None or not candidates:
        return {"exact_match": exact, "pred_synset": pred_synset, "resolved": False,
                "hp": 0.0, "hr": 0.0, "hf1": 0.0, "wup": 0.0,
                "matched_gt_name": gt_name_display(names[0]) if names else "",
                "matched_gt_synset": None,
                "n_gt_names": len(names)}

    best = None
    for name in candidates:
        gt_synset = name["wordnet_synset_id"]
        perfect = pred_synset == gt_synset
        hier = taxonomy_metrics.compute_hierarchical_metrics(
            graph, gt_synset, pred_synset, perfect_match=perfect)
        wup = taxonomy_metrics.compute_wup_similarity(gt_synset, pred_synset,
                                                      perfect_match=perfect)
        cand = {"exact_match": exact, "pred_synset": pred_synset, "resolved": True,
                "hp": hier["hp"], "hr": hier["hr"], "hf1": hier["hf1"], "wup": wup,
                "matched_gt_name": gt_name_display(name),
                "matched_gt_synset": gt_synset,
                "n_gt_names": len(names)}
        if best is None or (cand["hf1"], cand["wup"]) > (best["hf1"], best["wup"]):
            best = cand
    return best


# Hierarchical bar for second-name credit, ADDITIONAL to exact match (never a
# replacement — a free-text GT name has no synset and can only ever be reached
# via exact match, so that path stays live regardless of this default).
DEFAULT_SECOND_NAME_HF1 = 0.7


def _best_available_name(phrase, available_names, graph, hf1_threshold):
    """Best credit for `phrase` among `available_names` ONLY — the pool a
    caller has ALREADY narrowed to names not yet spent on some earlier
    prediction this image (see `second_name_credit`'s `spent_by_gi`).
    Restricting the pool, not just checking membership, is what stops a name
    being used to excuse more than one redundant prediction.

    TWO INDEPENDENT PATHS, exactly mirroring `score_label_agreement`'s own two
    views of "this phrase names that region" — checked in this order because a
    literal correct answer is stronger evidence than a merely-close one:
      1. EXACT — phrase_matches_terms against ANY available name's surface
         forms. The ONLY path a free-text name (no synset) can ever satisfy.
      2. HIERARCHICAL — resolve `phrase` to WordNet ANCHORLESS (identical to
         score_label_agreement's own anti-leakage resolution — the GT is only
         ever the comparison target, never an input to disambiguation), then
         take the available name with the highest hF1 against it; credited
         only if that hF1 clears `hf1_threshold`.

    Returns `(name_display, via)` — `via` is `"exact"` or `"hierarchical"`, so
    a reader can tell which mechanism actually fired — or `(None, None)`.
    """
    for n in available_names:
        terms = name_match_terms(n)
        if terms and phrase_matches_terms(phrase, terms):
            return gt_name_display(n), "exact"

    pred_synset = taxonomy_metrics.resolve_phrase_to_wordnet(phrase, anchor_synset_id=None)
    if pred_synset is None:
        return None, None
    best_name, best_hf1 = None, -1.0
    for n in available_names:
        gt_synset = n.get("wordnet_synset_id")
        if not gt_synset:
            continue
        perfect = pred_synset == gt_synset
        hier = taxonomy_metrics.compute_hierarchical_metrics(
            graph, gt_synset, pred_synset, perfect_match=perfect)
        if hier["hf1"] > best_hf1:
            best_name, best_hf1 = gt_name_display(n), hier["hf1"]
    if best_name is not None and best_hf1 >= hf1_threshold:
        return best_name, "hierarchical"
    return None, None


def second_name_credit(pi, ious, gt_objs, entities, matched_gt_name_by_gi,
                       threshold, spent_by_gi, graph, hf1_threshold):
    """Is unmatched prediction `pi` a second, EQUALLY VALID name for a GT
    region some OTHER prediction already matched?

    WHY THIS EXISTS. A multi-named GT region (`cloud` + `sky`) can draw TWO
    separate extractions from the model — one per name — both landing on
    essentially the same pixels, because that region genuinely is both things.
    One-to-one assignment can only credit ONE of them toward recall; without
    this check, the other clears the IoU bar, gets rejected by the matcher, and
    falls into `ggm.classify_false_positive`, which sees "overlaps an
    already-matched prediction" and calls it a `duplicate` — the same bucket
    as a model naming "tree" and "green tree" for one object. But this is not
    redundant extraction: the model correctly recognized the region's OTHER
    asserted identity, and penalizing that is scoring the annotation
    convention, not a model error.

    THE MATCH BAR mirrors `score_label_agreement`'s own credit for the WINNING
    prediction, not a stricter one: exact match, OR hF1 >= `hf1_threshold`
    (default `DEFAULT_SECOND_NAME_HF1`) against the SPECIFIC other name — see
    `_best_available_name`. It used to be exact-match-only, which held the
    LOSING prediction of a pair to a tighter bar than the one that won; a
    near-synonym second extraction ("cloudy sky" instead of "sky") was
    penalized as a duplicate even though the matched pair's own hF1 credit
    would have forgiven an equally loose match.

    ONE CREDIT PER (REGION, NAME) — the fix that makes loosening the bar safe.
    `spent_by_gi[gi]` starts as `{claimed}` (the name the ONE-TO-ONE match
    already used) and gains the credited name the FIRST time it fires; every
    later prediction that also lands on that same already-spent name is left
    to `classify_false_positive` as a genuine duplicate. Without this, a model
    that emits "sky" (matched) + "cloud" + "cloud" AGAIN as a third, separate
    mask would have BOTH "cloud" predictions excused — the second one is not
    evidence of a second identity, it is a plain duplicate of the credit the
    first "cloud" already earned. This was live even under strict exact match
    (harder to trigger) and would become a real gaming vector once the bar
    drops to hF1-based matching (many near-synonyms independently clearing it).

    Candidate regions are tried IoU-descending — a deterministic tie-break for
    the rare case a prediction clears the bar against more than one
    already-matched multi-named region.

    Returns the credited GT index, or None. Does NOT touch tp/fn — the region
    was already found by the OTHER prediction; this only removes `pi` from the
    false-positive count, via the same `excluded_pred` bucket COCO's own
    curated-vocabulary exemption uses (see `detection_metrics.detection_summary`:
    an excluded prediction counts toward neither precision's numerator nor its
    denominator).
    """
    phrase = entities[pi]["object"]
    candidates = sorted(matched_gt_name_by_gi.keys(), key=lambda gi: -ious[gi, pi])
    for gi in candidates:
        if ious[gi, pi] < threshold:
            continue
        names = gt_names(gt_objs[gi])
        if len(names) < 2:
            continue
        available = [n for n in names if gt_name_display(n) not in spent_by_gi[gi]]
        if not available:
            continue
        hit, _via = _best_available_name(phrase, available, graph, hf1_threshold)
        if hit is not None:
            spent_by_gi[gi].add(hit)
            return gi
    return None


def partition_second_name_credit(pred_idxs, ious, gt_objs, entities,
                                 matched_gt_name_by_gi, threshold, graph,
                                 hf1_threshold=DEFAULT_SECOND_NAME_HF1):
    """Split one threshold's unmatched predictions into (true_fp, credits).

    `credits` is `[(pred_idx, gt_idx), ...]` — every prediction excused under
    `second_name_credit`. `true_fp` is everything else, unchanged from before
    this existed, so a run with no multi-named GT regions scores identically.

    `spent_by_gi` is built FRESH here, per call — each threshold in the sweep
    gets its own independent "which names are already spoken for" bookkeeping,
    since which pairs are matched (and therefore which name is already
    claimed) changes with the threshold.
    """
    spent_by_gi = {gi: {claimed} for gi, claimed in matched_gt_name_by_gi.items()}
    true_fp, credits = [], []
    for pi in pred_idxs:
        gi = second_name_credit(pi, ious, gt_objs, entities,
                                matched_gt_name_by_gi, threshold, spent_by_gi,
                                graph, hf1_threshold)
        (credits.append((pi, gi)) if gi is not None else true_fp.append(pi))
    return true_fp, credits


def score_axis_agreement(pred_final, gt_obj):
    """biotic/material agreement on a matched pair, mirroring
    run_vlm_pipeline.score_axis_agreement.

    GT stores these as strings ("biotic"/"abiotic", "material"/"immaterial");
    the pipeline's hybrid finals are booleans with biotic/material as the
    positive class. An axis whose value is missing on EITHER side yields None
    (a missing value), never a fabricated disagreement, so the aggregate drops
    that pair from that axis's support.

    NATURE is deliberately absent — see this module's docstring.
    """
    gt_vals = {
        "biotic": (True if gt_obj.get("biotic") == "biotic"
                   else (False if gt_obj.get("biotic") == "abiotic" else None)),
        "material": (True if gt_obj.get("material") == "material"
                     else (False if gt_obj.get("material") == "immaterial" else None)),
    }
    out = {}
    for axis in ("biotic", "material"):
        gt_val = gt_vals[axis]
        pred_val = (pred_final or {}).get(f"final_{axis}")
        out[f"gt_{axis}"] = gt_val
        out[f"pred_{axis}"] = pred_val
        out[f"{axis}_agree"] = (bool(gt_val) == bool(pred_val)
                                if gt_val is not None and pred_val is not None else None)
    return out


# =============================================================================
# Loading
# =============================================================================
def load_gt(processed_dir):
    """`processed/{nature,no_nature}/*.json` -> {image_id: record}."""
    gt = {}
    for split in ("nature", "no_nature"):
        for path in glob.glob(os.path.join(processed_dir, split, "*.json")):
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            gt[rec["image_id"]] = rec
    if not gt:
        raise SystemExit(f"No GT records under {processed_dir}/{{nature,no_nature}} — "
                         "run scripts/convert_grounding_annotations.py first.")
    return gt


def load_predictions(artifact_paths):
    """One or more artifact JSON-Lines files -> {image basename: {...}}.

    SEVERAL artifacts are accepted because the grounding GT spans BOTH BIG-5
    platforms (84 twitter / 86 weibo images), while the pipeline writes one
    artifact per dataset name — `big5_twitter` and `big5_weibo` are separate
    runs. Passing only one would leave the other platform's images absent from
    the predictions and score every one of their GT regions as a false
    negative, silently halving recall. Records are keyed by image BASENAME, so
    merging is safe; a duplicate basename across artifacts is reported rather
    than silently overwritten.

    Only entities SAM3 actually CONFIRMED count as predictions:
    `grounded is None` means never attempted (not a nature entity) and
    `grounded is False` means SAM3 looked and no pixel cleared
    --mask_threshold. Neither is a claim about the image, so neither can be a
    false positive. This is the same filter `score_image_entities` applies on
    COCO.
    """
    if isinstance(artifact_paths, str):
        artifact_paths = [artifact_paths]
    preds, duplicates = {}, []
    # Per-artifact tally of whether the GROUNDING stage ever ran on it. An
    # artifact with image records but no `object_groundings` key anywhere is a
    # VLM-only artifact: scoring it silently yields zero predictions, hence
    # recall 0.0 and a plausible-looking but meaningless results file. That is
    # a setup mistake, not a model result, so it is detected and reported
    # loudly rather than left to be inferred from a suspicious number.
    grounded_seen = {}
    for artifact_path in artifact_paths:
        with open(artifact_path, encoding="utf-8") as fh:
          for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "image_path" not in rec:
                continue  # artifact header, not an image record
            tally = grounded_seen.setdefault(artifact_path, {"records": 0, "grounded": 0})
            tally["records"] += 1
            # KEY PRESENCE, not truthiness: an image with nothing to ground
            # (every no-nature image) correctly has `object_groundings: []`,
            # which is falsy but is NOT evidence that grounding was skipped.
            # `nature_relevance_score_coverage_ratio` is written for every
            # record the grounding stage processes, so it is the reliable mark.
            if ("nature_relevance_score_coverage_ratio" in rec
                    or "object_groundings" in rec):
                tally["grounded"] += 1
            finals = rec.get("object_finals") or []
            entities = []
            for idx, g in enumerate(rec.get("object_groundings") or []):
                if g.get("grounded") is not True or not g.get("mask_rle"):
                    continue
                entities.append({
                    "object": g.get("object") or "",
                    "object_idx": idx,
                    "mask_rle": g["mask_rle"],
                    "pixel_count": (g.get("pixel_count")
                                    or detection_metrics.rle_area(g["mask_rle"])),
                    "final": finals[idx] if idx < len(finals) else {},
                })
            key = os.path.basename(rec["image_path"])
            if key in preds:
                duplicates.append(key)
            preds[key] = {
                "entities": entities,
                "n_extracted": len(rec.get("objects") or []),
                "coverage_ratio": rec.get("nature_relevance_score_coverage_ratio"),
                "center_weighted": rec.get("nature_relevance_score_center_weighted"),
            }
    for path, tally in grounded_seen.items():
        if tally["records"] and not tally["grounded"]:
            print(f"\n  ERROR: {os.path.basename(path)} has {tally['records']} image "
                  f"records but NO object_groundings on any of them — the "
                  f"GROUNDING stage was never run on this artifact.\n"
                  f"         Scoring it would report recall 0.0 for setup "
                  f"reasons, not model reasons. Run the grounding pipeline "
                  f"(scripts/run_grounding_pipeline.py) over it first.",
                  file=sys.stderr)
    if duplicates:
        print(f"  NOTE: {len(duplicates)} image(s) appeared in more than one "
              f"artifact; the last occurrence wins: {duplicates[:5]}",
              file=sys.stderr)
    return preds


# =============================================================================
# Per-image scoring
# =============================================================================
def score_nature_image(gt_rec, pred_rec, graph, args):
    """One annotated image: match, then score localization and naming.

    Mirrors `run_vlm_pipeline.score_image_entities` step for step — same
    one-to-one assignment, same IoU sweep off one cached matrix, same size
    bucketing — with the void-aware IoU and the exhaustive-annotation FP rule
    substituted in.
    """
    gt_objs = gt_rec["objects"]
    gt_rles = [o["segmentation"] for o in gt_objs]
    H, W = gt_rec["height"], gt_rec["width"]

    entities = (pred_rec or {}).get("entities", [])
    pred_rles, dropped = [], []
    for ent in entities:
        size = list(ent["mask_rle"]["size"])
        if size != [H, W]:
            # A predicted mask at a different resolution cannot be compared
            # pixel-wise. Recorded and skipped rather than silently rescaled:
            # resampling a mask invents boundary pixels and would quietly
            # corrupt every IoU on the image.
            dropped.append({"object": ent["object"], "pred_size": size, "gt_size": [H, W]})
            continue
        pred_rles.append(ent)
    entities = pred_rles
    pred_rles = [e["mask_rle"] for e in entities]

    voids = ggm.build_void_rles(gt_rles) if args.void else None
    voided_px = sum(ggm._area(v) for v in voids if v is not None) if voids else 0

    ious = ggm.void_iou_matrix(gt_rles, pred_rles, voids)
    # Resolve EXACT geometric ties between identical GT regions (the derived
    # sky/cloud pairs) toward the label-consistent pairing. Epsilon-sized, so
    # it cannot change tp/fp/fn — only which of two indistinguishable pairings
    # is reported. See ggm.tie_break_identical_gt.
    ious = ggm.tie_break_identical_gt(
        ious, gt_rles, [[gt_name_text(n) for n in gt_names(o)] for o in gt_objs],
        [e["object"] for e in entities])
    # Prediction-vs-prediction overlap, for the false-positive taxonomy only.
    pp_iou = ggm.pred_pred_iou_matrix(pred_rles)

    # `match_boxes` needs only lengths when `ious` is supplied — the geometry
    # is already in the matrix. Passing the GT/pred bboxes would be misleading
    # here (matching is on MASKS), so placeholder lists of the right length are
    # passed instead, exactly as the COCO caller relies on the same behaviour.
    gt_ph = [None] * len(gt_rles)
    pred_ph = [None] * len(pred_rles)
    matches, unmatched_gt, unmatched_pred = detection_metrics.match_boxes(
        gt_ph, pred_ph, iou_threshold=args.iou_threshold, ious=ious)

    pair_rows, label_records, axis_records = [], [], []
    for gi, pi, iou in matches:
        naming = score_label_agreement(entities[pi]["object"], gt_objs[gi], graph)
        axis = score_axis_agreement(entities[pi].get("final"), gt_objs[gi])
        label_records.append(naming)
        axis_records.append(axis)
        pair_rows.append({
            "gt_index": gi, "pred_index": pi,
            # PRIMARY name for display; `gt_all_labels` shows the rest, and
            # `matched_gt_name` (from `naming`) says which one earned the
            # hierarchical credit.
            "gt_label": _primary_label(gt_objs[gi]),
            "gt_all_labels": _all_labels(gt_objs[gi]),
            "gt_synset": gt_objs[gi].get("wordnet_synset_id"),
            "gt_area": gt_objs[gi]["area"],
            "pred_object": entities[pi]["object"],
            "pred_pixels": entities[pi]["pixel_count"],
            "iou": iou,
            **naming, **axis,
        })

    # --- SECOND-NAME CREDIT, before classifying anything as a false positive -
    # A multi-named GT region can draw two extractions ("cloud" and "sky"),
    # both landing on the same pixels and both individually correct — but
    # one-to-one assignment only credits one of them. Without this, the other
    # falls into classify_false_positive and reads as a `duplicate`, which
    # penalizes the model for correctly recognizing the region's OTHER
    # asserted identity rather than for any real extraction redundancy. See
    # second_name_credit's docstring for the full argument.
    matched_gt_name_by_gi = {gi: rec["matched_gt_name"]
                             for (gi, _, _), rec in zip(matches, label_records)}
    true_fp, second_name_credits = partition_second_name_credit(
        unmatched_pred, ious, gt_objs, entities, matched_gt_name_by_gi,
        args.iou_threshold, graph, args.second_name_hf1_threshold)

    # --- WHY each false positive happened -----------------------------------
    # "precision 0.44" is not actionable on its own: a duplicate mask over an
    # already-matched region, a loose mask over the right object, and a mask
    # over unannotated content are three different failures with three
    # different fixes. See ggm.classify_false_positive. Second-name-credited
    # predictions never reach this classifier — they are not false positives
    # at all, so putting them through "duplicate" would relabel the same bug
    # this section exists to fix.
    matched_pred_idxs = [pi for _, pi, _ in matches]
    fp_rows = []
    for pi in true_fp:
        reason = ggm.classify_false_positive(
            pi, ious, matched_pred_idxs, pp_iou, args.iou_threshold,
            args.duplicate_iou)
        fp_rows.append({"pred_object": entities[pi]["object"],
                        "pred_pixels": entities[pi]["pixel_count"],
                        "best_gt_iou": (float(ious[:, pi].max())
                                        if ious.shape[0] else 0.0),
                        "reason": reason})
    second_name_rows = [
        {"pred_object": entities[pi]["object"], "gt_index": gi,
         "gt_label": _all_labels(gt_objs[gi]),
         "claimed_by_match": matched_gt_name_by_gi.get(gi)}
        for pi, gi in second_name_credits]

    # --- per-GT-label recall rows -------------------------------------------
    # ONE ROW PER GT NAME, not per GT region: a mask named both `cloud` and
    # `sky` answers "was sky found?" as well as "was cloud found?", and both
    # readers deserve a true answer. Supports therefore sum to MORE than the GT
    # region count when multi-named regions exist — documented at the point of
    # aggregation (ggm.label_breakdown) since it makes this table not a
    # partition of the GT.
    matched_by_gt = {gi: (pi, iou) for gi, pi, iou in matches}
    label_rows = []
    for gi, o in enumerate(gt_objs):
        hit = matched_by_gt.get(gi)
        # Looked up by GT INDEX, not by (label, phrase) strings: the converter
        # keeps two same-labelled objects separate when they differ on an axis
        # (a real `flower` and a depicted one both appear in
        # nTbNhYDVODIQDz1q_2.jpeg), so a label is NOT unique within an image and
        # a string match could read the wrong pair's verdict.
        pred_phrase = entities[hit[0]]["object"] if hit else None
        for name in gt_names(o):
            label = gt_name_display(name)
            if not label:
                continue  # a wholly empty name row — nothing to report under
            # Per-NAME exact match, deliberately narrower than the pair's
            # overall `exact_match` (which succeeds on ANY name): here the
            # question is whether the model used THIS name, so "sky recall 1.00,
            # exact name 0.20" correctly reads as "always found, usually called
            # something else".
            exact = None
            if hit:
                terms = name_match_terms(name)
                exact = phrase_matches_terms(pred_phrase, terms) if terms else False
            label_rows.append({"gt_label": label, "found": hit is not None,
                               "iou": hit[1] if hit else None,
                               "exact_match": exact})

    # --- IoU SWEEP off the SAME cached matrix, with the size partition -------
    # Almost every unmatched prediction is a false positive: these annotations
    # are exhaustive, so there is no curated-vocabulary exemption and no crowd
    # regions to suppress against (the annotation format has no iscrowd). The
    # ONE exemption that does apply is second-name credit (above) — reused via
    # the same `excluded_pred` bucket COCO's curated-vocabulary exemption uses,
    # recomputed at each rung since which pairs are matched (and therefore
    # which name is already "claimed") changes with the threshold.
    sweep, size_sweep = {}, {}

    def _buckets(m, unm_gt, unm_pred):
        b = {name: {"tp": 0, "fp": 0, "fn": 0} for name in detection_metrics.AREA_BUCKETS}
        for gi, _, _ in m:
            b[gt_objs[gi]["size_bucket"]]["tp"] += 1
        for gi in unm_gt:
            b[gt_objs[gi]["size_bucket"]]["fn"] += 1
        for pi in unm_pred:
            b[detection_metrics.area_bucket(entities[pi]["pixel_count"])]["fp"] += 1
        return b

    # Label agreement is cached per (gt, pred) PAIR and reused across every
    # rung of the ladder. The pair's naming verdict does not depend on the
    # threshold — only on WHICH pairs are matched — so resolving each phrase to
    # WordNet once instead of ten times keeps the sweep cheap.
    label_cache = {(gi, pi): rec for (gi, pi, _), rec in zip(matches, label_records)}
    sweep_labels = {}
    matches_by_t = {}

    for t in detection_metrics.COCO_AP_IOU_THRESHOLDS:
        if t == args.iou_threshold:
            t_m, t_ug, t_up = matches, unmatched_gt, unmatched_pred
        else:
            t_m, t_ug, t_up = detection_metrics.match_boxes(
                gt_ph, pred_ph, iou_threshold=t, ious=ious)
        matches_by_t[t] = t_m  # reused by the many-to-one sweep below
        recs = []
        t_matched_gt_name_by_gi = {}
        for gi, pi, _ in t_m:
            key = (gi, pi)
            if key not in label_cache:
                label_cache[key] = score_label_agreement(
                    entities[pi]["object"], gt_objs[gi], graph)
            recs.append(label_cache[key])
            t_matched_gt_name_by_gi[gi] = label_cache[key]["matched_gt_name"]
        sweep_labels[t] = recs

        t_true_fp, t_credits = partition_second_name_credit(
            t_up, ious, gt_objs, entities, t_matched_gt_name_by_gi, t,
            graph, args.second_name_hf1_threshold)
        sweep[t] = {"tp": len(t_m), "fp": len(t_true_fp), "fn": len(t_ug),
                    "excluded_pred": len(t_credits), "crowd_suppressed": 0}
        size_sweep[t] = _buckets(t_m, t_ug, t_true_fp)

    # --- many-to-one, SWEPT across the same ladder as tables 1/2 -------------
    # Recall and precision deliberately use DIFFERENT numerators, unlike
    # one-to-one's shared TP: recall counts DISTINCT GT REGIONS explained
    # (what "did we find it" means), precision counts every PREDICTION that
    # contributed to a successful group (a 3-mask group is one recall hit but
    # three "not-a-false-positive" predictions) — the mask-side twin of
    # crediting a whole many-to-one group's naming rather than just its base.
    #
    # NAMING is scored on the group's PRIMARY CONTRIBUTOR — the single member
    # whose own overlap with the GT accounts for the most GT pixels — never
    # best/mean across members: a group with one right member and two junk
    # ones should be scored as "the model called this X", not "best of three
    # guesses" or diluted by guesses that barely touched the region.
    def _many_to_one_at(t, t_matches, want_detail):
        leftover_t = [pi for pi in range(len(pred_rles))
                     if pi not in {p for _, p, _ in t_matches}]
        n_tp, used_pred, sizes, naming_records, detail_rows = 0, set(), [], [], []
        for gi in range(len(gt_objs)):
            base = next((p for g, p, _ in t_matches if g == gi), None)
            absorbed, merged_iou = ggm.absorb_predictions(
                gt_rles[gi], voids[gi] if voids else None, base, leftover_t,
                pred_rles, ious, gi, t, args.min_inside)
            members = ([base] if base is not None else []) + absorbed
            if merged_iou < t or not members:
                continue  # candidates NOT consumed — see the note below
            # Consumed from the pool ONLY on success. The single-pass version
            # of this code removed `absorbed` unconditionally, so a prediction
            # tried (and discarded) by an earlier, failed GT could starve a
            # LATER gi that would genuinely have needed it — a real bug,
            # confirmed on production data: the printed group-size
            # distribution summed to 285 while the reported matched-GT count
            # was 281. Each threshold now also gets its own fresh `leftover_t`
            # rather than sharing state, so this cannot leak across rungs either.
            for pi in absorbed:
                leftover_t.remove(pi)
            n_tp += 1
            used_pred.update(members)
            sizes.append(len(members))
            primary_pi = members[ggm.pick_primary_contributor(
                gt_rles[gi], voids[gi] if voids else None,
                [pred_rles[pi] for pi in members])]
            cache_key = (gi, primary_pi)
            if cache_key not in m2o_label_cache:
                m2o_label_cache[cache_key] = score_label_agreement(
                    entities[primary_pi]["object"], gt_objs[gi], graph)
            naming = m2o_label_cache[cache_key]
            naming_records.append(naming)
            if want_detail:
                detail_rows.append({
                    "gt_label": _primary_label(gt_objs[gi]),
                    "gt_all_labels": _all_labels(gt_objs[gi]),
                    "iou": merged_iou,
                    "group_size": len(members),
                    "members": [entities[pi]["object"] for pi in members],
                    "primary_member": entities[primary_pi]["object"],
                    **naming,
                })
        return n_tp, len(used_pred), sizes, naming_records, detail_rows

    m2o_label_cache = {}
    m2o_sweep, m2o_sweep_labels = {}, {}
    merged_rows, group_sizes, n_used_headline = [], [], 0
    for t in detection_metrics.COCO_AP_IOU_THRESHOLDS:
        n_tp, n_used, sizes, naming_records, detail = _many_to_one_at(
            t, matches_by_t[t], want_detail=(t == args.iou_threshold))
        m2o_sweep[t] = {"tp": n_tp, "fp": len(pred_rles) - n_used,
                        "fn": len(gt_objs) - n_tp,
                        "excluded_pred": 0, "crowd_suppressed": 0}
        m2o_sweep_labels[t] = naming_records
        if t == args.iou_threshold:
            merged_rows, group_sizes, n_used_headline = detail, sizes, n_used
    # Predictions many-to-one credits BEYOND what one-to-one already did.
    n_predictions_absorbed = n_used_headline - len(matches)

    return {
        "image_id": gt_rec["image_id"],
        "platform": gt_rec.get("platform"),
        "n_gt": len(gt_objs),
        "n_pred": len(entities),
        "matches": matches,
        "pair_rows": pair_rows,
        "label_records": label_records,
        "axis_records": axis_records,
        # `n_fp` EXCLUDES second-name-credited predictions — see
        # second_name_credit's docstring. `unmatched_pred` still names
        # everything the one-to-one matcher rejected; `true_fp` is the subset
        # that is actually a false positive.
        "n_fp": len(true_fp),
        "n_fn": len(unmatched_gt),
        "missed_gt": [_all_labels(gt_objs[gi]) for gi in unmatched_gt],
        "false_positives": [entities[pi]["object"] for pi in true_fp],
        "fp_rows": fp_rows,
        "second_name_rows": second_name_rows,
        "n_second_name_credits": len(second_name_credits),
        "label_rows": label_rows,
        "n_multi_named_gt": sum(1 for o in gt_objs if len(gt_names(o)) > 1),
        "n_identical_gt_groups": sum(
            1 for c in Counter(r["counts"] for r in gt_rles).values() if c > 1),
        "sweep": sweep,
        "size_sweep": size_sweep,
        "merged_rows": merged_rows,
        "group_sizes": group_sizes,
        "n_predictions_absorbed": n_predictions_absorbed,
        "m2o_sweep": m2o_sweep,
        "m2o_sweep_labels": m2o_sweep_labels,
        "voided_px_fraction": voided_px / float(H * W) if H * W else 0.0,
        "dropped_size_mismatch": dropped,
        "sweep_labels": sweep_labels,
        # The grounding pipeline's own per-image nature relevance scores, both
        # methods, carried through so the evaluation can report them beside the
        # localization numbers rather than requiring the raw .jsonl.
        "coverage_ratio": (pred_rec or {}).get("coverage_ratio"),
        "center_weighted": (pred_rec or {}).get("center_weighted"),
    }


def score_no_nature_image(gt_rec, pred_rec):
    """An image annotated as containing NO nature: GT is the empty mask set,
    treated as absolute, so every grounded nature mask is a false positive."""
    entities = (pred_rec or {}).get("entities", [])
    H, W = gt_rec["height"], gt_rec["width"]
    union = ggm._merge([e["mask_rle"] for e in entities], intersect=False) if entities else None
    return {
        "image_id": gt_rec["image_id"],
        "platform": gt_rec.get("platform"),
        "n_pred_masks": len(entities),
        "phrases": [e["object"] for e in entities],
        "nature_px_fraction": (ggm._area(union) / float(H * W)) if union and H * W else 0.0,
        "coverage_ratio": (pred_rec or {}).get("coverage_ratio"),
        "center_weighted": (pred_rec or {}).get("center_weighted"),
    }


# =============================================================================
# Aggregation
# =============================================================================
def aggregate(nature_rows, no_nature_rows, args):
    pooled_sweep, pooled_size, pooled_m2o_sweep = {}, {}, {}
    for t in detection_metrics.COCO_AP_IOU_THRESHOLDS:
        agg = {"tp": 0, "fp": 0, "fn": 0, "excluded_pred": 0, "crowd_suppressed": 0}
        m2o_agg = {"tp": 0, "fp": 0, "fn": 0, "excluded_pred": 0, "crowd_suppressed": 0}
        bucket_agg = {b: {"tp": 0, "fp": 0, "fn": 0}
                      for b in detection_metrics.AREA_BUCKETS}
        for r in nature_rows:
            for k in agg:
                agg[k] += r["sweep"][t].get(k, 0)
                m2o_agg[k] += r["m2o_sweep"][t].get(k, 0)
            for b, c in r["size_sweep"][t].items():
                for k in ("tp", "fp", "fn"):
                    bucket_agg[b][k] += c[k]
        pooled_sweep[t] = agg
        pooled_m2o_sweep[t] = m2o_agg
        pooled_size[t] = bucket_agg

    head = dict(pooled_sweep[args.iou_threshold])
    head.update({"iou_threshold": args.iou_threshold,
                 "n_gt_instances": sum(r["n_gt"] for r in nature_rows),
                 "n_pred_instances": sum(r["n_pred"] for r in nature_rows)})

    all_pairs = [p for r in nature_rows for p in r["pair_rows"]]
    ious = [p["iou"] for p in all_pairs]
    n_fp = sum(r["n_fp"] for r in nature_rows)
    n_fn = sum(r["n_fn"] for r in nature_rows)

    group_sizes = [s for r in nature_rows for s in r["group_sizes"]]

    return {
        "grounding": {
            **detection_metrics.detection_summary(head),
            "assignment": _assignment_method(),
            "void_handling": args.void,
            "prediction_source": "semantic_seg",
            "exhaustive_annotation": True,
            "note": (
                "Annotations are EXHAUSTIVE, so almost every unmatched "
                "prediction is a false positive — there is no curated-"
                "vocabulary exemption. The one exception: excluded_predictions "
                "here counts SECOND-NAME CREDITS — a multi-named GT region "
                "(e.g. 'cloud'+'sky') can draw two individually-correct "
                "extractions but one-to-one assignment matches only one, so "
                "the other is excused rather than charged as a duplicate false "
                "positive (see second_name_credit). Precision is therefore "
                "still STRICTER than the COCO detection block's, where 76% of "
                "predictions were exempted for a different reason; do not "
                "compare the two directly."),
        },
        "grounding_iou_sweep": detection_metrics.sweep_summary(pooled_sweep),
        "grounding_by_size": detection_metrics.size_summary(pooled_size),
        "grounding_by_size_note": (
            "COCO's own small/medium/large cut-offs (cocoeval segm areaRng). A "
            "TP/FN takes its GT region's bucket, an FP its own predicted "
            "region's. Unlike the COCO block, GT here is one hand-drawn region "
            "per entity, so the bucket is that region's true area with no "
            "merged-instance distortion."),
        "grounding_pq": {
            "localization": ggm.pq_summary(ious, None, n_fp, n_fn),
            "exact_synset": ggm.pq_summary(
                ious, [1.0 if p["exact_match"] else 0.0 for p in all_pairs], n_fp, n_fn),
            "hierarchical": ggm.pq_summary(
                ious, [p["hf1"] for p in all_pairs], n_fp, n_fn),
            "note": ("PQ-STYLE: standard PQ assumes a partitioned GT and this "
                     "annotation is not one. PQ = SQ x RQ holds exactly."),
        },
        "grounding_labels": detection_metrics.label_summary(
            [r for row in nature_rows for r in row["label_records"]]),
        "grounding_axes": detection_metrics.axis_agreement_summary(
            [r for row in nature_rows for r in row["axis_records"]]),
        "grounding_axes_note": (
            "NATURE is deliberately absent: only final_nature-True entities are "
            "grounded and every GT object here is nature, so agreement would be "
            "1.0 by construction. See grounding_no_nature for the real "
            "nature-axis evidence."),
        # Localization P/R/F1 swept across the SAME IoU ladder as table 1, so
        # the two are directly comparable rung-for-rung. RECALL's numerator is
        # distinct GT regions explained; PRECISION's is every prediction that
        # contributed to a successful group (a 3-mask group is one recall hit
        # but three predictions credited) — see score_nature_image's
        # `_many_to_one_at` for why these deliberately differ, unlike
        # one-to-one's single shared TP.
        "grounding_many_to_one_iou_sweep": detection_metrics.sweep_summary(pooled_m2o_sweep),
        # Naming quality at every rung, scored on each group's PRIMARY
        # CONTRIBUTOR (the member accounting for the most GT pixels) rather
        # than a best/mean across members — see ggm.pick_primary_contributor.
        "grounding_many_to_one_labels_by_iou": {
            t: {**detection_metrics.label_summary(
                    [r for row in nature_rows for r in row["m2o_sweep_labels"][t]]),
                "iou_threshold": t}
            for t in detection_metrics.COCO_AP_IOU_THRESHOLDS},
        "grounding_many_to_one": {
            "group_size_distribution": dict(sorted(Counter(group_sizes).items())),
            "n_decomposed_gt": int(sum(1 for s in group_sizes if s > 1)),
            # How many one-to-one false positives absorption actually explains.
            # This is the honest way to report many-to-one's effect on
            # precision: a group of 3 masks for 1 GT yields 1 TP and absorbs 2
            # predictions, and calling those 2 "not false positives" is a
            # judgement the reader should make with the number in hand.
            "n_predictions_absorbed": int(sum(
                r["n_predictions_absorbed"] for r in nature_rows)),
            "note": ("The headline-threshold P/R/F1/naming numbers now live in "
                     "grounding_many_to_one_iou_sweep and "
                     "grounding_many_to_one_labels_by_iou, which are STRICTLY "
                     "MORE PERMISSIVE than tables 1/2 — never quote them alone. "
                     "group_size_distribution IS the measurement of how often "
                     "the model split one annotated concept into several masks."),
        },
        # Naming quality at EVERY rung, not just the headline threshold. The
        # pairs change as the threshold tightens, so hF1 can move for two very
        # different reasons — better naming among survivors, or simply fewer
        # survivors — and `pairs` is reported alongside so the two are
        # distinguishable.
        "grounding_labels_by_iou": {
            t: {**detection_metrics.label_summary(
                    [r for row in nature_rows for r in row["sweep_labels"][t]]),
                "iou_threshold": t}
            for t in detection_metrics.COCO_AP_IOU_THRESHOLDS},
        # The grounding pipeline's own per-image relevance scores (both
        # methods), summarized here so they sit beside the localization
        # numbers instead of only in the raw artifact.
        "grounding_relevance": {
            "coverage_ratio": ggm.mean_std(
                [r["coverage_ratio"] for r in nature_rows
                 if r.get("coverage_ratio") is not None]),
            "center_weighted": ggm.mean_std(
                [r["center_weighted"] for r in nature_rows
                 if r.get("center_weighted") is not None]),
            "note": ("Per-image nature relevance from src/grounding_pipeline.py, "
                     "over the RLE-merged union of surviving nature masks. "
                     "coverage_ratio = nature px / total px; center_weighted "
                     "applies a Gaussian in normalized distance from centre. "
                     "These describe the PREDICTION only — they are not scored "
                     "against the GT."),
        },
        "grounding_fp_breakdown": _fp_breakdown(nature_rows),
        "grounding_by_gt_label": ggm.label_breakdown(
            [r for row in nature_rows for r in row["label_rows"]]),
        "grounding_no_nature": ggm.no_nature_summary(no_nature_rows),
        "grounding_no_nature_phrases": dict(Counter(
            ph for r in no_nature_rows for ph in r["phrases"]).most_common(30)),
        "grounding_diagnostics": {
            "voided_px_fraction": ggm.mean_std(
                [r["voided_px_fraction"] for r in nature_rows]),
            "n_nature_images": len(nature_rows),
            # How much of the run the multi-name rule actually touches. A
            # region with one name scores exactly as it did before this
            # existed, so this is the honest scope of the change.
            "n_multi_named_gt_regions": int(
                sum(r["n_multi_named_gt"] for r in nature_rows)),
            # Should now be 0: identical GT masks came from the removed
            # duplicate-the-region convention. A non-zero value means the
            # annotator drew two genuinely coincident regions, which
            # ggm.tie_break_identical_gt still handles.
            "n_identical_gt_mask_groups": int(
                sum(r["n_identical_gt_groups"] for r in nature_rows)),
            # How many unmatched predictions were excused as a second, equally
            # valid name for an already-matched multi-named region, rather than
            # charged as a duplicate false positive — the headline-threshold
            # count behind `grounding.excluded_predictions`. Only ever nonzero
            # on the rare regions counted above.
            "n_second_name_credits": int(
                sum(r["n_second_name_credits"] for r in nature_rows)),
            "images_with_no_prediction": int(
                sum(1 for r in nature_rows if r["n_pred"] == 0)),
            "masks_dropped_size_mismatch": int(
                sum(len(r["dropped_size_mismatch"]) for r in nature_rows)),
        },
    }


def _fp_breakdown(nature_rows):
    """Pool the per-image false-positive reasons into the reported dict.

    Also lists the phrases most responsible for each cause, because that is
    what makes the number actionable: a `duplicate` list full of near-synonyms
    ("tree"/"green tree") points at the EXTRACTION prompt, while a
    `hallucination` list full of one recurring word points at the taxonomy
    LABELLING of that concept.

    Second-name-credited predictions (`row["second_name_rows"]`) are NOT
    pooled here — they were already excluded from `fp_rows` before this
    function runs, precisely so they never contend for a `duplicate`
    classification. See `grounding_diagnostics.n_second_name_credits` for
    their count.
    """
    rows = [r for row in nature_rows for r in row["fp_rows"]]
    total = len(rows)
    by_reason = Counter(r["reason"] for r in rows)
    out = {"total_false_positives": total,
           "by_reason": dict(by_reason),
           "by_reason_fraction": {k: v / total for k, v in by_reason.items()} if total else {},
           "top_phrases_by_reason": {}}
    for reason in ("duplicate", "near_miss", "hallucination"):
        phrases = Counter(r["pred_object"] for r in rows if r["reason"] == reason)
        out["top_phrases_by_reason"][reason] = dict(phrases.most_common(15))
    out["note"] = (
        "duplicate = a second mask over an ALREADY-MATCHED region (extraction "
        "redundancy, pixels were right); near_miss = overlaps a GT region but "
        "below the IoU threshold (mask quality); hallucination = no meaningful "
        "GT overlap (perception/labelling). Only the last is nature claimed "
        "where the annotator saw none.")
    return out


def _assignment_method():
    """Which assignment `detection_metrics.match_boxes` actually used.

    It prefers SciPy's Hungarian solver and falls back to greedy when SciPy is
    not importable. That fallback is NOT cosmetic on this dataset: GT regions
    overlap up to IoU 0.747, so a prediction can genuinely be a candidate for
    two different GT objects, and greedy resolves that by iteration order while
    Hungarian maximizes total IoU. Recorded in the results so a run is never
    ambiguous about which one produced its numbers.
    """
    try:
        import scipy.optimize  # noqa: F401
        return "hungarian_scipy"
    except ImportError:
        return "greedy_fallback_no_scipy"


# =============================================================================
# Driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True, nargs="+",
                    help="one or more grounded vlm_responses_<model>.jsonl "
                         "files. Pass BOTH big5_twitter and big5_weibo "
                         "artifacts — the GT spans both platforms, and a "
                         "missing one scores as all-false-negative.")
    ap.add_argument("--gt_dir", required=True,
                    help="grounding_annotations/processed")
    # Resolved against the REPO ROOT, not the cwd, so this works both from the
    # repo root and from scripts/ (where every job_*.sh runs and where the
    # other entrypoints use a "../data/..." relative default).
    ap.add_argument("--excel_path",
                    default=os.path.join(_REPO_ROOT, "data", "big5_taxonomy",
                                         "flat_wordnet_tree_fixed.xlsx"))
    ap.add_argument("--excel_sheet", default="data corrected")
    ap.add_argument("--iou_threshold", type=float,
                    default=detection_metrics.DEFAULT_IOU_THRESHOLD)
    ap.add_argument("--duplicate_iou", type=float, default=ggm.DEFAULT_DUPLICATE_IOU,
                    help="overlap with an already-matched prediction above "
                         "which an unmatched one is called a DUPLICATE rather "
                         "than an independent false positive")
    ap.add_argument("--second_name_hf1_threshold", type=float,
                    default=DEFAULT_SECOND_NAME_HF1,
                    help="hierarchical hF1 bar (in addition to exact match) for "
                         "crediting an unmatched prediction as a second, "
                         "equally valid name for an already-matched multi-named "
                         "GT region, rather than charging it as a duplicate "
                         "false positive — see second_name_credit")
    ap.add_argument("--min_inside", type=float, default=ggm.DEFAULT_MIN_INSIDE,
                    help="fraction of an absorbed mask that must lie inside the "
                         "GT region for many-to-one grouping")
    ap.add_argument("--no_void", dest="void", action="store_false",
                    help="score overlapping GT pixels normally instead of "
                         "voiding them (comparison only — see the metrics "
                         "module docstring)")
    ap.add_argument("--out", default="grounding_gt_eval")
    args = ap.parse_args()

    from src.loaders.excel_loader import TaxonomyGraph
    graph = TaxonomyGraph()
    graph.load_excel(args.excel_path, sheet_name=args.excel_sheet)

    gt = load_gt(args.gt_dir)
    preds = load_predictions(args.artifact)
    print(f"GT {len(gt)} images | artifact {len(preds)} records | "
          f"void={args.void} | assignment={_assignment_method()}")
    print(f"artifacts: {', '.join(args.artifact)}")

    nature_rows, no_nature_rows, missing = [], [], []
    for image_id, rec in sorted(gt.items()):
        pred_rec = preds.get(image_id)
        if pred_rec is None:
            missing.append(image_id)
        if rec["split"] == "nature":
            nature_rows.append(score_nature_image(rec, pred_rec, graph, args))
        else:
            no_nature_rows.append(score_no_nature_image(rec, pred_rec))

    results = aggregate(nature_rows, no_nature_rows, args)
    results["config"] = {
        "artifacts": list(args.artifact), "gt_dir": args.gt_dir,
        "iou_threshold": args.iou_threshold, "void_handling": args.void,
        "min_inside": args.min_inside,
        "second_name_hf1_threshold": args.second_name_hf1_threshold,
        "n_missing_from_artifact": len(missing),
        "missing_from_artifact": missing[:50],
    }

    with open(f"{args.out}_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    _write_csv(f"{args.out}_per_image.csv", nature_rows, no_nature_rows)
    _print_summary(results, missing)
    print(f"\nwrote {args.out}_results.json and {args.out}_per_image.csv")


def _write_csv(path, nature_rows, no_nature_rows):
    """One row per image — the qualitative-review file, per CLAUDE.md's
    convention that a spot-check must never require opening the .jsonl."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "split", "platform", "n_gt", "n_pred", "tp", "fp", "fn",
                    "mean_iou", "exact_matches", "mean_hf1", "voided_px_fraction",
                    "nature_relevance_coverage_ratio",
                    "nature_relevance_center_weighted",
                    "matched_pairs", "missed_gt", "false_positives",
                    "false_positive_reasons",
                    # Predictions excused as a second, equally valid name for an
                    # already-matched multi-named region — NOT counted in `fp`,
                    # NOT in `false_positives`/`false_positive_reasons` above.
                    # See second_name_credit / grounding_diagnostics.
                    "second_name_credits",
                    # many_to_one_present: cheap boolean-ish column to FILTER
                    # on ("does this image have any decomposed GT at all") —
                    # answers the "in case there is a many-to-one assignment"
                    # ask directly, without parsing the descriptive column.
                    "many_to_one_present", "many_to_one_decomposed_count",
                    "many_to_one_groups"])
        for r in nature_rows:
            ious = [p["iou"] for p in r["pair_rows"]]
            hf1s = [p["hf1"] for p in r["pair_rows"]]
            decomposed = [m for m in r["merged_rows"] if m["group_size"] > 1]
            w.writerow([
                r["image_id"], "nature", r["platform"], r["n_gt"], r["n_pred"],
                len(r["pair_rows"]), r["n_fp"], r["n_fn"],
                round(float(np.mean(ious)), 4) if ious else 0.0,
                sum(1 for p in r["pair_rows"] if p["exact_match"]),
                round(float(np.mean(hf1s)), 4) if hf1s else 0.0,
                round(r["voided_px_fraction"], 5),
                r["coverage_ratio"], r["center_weighted"],
                # A multi-named GT region shows ALL its names plus, in
                # brackets, the one that supplied the hierarchical credit —
                # otherwise a `cloud`-vs-`cloud/sky` pair reads as a mismatch.
                " | ".join(
                    f'{p["gt_all_labels"] or p["gt_label"]}'
                    f'{"[->" + p["matched_gt_name"] + "]" if p.get("n_gt_names", 1) > 1 and p.get("matched_gt_name") else ""}'
                    f'~{p["pred_object"]}@{p["iou"]:.2f}'
                    for p in r["pair_rows"]),
                " | ".join(r["missed_gt"]),
                " | ".join(r["false_positives"]),
                " | ".join(f'{f["pred_object"]}:{f["reason"]}' for f in r["fp_rows"]),
                " | ".join(f'{s["pred_object"]}->{s["gt_label"]} '
                           f'(claimed by match: {s["claimed_by_match"]})'
                           for s in r["second_name_rows"]),
                bool(decomposed), len(decomposed),
                # Primary member (used for naming) + its hF1, so a many-to-one
                # row is self-explanatory without cross-referencing the JSON.
                " | ".join(f'{m["gt_label"]}<-[{"+".join(m["members"])}]'
                           f'(primary={m["primary_member"]}, hF1={m["hf1"]:.2f})'
                           f'@{m["iou"]:.2f}' for m in decomposed),
            ])
        for r in no_nature_rows:
            w.writerow([r["image_id"], "no_nature", r["platform"], 0, r["n_pred_masks"],
                        0, r["n_pred_masks"], 0, 0.0, 0, 0.0, 0.0,
                        r["coverage_ratio"], r["center_weighted"],
                        "", "", " | ".join(r["phrases"]), "", "", False, 0, ""])


def _table(headers, rows, indent="  "):
    """Fixed-width text table. Columns are sized to their widest cell so the
    numbers line up and can be scanned down a column, which is the whole point
    of showing a sweep rather than three isolated operating points."""
    cols = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            cols[i] = max(cols[i], len(str(c)))
    out = [indent + "  ".join(h.rjust(cols[i]) for i, h in enumerate(headers))]
    out.append(indent + "  ".join("-" * cols[i] for i in range(len(headers))))
    for r in rows:
        out.append(indent + "  ".join(str(c).rjust(cols[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def _print_summary(results, missing):
    g = results["grounding"]
    sweep = results["grounding_iou_sweep"]
    per_iou = sweep.get("per_iou", {})
    labels_by_iou = results["grounding_labels_by_iou"]

    print("\n" + "=" * 78)
    print(f"BIG-5 GROUNDING EVALUATION vs. hand-drawn annotations")
    print(f"  {g['n_gt_instances']} GT regions over "
          f"{results['grounding_diagnostics']['n_nature_images']} nature images"
          f"  |  {g['n_pred_instances']} predicted masks")
    print("=" * 78)

    # ---- 1. LOCALIZATION -----------------------------------------------
    print("\n1. LOCALIZATION — did the mask land on the right pixels?")
    print("   Matching is class-agnostic, so this measures WHERE only, not naming.")
    print("   Almost every unmatched prediction counts as a false positive "
          "(annotations are exhaustive) —")
    print("   'excl' is the one exception: a second, equally valid name for an "
          "already-matched")
    print("   multi-named region (e.g. both 'cloud' and 'sky' predicted), "
          "excused rather than")
    print("   charged as a duplicate false positive. See "
          "grounding_diagnostics.n_second_name_credits.\n")
    rows = []
    for t in sorted(per_iou):
        c = per_iou[t]
        rows.append([f"{t:.2f}", c["tp"], c["fp"], c["fn"],
                     c.get("excluded_predictions", 0),
                     f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1']:.3f}"])
    print(_table(["IoU", "TP", "FP", "FN", "excl", "precision", "recall", "F1"], rows))
    print(f"\n   mean over [.50:.95]   precision={sweep.get('precision_50_95', 0):.3f}  "
          f"recall={sweep.get('recall_50_95', 0):.3f}  F1={sweep.get('f1_50_95', 0):.3f}")
    print("   Reading the column downward: F1 that HOLDS from .50 to .75 means the")
    print("   masks genuinely trace their objects; F1 that COLLAPSES means loose blobs.")

    # ---- 2. NAMING -----------------------------------------------------
    print("\n\n2. NAMING — on the pairs that matched, was the entity named right?")
    print("   exact = the phrase names the GT class outright.")
    print("   hP/hR/hF1 = WordNet ancestral-closure overlap, so a predicted 'bull'")
    print("   against a GT 'cow' earns ~0.94 instead of a flat zero. WuP = Wu-Palmer.")
    print("   A GT region may carry SEVERAL names: exact match succeeds on any of")
    print("   them, and hP/hR/hF1/WuP are taken from the best-scoring one.\n")
    rows = []
    for t in sorted(labels_by_iou):
        L = labels_by_iou[t]
        if not L.get("support"):
            continue
        rows.append([f"{t:.2f}", L["support"], f"{L['exact_match_accuracy']:.3f}",
                     f"{L['hp']:.3f}", f"{L['hr']:.3f}", f"{L['hf1']:.3f}",
                     f"{L['wup']:.3f}", f"{L['resolution_failure_rate']:.3f}"])
    if rows:
        print(_table(["IoU", "pairs", "exact", "hP", "hR", "hF1", "WuP",
                      "unresolved"], rows))
        print("   NOTE 'pairs' shrinks as IoU tightens — a rise in hF1 down this table")
        print("   can mean better naming among survivors, not better naming overall.")

    # ---- 3. OBJECT SIZE ------------------------------------------------
    sz = results["grounding_by_size"]
    if sz:
        print("\n\n3. OBJECT SIZE — COCO's own small/medium/large buckets")
        print("   small < 32x32 px, medium < 96x96 px, large beyond.")
        print("   A TP/FN is bucketed by its GT region, an FP by its own mask.\n")
        rows = []
        for b in detection_metrics.AREA_BUCKETS:
            e = sz[b]
            rows.append([b, e["n_gt"], f"{e['precision_50']:.3f}",
                         f"{e['recall_50']:.3f}", f"{e['f1_50']:.3f}",
                         f"{e['f1_75']:.3f}", f"{e['f1_50_95']:.3f}"])
        print(_table(["bucket", "n_GT", "P@.50", "R@.50", "F1@.50", "F1@.75",
                      "F1@[.50:.95]"], rows))

    # ---- 4. TAXONOMY AXES ----------------------------------------------
    ax = results["grounding_axes"]
    print("\n\n4. TAXONOMY AXES — on matched pairs, does the label agree with GT?")
    print("   nature is absent on purpose: every matched pair is nature-vs-nature")
    print("   by construction, so a rate for it would report a tautology.\n")
    print(_table(["axis", "n pairs", "accuracy"],
                 [[a, ax[a]["support"], f"{ax[a]['accuracy']:.3f}"]
                  for a in ("biotic", "material")]))

    # ---- 5. FALSE POSITIVES --------------------------------------------
    fpb = results["grounding_fp_breakdown"]
    if fpb["total_false_positives"]:
        print(f"\n\n5. FALSE POSITIVES — why did {fpb['total_false_positives']} "
              f"predictions match nothing?")
        print("   duplicate     = a 2nd mask over an ALREADY-MATCHED region")
        print("                   (extraction emitted two names; the pixels were right)")
        print("   near_miss     = overlaps a GT region but below the IoU bar (mask quality)")
        print("   hallucination = no meaningful GT overlap (nature claimed where")
        print("                   the annotator saw none)\n")
        rows = []
        for reason in ("duplicate", "near_miss", "hallucination"):
            n = fpb["by_reason"].get(reason, 0)
            share = fpb["by_reason_fraction"].get(reason, 0.0)
            top = list(fpb["top_phrases_by_reason"].get(reason, {}).items())[:4]
            rows.append([reason, n, f"{share:.1%}",
                         ", ".join(f"{k}({v})" for k, v in top) or "-"])
        print(_table(["reason", "count", "share", "most common phrases"], rows))

    # ---- 6. PER-LABEL --------------------------------------------------
    bylab = results["grounding_by_gt_label"]
    shown = [(k, v) for k, v in bylab.items() if v["support"] >= 5]
    if shown:
        print("\n\n6. PER-LABEL RECALL — which annotated concepts get found?")
        print("   (labels with at least 5 annotated regions)")
        print("   One row PER NAME: a mask named both 'cloud' and 'sky' counts under")
        print("   each, so the supports do not partition the GT. 'exact name' is that")
        print("   NAME's own rate — high recall with a low exact rate means the region")
        print("   was found but usually called by its other name.\n")
        rows = [[k, v["support"], f"{v['recall']:.3f}",
                 f"{v['mean_iou_when_found']:.3f}",
                 f"{v['exact_name_rate_when_found']:.3f}"]
                for k, v in sorted(shown, key=lambda kv: -kv[1]["recall"])]
        print(_table(["label", "n_GT", "recall", "mean IoU (found)",
                      "exact name (found)"], rows))

    # ---- 7. MANY-TO-ONE ------------------------------------------------
    m2o = results["grounding_many_to_one"]
    m2o_sweep = results["grounding_many_to_one_iou_sweep"]
    m2o_per_iou = m2o_sweep.get("per_iou", {})
    m2o_labels_by_iou = results["grounding_many_to_one_labels_by_iou"]
    print("\n\n7. MANY-TO-ONE — allowing several masks to cover one GT region")
    print("   Handles the model splitting one annotated concept (GT 'greenery')")
    print("   into parts ('tree' + 'bush'). Same two tables as 1/2 above, so they")
    print("   can be read side by side — STRICTLY more permissive than 1/2;")
    print("   never quote alone. RECALL counts distinct GT regions found; ")
    print("   PRECISION counts every prediction credited (a 3-mask group is one")
    print("   recall hit but three predictions not charged as false positives).\n")
    rows = []
    for t in sorted(m2o_per_iou):
        c = m2o_per_iou[t]
        rows.append([f"{t:.2f}", c["tp"], c["fp"], c["fn"],
                     f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1']:.3f}"])
    if rows:
        print(_table(["IoU", "TP", "FP", "FN", "precision", "recall", "F1"], rows))
        print(f"\n   mean over [.50:.95]   precision={m2o_sweep.get('precision_50_95', 0):.3f}  "
              f"recall={m2o_sweep.get('recall_50_95', 0):.3f}  "
              f"F1={m2o_sweep.get('f1_50_95', 0):.3f}")

    print("\n   Naming, scored on each group's PRIMARY CONTRIBUTOR (the member")
    print("   accounting for the most GT pixels — never best/mean across members):\n")
    rows = []
    for t in sorted(m2o_labels_by_iou):
        L = m2o_labels_by_iou[t]
        if not L.get("support"):
            continue
        rows.append([f"{t:.2f}", L["support"], f"{L['exact_match_accuracy']:.3f}",
                     f"{L['hp']:.3f}", f"{L['hr']:.3f}", f"{L['hf1']:.3f}",
                     f"{L['wup']:.3f}"])
    if rows:
        print(_table(["IoU", "pairs", "exact", "hP", "hR", "hF1", "WuP"], rows))

    dist = m2o["group_size_distribution"]
    print(f"\n   GT split across >1 mask: {m2o['n_decomposed_gt']}   "
          f"predictions absorbed: {m2o['n_predictions_absorbed']}")
    if dist:
        print("   masks per GT region: " +
              ", ".join(f"{k} mask{'s' if int(k) > 1 else ''} -> {v} regions"
                        for k, v in sorted(dist.items(), key=lambda kv: int(kv[0]))))

    # ---- 8. RELEVANCE --------------------------------------------------
    rel = results["grounding_relevance"]
    print("\n\n8. NATURE RELEVANCE SCORE — how much of the image the pipeline")
    print("   considers nature. A property of the PREDICTION, not scored against GT.")
    print("   Per-image values are in the CSV; means over the nature images below.\n")
    print(_table(["method", "mean", "std", "images"],
                 [["coverage_ratio", f"{rel['coverage_ratio']['mean']:.4f}",
                   f"{rel['coverage_ratio']['std']:.4f}",
                   rel["coverage_ratio"]["support"]],
                  ["center_weighted", f"{rel['center_weighted']['mean']:.4f}",
                   f"{rel['center_weighted']['std']:.4f}",
                   rel["center_weighted"]["support"]]]))

    # ---- 9. NO-NATURE CONTROL ------------------------------------------
    nn = results["grounding_no_nature"]
    if nn.get("n_images"):
        print(f"\n\n9. NO-NATURE CONTROL — {nn['n_images']} images annotated as")
        print("   containing NO nature at all. GT is the empty mask set, taken as")
        print("   absolute, so ANY grounded nature mask here is a false positive.")
        print("   This is the only unambiguous false-positive evidence in the run:")
        print("   on a nature image an over-eager model still scores well on recall.\n")
        print(_table(["metric", "value"], [
            ["images with >=1 false positive",
             f"{nn['n_images_with_false_positive']} / {nn['n_images']} "
             f"({nn['image_false_positive_rate']:.1%})"],
            ["false-positive masks total", nn["n_false_positive_masks"]],
            ["mean nature px fraction (all images)",
             f"{nn['mean_nature_pixel_fraction']:.4f}"],
            ["mean nature px fraction (flagged only)",
             f"{nn['mean_nature_pixel_fraction_when_flagged']:.4f}"],
            ["mean center_weighted relevance",
             f"{nn['mean_relevance_center_weighted']:.4f}"
             if nn.get("mean_relevance_center_weighted") is not None else "n/a"],
        ]))
        top = list(results["grounding_no_nature_phrases"].items())[:8]
        if top:
            print("\n   phrases most often hallucinated: " +
                  ", ".join(f"{k}({v})" for k, v in top))

    # ---- diagnostics ----------------------------------------------------
    d = results["grounding_diagnostics"]
    print("\n\nDIAGNOSTICS")
    print(f"   void pixels (mean per image)      {d['voided_px_fraction']['mean']:.4f}")
    print(f"   nature images with no prediction  {d['images_with_no_prediction']}")
    print(f"   masks dropped (size mismatch)     {d['masks_dropped_size_mismatch']}")
    print(f"   GT regions with >1 name           {d['n_multi_named_gt_regions']}")
    print(f"   second-name credits (not FP)      {d['n_second_name_credits']}")
    print(f"   identical GT mask groups          {d['n_identical_gt_mask_groups']}"
          f"   (expected 0 — see tie_break_identical_gt)")
    print(f"   assignment                        {g['assignment']}")
    print(f"   void handling                     {g['void_handling']}")
    if missing:
        print(f"\nWARNING: {len(missing)} GT images absent from the artifact "
              f"(scored as all-FN): {missing[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
