"""
Paired bootstrap significance test for the caption-ablation comparison
(pipeline WITH caption vs WITHOUT caption), on the ranking metric actually
used for the ablation study:

    score(dataset) = 0.5 * nature_acc + 0.25 * (life_category_acc + tangibility_acc)
    final_score    = mean(score(twitter), score(weibo))

Reads the RAW vlm_responses_*.jsonl artifacts directly (NOT the predictions
CSV) and reuses the pipeline's OWN scoring functions/logic rather than
reimplementing them independently:

  - image_gt_nature / image_pred_nature: imported verbatim from
    scripts/run_vlm_pipeline.py.
  - biotic/material direction-aware "at least one matching entity" check:
    copied verbatim from phase_score's own inline logic in that same file
    (not factored into a reusable function upstream, so reproduced here
    exactly rather than approximated from the CSV's lossy formatted cells).

WHY NOT THE PREDICTIONS CSV: its image_gt_biotic/image_pred_biotic columns
are rendered via _fmt_big5_axis_cell, which collapses ANY 2-element
disagreement list (coders disagreed, e.g. [True, False]) to the FIXED string
"True/False (disagreement)" regardless of what the model actually predicted
for either direction -- both the GT and the predicted-list columns render
identically whenever their list has length 2, so the CSV alone cannot tell
whether the model matched zero, one, or both of the disagreement directions.
Reading the .jsonl artifact directly preserves full fidelity for these
(rare) images instead of silently dropping or mis-scoring them.

PER-IMAGE GRANULARITY: nature is always exactly one verdict per image. A
disagreement image contributes UP TO TWO ground-truth instances for
biotic/material (per CLAUDE.md's own convention for BIG-5 scoring) -- this
script keeps ONE row per image for the bootstrap (matching the requested
design) by averaging the per-instance correctness within an image, so a
disagreement image scores 0.0/0.5/1.0 on that axis instead of contributing
two separate bootstrap units. This only affects the rare images where human
coders themselves disagreed.

USAGE
    python significance_test_caption_ablation.py \\
        --twitter_with    <path to WITH-caption twitter artifact.jsonl> \\
        --twitter_without <path to WITHOUT-caption twitter artifact.jsonl> \\
        --weibo_with      <path to WITH-caption weibo artifact.jsonl> \\
        --weibo_without   <path to WITHOUT-caption weibo artifact.jsonl> \\
        [--n_boot 10000] [--seed 0]

Run from the scripts/ directory (imports run_vlm_pipeline as a sibling
module) -- or anywhere, with scripts/ on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np


# Copied VERBATIM from scripts/run_vlm_pipeline.py (image_gt_nature /
# image_pred_nature) rather than imported — that module pulls in
# torch/transformers/vllm at load time for its own CLI, which would make
# this lightweight, dependency-free stats script fragile (and slow) for no
# benefit, since both functions are tiny and pure Python. Keep these in sync
# with the originals if that file's logic ever changes.
def image_gt_nature(targets):
    """Image-level GT nature: True if ANY target is nature, False if all targets
    are explicitly non-nature, None if no target carries a nature label."""
    vals = [t.get("gt_nature") for t in targets if t.get("gt_nature") is not None]
    if not vals:
        return None
    return any(vals)


def image_pred_nature(object_final_labels):
    """Image-level predicted nature: True if ANY extracted object is labeled
    nature. An image with no objects predicts False (no nature found)."""
    return any(bool(o["final_nature"]) for o in object_final_labels)


# =============================================================================
# Per-image extraction from a raw vlm_responses_*.jsonl artifact
# =============================================================================
def _biotic_or_material_correct(targets, object_finals, axis: str) -> Optional[float]:
    """Mean per-GT-instance correctness for one image on `axis` ("biotic" or
    "material"), or None if this image has no usable GT for that axis.

    Verbatim reproduction of phase_score's own inline logic in
    run_vlm_pipeline.py (search for "has_biotic"/"has_abiotic" there) -- NOT
    an independent reimplementation, to avoid the two silently drifting
    apart. `object_finals` is each extracted object's own hybrid-resolved
    label dict (final_nature/final_biotic/final_material); `targets` is the
    image's GT target list, whose FIRST entry carries the BIG-5 holistic
    gt_biotic/gt_material lists (src.loaders.dataset_loader.load_big5).
    """
    gt_key = f"gt_{axis}"
    final_key = f"final_{axis}"

    t0 = targets[0] if targets else {}
    gt_vals = t0.get(gt_key)
    if not gt_vals:
        return None

    nature_entities = [fin for fin in object_finals if fin.get("final_nature") is True]
    has_true = any(fin.get(final_key) is True for fin in nature_entities)
    has_false = any(fin.get(final_key) is False for fin in nature_entities)

    per_instance_correct = []
    for gt_b in gt_vals:
        pred_b = has_true if gt_b else (not has_false)
        per_instance_correct.append(1.0 if bool(gt_b) == bool(pred_b) else 0.0)

    return float(np.mean(per_instance_correct))


def load_per_image_arrays(artifact_path: str) -> Dict[str, np.ndarray]:
    """Read one vlm_responses_*.jsonl BIG-5 artifact into the four aligned
    per-image arrays the bootstrap needs. Images with no usable GT nature
    label are skipped entirely (image_gt_nature returning None -- shouldn't
    normally happen per that function's own docstring, but handled the same
    defensive way it already is upstream)."""
    nature_correct: List[float] = []
    is_positive: List[bool] = []
    life_correct: List[float] = []
    tang_correct: List[float] = []

    with open(artifact_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "image_path" not in rec:
                continue  # header/footer record

            targets = rec.get("targets") or []
            g_nat = image_gt_nature(targets)
            if g_nat is None:
                continue

            # rec["object_finals"] is DIRECTLY the list of resolve_hybrid_label()
            # dicts (src/vlm_pipeline.py), each carrying "final_nature"/
            # "final_biotic"/"final_material" — confirmed against the writer,
            # not guessed.
            object_finals = rec.get("object_finals") or []
            p_nat = image_pred_nature(object_finals)

            nature_correct.append(1.0 if g_nat == p_nat else 0.0)
            pos = bool(g_nat)
            is_positive.append(pos)

            if pos:
                lc = _biotic_or_material_correct(targets, object_finals, "biotic")
                tc = _biotic_or_material_correct(targets, object_finals, "material")
                # A nature-positive image with no usable biotic/material GT
                # (shouldn't happen for BIG-5, but defensively excluded
                # rather than guessed) drops out of THOSE two axes only --
                # nature_correct/is_positive above are already recorded.
                life_correct.append(lc if lc is not None else np.nan)
                tang_correct.append(tc if tc is not None else np.nan)
            else:
                life_correct.append(np.nan)
                tang_correct.append(np.nan)

    return {
        "nature_correct": np.array(nature_correct, dtype=float),
        "is_nature_positive": np.array(is_positive, dtype=bool),
        "life_correct": np.array(life_correct, dtype=float),
        "tang_correct": np.array(tang_correct, dtype=float),
    }


# =============================================================================
# Bootstrap
# =============================================================================
def weighted_score(nature_correct, life_correct, tang_correct) -> float:
    nature_acc = float(np.mean(nature_correct))
    life_acc = float(np.mean(life_correct))
    tang_acc = float(np.mean(tang_correct))
    return 0.5 * nature_acc + 0.25 * (life_acc + tang_acc)


def _dataset_score(d: Dict[str, np.ndarray], idx: np.ndarray) -> float:
    """weighted_score for one dataset at the given (possibly resampled)
    indices. pos is cast to bool EXPLICITLY -- indexing an int/float array
    with a non-bool array is silent fancy-indexing, not a boolean mask, and
    would produce wrong-but-not-erroring results."""
    nc = d["nature_correct"][idx]
    pos = d["is_nature_positive"][idx].astype(bool)
    lc = d["life_correct"][idx][pos]
    tc = d["tang_correct"][idx][pos]
    # nan-safe: a nature-positive image with no usable biotic/material GT
    # (see load_per_image_arrays) is excluded from THIS axis's mean, not
    # counted as wrong.
    lc = lc[~np.isnan(lc)]
    tc = tc[~np.isnan(tc)]
    return weighted_score(nc, lc, tc)


def final_score(data: Dict[str, Dict[str, np.ndarray]]) -> float:
    """The real, non-bootstrapped statistic: mean over datasets of each
    dataset's weighted_score on its FULL (unresampled) data."""
    per_dataset = []
    for ds, d in data.items():
        idx_full = np.arange(len(d["nature_correct"]))
        per_dataset.append(_dataset_score(d, idx_full))
    return float(np.mean(per_dataset))


def paired_bootstrap(data_with, data_without, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    datasets = list(data_with.keys())
    assert datasets == list(data_without.keys()), "dataset keys must match and be in the same order"
    for ds in datasets:
        n_with = len(data_with[ds]["nature_correct"])
        n_without = len(data_without[ds]["nature_correct"])
        assert n_with == n_without, (
            f"{ds}: WITH has {n_with} images, WITHOUT has {n_without} -- these must be the "
            f"SAME images in the SAME order for the paired resample to be valid. Filter both "
            f"artifacts down to their common image set before running this."
        )

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        scores_with = []
        scores_without = []
        for ds in datasets:
            n = len(data_with[ds]["nature_correct"])
            idx = rng.integers(0, n, n)  # SAME resample for with/without -> paired
            scores_with.append(_dataset_score(data_with[ds], idx))
            scores_without.append(_dataset_score(data_without[ds], idx))
        diffs[b] = np.mean(scores_with) - np.mean(scores_without)

    observed = final_score(data_with) - final_score(data_without)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value: proportion of the resample distribution on
    # the OPPOSITE side of zero from the observed effect, doubled -- the
    # standard nonparametric shortcut equivalent to "does the (1-p) percentile
    # CI still exclude zero." Approximate, not a classical exact p-value.
    p_value = float(2 * min((diffs > 0).mean(), (diffs < 0).mean()))
    return observed, (float(ci_low), float(ci_high)), p_value, diffs


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twitter_with", required=True)
    ap.add_argument("--twitter_without", required=True)
    ap.add_argument("--weibo_with", required=True)
    ap.add_argument("--weibo_without", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading artifacts...")
    data_with = {
        "twitter": load_per_image_arrays(args.twitter_with),
        "weibo": load_per_image_arrays(args.weibo_with),
    }
    data_without = {
        "twitter": load_per_image_arrays(args.twitter_without),
        "weibo": load_per_image_arrays(args.weibo_without),
    }

    for ds in ("twitter", "weibo"):
        n_w = len(data_with[ds]["nature_correct"])
        n_wo = len(data_without[ds]["nature_correct"])
        print(f"  {ds}: with={n_w} images, without={n_wo} images")
        if n_w != n_wo:
            print(f"  WARNING: image counts differ for {ds} -- the two artifacts don't cover "
                  f"the exact same images. The pairing assumption is violated; fix this before "
                  f"trusting the result (e.g. restrict both to their common --split_file).")

    s_with = final_score(data_with)
    s_without = final_score(data_without)
    print(f"\nweighted_score WITH caption    = {s_with:.4f}")
    print(f"weighted_score WITHOUT caption = {s_without:.4f}")

    observed, (ci_low, ci_high), p_value, _ = paired_bootstrap(
        data_with, data_without, n_boot=args.n_boot, seed=args.seed)

    print(f"\nobserved difference (with - without) = {observed:+.4f}")
    print(f"95% CI                                = [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"two-sided bootstrap p-value           = {p_value:.4g}")
    if ci_low > 0 or ci_high < 0:
        print("-> CI excludes zero: significant at alpha=0.05")
    else:
        print("-> CI includes zero: NOT significant at alpha=0.05")


if __name__ == "__main__":
    main()
