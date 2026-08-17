"""
Turn VLM-pipeline artifacts into a rejection-sampling fine-tuning dataset.

  python fine_tuning/build_rft_dataset.py \
      --artifact results/vlm_pipeline/baseline/big5_twitter/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
      --artifact results/vlm_pipeline/baseline/big5_weibo/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
      --splits /home/pmonserrat/datasets/big_5/rft/splits/splits.json \
      --out /home/pmonserrat/datasets/big_5/rft/rft_gemma12b

Reads each artifact, keeps only images in the requested split whose IMAGE-LEVEL
prediction matches the human annotation (see rft_common.image_verdict), and
writes one JSON-Lines file of (system, user, image) -> assistant examples per
split, plus a stats.json.

THE IMBALANCE THIS SCRIPT EXISTS TO MANAGE
==========================================
Rejection sampling does not accept the two classes at the same rate. Measured
on the gemma-4-12B-it BIG-5 artifacts:

    GT nature      3397 / 3634 accepted   (93.5%)
    GT non-nature  1672 / 3029 accepted   (55.2%)

The gap is not incidental — it is the model's own failure mode showing through
the filter. The pipeline over-predicts nature, so images whose correct answer
is "no nature here" are exactly the ones it most often gets wrong, and exactly
the ones rejection sampling therefore discards. Training on the raw accepted
set means training on a 67% nature mixture drawn from a 55% nature dataset,
which pushes the model further in the direction it is already wrong.

Three treatments, chosen with --balance:

  none              Every accepted image. Most data, keeps the bias.
  downsample_nature Subsample accepted GT-nature images down to the number of
                    accepted GT-non-nature images, so the training mixture is
                    50/50. Costs data; the most direct correction.
  loss_weight       Every accepted image, but each example carries a `weight`
                    the trainer multiplies its loss by, inversely proportional
                    to its GT-nature class frequency in the accepted set
                    (normalized to mean 1.0 so the effective learning rate
                    does not change with the mixture). Keeps all the data.

Downsampling removes whole IMAGES rather than individual examples, so an image
is never half-present: dropping only some of an image's labeling calls would
train the model on a partial scene reading.

WHICH STAGES BECOME EXAMPLES (--stages)
=======================================
Default is extraction + both labeling calls. The free-form CAPTION stage is
available but off by default: the acceptance test says nothing about caption
quality (a correct image-level verdict can follow from a mediocre caption), it
is by far the longest generation in the chain, and training on it risks
dragging the neutral descriptive caption toward taxonomy vocabulary — which the
project deliberately keeps out of that call (CLAUDE.md's caption conventions).
Adding "caption" to --stages is a one-word change if you want the full
on-policy chain.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

from rft_common import (ACCEPT_RULES, STAGES, BuildStats, Example,
                        examples_for_record, image_verdict, read_artifact)
from make_splits import platform_of

BALANCE_MODES = ("none", "downsample_nature", "loss_weight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", action="append", required=True,
                    help="vlm_responses_*.jsonl providing the training signal. Repeat per "
                         "platform. For DISTILLATION, point these at a heavier model's "
                         "artifact — the examples are built the same way and tagged with "
                         "that model in `source_model`.")
    ap.add_argument("--splits", required=True, help="splits.json from make_splits.py")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--build_splits", nargs="+", default=["train", "val"],
                    help="Which splits to emit examples for. TEST is deliberately not a "
                         "default: it is evaluated by running the real pipeline, never by "
                         "scoring reconstructed examples.")
    ap.add_argument("--stages", nargs="+", default=["extraction", "label_full", "label_material"],
                    choices=list(STAGES) + ["caption"],
                    help="Pipeline calls to turn into examples.")
    ap.add_argument("--accept_rule", default="strict", choices=list(ACCEPT_RULES))
    ap.add_argument("--balance", default="downsample_nature", choices=list(BALANCE_MODES),
                    help="How to treat the nature/non-nature imbalance in the ACCEPTED set. "
                         "Applied to the train split only — val stays an unmodified sample "
                         "of accepted images so its loss stays comparable across modes.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if "caption" in args.stages:
        # Kept as an explicit opt-in rather than silently supported: the
        # caption call takes NO system prompt at all (see vlm_pipeline.
        # run_inference), which is a different example shape from every other
        # stage, so it needs its own handling in rft_common before it can be
        # emitted. Fail loudly instead of dropping it without a word.
        ap.error("--stages caption is not implemented yet: the caption call is the one "
                 "stage that runs with no system prompt, so it needs its own example "
                 "shape. Remove it, or add it to rft_common.examples_for_record first.")

    with open(args.splits, "r", encoding="utf-8") as fh:
        image_split: Dict[str, str] = json.load(fh)["image_split"]

    rng = random.Random(args.seed)
    stats = BuildStats()

    # ---- Pass 1: acceptance ------------------------------------------------
    # Records are held per split so balancing can be decided over the whole
    # accepted set before any example is emitted.
    accepted: Dict[str, List[tuple]] = defaultdict(list)   # split -> [(record, platform)]
    reject_reasons: Counter = Counter()
    source_models = set()

    for path in args.artifact:
        header, records = read_artifact(path)
        platform = platform_of(path, header)
        source_models.add(header.get("model_name") or "unknown")
        for rec in records:
            split = image_split.get(rec["image_path"])
            if split is None:
                stats.bump("images_not_in_splits")
                continue
            stats.bump(f"images_seen/{split}")
            verdict = image_verdict(rec, rule=args.accept_rule)
            if not verdict.accepted:
                reject_reasons[f"{split}/{verdict.reason}"] += 1
                stats.bump(f"images_rejected/{split}")
                continue
            stats.bump(f"images_accepted/{split}")
            stats.bump(f"images_accepted/{split}/gt_nature={verdict.gt_nature}")
            if split in args.build_splits:
                accepted[split].append((rec, platform))

    source_model = sorted(source_models)[0] if len(source_models) == 1 else "+".join(sorted(source_models))
    if len(source_models) > 1:
        print(f"⚠️  Artifacts come from DIFFERENT models ({sorted(source_models)}). That is "
              f"valid (e.g. one teacher per platform) but every example is tagged with the "
              f"combined name, so provenance per example is lost — prefer one model per build.")

    # ---- Balancing (train only) -------------------------------------------
    weights: Dict[bool, float] = {True: 1.0, False: 1.0}
    if "train" in accepted:
        train = accepted["train"]
        n_nat = sum(1 for rec, _ in train if (rec.get("targets") or [{}])[0].get("gt_nature") is True)
        n_non = len(train) - n_nat
        if args.balance == "downsample_nature":
            if n_nat > n_non:
                nature_recs = [t for t in train if (t[0].get("targets") or [{}])[0].get("gt_nature") is True]
                other_recs = [t for t in train if (t[0].get("targets") or [{}])[0].get("gt_nature") is not True]
                rng.shuffle(nature_recs)
                dropped = len(nature_recs) - n_non
                accepted["train"] = other_recs + nature_recs[:n_non]
                rng.shuffle(accepted["train"])
                stats.bump("train_images_dropped_by_downsampling", dropped)
            else:
                print(f"⚠️  --balance downsample_nature is a no-op: accepted nature images "
                      f"({n_nat}) do not outnumber non-nature ones ({n_non}).")
        elif args.balance == "loss_weight":
            # Inverse-frequency, normalized so the MEAN example weight is 1.0
            # — otherwise changing the mixture would also change the effective
            # learning rate, confounding the comparison this flag exists for.
            total = n_nat + n_non
            if n_nat and n_non:
                weights = {True: total / (2.0 * n_nat), False: total / (2.0 * n_non)}

    # ---- Pass 2: emit examples --------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    per_split_counts: Dict[str, Counter] = defaultdict(Counter)
    for split in args.build_splits:
        recs = accepted.get(split, [])
        out_path = os.path.join(args.out, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec, platform in recs:
                for ex in examples_for_record(rec, platform, source_model, tuple(args.stages), stats):
                    row = json.loads(ex.to_json())
                    if args.balance == "loss_weight" and split == "train":
                        row["weight"] = round(weights.get(bool(ex.gt_nature), 1.0), 6)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    per_split_counts[split][ex.stage] += 1
                    per_split_counts[split]["_total"] += 1
        print(f"Wrote {out_path}: {per_split_counts[split]['_total']} examples from {len(recs)} images")

    # ---- Stats -------------------------------------------------------------
    stats_path = os.path.join(args.out, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump({
            "artifacts": args.artifact,
            "source_model": source_model,
            "splits_file": args.splits,
            "accept_rule": args.accept_rule,
            "balance": args.balance,
            "loss_weights_by_gt_nature": {str(k): v for k, v in weights.items()},
            "stages": args.stages,
            "seed": args.seed,
            "counters": dict(sorted(stats.counts.items())),
            "rejection_reasons": dict(sorted(reject_reasons.items())),
            "examples_per_split": {k: dict(v) for k, v in per_split_counts.items()},
        }, fh, indent=1)

    print(f"\nWrote {stats_path}")
    print("Acceptance:")
    for split in sorted({k.split("/")[1] for k in stats.counts if k.startswith("images_seen/")}):
        seen = stats.counts.get(f"images_seen/{split}", 0)
        acc = stats.counts.get(f"images_accepted/{split}", 0)
        nat = stats.counts.get(f"images_accepted/{split}/gt_nature=True", 0)
        non = stats.counts.get(f"images_accepted/{split}/gt_nature=False", 0)
        print(f"  {split:5s} {acc}/{seen} accepted ({acc / seen:.1%} of split)  "
              f"[nature {nat}, non-nature {non}]")
    dropped = stats.counts.get("train_images_dropped_by_downsampling")
    if dropped:
        print(f"  train: dropped {dropped} accepted nature images to balance 50/50")
    skipped = {k: v for k, v in stats.counts.items() if "skipped" in k}
    if skipped:
        print("Examples skipped during reconstruction (per call, not per image):")
        for k, v in sorted(skipped.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
