"""
Build the BIG-5 train/val/test splits used by every fine-tuning experiment.

  python fine_tuning/make_splits.py \
      --artifact results/vlm_pipeline/baseline/big5_twitter/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
      --artifact results/vlm_pipeline/baseline/big5_weibo/responses/vlm_responses_google_gemma-4-12B-it.jsonl \
      --out /home/pmonserrat/datasets/big_5/rft/splits

TWITTER AND WEIBO ARE ONE DATASET HERE. The evaluation keeps them apart
(per-platform metrics), but the fine-tune treats them as a single pool, so one
split file covers both and every downstream step reads that one file.

GROUPED BY POST, NOT BY IMAGE — the decision worth understanding
===============================================================
A BIG-5 image is one slot of a social-media post: files are named
"<platform_id>_<slot>.<ext>" and a post carries up to 4 (Twitter) or 9 (Weibo)
of them, 3.4 on average. Images within a post are frequently near-duplicates —
burst shots of one scene, frames of one event, variants of one meme template.
Splitting at IMAGE level would put slot 0 in train and slot 1 in test, and the
test score would then partly measure memorization of a training image. So the
unit of assignment is the POST: every image of a post lands in the same split.

The cost is granularity — post sizes vary, so exact 70/10/20 image counts are
not reachable while keeping posts intact. The assignment below gets close by
placing each post (largest first) into whichever split is currently furthest
below its target share of IMAGES, which keeps the image-level proportions near
target rather than only the post-level ones.

STRATIFIED BY PLATFORM AND BY THE POST'S NATURE COMPOSITION, because the
nature/non-nature balance differs sharply between the two platforms and a
plain random split can drift several points on a dataset this size — which
would then be indistinguishable from a real effect of fine-tuning.

The split is a property of the DATA, not of any model: it is derived from
image paths and ground-truth labels only. Any artifact of any model over the
same images produces the same split, so one split file is valid for the
gemma-12B self-training run and for every future distillation run alike. The
artifacts are read here purely as the listing of which annotated images exist.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

from rft_common import group_key, read_artifact

SPLITS = ("train", "val", "test")


def platform_of(artifact_path: str, header: Dict) -> str:
    """The platform an artifact covers, taken from its own header (`dataset`
    is "big5_twitter"/"big5_weibo") and falling back to the path."""
    ds = str(header.get("dataset") or "")
    if ds.startswith("big5_"):
        return ds[len("big5_"):]
    low = artifact_path.lower()
    for name in ("twitter", "weibo"):
        if name in low:
            return name
    return "unknown"


def nature_bucket(n_nature: int, n_images: int) -> str:
    """A post's stratum by how much of it is annotated nature. Three buckets,
    not a continuous ratio: with 1968 posts, finer strata would leave some
    with too few posts to divide 70/10/20 meaningfully."""
    if n_nature == 0:
        return "no_nature"
    if n_nature == n_images:
        return "all_nature"
    return "mixed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", action="append", required=True,
                    help="vlm_responses_*.jsonl to read the image listing + GT from. "
                         "Repeat once per platform (Twitter and Weibo).")
    ap.add_argument("--out", required=True,
                    help="Output DIRECTORY. Writes splits.json, a per-split image-path "
                         "list (<split>_images.txt) for --split_file, and summary.json.")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.10, 0.20),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        ap.error(f"--ratios must sum to 1.0, got {args.ratios} summing to {sum(args.ratios)}")

    # ---- Collect every annotated image, grouped by post -------------------
    posts: Dict[str, Dict] = {}
    for path in args.artifact:
        header, records = read_artifact(path)
        platform = platform_of(path, header)
        for rec in records:
            targets = rec.get("targets") or []
            gt_nature = (targets[0] if targets else {}).get("gt_nature")
            if gt_nature is None:
                continue  # unannotated slot — not part of any split
            key = group_key(rec["image_path"], platform)
            post = posts.setdefault(key, {"platform": platform, "images": [], "n_nature": 0})
            post["images"].append(rec["image_path"])
            post["n_nature"] += int(bool(gt_nature))

    if not posts:
        raise SystemExit("No annotated images found in the given artifacts.")

    # ---- Stratify, then assign whole posts --------------------------------
    strata: Dict[tuple, List[str]] = defaultdict(list)
    for key, post in posts.items():
        strata[(post["platform"], nature_bucket(post["n_nature"], len(post["images"])))].append(key)

    rng = random.Random(args.seed)
    ratios = dict(zip(SPLITS, args.ratios))
    assignment: Dict[str, str] = {}
    # Running image counts per split, accumulated ACROSS strata so that a
    # stratum too small to divide cleanly on its own is compensated by the
    # next one rather than skewing the total.
    placed = {s: 0 for s in SPLITS}

    for stratum in sorted(strata):
        keys = strata[stratum]
        rng.shuffle(keys)
        # Largest posts first: placing a 9-image post after the small ones
        # would overshoot whichever split happened to be last to fill.
        keys.sort(key=lambda k: len(posts[k]["images"]), reverse=True)
        for key in keys:
            n_img = len(posts[key]["images"])
            total = sum(placed.values()) + n_img
            # Deficit = how many images this split still owes relative to its
            # target share of everything placed so far. Most-owed split wins.
            split = max(SPLITS, key=lambda s: ratios[s] * total - placed[s])
            assignment[key] = split
            placed[split] += n_img

    # ---- Write ------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    image_split: Dict[str, str] = {}
    per_split_images: Dict[str, List[str]] = {s: [] for s in SPLITS}
    for key, post in posts.items():
        split = assignment[key]
        for img in sorted(post["images"]):
            image_split[img] = split
            per_split_images[split].append(img)

    splits_path = os.path.join(args.out, "splits.json")
    with open(splits_path, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": args.seed,
            "ratios": dict(zip(SPLITS, args.ratios)),
            "grouped_by": "post (<platform_id>_<slot> filename stem)",
            "artifacts": args.artifact,
            "post_split": assignment,
            "image_split": image_split,
        }, fh, indent=1)

    for split in SPLITS:
        with open(os.path.join(args.out, f"{split}_images.txt"), "w", encoding="utf-8") as fh:
            for img in sorted(per_split_images[split]):
                fh.write(img + "\n")

    # ---- Summarize --------------------------------------------------------
    summary = {}
    n_images_total = len(image_split)
    for split in SPLITS:
        imgs = per_split_images[split]
        n_posts = sum(1 for k, s in assignment.items() if s == split)
        by_platform = Counter()
        n_nature = 0
        for key, post in posts.items():
            if assignment[key] != split:
                continue
            by_platform[post["platform"]] += len(post["images"])
            n_nature += post["n_nature"]
        summary[split] = {
            "images": len(imgs),
            "image_share": round(len(imgs) / n_images_total, 4),
            "posts": n_posts,
            "gt_nature": n_nature,
            "gt_nature_share": round(n_nature / len(imgs), 4) if imgs else 0.0,
            "by_platform": dict(by_platform),
        }
    summary["_totals"] = {"images": n_images_total, "posts": len(posts)}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print(f"Wrote {splits_path}")
    for split in SPLITS:
        s = summary[split]
        print(f"  {split:5s}  {s['images']:5d} images ({s['image_share']:.1%})  "
              f"{s['posts']:4d} posts  nature {s['gt_nature_share']:.1%}  {s['by_platform']}")


if __name__ == "__main__":
    main()
