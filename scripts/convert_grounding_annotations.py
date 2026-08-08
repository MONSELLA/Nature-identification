#!/usr/bin/env python3
"""Convert the hand-drawn BIG-5 grounding annotations into the representation
the grounding evaluation actually scores against.

The annotation tool writes one JSON per image with polygons in NORMALIZED
[0,1] coordinates, an outer/hole flag per polygon, and free-text + WordNet
labels per object. The grounding pipeline, meanwhile, stores its predictions
as pycocotools RLE at native image resolution (src/grounding_pipeline.py's
`encode_rle`). This script closes that gap ONCE, offline, so the scorer only
ever deals with RLE-vs-RLE.

What it does, per image:
  1. Skips any image with a `skip_reason` (annotator could not label it) —
     those are excluded from the evaluation entirely, not counted as
     no-nature (see the n=170 decision).
  2. Reads the image's real WxH (PIL header read only, no decode).
  3. Denormalizes every polygon and rasterizes it, subtracting HOLE polygons
     from the outer union. pycocotools' own `frPyObjects` UNIONS every polygon
     it is given, so holes must be handled explicitly here or they'd be
     silently filled.
  4. Merges objects that share the same (label, synset, axis labels) into ONE
     concept mask, because SAM3's `semantic_seg` is concept-level: prompting
     "flower" returns a single mask covering every flower, so an image with
     4 separately-drawn flower objects must become one GT flower mask.
  5. Emits RLE + COCO-style abs-xywh bbox + area.

It also materializes the no-nature half of the evaluation set: images the
majority-vote annotators marked `nature=No`, whose GT is the EMPTY mask set
(treated as absolute — any grounded nature mask there is a false positive).
These are sampled to match the nature half's size and platform composition.

Outputs, under grounding_annotations/:
  raw/                        verbatim copy of the annotator's own files
  processed/nature/*.json     per-image RLE GT (evaluation input)
  processed/no_nature/*.json  per-image empty GT (evaluation input)
  processed/manifest.json     the frozen evaluation split + conversion stats
  processed/coco_instances.json  portable COCO-format GT (not used by the
                              scorer — COCOeval is the wrong tool here, no
                              confidence scores exist — but it makes the
                              annotations reusable outside this project)
"""

import argparse
import csv
import glob
import json
import os
import random
import shutil
import sys
import unicodedata
from collections import Counter, OrderedDict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# =============================================================================
# Paths
# =============================================================================
DEFAULT_ROOT = "/home/pau/Documentos/Universidad/Master/TFM/BIG_5_Datasets"


def subset_dirs(root):
    """The subset folders that hold both the images and the majority-vote CSV.

    Subset_3_proportional is deliberately EXCLUDED: none of its images were
    grounding-annotated, so it stays untouched as a potential held-out set.
    """
    return [
        (os.path.join(root, "Subset_1", "images"),
         os.path.join(root, "Subset_1", "annotations_subset_1.csv")),
        (os.path.join(root, "Subset_2", "images"),
         os.path.join(root, "Subset_2", "annotations_subset_2.csv")),
    ]


# =============================================================================
# Polygon -> RLE
# =============================================================================
def _denormalize(points, width, height):
    """Flat [x,y,x,y,...] in [0,1] -> flat [x,y,...] in absolute pixels.

    Coordinates are CLAMPED to the image box: the annotation tool lets a
    stroke run a hair outside the canvas (several masks have points at
    exactly 0.0/1.0 or a fraction beyond), and pycocotools silently produces
    a garbage mask rather than erroring on out-of-range input.
    """
    out = []
    for i in range(0, len(points) - 1, 2):
        x = min(max(float(points[i]) * width, 0.0), float(width))
        y = min(max(float(points[i + 1]) * height, 0.0), float(height))
        out.extend([x, y])
    return out


def _rasterize(polygons, width, height):
    """Union of a list of flat absolute-coordinate polygons -> bool HxW array.

    Returns an all-False array when the list is empty, so the caller can
    subtract an empty hole set without special-casing it.
    """
    valid = [p for p in polygons if len(p) >= 6]  # need >=3 points to have area
    if not valid:
        return np.zeros((height, width), dtype=bool)
    rles = mask_utils.frPyObjects(valid, height, width)
    merged = mask_utils.merge(rles)  # frPyObjects returns one RLE per polygon
    return mask_utils.decode(merged).astype(bool)


def object_mask(obj, width, height):
    """Rasterize one annotated object: outer polygons MINUS hole polygons."""
    outer, holes = [], []
    for poly in obj.get("mask") or []:
        pts = _denormalize(poly.get("points") or [], width, height)
        (holes if poly.get("hole") else outer).append(pts)
    mask = _rasterize(outer, width, height)
    if holes:
        # `& ~holes` rather than a second merge: pycocotools has no "subtract".
        mask = mask & ~_rasterize(holes, width, height)
    return mask


def encode_rle(mask):
    """Bool HxW -> JSON-safe pycocotools RLE.

    Mirrors src/grounding_pipeline.py's own encode_rle exactly (Fortran order,
    `counts` decoded from bytes to str) so GT and predictions are byte-level
    comparable by the same pycocotools calls.
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# =============================================================================
# Concept merging
# =============================================================================
def _norm_label(text):
    """Lowercase + strip diacritics, matching src/vlm_pipeline._normalize_object
    so a GT "café" and a predicted "cafe" group the same way."""
    text = unicodedata.normalize("NFKD", (text or "").strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def merge_key(obj):
    """Objects merge into one concept mask only if they agree on EVERYTHING
    that gets scored — label, synset, and all three taxonomy axes.

    Deliberately conservative: two objects both labeled "tree" but annotated
    with different material values are a real distinction (a real tree and a
    drawn one), so they stay separate rather than being averaged away.
    """
    return (
        _norm_label(obj.get("entity_label") or obj.get("entity_label_freetext")),
        obj.get("wordnet_synset_id"),
        bool(obj.get("nature")),
        obj.get("biotic"),
        obj.get("material"),
    )


# =============================================================================
# Conversion
# =============================================================================
def convert_image(record, image_path, stats):
    """One annotator JSON -> one processed GT record (or None if skipped)."""
    if record.get("skip_reason"):
        stats["skipped"] += 1
        return None

    with Image.open(image_path) as im:  # header read only, no pixel decode
        width, height = im.size

    groups = OrderedDict()
    for obj in record.get("objects") or []:
        key = merge_key(obj)
        groups.setdefault(key, []).append(obj)

    out_objects = []
    for key, members in groups.items():
        label, synset, nature, biotic, material = key

        mask = np.zeros((height, width), dtype=bool)
        for obj in members:
            mask |= object_mask(obj, width, height)

        area = int(mask.sum())
        if area == 0:
            # A mask that rasterizes to nothing cannot be matched or scored;
            # dropping it silently would hide an annotation problem, so it is
            # counted and reported rather than ignored.
            stats["empty_masks"] += 1
            stats["empty_mask_detail"].append(
                {"image_id": record.get("image_id"), "entity_label": label})
            continue

        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        out_objects.append({
            "gt_id": f"{record.get('image_id')}::{len(out_objects)}",
            "entity_label": members[0].get("entity_label") or "",
            "entity_label_freetext": members[0].get("entity_label_freetext") or "",
            "wordnet_synset_id": synset,
            "in_taxonomy": bool(members[0].get("in_taxonomy")),
            "taxonomy_node": members[0].get("taxonomy_node"),
            "unmapped_from_wordnet": bool(members[0].get("unmapped_from_wordnet")),
            "ambiguous": any(m.get("ambiguous") for m in members),
            "nature": nature,
            "biotic": biotic,
            "material": material,
            "n_source_objects": len(members),   # >1 == concept-merged
            "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],  # COCO abs xywh
            "area": area,
            "segmentation": encode_rle(mask),
        })
        if len(members) > 1:
            stats["merged_groups"] += 1

    stats["converted"] += 1
    stats["objects_out"] += len(out_objects)
    return {
        "image_id": record.get("image_id"),
        "width": width,
        "height": height,
        "split": "nature",
        "nature_present": bool(record.get("nature_present")),
        "annotator_id": record.get("annotator_id"),
        "objects": out_objects,
    }


# =============================================================================
# No-nature half of the evaluation set
# =============================================================================
def load_subset_rows(root):
    """filename -> row, across Subset_1 + Subset_2's majority-vote CSVs."""
    rows = {}
    for images_dir, csv_path in subset_dirs(root):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["_images_dir"] = images_dir
                rows[row["filename"]] = row
    return rows


def select_no_nature(rows, nature_platform_counts, seed):
    """Sample no-nature images matching the nature half's platform composition.

    Matching on platform (rather than taking the first N) keeps the false-
    positive half comparable to the true-positive half: Twitter and Weibo
    images differ enough in content that an unmatched split would confound
    platform with the nature/no-nature contrast.
    """
    pool = {}
    for fname, row in rows.items():
        if row.get("nature") == "No":
            pool.setdefault(row.get("platform"), []).append(fname)

    rng = random.Random(seed)
    selected = []
    for platform, n_wanted in sorted(nature_platform_counts.items()):
        available = sorted(pool.get(platform, []))  # sorted == seed-reproducible
        if len(available) < n_wanted:
            print(f"  WARNING: only {len(available)} no-nature {platform} images "
                  f"available, wanted {n_wanted}", file=sys.stderr)
            n_wanted = len(available)
        selected.extend(rng.sample(available, n_wanted))
    return sorted(selected)


# =============================================================================
# COCO export (portable artifact; the scorer does NOT read this)
# =============================================================================
def build_coco(nature_records, no_nature_records):
    """COCO-format instances JSON over the nature half.

    Categories are keyed by WordNet synset id — the thing the evaluation
    actually matches on — not by the free-text label. No-nature images are
    included as images with ZERO annotations, which is exactly what "absolute
    no-nature GT" means in COCO's own encoding.
    """
    synsets = sorted({o["wordnet_synset_id"] for r in nature_records
                      for o in r["objects"] if o["wordnet_synset_id"]})
    cat_id = {s: i + 1 for i, s in enumerate(synsets)}
    categories = [{"id": i, "name": s, "supercategory": "nature"}
                  for s, i in cat_id.items()]

    images, annotations, img_id = [], [], 0
    for rec in nature_records + no_nature_records:
        img_id += 1
        images.append({
            "id": img_id,
            "file_name": rec["image_id"],
            "width": rec["width"],
            "height": rec["height"],
            "split": rec["split"],
        })
        for obj in rec["objects"]:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": img_id,
                "category_id": cat_id.get(obj["wordnet_synset_id"], 0),
                "segmentation": obj["segmentation"],   # RLE, not polygons
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0,
                "entity_label": obj["entity_label"],
                "nature": obj["nature"],
                "biotic": obj["biotic"],
                "material": obj["material"],
            })
    return {
        "info": {"description": "BIG-5 grounding GT (nature + absolute no-nature)"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# =============================================================================
# Driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="BIG_5_Datasets root")
    ap.add_argument("--annotations_dir", default=None,
                    help="grounding_annotations dir (default: <root>/Annotations/grounding_annotations)")
    ap.add_argument("--source_subdir", default=None,
                    help="subfolder of annotations_dir holding the annotator's "
                         "JSONs; default auto-detects, since the annotation "
                         "tool moves them between the top level and _shared/")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the no-nature sample (frozen in the manifest)")
    ap.add_argument("--no_coco", action="store_true",
                    help="skip the portable COCO-format export")
    args = ap.parse_args()

    ann_dir = args.annotations_dir or os.path.join(
        args.root, "Annotations", "grounding_annotations")
    raw_dir = os.path.join(ann_dir, "raw")
    proc_dir = os.path.join(ann_dir, "processed")
    nat_dir = os.path.join(proc_dir, "nature")
    non_dir = os.path.join(proc_dir, "no_nature")

    # The annotation tool has been observed writing to both `_shared/` and the
    # top level, so try the candidates in order rather than assuming a layout.
    # raw/ is skipped as a source: it is THIS script's own output, and reading
    # it back would make a rerun convert a stale copy instead of the live one.
    if args.source_subdir is not None:
        candidates = [os.path.join(ann_dir, args.source_subdir)]
    else:
        candidates = [os.path.join(ann_dir, "_shared"), ann_dir]

    src_dir, src_files = None, []
    for cand in candidates:
        found = sorted(glob.glob(os.path.join(cand, "*.json")))
        if found:
            src_dir, src_files = cand, found
            break
    if not src_files:
        sys.exit(f"No annotation JSONs found under any of: {candidates}")
    print(f"source:    {src_dir}")

    # --- 1. raw/ : verbatim copy of the annotator's files -------------------
    os.makedirs(raw_dir, exist_ok=True)
    for path in src_files:
        shutil.copy2(path, os.path.join(raw_dir, os.path.basename(path)))
    print(f"raw/       {len(src_files)} annotation files copied")

    # --- 2. processed/nature/ ----------------------------------------------
    for d in (nat_dir, non_dir):
        os.makedirs(d, exist_ok=True)
    for stale in glob.glob(os.path.join(nat_dir, "*.json")) + \
                 glob.glob(os.path.join(non_dir, "*.json")):
        os.remove(stale)  # a rerun must not leave a previous split's files behind

    rows = load_subset_rows(args.root)
    stats = Counter()
    stats["empty_mask_detail"] = []
    nature_records, missing_images, skipped_ids = [], [], []

    for path in src_files:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        image_id = record.get("image_id") or os.path.basename(path)[:-5]

        row = rows.get(image_id)
        if row is None:
            missing_images.append(image_id)
            continue
        image_path = os.path.join(row["_images_dir"], image_id)
        if not os.path.exists(image_path):
            missing_images.append(image_id)
            continue

        converted = convert_image(record, image_path, stats)
        if converted is None:
            skipped_ids.append(image_id)
            continue
        converted["platform"] = row.get("platform")
        nature_records.append(converted)
        with open(os.path.join(nat_dir, image_id + ".json"), "w", encoding="utf-8") as fh:
            json.dump(converted, fh)

    print(f"processed/nature/     {len(nature_records)} images, "
          f"{stats['objects_out']} objects "
          f"({stats['merged_groups']} concept-merged, {stats['skipped']} skipped)")

    # --- 3. processed/no_nature/ -------------------------------------------
    platform_counts = Counter(r["platform"] for r in nature_records)
    selected = select_no_nature(rows, platform_counts, args.seed)

    no_nature_records = []
    for fname in selected:
        row = rows[fname]
        image_path = os.path.join(row["_images_dir"], fname)
        if not os.path.exists(image_path):
            missing_images.append(fname)
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        rec = {
            "image_id": fname,
            "width": width,
            "height": height,
            "split": "no_nature",
            "platform": row.get("platform"),
            # ABSOLUTE no-nature GT: the empty mask set. Any grounded nature
            # mask on these images is a false positive, with no area threshold.
            "nature_present": False,
            "objects": [],
        }
        no_nature_records.append(rec)
        with open(os.path.join(non_dir, fname + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh)

    print(f"processed/no_nature/  {len(no_nature_records)} images "
          f"({dict(Counter(r['platform'] for r in no_nature_records))})")

    # --- 4. manifest + COCO -------------------------------------------------
    manifest = {
        "seed": args.seed,
        "source_dir": src_dir,
        "n_nature_images": len(nature_records),
        "n_nature_objects": stats["objects_out"],
        "n_no_nature_images": len(no_nature_records),
        "n_skipped": stats["skipped"],
        "skipped_image_ids": sorted(skipped_ids),
        "nature_platform_counts": dict(platform_counts),
        "no_nature_platform_counts": dict(
            Counter(r["platform"] for r in no_nature_records)),
        "nature_image_ids": sorted(r["image_id"] for r in nature_records),
        "no_nature_image_ids": sorted(r["image_id"] for r in no_nature_records),
        "concept_merged_groups": stats["merged_groups"],
        "empty_masks_dropped": stats["empty_masks"],
        "empty_mask_detail": stats["empty_mask_detail"],
        "missing_images": sorted(missing_images),
    }
    with open(os.path.join(proc_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"processed/manifest.json written")

    if not args.no_coco:
        coco = build_coco(nature_records, no_nature_records)
        with open(os.path.join(proc_dir, "coco_instances.json"), "w", encoding="utf-8") as fh:
            json.dump(coco, fh)
        print(f"processed/coco_instances.json  {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations, "
              f"{len(coco['categories'])} synset categories")

    if missing_images:
        print(f"\nWARNING: {len(missing_images)} images could not be located: "
              f"{missing_images[:5]}", file=sys.stderr)
    if stats["empty_masks"]:
        print(f"\nWARNING: {stats['empty_masks']} object(s) rasterized to zero "
              f"pixels and were dropped: {stats['empty_mask_detail']}", file=sys.stderr)


if __name__ == "__main__":
    main()
