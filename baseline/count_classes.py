#!/usr/bin/env python3
"""
Calculate ground-truth class mappings for ImageNet, COCO, and Places365 against
the BIG-5 nature taxonomy.

This script loads the taxonomy from the provided Excel file and evaluates how
many classes in each dataset map to:
  - Nature / No-Nature / Unmapped
  - Biotic / Abiotic
  - Material / Immaterial

The results are printed to the console and saved to a text file.

WHAT IS THIS SCRIPT FOR? It's a pure DIAGNOSTIC/COVERAGE tool — it doesn't run
any model or evaluate any predictions. It just answers "of ImageNet's 1000
classes (or COCO's 80, or Places365's 365), how many actually resolve to a
taxonomy label at all, and how do they split across nature/biotic/material?"
This is useful context BEFORE running the actual closed-set or VLM
evaluations, since it tells you what fraction of each dataset's "mapped
subset" (the convention used throughout this project) you can expect.

SELF-CONTAINED ON PURPOSE: this used to import TaxonomyLookup/safe_binary_map/
COCO_TO_WNSYNSET from baseline.common, but that module's shape changed on the
cluster (TaxonomyLookup no longer exists there) and broke this script. Rather
than re-couple to common.py's current shape (unknown, and liable to keep
changing out from under a diagnostic script nobody else depends on), the
small pieces this script actually needs are reimplemented locally below:
  - _TaxonomyAnnotations: a DIRECT-only annotation lookup (no ancestor/
    descendant hop resolution) parsed straight from the Excel, matching the
    same row format `src/loaders/excel_loader.py`'s TaxonomyGraph uses.
  - _safe_binary_map: normalizes a free-text annotation cell to 1/0/None.
  - _COCO_TO_WNSYNSET: copied verbatim from `src/loaders/dataset_loader.py`.
This file now has no import-time dependency on any other project module.
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from torchvision import datasets

try:
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# Matches a WordNet synset id string like "golden_retriever.n.01".
_SYNSET_PATTERN = re.compile(r"([\w\-']+\.[nvasr]\.[0-9]+)")


class _TaxonomyAnnotations:
    """DIRECT-only lookup of the Excel's own hand-labeled synsets — a class
    only counts here if it was annotated on its own row, never inherited from
    an ancestor/descendant (unlike TaxonomyGraph.resolve_labels elsewhere in
    the project, which this script deliberately does NOT use)."""

    def __init__(self):
        self._nodes = {}  # synset_str -> {"is_nature", "biotic_abiotic", "material_immaterial"}

    def load(self, excel_path, bio_col="Biotic/abiotic", mat_col="Material/immaterial",
              sheet_name="data corrected"):
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        for index, row in df.iterrows():
            bio_val = row[bio_col]
            mat_val = row[mat_col]
            incoming_bio = str(bio_val).strip().lower() if pd.notna(bio_val) and str(bio_val).strip() != "" else None
            incoming_mat = str(mat_val).strip().lower() if pd.notna(mat_val) and str(mat_val).strip() != "" else None
            is_nature = incoming_mat is not None

            raw_synset = None
            for val in row.drop(labels=[bio_col, mat_col]):
                if pd.isna(val) or str(val).strip() == "":
                    break
                raw_synset = str(val).strip()
            if not raw_synset:
                continue

            match = _SYNSET_PATTERN.search(raw_synset)
            if not match:
                continue
            synset_str = match.group(1)

            entry = self._nodes.setdefault(synset_str, {})
            entry["is_nature"] = is_nature
            if incoming_bio:
                entry["biotic_abiotic"] = incoming_bio
            if incoming_mat:
                entry["material_immaterial"] = incoming_mat

    def get_node_attributes(self, synset_str):
        return self._nodes.get(synset_str)

    def get_synset_str_from_wnid(self, wnid):
        """Converts an ImageFolder-style WNID (e.g. 'n02124278') into its
        WordNet synset string (e.g. 'leopard.n.01')."""
        try:
            return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:])).name()
        except Exception:
            return None


def _safe_binary_map(value, positive_label, negative_label):
    """Normalizes a free-text annotation cell into 1 (positive_label), 0
    (negative_label), or None (blank/unrecognized)."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == positive_label:
        return 1
    if normalized == negative_label:
        return 0
    return None


# Copied verbatim from src/loaders/dataset_loader.py's COCO_TO_WNSYNSET (COCO's
# 80 category ids -> the WordNet synset that best represents each).
_COCO_TO_WNSYNSET = {
    1: 'person.n.01', 2: 'bicycle.n.01', 3: 'car.n.01', 4: 'motorcycle.n.01', 5: 'airplane.n.01',
    6: 'bus.n.01', 7: 'train.n.01', 8: 'truck.n.01', 9: 'boat.n.01', 10: 'traffic_light.n.01',
    11: 'fireplug.n.01', 13: 'street_sign.n.01', 14: 'parking_meter.n.01', 15: 'bench.n.01',
    16: 'bird.n.01', 17: 'cat.n.01', 18: 'dog.n.01', 19: 'horse.n.01', 20: 'sheep.n.01',
    21: 'cow.n.01', 22: 'elephant.n.01', 23: 'bear.n.01', 24: 'zebra.n.01', 25: 'giraffe.n.01',
    27: 'backpack.n.01', 28: 'umbrella.n.01', 31: 'bag.n.04', 32: 'necktie.n.01', 33: 'bag.n.06',
    34: 'frisbee.n.01', 35: 'ski.n.01', 36: 'snowboard.n.01', 37: 'ball.n.01', 38: 'kite.n.03',
    39: 'baseball_bat.n.01', 40: 'baseball_glove.n.01', 41: 'skateboard.n.01', 42: 'surfboard.n.01',
    43: 'tennis_racket.n.01', 44: 'bottle.n.01', 46: 'wineglass.n.01', 47: 'cup.n.01', 48: 'fork.n.01',
    49: 'knife.n.01', 50: 'spoon.n.01', 51: 'bowl.n.01', 52: 'banana.n.02', 53: 'apple.n.01',
    54: 'sandwich.n.01', 55: 'orange.n.01', 56: 'broccoli.n.02', 57: 'carrot.n.01', 58: 'hotdog.n.02',
    59: 'pizza.n.01', 60: 'doughnut.n.02', 61: 'cake.n.03', 62: 'chair.n.01', 63: 'sofa.n.01',
    64: 'pot_plant.n.01', 65: 'bed.n.01', 67: 'dining_table.n.01', 70: 'toilet.n.02',
    72: 'television_receiver.n.01', 73: 'laptop.n.01', 74: 'mouse.n.04', 75: 'remote_control.n.01',
    76: 'computer_keyboard.n.01', 77: 'cellular_telephone.n.01', 78: 'microwave.n.02', 79: 'oven.n.01',
    80: 'toaster.n.02', 81: 'sink.n.01', 82: 'refrigerator.n.01', 84: 'book.n.02', 85: 'clock.n.01',
    86: 'vase.n.01', 87: 'scissors.n.01', 88: 'teddy.n.01', 89: 'hand_blower.n.01', 90: 'toothbrush.n.01',
}

# Accumulates every dataset's stats dict (keyed by the same lowercase name
# used in --dataset), populated as each process_* function runs, so main()
# can dump one combined JSON at the end without re-deriving anything.
ALL_STATS = {}

def log_statistics(dataset_name, total_classes, stats, output_file=None):
    """Formats the statistics, prints them to the console, and appends to a text file."""
    ALL_STATS[dataset_name.lower()] = {"total_classes": total_classes, **stats}
    output_str = (
        f"\n{'=' * 60}\n"
        f"📊 TAXONOMY MAPPING STATISTICS: {dataset_name.upper()}\n"
        f"{'=' * 60}\n"
        f"Total Dataset Classes:   {total_classes}\n"
        f"{'-' * 60}\n"
        f"--- Top-Level Split ---\n"
        f"  Nature:                {stats['nature']}\n"
        f"  No-Nature:             {stats['no_nature']}\n"
        f"  Unmapped:              {stats['unmapped']}\n\n"
        f"--- Sub-Categories (Nature Branch) ---\n"
        f"  Biotic:                {stats['biotic']}\n"
        f"  Abiotic:               {stats['abiotic']}\n"
        f"  Material:              {stats['material']}\n"
        f"  Immaterial:            {stats['immaterial']}\n"
        f"{'=' * 60}\n"
    )

    # Print to console
    print(output_str)

    # Write to file if specified
    # Opened in "append" mode ("a") rather than "write" ("w") — each dataset's
    # stats are appended one after another into the SAME growing report file,
    # rather than each call overwriting the previous dataset's results.
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(output_str)

# ============================================================================
# DATASET PROCESSORS
# ============================================================================
# Each `process_*` function below follows the identical pattern: build a
# {id: class_name} table for the dataset, resolve each class to a WordNet
# synset, look up that synset's taxonomy label via the taxonomy, and tally
# how many classes fall into each nature/biotic/material bucket (or are
# unmapped entirely).

def process_imagenet(taxonomy, imagenet_dir, output_file):
    """Tally taxonomy coverage across every class folder in an ImageNet-style
    directory (ImageFolder layout: one subdirectory per class, named by its
    WordNet id, e.g. "n02124278")."""
    print(f"[INFO] Extracting ImageNet classes from {imagenet_dir}...")
    # torchvision's ImageFolder auto-discovers class subdirectories; we only
    # need the folder-name <-> class-index mapping here, not the actual images.
    full_dataset = datasets.ImageFolder(imagenet_dir)
    idx_to_wnid = {v: k for k, v in full_dataset.class_to_idx.items()}

    stats = {"nature": 0, "no_nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}

    for idx, wnid in idx_to_wnid.items():
        # Convert the WordNet-id-shaped folder name (e.g. "n02124278") into
        # the actual synset string (e.g. "leopard.n.01").
        synset_str = taxonomy.get_synset_str_from_wnid(wnid)
        # Look up whatever taxonomy attributes (if any) were recorded on this
        # exact synset node — note this does NOT walk up to ancestors the way
        # `TaxonomyGraph.resolve_labels` does elsewhere in the project; a
        # class only counts here if it was DIRECTLY annotated, or has
        # otherwise become a graph node with these attributes set.
        node_attrs = taxonomy.get_node_attributes(synset_str)

        if not node_attrs:
            stats["unmapped"] += 1
            continue

        is_nature = node_attrs.get('is_nature')
        if is_nature:
            stats["nature"] += 1
        else:
            stats["no_nature"] += 1

        # Biotic / Abiotic
        bio_val = node_attrs.get('biotic_abiotic')
        bio_bin = _safe_binary_map(bio_val, "biotic", "abiotic")
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        # Material / Immaterial
        mat_val = node_attrs.get('material_immaterial')
        mat_bin = _safe_binary_map(mat_val, "material", "immaterial")
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    log_statistics("ImageNet", len(idx_to_wnid), stats, output_file)

def process_coco(taxonomy, instances_json, output_file):
    """Tally taxonomy coverage across COCO's fixed 80 (or fewer, if a subset
    JSON is given) object categories."""
    print(f"[INFO] Extracting COCO classes from {instances_json}...")
    with open(instances_json, "r") as f:
        data = json.load(f)

    # COCO's instances_*.json always has a top-level "categories" list
    # describing every category id/name pair used in this annotation file.
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    stats = {"nature": 0, "no_nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}

    for cid, name in id_to_name.items():
        synset_str = _COCO_TO_WNSYNSET.get(cid)
        node_attrs = taxonomy.get_node_attributes(synset_str) if synset_str else None

        if not node_attrs:
            stats["unmapped"] += 1
            continue

        is_nature = node_attrs.get('is_nature')
        if is_nature:
            stats["nature"] += 1
        else:
            stats["no_nature"] += 1

        bio_bin = _safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        mat_bin = _safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    log_statistics("COCO", len(id_to_name), stats, output_file)

def process_coco_dense(taxonomy, instances_json, output_file=None):
    """COCO's row for the thesis's DENSE ground-truth table (paired with
    BIG-5-Dense), NOT another view of `process_coco`'s class-coverage table
    above — different question, different granularity.

    SCOPE: restricted to COCO's NATURE-mapped classes ONLY (of the 80 total,
    the same subset `process_coco` reports as "Nature Yes" — 21 at the time
    of writing; this function computes the count itself rather than
    hardcoding it, so a taxonomy update can't silently go stale here). A
    non-nature class (car, laptop, ...) is not a "dense nature entity" any
    more than an unmapped BIG-5-Dense object would be.

    GRANULARITY: SEMANTIC-SEGMENTATION style, matching how the VLM+grounding
    pipeline actually evaluates COCO elsewhere in this project (SAM3's
    semantic head; `run_vlm_pipeline.py`'s `score_image_entities`/
    `gt_by_class`, which merges every instance of ONE class within an image
    into a SINGLE region before scoring). Every same-class instance in one
    image is therefore counted as ONE entity here too — a bowl of 10 oranges
    is one "orange" entity, not 10 — so this table's "No. of Entities" answers
    "how many semantic masks would this image produce", the same quantity the
    pipeline is actually scored against, NOT a raw instance count.

    `iscrowd` annotations are excluded throughout (COCO's own convention: a
    crowd region stands in for an unknown-sized group, not one localized
    entity — the same exclusion the real detection evaluation applies).
    """
    print(f"[INFO] Extracting COCO dense (semantic) nature-entity stats from {instances_json}...")
    with open(instances_json, "r") as f:
        data = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    # Which of COCO's 80 categories are nature, via the SAME direct-lookup
    # taxonomy resolution process_coco uses above (so this can never disagree
    # with the class-coverage table's own "Nature Yes" count).
    nature_cats = {}  # category_id -> {"class_name", "bio_bin", "mat_bin"}
    n_bio_unresolved = n_mat_unresolved = 0
    for cid, name in id_to_name.items():
        synset_str = _COCO_TO_WNSYNSET.get(cid)
        node_attrs = taxonomy.get_node_attributes(synset_str) if synset_str else None
        if not node_attrs or not node_attrs.get("is_nature"):
            continue
        bio_bin = _safe_binary_map(node_attrs.get("biotic_abiotic"), "biotic", "abiotic")
        mat_bin = _safe_binary_map(node_attrs.get("material_immaterial"), "material", "immaterial")
        if bio_bin is None:
            n_bio_unresolved += 1
        if mat_bin is None:
            n_mat_unresolved += 1
        nature_cats[cid] = {"class_name": name, "bio_bin": bio_bin, "mat_bin": mat_bin}

    # image_id -> set of nature category_ids present (non-crowd). This IS the
    # "join different masks corresponding to different instances of the same
    # class into a unified semantic mask" step: the set, not a per-instance
    # list, is what turns repeated instances of one class into one entity.
    entities_per_image = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        cid = ann["category_id"]
        if cid not in nature_cats:
            continue
        entities_per_image.setdefault(ann["image_id"], set()).add(cid)

    per_image_counts = [len(cats) for cats in entities_per_image.values() if cats]
    all_entity_cids = [cid for cats in entities_per_image.values() for cid in cats]
    arr = np.array(per_image_counts) if per_image_counts else np.zeros(1)

    n_images = len(per_image_counts)
    n_entities = len(all_entity_cids)
    distinct_classes = len({cid for cats in entities_per_image.values() for cid in cats})
    biotic = sum(1 for cid in all_entity_cids if nature_cats[cid]["bio_bin"] == 1)
    abiotic = sum(1 for cid in all_entity_cids if nature_cats[cid]["bio_bin"] == 0)
    material = sum(1 for cid in all_entity_cids if nature_cats[cid]["mat_bin"] == 1)
    immaterial = sum(1 for cid in all_entity_cids if nature_cats[cid]["mat_bin"] == 0)

    output_str = (
        f"\n{'=' * 60}\n"
        f"📊 COCO DENSE (SEMANTIC) NATURE-ENTITY STATISTICS\n"
        f"{'=' * 60}\n"
        f"Nature classes in scope:     {len(nature_cats)} of 80\n"
        f"No. of Images:                {n_images}\n"
        f"No. of Entities:              {n_entities}\n"
        f"Entities/Image (mean +/- population std): "
        f"{arr.mean():.2f} +/- {arr.std():.2f}  (min={int(arr.min())}, max={int(arr.max())})\n"
        f"Distinct Classes (present):   {distinct_classes}\n"
        f"Biotic:                       {biotic}\n"
        f"Abiotic:                      {abiotic}\n"
        f"Material:                     {material}\n"
        f"Immaterial:                   {immaterial}\n"
        f"{'=' * 60}\n"
    )
    if n_bio_unresolved or n_mat_unresolved:
        output_str += (
            f"NOTE: {n_bio_unresolved} nature class(es) have no direct biotic/abiotic "
            f"annotation and {n_mat_unresolved} no direct material/immaterial "
            f"annotation on the Excel — Biotic+Abiotic and/or Material+Immaterial "
            f"will NOT sum to No. of Entities as a result (unlike BIG-5-Dense, "
            f"where both axes are mandatory per entity).\n"
        )

    print(output_str)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(output_str)

    ALL_STATS["coco_dense"] = {
        "nature_classes": len(nature_cats), "n_images": n_images, "n_entities": n_entities,
        "entities_per_image_mean": float(arr.mean()), "entities_per_image_std": float(arr.std()),
        "distinct_classes": distinct_classes, "biotic": biotic, "abiotic": abiotic,
        "material": material, "immaterial": immaterial,
    }


def process_places(taxonomy, excel_path, categories_txt, sourcekey_sheet, missing_sheet, output_file):
    """Tally taxonomy coverage across Places365's ~365 scene categories.
    Unlike ImageNet/COCO, Places365 category names have no built-in WordNet
    id — this function reconstructs a best-effort mapping (see
    resolve_via_wordnet below), restricted to synsets the Excel confirms came
    from Places365's own source ("MIT")."""
    print(f"[INFO] Extracting Places365 classes from {categories_txt}...")

    # 1. Load Places categories
    # Parses lines like "/a/airfield 0" into {0: "airfield"} — stripping the
    # leading "/x/" alphabetical grouping prefix Places365 uses.
    id_to_name = {}
    with open(categories_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            path, idx_str = line.rsplit(" ", 1)
            name = path[3:] if path.startswith("/") and len(path) > 3 and path[2] == "/" else path.lstrip("/")
            id_to_name[int(idx_str)] = name

    # 2. Reconstruct WordNet mappings
    # The "sourcekey" sheet records which external dataset (if any) each
    # taxonomy synset is confirmed to belong to — we only want the ones
    # explicitly tagged "MIT" (Places365's origin institution), to avoid
    # matching a Places scene name to some unrelated synset that happens to
    # share the same word.
    df_source = pd.read_excel(excel_path, sheet_name=sourcekey_sheet, header=0)
    taxonomy_synsets = set()
    for _, row in df_source.iterrows():
        raw = row.iloc[0]
        if pd.isna(raw) or not str(raw).strip(): continue
        synset = str(raw).strip().split(' ')[0]
        source = "" if pd.isna(row.iloc[1]) else str(row.iloc[1]).strip()
        if 'MIT' in source: taxonomy_synsets.add(synset)

    # This sheet lists Places category names already manually confirmed to
    # have NO usable taxonomy synset at all — skip them outright rather than
    # let the heuristic resolver below guess something wrong.
    df_missing = pd.read_excel(excel_path, sheet_name=missing_sheet, header=None)
    exclusion = {str(val).strip() for val in df_missing.iloc[:, 0] if pd.notna(val) and str(val).strip()}

    from nltk.corpus import wordnet as wn
    def resolve_via_wordnet(cls, tax_synsets):
        """Try a few different readings of a Places category name (the whole
        thing, just its first segment, just its last segment) against
        WordNet, accepting the first noun sense that happens to be in our
        restricted MIT-tagged synset set."""
        base = cls.replace('/', '_')
        head = cls.split('/')[0]
        candidates = [base, head]
        if '/' in cls: candidates.append(cls.split('/')[-1])
        seen = set()
        for c in candidates:
            key = c.replace(' ', '_')
            if key in seen: continue
            seen.add(key)
            for s in wn.synsets(key, pos='n'):
                if s.name() in tax_synsets: return s.name()
        return None

    stats = {"nature": 0, "no_nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}

    for cid, name in id_to_name.items():
        synset_str = None if name in exclusion else resolve_via_wordnet(name, taxonomy_synsets)
        node_attrs = taxonomy.get_node_attributes(synset_str) if synset_str else None

        if not node_attrs:
            stats["unmapped"] += 1
            continue

        is_nature = node_attrs.get('is_nature')
        if is_nature:
            stats["nature"] += 1
        else:
            stats["no_nature"] += 1

        bio_bin = _safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        mat_bin = _safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    log_statistics("Places365", len(id_to_name), stats, output_file)

# ============================================================================
# MAIN SCRIPT
# ============================================================================
def parse_args():
    """Command-line flags: which taxonomy Excel/sheet to use, which
    dataset(s) to process, and where each dataset's files live."""
    parser = argparse.ArgumentParser(description="Extract taxonomy class statistics for ImageNet, COCO, and Places365")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx",
                        help="Path to the taxonomy workbook.")
    parser.add_argument("--dataset", type=str,
                        choices=["imagenet", "coco", "coco_dense", "places", "all"], required=True,
                        help="Which dataset to process. 'coco_dense' is the "
                             "semantic-segmentation-style dense nature-entity table "
                             "(paired with BIG-5-Dense), NOT part of 'all' — a different "
                             "question from 'coco's class-coverage table, opt-in only.")

    # Dataset specific paths
    parser.add_argument("--imagenet_dir", type=str, default=None,
                        help="Path to ImageNet validation split (required if dataset is 'imagenet' or 'all').")
    parser.add_argument("--coco_instances_json", type=str, default=None,
                        help="Path to COCO instances_val2017.json (required if dataset is 'coco' or 'all').")
    parser.add_argument("--places_categories_txt", type=str, default=None,
                        help="Path to categories_places365.txt (required if dataset is 'places' or 'all').")

    # Places specific sheets
    parser.add_argument("--places_sourcekey_sheet", type=str, default="sourcekey",
                        help="Sheet in --excel_path for Places WordNet resolution.")
    parser.add_argument("--places_missing_sheet", type=str, default="still missing MIT Places",
                        help="Sheet in --excel_path listing unmapped Places classes.")

    # Output text file (optional — omit to only print to the console/log).
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to also save the output statistics as a text file.")

    return parser.parse_args()

DISPLAY_NAMES = {"imagenet": "ImageNet", "coco": "COCO", "places365": "Places365"}
TABLE_COLUMNS = [
    ("total_classes", "Total Classes"), ("unmapped", "Unmapped"),
    ("nature", "Nature Yes"), ("no_nature", "Nature No"),
    ("biotic", "Biotic"), ("abiotic", "Abiotic"),
    ("material", "Material"), ("immaterial", "Immaterial"),
]

def print_combined_table(all_stats):
    """Prints every processed dataset's stats as one Markdown table, plus a
    Combined row summing across datasets, straight to the console/log.

    EXCLUDES "coco_dense": it shares the "biotic"/"abiotic"/"material"/
    "immaterial" key names with the class-coverage stats dicts by
    coincidence, but counts a completely different thing (semantic ENTITIES,
    not CLASSES) — folding it in here would silently corrupt the Combined
    row's sum, mixing two incompatible units into one number.
    """
    all_stats = {k: v for k, v in all_stats.items() if k != "coco_dense"}
    if not all_stats:
        return
    combined = {key: sum(s.get(key, 0) for s in all_stats.values()) for key, _ in TABLE_COLUMNS}
    rows = [(DISPLAY_NAMES.get(name, name), stats) for name, stats in all_stats.items()]
    rows.append(("Combined", combined))

    header = ["Dataset"] + [label for _, label in TABLE_COLUMNS]
    lines = ["\n" + " | ".join(header), "-" * 60]
    for name, stats in rows:
        lines.append(" | ".join([name] + [str(stats.get(key, 0)) for key, _ in TABLE_COLUMNS]))
    print("\n".join(lines))

def main():
    args = parse_args()

    print(f"[INFO] Initializing Taxonomy Pipeline from {args.excel_path}...")
    taxonomy = _TaxonomyAnnotations()
    taxonomy.load(args.excel_path)

    # Clear/Initialize the output text file if it's going to be used
    # Opened in "write" mode ("w") here specifically to RESET the file at the
    # start of the run (each `log_statistics` call above then APPENDS to it),
    # so re-running this script doesn't just keep growing a stale file
    # forever with old results mixed in.
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("TAXONOMY MAPPING STATISTICS REPORT\n")
            f.write(f"Source Excel: {args.excel_path}\n")

    if args.dataset in ["imagenet", "all"]:
        if not args.imagenet_dir:
            raise ValueError("Error: --imagenet_dir is required to process ImageNet.")
        process_imagenet(taxonomy, args.imagenet_dir, args.output_file)

    if args.dataset in ["coco", "all"]:
        if not args.coco_instances_json:
            raise ValueError("Error: --coco_instances_json is required to process COCO.")
        process_coco(taxonomy, args.coco_instances_json, args.output_file)

    if args.dataset == "coco_dense":
        if not args.coco_instances_json:
            raise ValueError("Error: --coco_instances_json is required to process COCO.")
        process_coco_dense(taxonomy, args.coco_instances_json, args.output_file)

    if args.dataset in ["places", "all"]:
        if not args.places_categories_txt:
            raise ValueError("Error: --places_categories_txt is required to process Places365.")
        process_places(taxonomy, args.excel_path, args.places_categories_txt,
                       args.places_sourcekey_sheet, args.places_missing_sheet, args.output_file)

    if args.output_file:
        print(f"\n💾 Statistics successfully saved to {args.output_file}")

    print_combined_table(ALL_STATS)

if __name__ == "__main__":
    main()
