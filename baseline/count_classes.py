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

NOTE ON RUNNABILITY: this script imports `TaxonomyEvaluationPipeline` from a
`first_tests.evaluation` module (see the sys.path manipulation just below).
That module is not present in the current repository tree — the maintained
taxonomy resolver used elsewhere in the project now lives at
src/loaders/excel_loader.py's `TaxonomyGraph` instead (see e.g.
scripts/run_vlm_pipeline.py). This file is left as-is/commented for
reference, since it documents useful class-coverage statistics logic, but it
will raise `ModuleNotFoundError` if run against the current tree unless that
module is restored or this script is repointed at `TaxonomyGraph`.
"""

import os
import sys
import json
import argparse
import pandas as pd
from torchvision import datasets

# Add parent directory to path to import the evaluation pipeline
# Python only looks for importable modules in directories listed in
# `sys.path`. This script lives in `baseline/`, but the module it needs
# (`first_tests.evaluation`) would live one level up at the repo root — so we
# compute that parent directory and insert it at the FRONT of sys.path (if
# it isn't already there) before attempting the import below.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from first_tests.evaluation import TaxonomyEvaluationPipeline

# ============================================================================
# COCO DICTIONARY (From evaluate_coco.py)
# ============================================================================
# Hand-built mapping from COCO's numeric category ids to the WordNet synset
# that best represents each one (COCO doesn't ship its own WordNet mapping,
# unlike ImageNet whose folder names ARE WordNet ids). Kept identical to the
# same dict used by lib/dataset_loader.py's COCO_TO_WNSYNSET and the other
# baseline scripts, so every part of the project agrees on these synsets.
COCO_TO_WNSYNSET = {
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def safe_binary_map(val, positive_str, negative_str):
    """Turn a raw annotation cell value into 1 (matches positive_str), 0
    (matches negative_str), or None (missing/unrecognized/not a string at
    all) — case-insensitive comparison."""
    if not isinstance(val, str): return None
    val = val.strip().lower()
    if val == positive_str.lower(): return 1
    if val == negative_str.lower(): return 0
    return None

def log_statistics(dataset_name, total_classes, stats, output_file=None):
    """Formats the statistics, prints them to the console, and appends to a text file."""
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
# synset, look up that synset's taxonomy label via the pipeline, and tally
# how many classes fall into each nature/biotic/material bucket (or are
# unmapped entirely).

def process_imagenet(pipeline, imagenet_dir, output_file):
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
        synset_str = pipeline.get_synset_str_from_wnid(wnid)
        # Look up whatever taxonomy attributes (if any) were recorded on this
        # exact synset node — note this does NOT walk up to ancestors the way
        # `TaxonomyGraph.resolve_labels` does elsewhere in the project; a
        # class only counts here if it was DIRECTLY annotated, or has
        # otherwise become a graph node with these attributes set.
        node_attrs = pipeline.get_node_attributes(synset_str)

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
        bio_bin = safe_binary_map(bio_val, "biotic", "abiotic")
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        # Material / Immaterial
        mat_val = node_attrs.get('material_immaterial')
        mat_bin = safe_binary_map(mat_val, "material", "immaterial")
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    log_statistics("ImageNet", len(idx_to_wnid), stats, output_file)

def process_coco(pipeline, instances_json, output_file):
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
        synset_str = COCO_TO_WNSYNSET.get(cid)
        node_attrs = pipeline.get_node_attributes(synset_str) if synset_str else None

        if not node_attrs:
            stats["unmapped"] += 1
            continue

        is_nature = node_attrs.get('is_nature')
        if is_nature:
            stats["nature"] += 1
        else:
            stats["no_nature"] += 1

        bio_bin = safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        mat_bin = safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    log_statistics("COCO", len(id_to_name), stats, output_file)

def process_places(pipeline, excel_path, categories_txt, sourcekey_sheet, missing_sheet, output_file):
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
        node_attrs = pipeline.get_node_attributes(synset_str) if synset_str else None

        if not node_attrs:
            stats["unmapped"] += 1
            continue

        is_nature = node_attrs.get('is_nature')
        if is_nature:
            stats["nature"] += 1
        else:
            stats["no_nature"] += 1

        bio_bin = safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        mat_bin = safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
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
    parser.add_argument("--dataset", type=str, choices=["imagenet", "coco", "places", "all"], required=True,
                        help="Which dataset to process.")

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

    # Output text file
    parser.add_argument("--output_file", type=str, default="taxonomy_statistics.txt",
                        help="Path to save the output statistics as a text file.")

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] Initializing Taxonomy Pipeline from {args.excel_path}...")
    pipeline = TaxonomyEvaluationPipeline()
    df_taxonomy = pd.read_excel(args.excel_path, sheet_name="data corrected")
    pipeline.load_custom_excel_annotations(df_taxonomy, "Biotic/abiotic", "Material/immaterial")

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
        process_imagenet(pipeline, args.imagenet_dir, args.output_file)

    if args.dataset in ["coco", "all"]:
        if not args.coco_instances_json:
            raise ValueError("Error: --coco_instances_json is required to process COCO.")
        process_coco(pipeline, args.coco_instances_json, args.output_file)

    if args.dataset in ["places", "all"]:
        if not args.places_categories_txt:
            raise ValueError("Error: --places_categories_txt is required to process Places365.")
        process_places(pipeline, args.excel_path, args.places_categories_txt,
                       args.places_sourcekey_sheet, args.places_missing_sheet, args.output_file)

    if args.output_file:
        print(f"\n💾 Statistics successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()
