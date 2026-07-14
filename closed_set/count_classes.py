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
"""

import os
import sys
import json
import argparse
import pandas as pd
from torchvision import datasets

# Add parent directory to path to import the evaluation pipeline
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from first_tests.evaluation import TaxonomyEvaluationPipeline

# ============================================================================
# COCO DICTIONARY (From evaluate_coco.py)
# ============================================================================
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
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(output_str)

# ============================================================================
# DATASET PROCESSORS
# ============================================================================
def process_imagenet(pipeline, imagenet_dir, output_file):
    print(f"[INFO] Extracting ImageNet classes from {imagenet_dir}...")
    full_dataset = datasets.ImageFolder(imagenet_dir)
    idx_to_wnid = {v: k for k, v in full_dataset.class_to_idx.items()}
    
    stats = {"nature": 0, "no_nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}
    
    for idx, wnid in idx_to_wnid.items():
        synset_str = pipeline.get_synset_str_from_wnid(wnid)
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
    print(f"[INFO] Extracting COCO classes from {instances_json}...")
    with open(instances_json, "r") as f:
        data = json.load(f)
    
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
    print(f"[INFO] Extracting Places365 classes from {categories_txt}...")
    
    # 1. Load Places categories
    id_to_name = {}
    with open(categories_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            path, idx_str = line.rsplit(" ", 1)
            name = path[3:] if path.startswith("/") and len(path) > 3 and path[2] == "/" else path.lstrip("/")
            id_to_name[int(idx_str)] = name

    # 2. Reconstruct WordNet mappings
    df_source = pd.read_excel(excel_path, sheet_name=sourcekey_sheet, header=0)
    taxonomy_synsets = set()
    for _, row in df_source.iterrows():
        raw = row.iloc[0]
        if pd.isna(raw) or not str(raw).strip(): continue
        synset = str(raw).strip().split(' ')[0]
        source = "" if pd.isna(row.iloc[1]) else str(row.iloc[1]).strip()
        if 'MIT' in source: taxonomy_synsets.add(synset)

    df_missing = pd.read_excel(excel_path, sheet_name=missing_sheet, header=None)
    exclusion = {str(val).strip() for val in df_missing.iloc[:, 0] if pd.notna(val) and str(val).strip()}

    from nltk.corpus import wordnet as wn
    def resolve_via_wordnet(cls, tax_synsets):
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