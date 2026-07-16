"""
lib/dataset_loader.py

Unified dataset loader for COCO, ImageNet, Places365, and BIG-5.
Provides a standardized interface for loading images and projecting their ground-truth 
labels onto the BIG-5 taxonomy.

Yields a list of dictionaries in the format:
{
    "image_path": str,
    "targets": [
        {
            "class_name": str,
            "gt_nature": bool,
            "gt_biotic": bool,
            "gt_material": bool
        },
        ...
    ]
}
"""

import os
import ast
import urllib.request
import pandas as pd
from pathlib import Path

# ============================================================================
# COCO MAPPING
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
# HELPER: GRAPH RESOLUTION
# ============================================================================
def get_gt_from_graph(synset_str, taxonomy_graph):
    if not synset_str:
        return None
    labels = taxonomy_graph.resolve_labels(synset_str)
    if not labels:
        return None

    resolved_node = labels["resolved_from_node"]
    node_attrs = taxonomy_graph.graph.nodes.get(resolved_node, {})
    mat_val = node_attrs.get("material_immaterial")
    
    is_nature = labels["is_nature"]
    
    # Critical Fix: If nature is False, downstream labels MUST be None (N/A)
    # to match the prompt rules we are enforcing on the VLM.
    if is_nature:
        gt_biotic = labels.get("biotic_abiotic") == "biotic" if labels.get("biotic_abiotic") else None
        gt_material = mat_val == "material" if mat_val else True # Default to True for real datasets if not explicitly modeled
    else:
        gt_biotic = None
        gt_material = None

    return {
        "gt_nature": is_nature,
        "gt_biotic": gt_biotic,
        "gt_material": gt_material
    }

# ============================================================================
# LOADERS
# ============================================================================
def load_imagenet(data_dir, taxonomy_graph):
    from torchvision.datasets import ImageFolder
    import nltk
    from nltk.corpus import wordnet as wn
    
    try:
        wn.synsets('dog')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    dataset = ImageFolder(data_dir)
    idx_to_wnid = {v: k for k, v in dataset.class_to_idx.items()}
    class_to_target = {}

    for idx, wnid in idx_to_wnid.items():
        offset = int(wnid[1:])
        pos = wnid[0]
        try:
            synset_name = wn.synset_from_pos_and_offset(pos, offset).name()
        except Exception:
            continue

        gt = get_gt_from_graph(synset_name, taxonomy_graph)
        if gt:
            class_name = synset_name.split('.')[0].replace('_', ' ')
            class_to_target[idx] = {"class_name": class_name, **gt}

    results = []
    for path, target_idx in dataset.samples:
        if target_idx in class_to_target:
            results.append({
                "image_path": path,
                "targets": [class_to_target[target_idx]]
            })
    return results


def load_coco(images_dir, instances_json, taxonomy_graph):
    from pycocotools.coco import COCO
    coco = COCO(instances_json)
    
    cat_to_target = {}
    for cat_id, synset_str in COCO_TO_WNSYNSET.items():
        gt = get_gt_from_graph(synset_str, taxonomy_graph)
        if gt:
            class_name = synset_str.split('.')[0].replace('_', ' ')
            cat_to_target[cat_id] = {"class_name": class_name, **gt}

    results = []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(images_dir, info["file_name"])
        
        targets = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            cat_id = ann["category_id"]
            if cat_id in cat_to_target:
                targets.append(cat_to_target[cat_id])
                
        if targets:
            # Remove duplicate classes per image
            unique_targets = {t["class_name"]: t for t in targets}.values()
            results.append({
                "image_path": path,
                "targets": list(unique_targets)
            })
    return results


def load_places365(data_dir, categories_txt, excel_path, taxonomy_graph):
    from torchvision.datasets import ImageFolder
    
    def resolve_via_wordnet(cls, taxonomy_synsets):
        from nltk.corpus import wordnet as wn
        base = cls.replace('/', '_')
        head = cls.split('/')[0]
        candidates = [base, head]
        if '/' in cls:
            candidates.append(cls.split('/')[-1])
        seen = set()
        for c in candidates:
            key = c.replace(' ', '_')
            if key in seen: continue
            seen.add(key)
            for s in wn.synsets(key, pos='n'):
                if s.name() in taxonomy_synsets: return s.name()
        return None

    # Load categories and exclusion set
    id_to_name = {}
    with open(categories_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            path, idx_str = line.rsplit(" ", 1)
            name = path[3:] if (path.startswith("/") and len(path) > 3 and path[2] == "/") else path.lstrip("/")
            id_to_name[int(idx_str)] = name

    exclusion_df = pd.read_excel(excel_path, sheet_name="still missing MIT Places", header=None)
    exclusion = {str(val).strip() for val in exclusion_df.iloc[:, 0] if pd.notna(val) and str(val).strip()}

    sourcekey_df = pd.read_excel(excel_path, sheet_name="sourcekey", header=0)
    taxonomy_synsets = {
        str(row[0]).strip().split(' ')[0] 
        for _, row in sourcekey_df.iterrows() 
        if pd.notna(row[0]) and pd.notna(row[1]) and 'MIT' in str(row[1])
    }

    class_to_target = {}
    for cid, name in id_to_name.items():
        if name in exclusion: continue
        synset = resolve_via_wordnet(name, taxonomy_synsets)
        if synset:
            gt = get_gt_from_graph(synset, taxonomy_graph)
            if gt:
                class_name = name.replace('/', ' ').replace('_', ' ')
                class_to_target[cid] = {"class_name": class_name, **gt}

    dataset = ImageFolder(data_dir)
    idx_to_places_id = {dl_idx: int(name) for name, dl_idx in dataset.class_to_idx.items()}

    results = []
    for path, dl_idx in dataset.samples:
        places_id = idx_to_places_id.get(dl_idx)
        if places_id in class_to_target:
            results.append({
                "image_path": path,
                "targets": [class_to_target[places_id]]
            })
    return results


def load_big5(en_gt, es_gt, en_media, es_media, cache_dir):
    def map_yn(val): return 1 if str(val).strip().lower() == 'yes' else (0 if str(val).strip().lower() == 'no' else None)
    def map_mat(val): return 1 if str(val).strip().lower() == 'material' else (0 if str(val).strip().lower() == 'immaterial' else None)
    def map_bio(val): return 1 if str(val).strip().lower() == 'biotic' else (0 if str(val).strip().lower() == 'abiotic' else None)

    IMG_BASE_URL = "https://big5.cssh.bsc.es/STATIC/phase1-media/twitter_1_all/scaled/"
    os.makedirs(cache_dir, exist_ok=True)
    results = []

    for lang, gt_csv, media_csv in [("en", en_gt, en_media), ("es", es_gt, es_media)]:
        if not gt_csv or not media_csv: continue
        gt = pd.read_csv(gt_csv)
        media = pd.read_csv(media_csv)
        joined = gt.merge(media, on='platform_id', how='inner')

        for _, row in joined.iterrows():
            media_files_raw = row.get('media_files')
            if pd.isna(media_files_raw): continue
            try: filenames = ast.literal_eval(media_files_raw)
            except: continue

            for idx in range(4):
                if idx >= len(filenames): continue
                nat = map_yn(row.get(f'nature_visual_{idx}'))
                if nat is None: continue

                mat = map_mat(row.get(f'nep_materiality_visual_{idx}'))
                bio = map_bio(row.get(f'nep_biological_visual_{idx}'))
                if nat == 0 and (mat is not None or bio is not None):
                    mat, bio = None, None

                filename = filenames[idx]
                local_path = os.path.join(cache_dir, filename)
                if not os.path.isfile(local_path):
                    try: urllib.request.urlretrieve(IMG_BASE_URL + filename, local_path)
                    except: continue

                results.append({
                    "image_path": local_path,
                    "targets": [{
                        "class_name": "scene", # BIG-5 annotations apply to the entire scene
                        "gt_nature": nat == 1,
                        "gt_biotic": bio == 1 if bio is not None else None,
                        "gt_material": mat == 1 if mat is not None else None
                    }]
                })
    return results

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def load_dataset(dataset_name, taxonomy_graph, **kwargs):
    if dataset_name == "imagenet":
        return load_imagenet(kwargs.get("data_dir"), taxonomy_graph)
    elif dataset_name == "coco":
        return load_coco(kwargs.get("data_dir"), kwargs.get("instances_json"), taxonomy_graph)
    elif dataset_name == "places365":
        return load_places365(kwargs.get("data_dir"), kwargs.get("places_categories_txt"), kwargs.get("excel_path"), taxonomy_graph)
    elif dataset_name == "big5":
        return load_big5(kwargs.get("en_gt"), kwargs.get("es_gt"), kwargs.get("en_media"), kwargs.get("es_media"), kwargs.get("cache_dir"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")