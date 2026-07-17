"""
src/loaders/dataset_loader.py

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

WHY DOES THIS FILE EXIST?
Each of the four datasets we evaluate on stores its images and labels in a
totally different format (ImageFolder-style directories with WordNet ids,
a COCO-style JSON with numeric category ids, a Places365 category text file,
BIG-5's own CSVs of tweet annotations). This module's job is to hide all of
that format-specific mess behind ONE consistent output shape — a plain Python
list of "one entry per image" dicts — so the rest of the pipeline
(scripts/run_vlm_pipeline.py etc.) never needs to know which dataset it's
actually working with; it just reads `image_path` and `targets` the same way
every time.

The ground-truth (GT) labels these loaders attach to each image (gt_nature/
gt_biotic/gt_material) come from looking up each dataset's own class name in
the BIG-5 taxonomy graph (see src/loaders/excel_loader.py) via
`get_gt_from_graph` below.
"""

import os
import ast
import urllib.request
import pandas as pd
from pathlib import Path

# ============================================================================
# COCO MAPPING
# ============================================================================
# COCO identifies its 80 object categories by small integer ids (not always
# consecutive — some historical category ids were removed, which is why the
# numbers below skip around, e.g. 12/26/29/30 etc. are simply missing). This
# dict hand-maps each COCO category id to the WordNet synset that best
# represents it, so we can look up its taxonomy label the same way we do for
# ImageNet/Places classes.
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
    """
    Look up a single WordNet synset in the taxonomy graph and return its
    ground-truth taxonomy labels, or None if it can't be resolved at all
    (per project convention: unmapped classes get dropped, not defaulted).

    This is the ONE function all four loaders below call to turn "a dataset's
    class name" into "does this class count as nature/biotic/material".
    """
    if not synset_str:
        return None
    # `resolve_labels` (see excel_loader.py) walks UP the WordNet hierarchy
    # from this synset until it finds the nearest ancestor that was actually
    # hand-labeled in the BIG-5 Excel file. Returns None if nothing along that
    # chain was ever labeled.
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
        # `labels.get("biotic_abiotic")` is the string "biotic"/"abiotic"/None;
        # convert it into True/False/None the same way `label_to_bool` does
        # for VLM answers elsewhere in the pipeline (kept as inline logic here
        # rather than a shared helper since this file predates that one).
        gt_biotic = labels.get("biotic_abiotic") == "biotic" if labels.get("biotic_abiotic") else None
        gt_material = mat_val == "material" if mat_val else True # Default to True for real datasets if not explicitly modeled
    else:
        # This class isn't nature at all — biotic/material simply don't
        # apply, so both are None (never False), matching the "n/a" answer
        # the VLM itself is instructed to give in that situation.
        gt_biotic = None
        gt_material = None

    # `synset_id` is the ORIGINAL target synset (e.g. "leopard.n.01"), not the
    # nearest labeled ancestor it resolved from — ClipMatch / hierarchical
    # metrics need the leaf GT synset, so we thread it through here.
    return {
        "synset_id": synset_str,
        "gt_nature": is_nature,
        "gt_biotic": gt_biotic,
        "gt_material": gt_material
    }

# ============================================================================
# LOADERS
# ============================================================================
def _wnid_to_synset(wnid):
    """Convert an ImageNet folder id (e.g. "n02124278") to a WordNet synset
    string ("leopard.n.01"), or None if it can't be resolved."""
    from nltk.corpus import wordnet as wn
    try:
        # ImageNet's folder names follow WordNet's own internal id format: a
        # single letter for part-of-speech ("n" for noun, always the case for
        # ImageNet) followed by an 8-digit numeric "offset" that WordNet uses
        # to look up the exact synset. `wnid[0]` = the letter, `wnid[1:]` =
        # the offset digits (converted to an int).
        return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:])).name()
    except Exception:
        # Not a valid WordNet offset for some reason — nothing we can resolve.
        return None


def load_imagenet(data_dir, taxonomy_graph):
    """Load ImageNet: one folder per class (named by WordNet id), one label
    per image (single-label classification)."""
    from torchvision.datasets import ImageFolder
    import nltk
    from nltk.corpus import wordnet as wn

    # Make sure the WordNet corpus data is actually available before we try
    # to use it (same "probe, download if missing" pattern as excel_loader.py).
    try:
        wn.synsets('dog')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # torchvision's ImageFolder auto-discovers one "class" per subdirectory
    # inside data_dir, and assigns each an integer index. `class_to_idx` maps
    # folder-name -> index; we build the REVERSE mapping (index -> folder
    # name, i.e. the WNID) since that's what we need to look up class info.
    dataset = ImageFolder(data_dir)
    idx_to_wnid = {v: k for k, v in dataset.class_to_idx.items()}
    class_to_target = {}

    # Precompute EACH CLASS's taxonomy label just once (rather than doing this
    # lookup again for every single image, which could number in the tens of
    # thousands) — then just look up the pre-computed answer per image below.
    for idx, wnid in idx_to_wnid.items():
        synset_name = _wnid_to_synset(wnid)
        if synset_name is None:
            continue

        gt = get_gt_from_graph(synset_name, taxonomy_graph)
        if gt:
            # e.g. "golden_retriever.n.01" -> "golden retriever" (human-readable,
            # used later as the text fed to the VLM's classification prompt).
            class_name = synset_name.split('.')[0].replace('_', ' ')
            class_to_target[idx] = {"class_name": class_name, **gt}

    results = []
    # `dataset.samples` is torchvision's full list of (file_path, class_index)
    # pairs for every single image in the folder tree.
    for path, target_idx in dataset.samples:
        if target_idx in class_to_target:
            # Only include images whose class actually resolved to a taxonomy
            # label — per project convention, unmapped classes are dropped
            # entirely rather than kept with a missing/default label.
            results.append({
                "image_path": path,
                "targets": [class_to_target[target_idx]]
            })
    return results


def load_coco(images_dir, instances_json, taxonomy_graph):
    """Load COCO: one JSON annotation file describing possibly MULTIPLE
    labeled objects per image (multi-label, unlike ImageNet/Places)."""
    from pycocotools.coco import COCO
    # pycocotools' COCO class parses the (often huge) instances_*.json
    # annotation file and gives us convenient lookup methods (getImgIds,
    # loadImgs, getAnnIds, loadAnns) instead of us having to parse the raw
    # JSON structure ourselves.
    coco = COCO(instances_json)

    # As with ImageNet above, resolve each of COCO's (fixed, small) 80
    # categories to its taxonomy label just once up front.
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

        # Unlike ImageNet/Places, a single COCO image can have MANY labeled
        # objects (e.g. a photo with both a "dog" and a "person" annotated) —
        # gather every annotated object whose category resolved to a taxonomy
        # label.
        targets = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            cat_id = ann["category_id"]
            if cat_id in cat_to_target:
                targets.append(cat_to_target[cat_id])

        if targets:
            # Remove duplicate classes per image
            # A photo might have 3 separate "person" annotations (3 different
            # people in the photo) — for our purposes we only care about
            # WHICH CLASSES are present, not how many instances of each, so
            # collapse duplicates by class_name. Using a dict keyed by
            # class_name naturally keeps only the LAST entry per name (they're
            # identical anyway, since the label only depends on the class).
            unique_targets = {t["class_name"]: t for t in targets}.values()
            results.append({
                "image_path": path,
                "targets": list(unique_targets)
            })
    return results


# ---------------------------------------------------------------------------
# Places365 helpers (module-level so both load_places365 and get_candidate_vocab
# resolve scene names to synsets identically).
# ---------------------------------------------------------------------------
def _resolve_places_name_to_synset(cls, taxonomy_synsets):
    """Resolve a Places scene name (e.g. "forest/broadleaf") to a MIT-tagged
    taxonomy synset string, or None if it doesn't resolve. Heuristic — see the
    lossy-reconstruction caveat in baseline/evaluate_places.py.

    Unlike ImageNet (whose folder names ARE WordNet ids directly) or COCO
    (which has a small hand-built mapping table), Places365's category names
    are plain English scene descriptions like "forest/broadleaf" or
    "restaurant" with no built-in WordNet id — so we have to GUESS which
    WordNet synset each one corresponds to, restricted to only the synsets
    the BIG-5 project's sourcekey sheet already tags as coming from "MIT"
    (i.e. Places365's source institution), to reduce false matches.
    """
    from nltk.corpus import wordnet as wn
    # Places category names sometimes look like "bus_station/indoor" (a
    # broad category plus a sub-variant). Try a few different ways of reading
    # this string as a lookup key: the whole thing with '/' turned into '_'
    # (matching WordNet's underscore convention), just the part before the
    # first '/', and just the part after the last '/'.
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
        # For each candidate string, ask WordNet for every noun sense, and
        # accept the FIRST one that happens to be in our restricted
        # "MIT-tagged" synset set — i.e. a synset the taxonomy Excel actually
        # knows about, rather than an arbitrary unrelated WordNet sense.
        for s in wn.synsets(key, pos='n'):
            if s.name() in taxonomy_synsets: return s.name()
    return None


def _load_places_categories(categories_txt):
    """Parse categories_places365.txt into {places_id: scene_name}."""
    # This file's format is one line per category: "/a/airfield 0" (a path-like
    # category name, a space, then its numeric id).
    id_to_name = {}
    with open(categories_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # rsplit(" ", 1) splits from the RIGHT, at most once — so
            # "/a/airfield 0" becomes ["/a/airfield", "0"] even if the
            # category name itself might (in principle) contain spaces.
            path, idx_str = line.rsplit(" ", 1)
            # Strip the leading "/x/" category-letter prefix (Places365 groups
            # scenes alphabetically into /a/, /b/, /c/, ... subfolders) to get
            # just the plain scene name, e.g. "/a/airfield" -> "airfield".
            name = path[3:] if (path.startswith("/") and len(path) > 3 and path[2] == "/") else path.lstrip("/")
            id_to_name[int(idx_str)] = name
    return id_to_name


def _load_places_taxonomy_synsets(excel_path):
    """Return (exclusion_set, taxonomy_synsets) from the BIG-5 Excel — the set
    of scene names to skip and the set of MIT-tagged taxonomy synset strings."""
    # This sheet lists Places365 category names the project has manually
    # confirmed have NO good taxonomy synset at all — skip these entirely
    # rather than let the heuristic resolver guess something wrong for them.
    exclusion_df = pd.read_excel(excel_path, sheet_name="still missing MIT Places", header=None)
    exclusion = {str(val).strip() for val in exclusion_df.iloc[:, 0] if pd.notna(val) and str(val).strip()}

    # This sheet records, for various synsets, which external SOURCE dataset
    # they were confirmed to correspond to. We only want the ones tagged as
    # coming from "MIT" (Places365's source), so the heuristic resolver above
    # only ever picks synsets that are actually meant for Places scenes.
    sourcekey_df = pd.read_excel(excel_path, sheet_name="sourcekey", header=0)
    taxonomy_synsets = {
        str(row[0]).strip().split(' ')[0]
        for _, row in sourcekey_df.iterrows()
        if pd.notna(row[0]) and pd.notna(row[1]) and 'MIT' in str(row[1])
    }
    return exclusion, taxonomy_synsets


def load_places365(data_dir, categories_txt, excel_path, taxonomy_graph):
    """Load Places365: one folder per scene category, one label per image
    (single-label, like ImageNet)."""
    from torchvision.datasets import ImageFolder

    id_to_name = _load_places_categories(categories_txt)
    exclusion, taxonomy_synsets = _load_places_taxonomy_synsets(excel_path)

    # As with ImageNet/COCO above: resolve each of the (up to) 365 scene
    # categories to a taxonomy label just once, rather than per-image.
    class_to_target = {}
    for cid, name in id_to_name.items():
        if name in exclusion: continue
        synset = _resolve_places_name_to_synset(name, taxonomy_synsets)
        if synset:
            gt = get_gt_from_graph(synset, taxonomy_graph)
            if gt:
                class_name = name.replace('/', ' ').replace('_', ' ')
                class_to_target[cid] = {"class_name": class_name, **gt}

    # Places365's ImageFolder subdirectories are expected to be NAMED by
    # their numeric places-id (e.g. a folder literally called "0", "1", ...)
    # rather than by scene name — this line converts the folder-name string
    # back into an int so it matches the ids used in categories_places365.txt.
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
    """Load the BIG-5 social-media dataset itself: human-annotated tweet
    images (English + Spanish), each with its OWN direct nature/biotic/
    material annotation (no WordNet class lookup needed — these are already
    the ground truth, straight from human annotators)."""
    # Small helper functions to turn the raw CSV cell text ("Yes"/"No",
    # "Material"/"Immaterial", "Biotic"/"Abiotic") into 1/0/None. Kept as
    # local closures since they're only used within this one function.
    def map_yn(val): return 1 if str(val).strip().lower() == 'yes' else (0 if str(val).strip().lower() == 'no' else None)
    def map_mat(val): return 1 if str(val).strip().lower() == 'material' else (0 if str(val).strip().lower() == 'immaterial' else None)
    def map_bio(val): return 1 if str(val).strip().lower() == 'biotic' else (0 if str(val).strip().lower() == 'abiotic' else None)

    # BIG-5 images are hosted remotely; we download and cache them locally the
    # first time each one is needed, rather than re-downloading on every run.
    IMG_BASE_URL = "https://big5.cssh.bsc.es/STATIC/phase1-media/twitter_1_all/scaled/"
    os.makedirs(cache_dir, exist_ok=True)
    results = []

    # BIG-5 data comes as two SEPARATE CSV pairs (one per language): a
    # "ground truth" CSV (the human annotations) and a "media" CSV (which
    # tweet has which image files attached). We process English and Spanish
    # identically, just looping over both pairs in turn.
    for lang, gt_csv, media_csv in [("en", en_gt, en_media), ("es", es_gt, es_media)]:
        if not gt_csv or not media_csv: continue
        gt = pd.read_csv(gt_csv)
        media = pd.read_csv(media_csv)
        # Join the two tables on their shared tweet id ("platform_id"), so
        # each row afterward has BOTH the annotation columns and the media
        # filename columns together.
        joined = gt.merge(media, on='platform_id', how='inner')

        for _, row in joined.iterrows():
            media_files_raw = row.get('media_files')
            if pd.isna(media_files_raw): continue
            try:
                # The media_files column stores a Python-list-looking string,
                # e.g. "['photo1.jpg', 'photo2.jpg']" — ast.literal_eval safely
                # parses that string back into an actual Python list (safer
                # than eval() since it only allows literal data, no code execution).
                filenames = ast.literal_eval(media_files_raw)
            except: continue

            # A single tweet can have up to 4 attached images/videos; the
            # annotation columns are suffixed _0, _1, _2, _3 for each one.
            for idx in range(4):
                if idx >= len(filenames): continue
                nat = map_yn(row.get(f'nature_visual_{idx}'))
                if nat is None: continue

                mat = map_mat(row.get(f'nep_materiality_visual_{idx}'))
                bio = map_bio(row.get(f'nep_biological_visual_{idx}'))
                if nat == 0 and (mat is not None or bio is not None):
                    # Safety net: if a human annotator somehow marked this
                    # media as NOT nature but still filled in a biotic/material
                    # value (a data-entry inconsistency), force both back to
                    # "not applicable" rather than trusting the contradictory
                    # values — matches the same rule enforced elsewhere in the
                    # pipeline (no biotic/material label when nature is False).
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
                        "synset_id": None,      # holistic scene label — no WordNet synset
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
    """Dispatch to the correct loader by name — the single function every
    other script calls, so callers never need to import the individual
    load_imagenet/load_coco/etc. functions directly."""
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


# ============================================================================
# CANDIDATE VOCABULARY (for ClipMatch + hierarchical metrics — ImageNet/Places)
# ============================================================================
def get_candidate_vocab(dataset_name, **kwargs):
    """
    Returns the FIXED closed-set candidate class list for the ClipMatch and
    hierarchical (hP/hR) metrics, as a list of dicts:
        [{"class_name": str, "synset_id": str}, ...]

    Only ImageNet and Places365 define such a vocabulary (single-label, closed
    class set). COCO (multi-label) and BIG-5 (no closed class vocabulary) return
    None — ClipMatch/hP/hR are not run on them, per the project conventions.

    Every returned candidate carries a resolvable WordNet synset, so the
    ClipMatch-predicted class always has a synset for the hierarchical metrics:
      - ImageNet: every folder WNID converts losslessly via wn.synset_from_pos_and_offset.
      - Places365: scene names are resolved heuristically (MIT-tagged synsets);
        names that do not resolve are dropped from the candidate set.

    NOTE: unlike get_gt_from_graph's callers above, this function does NOT
    filter by whether the class resolves to a taxonomy LABEL — ClipMatch needs
    the full candidate class list (so the model has every possible class to
    choose from when predicting), regardless of whether that class happens to
    be "mapped" for nature/biotic/material purposes.
    """
    if dataset_name == "imagenet":
        from torchvision.datasets import ImageFolder
        classes = ImageFolder(kwargs.get("data_dir")).classes  # sorted WNIDs
        vocab = []
        seen = set()
        for wnid in classes:
            synset_name = _wnid_to_synset(wnid)
            if synset_name is None or synset_name in seen:
                continue
            seen.add(synset_name)
            class_name = synset_name.split('.')[0].replace('_', ' ')
            vocab.append({"class_name": class_name, "synset_id": synset_name})
        return vocab

    if dataset_name == "places365":
        id_to_name = _load_places_categories(kwargs.get("places_categories_txt"))
        exclusion, taxonomy_synsets = _load_places_taxonomy_synsets(kwargs.get("excel_path"))
        vocab = []
        seen = set()
        for _, name in sorted(id_to_name.items()):
            if name in exclusion:
                continue
            synset = _resolve_places_name_to_synset(name, taxonomy_synsets)
            if synset is None or synset in seen:
                continue
            seen.add(synset)
            class_name = name.replace('/', ' ').replace('_', ' ')
            vocab.append({"class_name": class_name, "synset_id": synset})
        return vocab

    # COCO (multi-label) and BIG-5 (open scene) have no closed candidate vocab.
    return None


# ============================================================================
# MAPPING VOCABULARY (authoritative class_name -> synset, for the hybrid
# object-labeling step — nature/biotic WordNet mapping vs VLM fallback)
# ============================================================================
def build_mapping_vocab(dataset_name, **kwargs):
    """
    Returns {normalized_class_name: synset_id} for the dataset — the authoritative
    lookup used to map an EXTRACTED OBJECT phrase onto a WordNet synset WITHOUT
    word-sense guessing. Synsets come from the dataset's own class tables
    (ImageNet WNIDs, COCO_TO_WNSYNSET, Places MIT-resolved), so e.g. "tiger"
    maps to tiger.n.02 (the animal), never tiger.n.01 ("a fierce person").

    Objects whose (normalized) phrase is not a key here are UNMAPPED and go to
    the image-supported VLM fallback. BIG-5 has no class vocabulary -> {}.

    Names are lowercased/space-normalized. Cross-dataset union mapping (recap §9)
    is a deliberate future enhancement, not done here.

    WHY IS THIS DIFFERENT FROM get_candidate_vocab? get_candidate_vocab gives
    ClipMatch a list of classes to CHOOSE FROM (for a specific single-label
    dataset). build_mapping_vocab instead gives the hybrid labeling step (see
    src/vlm_pipeline.py's map_object_to_taxonomy) a plain word->synset LOOKUP
    TABLE, used regardless of which axis-scoring metric is running — this is
    why COCO (which has no ClipMatch candidate vocab at all) still gets a
    mapping vocab here, built straight from COCO_TO_WNSYNSET.
    """
    vocab = {}

    if dataset_name in ("imagenet", "places365"):
        # Reuse the already-built candidate vocab and just reshape it into a
        # {name: synset} dict instead of a list of dicts.
        for entry in (get_candidate_vocab(dataset_name, **kwargs) or []):
            vocab[entry["class_name"].strip().lower()] = entry["synset_id"]
        return vocab

    if dataset_name == "coco":
        for synset in COCO_TO_WNSYNSET.values():
            name = synset.split('.')[0].replace('_', ' ').strip().lower()
            # `setdefault` only inserts if the key isn't already present —
            # guards against two different COCO synsets accidentally
            # producing the same display name (unlikely here, but safe).
            vocab.setdefault(name, synset)
        return vocab

    # big5 (holistic scene) — no closed class vocabulary.
    return vocab
