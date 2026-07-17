#!/usr/bin/env python3
"""
Evaluate any of the four model families built so far (ImageNet-style
classifier, Places365-style classifier, Query2Label COCO multi-label
classifier, or Paula Feliu's direct multitask model) against the BIG-5
Twitter dataset, projecting/comparing predictions against BIG-5's OWN
direct nature/biotic/material annotations.

WHAT IS THIS SCRIPT FOR? All the other baseline/evaluate_*.py scripts test a
model against a STANDARD benchmark dataset (ImageNet/COCO/Places365), each
with its own native ground truth. This script instead evaluates against the
BIG-5 project's OWN target domain: real social-media (Twitter) photos, with
human annotators directly labeling nature/biotic/material for each one. This
is the real "does this actually work on the images this whole project cares
about" test — the other scripts mostly probe "how good is this off-the-shelf
model on its home turf, once its predictions are projected onto our taxonomy."

------------------------------------------------------------------------------
HOW THIS DIFFERS FROM evaluate_imagenet.py / evaluate_places.py / evaluate_coco.py
------------------------------------------------------------------------------
In those three scripts, ground truth comes from each dataset's OWN native
labels (ImageNet class, Places category, COCO object annotations), which then
get PROJECTED onto the taxonomy via a WordNet synset mapping. Here, ground
truth is different in kind: BIG-5's annotators labeled each image directly
with nature/materiality/biological -- there is no projection needed on the
ground-truth side at all. The projection logic (predicted class -> synset ->
taxonomy) is still needed on the PREDICTION side for three of the four model
families (everything except multitask_direct, which already predicts the
taxonomy dimensions directly).

------------------------------------------------------------------------------
DATA SOURCES AND JOIN MECHANICS -- verified against actual files, not guessed
------------------------------------------------------------------------------
Two files per language are needed:
  1. table_for_pau_twitter-{en,es}-6.csv: the ground-truth annotation table
     (protocol v6, majority vote + 3rd-coder disagreement resolution per
     Ramin's email). Columns: platform_id, n_coders, nature_visual_any,
     nature_visual_0..3, nep_materiality_visual_0..3,
     nep_immaterial_specific_visual_0..3 (unused for this taxonomy -- a
     sub-categorization of WHICH kind of immaterial content it is, e.g.
     illustration/videogame/plain_text/infographic/other, per the BIG-5
     protocol's documented immaterial subcategories. "other" dominates
     (~96% of populated values across the ES data) but is not exclusive.
     Not needed here since nep_materiality_visual_N already gives the
     top-level material/immaterial call our taxonomy actually uses),
     nep_biological_visual_0..3.
  2. phase-1_twitter-{en,es}.csv: platform_id -> media_files, a string that
     looks like a Python list literal, e.g. "['123_0.jpg', '123_1.jpg']".
     VERIFIED: the numeric suffix in each filename (_0, _1, ...) matches the
     nature_visual_N / nep_*_visual_N index positionally, and list order
     matches index order. VERIFIED: inner-joining both files on platform_id
     for the ES data produced a 100% match (885/885), with no case of
     nature_visual_any=='Yes' lacking a populated media_files entry.
Images are downloaded from:
    https://big5.cssh.bsc.es/STATIC/phase1-media/twitter_1_all/scaled/<FILENAME>
and cached locally (see --images_cache_dir) so repeated runs don't re-download.

------------------------------------------------------------------------------
DATA QUALITY NOTE: 24 inconsistent image-slots (out of 3534 checked, ~0.7%)
------------------------------------------------------------------------------
A small number of image-slots have nature_visual_N == "No" while
nep_materiality_visual_N / nep_biological_visual_N are still populated. Per
user decision: nature_visual_N is treated as authoritative, and the stray
materiality/biological values are dropped (treated as None) in this case.
This is a data cleaning decision, not a taxonomy design choice -- if a
majority-vote nature label says "No", the detail fields for that slot
shouldn't apply, regardless of what got filled in for it.

------------------------------------------------------------------------------
IMAGENET WNID MAPPING -- verified against a canonical, well-established source
------------------------------------------------------------------------------
Unlike evaluate_imagenet.py (which derives idx->WNID from ImageFolder reading
real ImageNet validation directory names off disk), there's no such directory
here -- we're feeding arbitrary Twitter photos. imagenet_idx_to_wnid() below
uses timm.data.ImageNetInfo, which bundles the canonical index->WNID mapping
as installed package data (not a network fetch to some small personal repo --
the exact failure mode that cost significant time earlier with the
ML-Decoder weights bucket). timm is already installed on the cluster (seen
in this project's own Q2L run logs). The mapping is sanity-checked on first
use against two well-known reference facts (index 0 = tench = n01440764,
index 386 = African_elephant = n02504458) so a future timm version change
that altered this data would be caught immediately.

------------------------------------------------------------------------------
MODEL FAMILIES -- construction/preprocessing/prediction-projection logic is
reused VERBATIM from the already-verified scripts, not reimplemented
------------------------------------------------------------------------------
  --model_family imagenet         : same as evaluate_imagenet.py's torchvision path
  --model_family places           : same as evaluate_places.py's torchvision path
  --model_family coco_q2l         : same as evaluate_coco.py's q2l path
  --model_family multitask_direct : same as all three scripts' multitask_direct path
"""

import os
import sys
import ast
import json
import argparse
import urllib.request
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Make both the repo root (for the missing `first_tests` module) and this
# script's own directory importable — see count_classes.py's comment.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
this_dir = os.path.abspath(os.path.dirname(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from first_tests.evaluation import TaxonomyEvaluationPipeline

# Where BIG-5's Twitter images are hosted; downloaded and cached locally on
# first use (see download_image_if_missing below).
IMG_BASE_URL = "https://big5.cssh.bsc.es/STATIC/phase1-media/twitter_1_all/scaled/"


# ============================================================================
# CANONICAL IMAGENET INDEX -> WNID MAPPING (verified, see docstring above)
# ============================================================================
# ============================================================================
# IMAGENET INDEX -> WNID, VIA timm (already installed on the cluster, bundled
# as package data -- not a network fetch, not a hardcoded dict; see the
# module docstring for why this was chosen over embedding a static mapping).
# ============================================================================
def imagenet_idx_to_wnid(idx):
    """Return the WNID for a given ImageNet-1k class index (0..999), via
    timm.data.ImageNetInfo. Lazily imported and cached on first call."""
    # `hasattr(fn, "_info")` is a "memoize on the function object itself"
    # trick: the first call builds and stores `_info` as an attribute
    # ATTACHED TO THE FUNCTION, so subsequent calls skip straight to using it
    # instead of re-importing/re-validating timm every single call.
    if not hasattr(imagenet_idx_to_wnid, "_info"):
        try:
            from timm.data import ImageNetInfo
        except ImportError as e:
            raise ImportError(
                "timm is required for --model_family imagenet (it supplies the canonical "
                "ImageNet index->WNID mapping as bundled package data). Install it with "
                "`pip install timm`. Original error: " + str(e)
            )
        imagenet_idx_to_wnid._info = ImageNetInfo('imagenet-1k')
        # Sanity check against two well-known reference facts, so a timm version
        # change that alters this data would be caught immediately rather than
        # silently mismapping every prediction.
        assert imagenet_idx_to_wnid._info.index_to_label_name(0) == "n01440764", \
            "timm's ImageNetInfo index 0 does not match the expected WNID (n01440764, tench)."
        assert imagenet_idx_to_wnid._info.index_to_label_name(386) == "n02504458", \
            "timm's ImageNetInfo index 386 does not match the expected WNID (n02504458, African_elephant)."
    return imagenet_idx_to_wnid._info.index_to_label_name(idx)




# ============================================================================
# COCO -> WORDNET SYNSET MAPPING (identical to evaluate_coco.py, needed for
# --model_family coco_q2l's prediction projection)
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
    72: 'television_receiver.n.01',  # corrected from 'television.n.02'
    73: 'laptop.n.01', 74: 'mouse.n.04', 75: 'remote_control.n.01', 76: 'computer_keyboard.n.01',
    77: 'cellular_telephone.n.01', 78: 'microwave.n.02', 79: 'oven.n.01', 80: 'toaster.n.02',
    81: 'sink.n.01', 82: 'refrigerator.n.01', 84: 'book.n.02', 85: 'clock.n.01', 86: 'vase.n.01',
    87: 'scissors.n.01', 88: 'teddy.n.01', 89: 'hand_blower.n.01', 90: 'toothbrush.n.01',
}


# ============================================================================
# PLACES365 HELPERS (verbatim from evaluate_places.py / places_taxonomy_mapping.py
# -- needed to interpret --model_family places' predictions)
# ============================================================================
def load_places365_categories(categories_txt):
    """Parse categories_places365.txt into an id-ordered list of 365 scene
    names — same logic as evaluate_places.py's identically-named function
    (see there for a fully-commented line-by-line walkthrough)."""
    if not os.path.isfile(categories_txt):
        raise FileNotFoundError(
            f"Could not find categories_places365.txt at '{categories_txt}'. "
            f"Pass its correct location via --places_categories_txt."
        )
    id_to_name = {}
    with open(categories_txt, "r") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                path, idx_str = line.rsplit(" ", 1)
                idx = int(idx_str)
            except ValueError:
                raise ValueError(f"{categories_txt}:{line_no}: could not parse line '{raw_line.rstrip()}'.")
            if path.startswith("/") and len(path) > 3 and path[2] == "/":
                name = path[3:]
            else:
                name = path.lstrip("/")
            id_to_name[idx] = name
    if set(id_to_name.keys()) != set(range(365)):
        raise ValueError(f"{categories_txt} does not contain exactly ids 0..364.")
    return [id_to_name[i] for i in range(365)]


def _raise_sheet_not_found(excel_path, sheet_name, original_error):
    """Re-raise a pandas sheet-not-found error with the list of ACTUALLY
    available sheet names attached, so a typo'd sheet-name flag is easy to spot."""
    try:
        available = pd.ExcelFile(excel_path).sheet_names
    except Exception:
        available = ["<could not list sheets>"]
    raise ValueError(
        f"Could not read sheet '{sheet_name}' from {excel_path} ({original_error}). "
        f"Available sheets: {available}."
    )


def load_places_taxonomy_synsets(excel_path, sheet_name, mit_only=True):
    """Read the taxonomy's 'sourcekey' sheet and return the set of synsets
    tagged as coming from Places365 ('MIT') — see evaluate_places.py's
    identically-named function for the full rationale."""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
    except ValueError as e:
        _raise_sheet_not_found(excel_path, sheet_name, e)
    synset_col, source_col = df.columns[0], df.columns[1]
    tax = set()
    for _, row in df.iterrows():
        raw = row[synset_col]
        if pd.isna(raw) or not str(raw).strip():
            continue
        synset = str(raw).strip().split(' ')[0]
        source = "" if pd.isna(row[source_col]) else str(row[source_col]).strip()
        if mit_only and 'MIT' not in source:
            continue
        tax.add(synset)
    return tax


def load_places_exclusion_set(excel_path, sheet_name):
    """Read the 'still missing MIT Places' sheet: Places365 classes with no
    usable taxonomy synset at all — see evaluate_places.py's identically-
    named function."""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    except ValueError as e:
        _raise_sheet_not_found(excel_path, sheet_name, e)
    missing = set()
    for val in df.iloc[:, 0]:
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            missing.add(s)
    return missing


def resolve_places_via_wordnet(cls, taxonomy_synsets):
    """Try a few readings of a Places scene name against WordNet, accepting
    the first noun sense that's in the restricted MIT-tagged synset set —
    same heuristic as evaluate_places.py's resolve_via_wordnet."""
    from nltk.corpus import wordnet as wn
    base = cls.replace('/', '_')
    head = cls.split('/')[0]
    candidates = [base, head]
    if '/' in cls:
        candidates.append(cls.split('/')[-1])
    seen = set()
    for c in candidates:
        key = c.replace(' ', '_')
        if key in seen:
            continue
        seen.add(key)
        for s in wn.synsets(key, pos='n'):
            if s.name() in taxonomy_synsets:
                return s.name()
    return None


def load_places_explicit_mapping(mapping_csv):
    """Load an authoritative places_name|places_id -> synset override CSV, if
    the user supplied one, bypassing WordNet heuristics entirely."""
    df = pd.read_csv(mapping_csv)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    syn_col = next((c for c in cols if 'synset' in c or 'wordnet' in c or 'wnid' in c), cols[-1])
    key_cols = [c for c in cols if c != syn_col]
    mapping = {}
    for _, row in df.iterrows():
        syn = str(row[syn_col]).strip()
        if not syn or syn.lower() == 'nan':
            continue
        for kc in key_cols:
            k = str(row[kc]).strip()
            if k and k.lower() != 'nan':
                mapping[k] = syn
    return mapping


def build_places_id_to_synset(places_categories, excel_path=None, sourcekey_sheet="sourcekey",
                               missing_sheet="still missing MIT Places", mapping_csv=None):
    """Build {places_category_id: synset_string} either from an explicit
    --places_mapping_csv (authoritative) or by WordNet reconstruction
    restricted to MIT-tagged synsets — same logic/priority order as
    evaluate_places.py's build_places_id_to_synset (see there for the fully
    commented walkthrough); this copy skips the argparse-object indirection
    since it's called with plain keyword arguments here instead."""
    explicit = load_places_explicit_mapping(mapping_csv) if mapping_csv else None
    try:
        exclusion = load_places_exclusion_set(excel_path, missing_sheet)
    except Exception as e:
        if explicit is None:
            raise ValueError(f"Could not read '{missing_sheet}' from {excel_path} ({e}).")
        exclusion = set()

    taxonomy_synsets = None
    if explicit is None:
        taxonomy_synsets = load_places_taxonomy_synsets(excel_path, sourcekey_sheet, mit_only=True)

    id_to_synset, excluded, unresolved = {}, [], []
    for cid, name in enumerate(places_categories):
        if explicit is not None:
            syn = explicit.get(name) or explicit.get(str(cid))
            if syn:
                id_to_synset[cid] = syn
            elif name in exclusion:
                excluded.append((cid, name))
            else:
                unresolved.append((cid, name))
            continue
        if name in exclusion:
            excluded.append((cid, name))
            continue
        syn = resolve_places_via_wordnet(name, taxonomy_synsets)
        if syn is not None:
            id_to_synset[cid] = syn
        else:
            unresolved.append((cid, name))

    return id_to_synset, {"n_mapped": len(id_to_synset), "n_excluded_still_missing": len(excluded),
                          "n_unresolved": len(unresolved), "excluded": excluded, "unresolved": unresolved}


def _replace_head_365(model, model_name):
    """Swap a torchvision classifier head to 365 outputs — see
    evaluate_places.py's identically-named function for the full explanation
    of why each architecture family needs a different attribute path."""
    name = model_name.lower()
    if name.startswith("convnext"):
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, 365)
    elif name.startswith("swin"):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, 365)
    elif name.startswith("vit"):
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, 365)
    elif name.startswith("resnet"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 365)
    else:
        raise ValueError(f"Don't know how to swap the head for '{model_name}'.")


# ============================================================================
# QUERY2LABEL HELPERS (verbatim from evaluate_coco.py -- needed to build/run
# the model for --model_family coco_q2l)
# ============================================================================
def build_q2l_args(config_path, img_size_override=None, num_class_override=None):
    """Build the Q2L model-construction args (defaults + config.json overlay)
    — see evaluate_coco.py's identically-named function for the full
    explanation of why this exact override sequence matters."""
    args = SimpleNamespace(
        dataname='coco14', dataset_dir='/comp_robot/liushilong/data/COCO14/', img_size=448,
        arch='Q2L-TResL_22k-448', output=None, loss='asl', num_class=80, workers=8, batch_size=16,
        print_freq=10, resume=None, pretrained=False, eps=1e-5, world_size=-1, rank=-1,
        dist_url='tcp://127.0.0.1:3451', seed=None, local_rank=0, amp=False, orid_norm=False,
        enc_layers=1, dec_layers=2, dim_feedforward=256, hidden_dim=128, dropout=0.1, nheads=4,
        pre_norm=False, position_embedding='sine', backbone='resnet101',
        keep_other_self_attn_dec=False, keep_first_self_attn_dec=False, keep_input_proj=False,
    )
    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)
    for k, v in cfg_dict.items():
        setattr(args, k, v)
    if img_size_override is not None:
        args.img_size = img_size_override
    if num_class_override is not None:
        args.num_class = num_class_override
    return args


def clean_state_dict_fallback(state_dict):
    """Strip a leading 'module.' DataParallel/DDP prefix from checkpoint keys
    — see evaluate_coco.py's identically-named function."""
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
            for k, v in state_dict.items()}


def install_inplace_abn_shim_if_missing(verbose=False):
    """Register a numerically-identical eval-only substitute for the
    inplace_abn package's InPlaceABNSync (which needs a CUDA build toolchain
    to install) if the real package isn't available — see evaluate_coco.py's
    identically-named function for the full rationale."""
    try:
        import inplace_abn  # noqa: F401
        if verbose:
            print("[INFO] Real 'inplace_abn' package found; not installing shim.")
        return
    except ImportError:
        pass
    import types as _types
    import torch.nn.functional as F

    class InPlaceABNSyncShim(nn.BatchNorm2d):
        """Drop-in stand-in matching InPlaceABNSync's constructor signature
        and state_dict key names, implemented as plain BatchNorm2d + a
        manually-applied activation (mathematically identical for
        inference)."""
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                     activation="leaky_relu", activation_param=0.01, group=None):
            super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
            self._shim_activation = activation
            self._shim_activation_param = activation_param

        def forward(self, x):
            x = super().forward(x)
            act = self._shim_activation
            if act == "leaky_relu":
                return F.leaky_relu(x, negative_slope=self._shim_activation_param, inplace=True)
            elif act in (None, "identity", "none"):
                return x
            elif act == "relu":
                return F.relu(x, inplace=True)
            elif act == "elu":
                return F.elu(x, alpha=self._shim_activation_param, inplace=True)
            else:
                raise ValueError(f"Unsupported activation '{act}' in InPlaceABNSync shim")

    shim_module = _types.ModuleType("inplace_abn")
    shim_module.InPlaceABNSync = InPlaceABNSyncShim
    shim_module.InPlaceABN = InPlaceABNSyncShim
    sys.modules["inplace_abn"] = shim_module
    if verbose:
        print("[INFO] Real 'inplace_abn' not found; installed an eval-only shim.")


# ============================================================================
# MULTITASK DIRECT-TAXONOMY MODEL (Paula Feliu's TFG) -- verbatim from the
# other three scripts.
# ============================================================================
class CustomBackbone(nn.Module):
    """Swappable CNN feature-extractor — see evaluate_imagenet.py for full comments."""
    def __init__(self, model_choice='ResNet18'):
        super(CustomBackbone, self).__init__()
        self.model_choice = model_choice
        if model_choice == 'DenseNet121':
            model_base = tv_models.densenet121(weights=None)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1024
        elif model_choice == 'ResNet18':
            model_base = tv_models.resnet18(weights=None)
            model_base.fc = nn.Identity()
            self.feature_dim = 512
        elif model_choice == 'EfficientNetB0':
            model_base = tv_models.efficientnet_b0(weights=None)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            model_base = tv_models.resnet50(weights=None)
            model_base.fc = nn.Identity()
            self.feature_dim = 2048
        self.backbone = model_base

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


class MultiTaskModel(nn.Module):
    """Wraps a CustomBackbone with four independent linear prediction heads
    sharing the same features — see evaluate_imagenet.py for full comments."""
    def __init__(self, backbone, feature_dim):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.fc_nature = nn.Linear(feature_dim, 2)
        self.fc_materiality = nn.Linear(feature_dim, 3)
        self.fc_biological = nn.Linear(feature_dim, 3)
        self.fc_landscape = nn.Linear(feature_dim, 8)

    def forward(self, x):
        features = self.backbone(x)
        return (self.fc_nature(features), self.fc_materiality(features),
                self.fc_biological(features), self.fc_landscape(features))


# Translate the multitask model's own 0/1 class indices into this project's
# convention — see evaluate_imagenet.py for the full source citation.
MULTITASK_MATERIALITY_TO_OURS = {0: 1, 1: 0}
MULTITASK_BIOLOGICAL_TO_OURS = {0: 1, 1: 0}


def coco_readable_name(cid):
    """Human-readable name derived from the synset string, e.g. 'person.n.01' -> 'person'."""
    synset = COCO_TO_WNSYNSET.get(cid, "")
    if not synset:
        return str(cid)
    word = synset.rsplit('.', 2)[0]  # strip '.n.01'
    return word.replace('_', ' ')


# ============================================================================
# BIG-5 GROUND TRUTH: build one row per actual image (not per post)
# ============================================================================
def _map_yn(val):
    """'Yes'/'No' -> 1/0, anything else (blank, NaN) -> None."""
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if v == 'yes':
        return 1
    if v == 'no':
        return 0
    return None


def _map_materiality(val):
    """'material'/'immaterial' -> our convention (material=1), blank -> None."""
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if v == 'material':
        return 1
    if v == 'immaterial':
        return 0
    return None


def _map_biological(val):
    """'biotic'/'abiotic' -> our convention (biotic=1), blank -> None."""
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if v == 'biotic':
        return 1
    if v == 'abiotic':
        return 0
    return None


def build_big5_image_records(gt_csv, media_csv, language):
    """
    Join gt_csv (table_for_pau_twitter-{lang}-6.csv) with media_csv
    (phase-1_twitter-{lang}.csv) on platform_id, and explode into one record
    per actual image (0..3). Returns a list of dicts:
        {language, platform_id, image_index, filename, gt_nature, gt_biotic, gt_material}
    Applies the verified correction rule: if nature_visual_N == 'No' but
    materiality/biological are still populated, the stray detail is dropped
    (nature_visual_N is authoritative) -- per user decision.

    WHY "EXPLODE"? Each ROW in the GT csv describes one Twitter POST, which
    can have UP TO 4 attached images (indices 0-3, each with its own
    nature/materiality/biological annotation column suffixed _0.._3). This
    function turns "N posts, each with up to 4 image-slots" into a flat list
    of "one dict per actual image" — the natural unit for evaluation, since
    each image gets its own model prediction.
    """
    gt = pd.read_csv(gt_csv)
    media = pd.read_csv(media_csv)
    # Inner join on platform_id: only posts present in BOTH the GT annotation
    # table AND the media-file listing survive — a post with no matching
    # media_files entry has nothing to download/evaluate anyway.
    joined = gt.merge(media, on='platform_id', how='inner')

    records = []
    n_dropped_stray_detail = 0
    for _, row in joined.iterrows():
        media_files_raw = row.get('media_files')
        if pd.isna(media_files_raw):
            continue
        try:
            # media_files is stored as a STRING that looks like a Python list
            # literal (e.g. "['123_0.jpg', '123_1.jpg']") — ast.literal_eval
            # safely parses it back into an actual Python list without using
            # eval() (which would allow arbitrary code execution).
            filenames = ast.literal_eval(media_files_raw)
        except (ValueError, SyntaxError):
            continue

        for idx in range(4):
            if idx >= len(filenames):
                continue  # no image at this slot for this post
            nat = _map_yn(row.get(f'nature_visual_{idx}'))
            if nat is None:
                continue  # this slot wasn't coded at all

            mat_raw = row.get(f'nep_materiality_visual_{idx}')
            bio_raw = row.get(f'nep_biological_visual_{idx}')
            mat = _map_materiality(mat_raw)
            bio = _map_biological(bio_raw)

            if nat == 0 and (mat is not None or bio is not None):
                # Verified data-quality correction (see module docstring):
                # nature_visual_N is authoritative; drop stray detail values.
                n_dropped_stray_detail += 1
                mat, bio = None, None

            records.append({
                "language": language,
                "platform_id": row['platform_id'],
                "image_index": idx,
                "filename": filenames[idx],
                "gt_nature": nat,
                "gt_biotic": bio,
                "gt_material": mat,
            })

    return records, n_dropped_stray_detail


def select_diagnostic_sample(all_records, size=20, seed=42):
    """
    Stratified (not purely random) selection covering diverse ground-truth
    combinations, verified against real counts in the ES data before design:
    even the rarest combination (abiotic+immaterial nature) has 7 examples in
    ES alone, so every bucket below is realistically fillable.

    Buckets (target counts sum to `size`, scaled proportionally if size != 20):
      nature=0                                            (no nature at all)
      nature=1, biotic=1, material=1                      (typical real nature photo)
      nature=1, biotic=1, material=0                      (depicted/illustrated biotic content)
      nature=1, biotic=0, material=1                      (real abiotic nature content)
      nature=1, biotic=0, material=0                      (depicted/illustrated abiotic content)
      nature=1, (biotic is None or material is None)      (annotation gap -- coder said "yes,
                                                             nature" but left a sub-label blank)

    WHY "STRATIFIED" INSTEAD OF PLAIN RANDOM? A purely random 20-image sample
    from real data (which skews heavily toward one or two common
    combinations) could easily end up with zero examples of some rarer
    category, making it useless for eyeballing how a model handles that case.
    This function instead guarantees a MINIMUM presence from each
    meaningfully-different ground-truth combination.
    """
    import random
    rnd = random.Random(seed)

    buckets = {
        "nature0": [], "bio1_mat1": [], "bio1_mat0": [], "bio0_mat1": [], "bio0_mat0": [], "gap": [],
    }
    for r in all_records:
        if r["gt_nature"] == 0:
            buckets["nature0"].append(r)
        elif r["gt_biotic"] is None or r["gt_material"] is None:
            buckets["gap"].append(r)
        else:
            buckets[f"bio{r['gt_biotic']}_mat{r['gt_material']}"].append(r)

    for items in buckets.values():
        rnd.shuffle(items)

    # Target allocation for size=20; scaled proportionally for other sizes.
    base_targets = {"nature0": 4, "bio1_mat1": 4, "bio1_mat0": 3, "bio0_mat1": 3, "bio0_mat0": 3, "gap": 3}
    scale = size / sum(base_targets.values())
    targets = {k: max(1, round(v * scale)) for k, v in base_targets.items()}

    selected = []
    for name, target in targets.items():
        selected.extend(buckets[name][:target])

    if len(selected) < size:
        # Rounding the per-bucket targets can leave the total slightly short
        # of `size` — top up from whatever's left over across ALL buckets
        # (beyond each bucket's own target count), shuffled together so the
        # top-up isn't biased toward any one bucket.
        leftovers = []
        for name, target in targets.items():
            leftovers.extend(buckets[name][target:])
        rnd.shuffle(leftovers)
        selected.extend(leftovers[:size - len(selected)])

    selected = selected[:size]
    rnd.shuffle(selected)  # don't present them grouped by bucket
    return selected


def load_or_create_diagnostic_sample(all_records, sample_file, size=20):
    """
    Loads a previously-persisted diagnostic sample if it exists, so different
    --model_family runs compare predictions on the EXACT SAME images. If it
    doesn't exist yet, creates one via stratified sampling and saves it, so
    subsequent runs (regardless of model) reuse this same fixed set.
    """
    if os.path.isfile(sample_file):
        with open(sample_file, "r") as f:
            return json.load(f)
    sample = select_diagnostic_sample(all_records, size=size)
    to_save = [{
        "filename": r["filename"], "language": r["language"], "platform_id": str(r["platform_id"]),
        "image_index": r["image_index"], "gt_nature": r["gt_nature"],
        "gt_biotic": r["gt_biotic"], "gt_material": r["gt_material"],
    } for r in sample]
    with open(sample_file, "w") as f:
        json.dump(to_save, f, indent=2)
    return to_save


def update_comparison_file(comparison_file, diagnostic_sample, results_by_filename, model_id):
    """
    Persistent, cross-run JSON: {filename: {gt_nature, gt_biotic, gt_material,
    predictions: {model_id: {...}, ...}}}. Loads any existing file, OVERWRITES
    only this model_id's entry for each of the diagnostic sample's images
    (Python dict assignment naturally "deletes old, saves new" for the same
    key), leaves every other model's entries untouched, and saves back.
    """
    if os.path.isfile(comparison_file):
        with open(comparison_file, "r") as f:
            comparison = json.load(f)
    else:
        comparison = {}

    for d in diagnostic_sample:
        filename = d["filename"]
        entry = comparison.setdefault(filename, {
            "language": d["language"], "gt_nature": d["gt_nature"],
            "gt_biotic": d["gt_biotic"], "gt_material": d["gt_material"],
            "predictions": {},
        })
        r = results_by_filename.get(filename)
        if r is None:
            entry["predictions"][model_id] = {"missing_from_this_run": True}
        else:
            entry["predictions"][model_id] = {
                "raw_prediction": r["raw_prediction"],
                "pred_nature": r["pred_nature"],
                "pred_biotic": r["pred_biotic"],
                "pred_material": r["pred_material"],
                "no_taxonomy_match": r["no_taxonomy_match"],
            }

    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def download_image_if_missing(filename, cache_dir):
    """Download from IMG_BASE_URL into cache_dir if not already present. Returns
    the local path, or None if the download failed (skip that image)."""
    local_path = os.path.join(cache_dir, filename)
    if os.path.isfile(local_path):
        return local_path
    os.makedirs(cache_dir, exist_ok=True)
    url = IMG_BASE_URL + filename
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"⚠️  Failed to download {url}: {e}")
        return None


class Big5ImageDataset(Dataset):
    """One instance per actual image. __getitem__ returns
    (image_tensor, gt_nature, gt_biotic, gt_material, language, platform_id, image_index).
    GT fields are plain ints or None -- collated as lists via big5_collate_fn,
    NOT stacked into tensors, so None passes through unchanged."""

    def __init__(self, records, cache_dir, transform):
        self.records = records
        self.cache_dir = cache_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        local_path = download_image_if_missing(rec["filename"], self.cache_dir)
        if local_path is None:
            return None  # filtered out by collate_fn
        try:
            image = Image.open(local_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Could not open {local_path}: {e}")
            return None
        if self.transform is not None:
            image = self.transform(image)
        return (image, rec["gt_nature"], rec["gt_biotic"], rec["gt_material"],
                rec["language"], rec["platform_id"], rec["image_index"], rec["filename"])


def big5_collate_fn(batch):
    """Drops failed downloads/opens (None entries), stacks images into a
    tensor, keeps GT/metadata as plain Python lists so None values survive.

    WHY A CUSTOM COLLATE FUNCTION? PyTorch's DEFAULT collate function tries
    to stack every field of a batch into a tensor — which fails outright on
    a list containing `None` (a missing biotic/material label) mixed with
    integers, since there's no tensor dtype that represents "some values are
    just absent." This custom function only stacks the IMAGE tensors (which
    are always present and same-shaped) and leaves the GT/metadata fields as
    plain Python lists, so a None passes through completely unchanged.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b[0] for b in batch])
    gt_nature = [b[1] for b in batch]
    gt_biotic = [b[2] for b in batch]
    gt_material = [b[3] for b in batch]
    languages = [b[4] for b in batch]
    platform_ids = [b[5] for b in batch]
    image_indices = [b[6] for b in batch]
    filenames = [b[7] for b in batch]
    return images, gt_nature, gt_biotic, gt_material, languages, platform_ids, image_indices, filenames


def parse_args():
    """Command-line flags: BIG-5 data/cache paths, diagnostic-sample /
    cross-model comparison bookkeeping, taxonomy path, which model family to
    run and its family-specific arguments, and generation/testing/logging options."""
    parser = argparse.ArgumentParser(description="Evaluate a model family against the BIG-5 Twitter dataset")
    parser.add_argument("--twitter_en_gt_csv", type=str, default=None, help="table_for_pau_twitter-en-6.csv")
    parser.add_argument("--twitter_es_gt_csv", type=str, default=None, help="table_for_pau_twitter-es-6.csv")
    parser.add_argument("--twitter_en_media_csv", type=str, default=None, help="phase-1_twitter-en.csv")
    parser.add_argument("--twitter_es_media_csv", type=str, default=None, help="phase-1_twitter-es.csv")
    parser.add_argument("--images_cache_dir", type=str, default="/home/pmonserrat/datasets/big_5",
                        help="Where downloaded images are cached (avoids re-downloading every run).")
    parser.add_argument("--diagnostic_sample_file", type=str, default="big5_diagnostic_sample.json",
                        help="Path to persist/reuse a FIXED set of images for qualitative comparison "
                             "across different --model_family runs. Created via stratified sampling "
                             "(covering diverse nature/biotic/material combinations) on first run, "
                             "then reused identically on every subsequent run regardless of which "
                             "model you're testing, so predictions are directly comparable.")
    parser.add_argument("--diagnostic_sample_size", type=int, default=20)
    parser.add_argument("--model_id", type=str, default=None,
                        help="Explicit identifier for THIS specific model, used as the key in "
                             "--comparison_file. Defaults to an auto-derived name per family "
                             "(e.g. --model_name for imagenet, --places_model_name for places) if "
                             "not given. Set this explicitly if you're comparing multiple models "
                             "within the same family (e.g. convnext_base vs vit_b_16), since the "
                             "auto-derived default would otherwise collide for same-named args.")
    parser.add_argument("--comparison_file", type=str, default=None,
                        help="Path to a persistent JSON file accumulating every model's predictions "
                             "on the fixed diagnostic sample, indexed by image filename then by "
                             "--model_id. Re-running the same --model_id overwrites just that "
                             "model's entry; other models' entries are preserved. Defaults to "
                             "'big5_model_comparison.json' next to --diagnostic_sample_file if not given.")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx",
                        help="Taxonomy workbook. Needed for imagenet/places/coco_q2l families "
                             "(to project their predictions onto the taxonomy); not used for "
                             "multitask_direct, which predicts the taxonomy dimensions directly.")

    parser.add_argument("--model_family", type=str, required=True,
                        choices=["imagenet", "places", "coco_q2l", "multitask_direct"])

    # imagenet family
    parser.add_argument("--model_name", type=str, default=None,
                        help="[imagenet family] torchvision model name, e.g. convnext_base.")
    # places family
    parser.add_argument("--places_model_name", type=str, default="resnet50",
                        help="[places family] torchvision architecture name.")
    parser.add_argument("--places_weights", type=str, default=None,
                        help="[places family] Path to a Places365 checkpoint.")
    parser.add_argument("--places_categories_txt", type=str, default=None,
                        help="[places family] Path to categories_places365.txt.")
    parser.add_argument("--places_sourcekey_sheet", type=str, default="sourcekey")
    parser.add_argument("--places_missing_sheet", type=str, default="still missing MIT Places")
    parser.add_argument("--places_mapping_csv", type=str, default=None,
                        help="[places family] Optional authoritative places->synset mapping CSV.")
    # coco_q2l family
    parser.add_argument("--q2l_repo_path", type=str, default=None)
    parser.add_argument("--q2l_config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="[coco_q2l family] Sigmoid threshold for 'predicted positive'.")
    # multitask_direct family
    parser.add_argument("--multitask_checkpoint_path", type=str, default=None)
    parser.add_argument("--multitask_backbone_choice", type=str, default="DenseNet121",
                        choices=["DenseNet121", "ResNet18", "EfficientNetB0", "ResNet50"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="big5_baseline.json")
    parser.add_argument("--eyeball_samples_per_bucket", type=int, default=3,
                        help="How many example images to print per qualitative category "
                             "(nature TP/FP/TN/FN, no-taxonomy-match, biotic/material errors) "
                             "when --verbose is set.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Each model family has its own conditionally-required arguments (argparse
    # itself can't express "required only when --model_family is X" directly),
    # so these are validated manually right after parsing.
    if args.model_family == "imagenet" and not args.model_name:
        parser.error("--model_family imagenet requires --model_name")
    if args.model_family == "places" and not args.places_weights:
        parser.error("--model_family places requires --places_weights")
    if args.model_family == "coco_q2l":
        missing = [n for n in ("q2l_repo_path", "q2l_config", "checkpoint_path") if getattr(args, n) is None]
        if missing:
            parser.error(f"--model_family coco_q2l requires: {', '.join('--' + m for m in missing)}")
    if args.model_family == "multitask_direct" and not args.multitask_checkpoint_path:
        parser.error("--model_family multitask_direct requires --multitask_checkpoint_path")

    # Auto-derive --model_id if not explicitly given. Specific enough to
    # distinguish different models WITHIN the same family (e.g. convnext_base
    # vs vit_b_16 are both "imagenet" family, but different models).
    if args.model_id is None:
        if args.model_family == "imagenet":
            args.model_id = args.model_name
        elif args.model_family == "places":
            args.model_id = f"places_{args.places_model_name}"
        elif args.model_family == "coco_q2l":
            ckpt_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
            cfg_dir = os.path.basename(os.path.dirname(os.path.abspath(args.q2l_config)))
            args.model_id = f"q2l_{cfg_dir}" if cfg_dir else f"q2l_{ckpt_name}"
        else:  # multitask_direct
            args.model_id = f"multitask_{args.multitask_backbone_choice}"

    if args.comparison_file is None:
        base_dir = os.path.dirname(os.path.abspath(args.diagnostic_sample_file))
        args.comparison_file = os.path.join(base_dir, "big5_model_comparison.json")

    return args


def safe_binary_map(val, positive_str, negative_str):
    """Safely converts string annotations to binary labels (identical to the
    other baseline scripts' identically-named function)."""
    if not isinstance(val, str):
        return None
    val = val.strip().lower()
    if val == positive_str.lower():
        return 1
    if val == negative_str.lower():
        return 0
    return None


def compute_binary_metrics(gts, preds):
    """gts/preds: parallel lists, entries may be None (excluded). A None
    PREDICTION is penalized as wrong, matching convention used throughout
    this project. A None GROUND TRUTH excludes that instance entirely."""
    valid_gt, valid_pred = [], []
    for gt, pred in zip(gts, preds):
        if gt is None:
            # No usable ground truth for this instance at all — can't score
            # it either way, so it's excluded entirely (not penalized).
            continue
        valid_gt.append(gt)
        # A missing PREDICTION (None — e.g. "no taxonomy match") is instead
        # scored as the OPPOSITE of ground truth, guaranteeing it counts as
        # an error rather than being silently dropped from the metric.
        valid_pred.append((1 - gt) if pred is None else pred)
    if not valid_gt:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}
    acc = accuracy_score(valid_gt, valid_pred)
    p, r, f1, _ = precision_recall_fscore_support(valid_gt, valid_pred, average='binary', zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
            "f1": float(f1), "support": len(valid_gt)}


def stratified_eyeball_examples(results, n_per_bucket=3, seed=42):
    """
    Pull examples from meaningful categories rather than pure random sampling,
    so a small qualitative sample actually covers the different kinds of
    cases worth eyeballing: the four points of the nature confusion matrix,
    no-taxonomy-match cases specifically, and biotic/material errors among
    nature-positive images. Returns an ordered dict of {bucket_name: [records]}.
    Buckets with fewer than n_per_bucket available examples just return what
    exists (including empty), rather than erroring.
    """
    import random
    rng = random.Random(seed)

    def sample_bucket(pool):
        return rng.sample(pool, min(n_per_bucket, len(pool)))

    # Each bucket below corresponds to one cell of the standard binary
    # confusion matrix (True/False Positive/Negative) for the nature axis,
    # plus two extra diagnostic categories (no-taxonomy-match, and
    # biotic/material errors) that the plain confusion matrix doesn't surface.
    buckets = {}
    buckets["Nature: True Positive (gt=Yes, pred=Yes)"] = sample_bucket(
        [r for r in results if r["gt_nature"] == 1 and r["pred_nature"] == 1])
    buckets["Nature: False Negative (gt=Yes, pred=No)"] = sample_bucket(
        [r for r in results if r["gt_nature"] == 1 and r["pred_nature"] == 0])
    buckets["Nature: False Positive (gt=No, pred=Yes)"] = sample_bucket(
        [r for r in results if r["gt_nature"] == 0 and r["pred_nature"] == 1])
    buckets["Nature: True Negative (gt=No, pred=No)"] = sample_bucket(
        [r for r in results if r["gt_nature"] == 0 and r["pred_nature"] == 0])
    buckets["No taxonomy match (any ground truth)"] = sample_bucket(
        [r for r in results if r["no_taxonomy_match"]])
    buckets["Biotic/Abiotic: wrong prediction (nature-positive only)"] = sample_bucket(
        [r for r in results if r["gt_biotic"] is not None and r["pred_biotic"] is not None
         and r["gt_biotic"] != r["pred_biotic"]])
    buckets["Material/Immaterial: wrong prediction (nature-positive only)"] = sample_bucket(
        [r for r in results if r["gt_material"] is not None and r["pred_material"] is not None
         and r["gt_material"] != r["pred_material"]])
    return buckets


def print_eyeball_examples(buckets, images_cache_dir):
    """Pretty-print each category's sampled examples: local file path, the
    model's raw output, ground truth, and the mapped/final prediction."""
    for bucket_name, examples in buckets.items():
        print(f"\n--- {bucket_name} ({len(examples)} shown) ---")
        if not examples:
            print("  (none found -- this category is empty in this run's results)")
            continue
        for r in examples:
            local_path = os.path.join(images_cache_dir, r["filename"])
            print(f"  {local_path}")
            print(f"    raw model output: {r['raw_prediction']}")
            print(f"    gt:              nature={r['gt_nature']} biotic={r['gt_biotic']} material={r['gt_material']}")
            print(f"    mapped pred:     nature={r['pred_nature']} biotic={r['pred_biotic']} "
                  f"material={r['pred_material']}"
                  f"{'  [NO TAXONOMY MATCH]' if r['no_taxonomy_match'] else ''}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting BIG-5 Baseline ({args.model_family}) on {device.upper()}")

    if args.wandb:
        import wandb
        wandb.init(
            entity="paumonserrat03-universitat-aut-noma-de-barcelona",
            project="TFM_Closed-set",
            config=vars(args),
            name=f"big5_baseline_{args.model_family}",
        )

    # ==========================================
    # 1. BUILD BIG-5 GROUND TRUTH (both languages)
    # ==========================================
    all_records = []
    for lang, gt_csv, media_csv in [
        ("en", args.twitter_en_gt_csv, args.twitter_en_media_csv),
        ("es", args.twitter_es_gt_csv, args.twitter_es_media_csv),
    ]:
        if gt_csv is None or media_csv is None:
            continue
        recs, n_dropped = build_big5_image_records(gt_csv, media_csv, lang)
        all_records.extend(recs)
        if args.verbose:
            print(f"[INFO] {lang}: {len(recs)} image instances built "
                  f"({n_dropped} stray materiality/biological values dropped per the nature_visual "
                  f"correction rule).")

    if not all_records:
        raise ValueError("No BIG-5 records built. Pass at least one --twitter_{en,es}_gt_csv "
                         "+ matching --twitter_{en,es}_media_csv pair.")

    # Built from the FULL record set, before any --max_samples subsetting, so the
    # fixed diagnostic sample stays identical across runs regardless of --max_samples.
    _sample_file_existed_before = os.path.isfile(args.diagnostic_sample_file)
    diagnostic_sample = load_or_create_diagnostic_sample(
        all_records, args.diagnostic_sample_file, args.diagnostic_sample_size)
    if args.verbose:
        print(f"[INFO] Diagnostic sample: {len(diagnostic_sample)} images "
              f"({'loaded from' if _sample_file_existed_before else 'created and saved to'} "
              f"{args.diagnostic_sample_file}).")

    if args.max_samples is not None and len(all_records) > args.max_samples:
        import random
        random.seed(42)
        all_records = random.sample(all_records, args.max_samples)
        # Make sure the diagnostic sample's images are always evaluated, even
        # under --max_samples subsetting -- otherwise a small test run would
        # silently drop the fixed comparison images.
        already_in = {r["filename"] for r in all_records}
        missing_diag = [d for d in diagnostic_sample if d["filename"] not in already_in]
        if missing_diag:
            # Find the actual full records (with all fields) for the missing diagnostic filenames.
            # We have to rebuild ALL records again here (rather than reuse the
            # already-subsetted `all_records`) because the diagnostic sample's
            # own JSON only stored a SUMMARY (filename/GT), not the full record
            # dict `Big5ImageDataset` needs — so we look the missing ones back
            # up by filename from a fresh full build.
            by_filename = {}
            for lang, gt_csv, media_csv in [
                ("en", args.twitter_en_gt_csv, args.twitter_en_media_csv),
                ("es", args.twitter_es_gt_csv, args.twitter_es_media_csv),
            ]:
                if gt_csv is None or media_csv is None:
                    continue
                recs, _ = build_big5_image_records(gt_csv, media_csv, lang)
                for r in recs:
                    by_filename[r["filename"]] = r
            for d in missing_diag:
                r = by_filename.get(d["filename"])
                if r is not None:
                    all_records.append(r)

    if args.verbose:
        n_nature = sum(1 for r in all_records if r["gt_nature"] == 1)
        n_biotic = sum(1 for r in all_records if r["gt_biotic"] == 1)
        n_abiotic = sum(1 for r in all_records if r["gt_biotic"] == 0)
        n_material = sum(1 for r in all_records if r["gt_material"] == 1)
        n_immaterial = sum(1 for r in all_records if r["gt_material"] == 0)
        print(f"\n[INFO] --- BIG-5 ground truth statistics ({len(all_records)} total images) ---")
        print(f"  Nature:                {n_nature} / {len(all_records) - n_nature}")
        print(f"  Biotic/Abiotic:        {n_biotic} / {n_abiotic}")
        print(f"  Material/Immaterial:   {n_material} / {n_immaterial}")
        print("--------------------------------------------------------------\n")

    # ==========================================
    # 2. LOAD TAXONOMY (needed for imagenet/places/coco_q2l; unused for multitask_direct)
    # ==========================================
    pipeline = None
    if args.model_family in ("imagenet", "places", "coco_q2l"):
        if args.verbose:
            print(f"[INFO] Loading Taxonomy from {args.excel_path}...")
        pipeline = TaxonomyEvaluationPipeline()
        df_taxonomy = pd.read_excel(args.excel_path, sheet_name="data corrected")
        pipeline.load_custom_excel_annotations(df_taxonomy, "Biotic/abiotic", "Material/immaterial")

    # ==========================================
    # 3. BUILD MODEL + PREPROCESSING (per family)
    # ==========================================
    q2l_col_nature = q2l_col_biotic = q2l_col_material = None
    q2l_ordered_cat_ids = None
    places_id_to_synset = None

    if args.model_family == "imagenet":
        weights = tv_models.get_model_weights(args.model_name).DEFAULT
        model = tv_models.get_model(args.model_name, weights=weights).to(device)
        model.eval()
        preprocess = weights.transforms()

    elif args.model_family == "places":
        if args.places_categories_txt is None:
            raise ValueError("--model_family places requires --places_categories_txt")
        weights = tv_models.get_model_weights(args.places_model_name).DEFAULT
        preprocess = weights.transforms()
        model = tv_models.get_model(args.places_model_name, weights=None)
        _replace_head_365(model, args.places_model_name)
        checkpoint = torch.load(args.places_weights, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        model.eval()

        places_categories = load_places365_categories(args.places_categories_txt)
        places_id_to_synset, places_report = build_places_id_to_synset(
            places_categories, excel_path=args.excel_path,
            sourcekey_sheet=args.places_sourcekey_sheet, missing_sheet=args.places_missing_sheet,
            mapping_csv=args.places_mapping_csv,
        )
        if args.verbose:
            print(f"[INFO] Places mapping: {places_report['n_mapped']} mapped, "
                  f"{places_report['n_excluded_still_missing']} excluded, "
                  f"{places_report['n_unresolved']} unresolved.")

    elif args.model_family == "coco_q2l":
        repo_root = os.path.abspath(args.q2l_repo_path)
        repo_lib = os.path.join(repo_root, "lib")
        for p in (repo_root, repo_lib):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        install_inplace_abn_shim_if_missing(verbose=args.verbose)
        try:
            from models.query2label import build_q2l
        except ImportError as e:
            raise ImportError(f"Could not import build_q2l. Check --q2l_repo_path. Original error: {e}")
        try:
            from utils.misc import clean_state_dict
        except ImportError:
            clean_state_dict = clean_state_dict_fallback

        q2l_args = build_q2l_args(args.q2l_config)
        model = build_q2l(q2l_args).to(device)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        raw_state = checkpoint.get("state_dict", checkpoint.get("model"))
        if raw_state is None:
            raise ValueError(f"Checkpoint has neither 'state_dict' nor 'model' key.")
        state_dict = clean_state_dict(raw_state)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e_strict:
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"⚠️  strict=True failed ({e_strict}); loaded with strict=False instead.")
            except RuntimeError as e_loose:
                raise RuntimeError(f"Could not load checkpoint even with strict=False: {e_loose}") from e_loose
        model.eval()

        if q2l_args.orid_norm:
            normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((q2l_args.img_size, q2l_args.img_size)),
            transforms.ToTensor(), normalize,
        ])

        # COCO category id -> synset -> taxonomy, aligned to a fixed column order
        q2l_ordered_cat_ids = sorted(COCO_TO_WNSYNSET.keys())
        q2l_col_nature, q2l_col_biotic, q2l_col_material = [], [], []
        for cid in q2l_ordered_cat_ids:
            synset_str = COCO_TO_WNSYNSET[cid]
            node_attrs = pipeline.get_node_attributes(synset_str)
            if not node_attrs:
                q2l_col_nature.append(None); q2l_col_biotic.append(None); q2l_col_material.append(None)
                continue
            is_nature = node_attrs.get('is_nature')
            q2l_col_nature.append(1 if is_nature else 0)
            q2l_col_biotic.append(safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic"))
            q2l_col_material.append(safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial"))

    else:  # multitask_direct
        backbone = CustomBackbone(model_choice=args.multitask_backbone_choice)
        model = MultiTaskModel(backbone, backbone.feature_dim).to(device)
        state = torch.load(args.multitask_checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ==========================================
    # 4. DATASET / DATALOADER
    # ==========================================
    dataset = Big5ImageDataset(all_records, args.images_cache_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=big5_collate_fn)

    # ==========================================
    # 5. INFERENCE LOOP
    # ==========================================
    results = []  # list of dicts: language, gt_nature, gt_biotic, gt_material, pred_nature, pred_biotic, pred_material
    print(f"Running inference over {len(dataset)} BIG-5 images "
          f"(downloading/caching to {args.images_cache_dir} as needed)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not args.verbose):
            if batch is None:
                # every single image in this batch failed to download/open
                continue
            images, gt_nature, gt_biotic, gt_material, languages, platform_ids, image_indices, filenames = batch
            images = images.to(device)

            if args.model_family == "imagenet":
                outputs = model(images)
                preds_idx = outputs.argmax(dim=1).cpu().tolist()
                for i, pred_idx in enumerate(preds_idx):
                    raw_pred_name = weights.meta['categories'][pred_idx]  # torchvision's own name for this index
                    wnid = imagenet_idx_to_wnid(pred_idx)
                    synset_str = pipeline.get_synset_str_from_wnid(wnid) if wnid else None
                    node_attrs = pipeline.get_node_attributes(synset_str) if synset_str else None
                    no_match = not node_attrs
                    if not node_attrs:
                        pred_nat = pred_bio = pred_mat = None
                    else:
                        pred_nat = 1 if node_attrs.get('is_nature') else 0
                        pred_bio = safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
                        pred_mat = safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
                    results.append({"language": languages[i], "gt_nature": gt_nature[i], "gt_biotic": gt_biotic[i],
                                    "gt_material": gt_material[i], "pred_nature": pred_nat,
                                    "pred_biotic": pred_bio, "pred_material": pred_mat,
                                    "filename": filenames[i], "no_taxonomy_match": no_match,
                                    "raw_prediction": f"{raw_pred_name} ({wnid})"})

            elif args.model_family == "places":
                outputs = model(images)
                preds_idx = outputs.argmax(dim=1).cpu().tolist()  # logit index == Places category id
                for i, pred_id in enumerate(preds_idx):
                    raw_pred_name = places_categories[pred_id]
                    synset_str = places_id_to_synset.get(pred_id)
                    node_attrs = pipeline.get_node_attributes(synset_str) if synset_str else None
                    no_match = not node_attrs
                    if not node_attrs:
                        pred_nat = pred_bio = pred_mat = None
                    else:
                        pred_nat = 1 if node_attrs.get('is_nature') else 0
                        pred_bio = safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
                        pred_mat = safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
                    results.append({"language": languages[i], "gt_nature": gt_nature[i], "gt_biotic": gt_biotic[i],
                                    "gt_material": gt_material[i], "pred_nature": pred_nat,
                                    "pred_biotic": pred_bio, "pred_material": pred_mat,
                                    "filename": filenames[i], "no_taxonomy_match": no_match,
                                    "raw_prediction": f"{raw_pred_name} (id={pred_id})"})

            elif args.model_family == "coco_q2l":
                scores = torch.sigmoid(model(images)).cpu().numpy()
                preds_bin = (scores >= args.threshold).astype(int)
                # Which of the 80 COCO class COLUMNS are tagged nature/biotic/
                # material — computed once outside this loop's per-image work.
                nat_pos_cols = [c for c, lab in enumerate(q2l_col_nature) if lab == 1]
                bio_pos_cols = [c for c, lab in enumerate(q2l_col_biotic) if lab == 1]
                mat_pos_cols = [c for c, lab in enumerate(q2l_col_material) if lab == 1]
                for i in range(preds_bin.shape[0]):
                    row = preds_bin[i]
                    detected = [coco_readable_name(q2l_ordered_cat_ids[c]) for c in range(len(row)) if row[c] == 1]
                    raw_pred_name = ", ".join(detected) if detected else "(nothing detected above threshold)"
                    # "no taxonomy match" for a multi-label detector = it detected
                    # NOTHING at all above threshold across all 80 COCO classes.
                    no_match = bool(row.sum() == 0)
                    # This image counts as "predicted nature" if ANY of its
                    # detected (above-threshold) classes is nature-tagged.
                    pred_nat = 1 if (nat_pos_cols and row[nat_pos_cols].sum() > 0) else 0
                    # Biotic/material only get a real 0/1 answer when the image
                    # was predicted nature at all; otherwise (pred_nat == 0)
                    # they correctly stay None ("not applicable"), matching the
                    # same rule enforced everywhere else in this project.
                    pred_bio = 1 if (bio_pos_cols and row[bio_pos_cols].sum() > 0) else (0 if pred_nat else None)
                    pred_mat = 1 if (mat_pos_cols and row[mat_pos_cols].sum() > 0) else (0 if pred_nat else None)
                    results.append({"language": languages[i], "gt_nature": gt_nature[i], "gt_biotic": gt_biotic[i],
                                    "gt_material": gt_material[i], "pred_nature": pred_nat,
                                    "pred_biotic": pred_bio, "pred_material": pred_mat,
                                    "filename": filenames[i], "no_taxonomy_match": no_match,
                                    "raw_prediction": raw_pred_name})

            else:  # multitask_direct -- always answers directly, no "unmapped" concept applies
                out_nature, out_materiality, out_biological, _out_landscape = model(images)
                nature_argmax = out_nature.argmax(dim=1).cpu().tolist()
                materiality_argmax = out_materiality.argmax(dim=1).cpu().tolist()
                biological_argmax = out_biological.argmax(dim=1).cpu().tolist()
                pred_nat_batch = nature_argmax
                pred_mat_batch = [MULTITASK_MATERIALITY_TO_OURS.get(p) for p in materiality_argmax]
                pred_bio_batch = [MULTITASK_BIOLOGICAL_TO_OURS.get(p) for p in biological_argmax]
                # Raw class choice, in the model's OWN 3-way vocabulary (material/immaterial/N-A,
                # biotic/abiotic/N-A), before remapping -- shows exactly what the model said,
                # including when it output "not applicable" (class 2).
                MAT_NAMES = {0: "material", 1: "immaterial", 2: "N/A"}
                BIO_NAMES = {0: "biotic", 1: "abiotic", 2: "N/A"}
                for i in range(len(pred_nat_batch)):
                    raw_pred_name = (f"nature={'Yes' if nature_argmax[i] == 1 else 'No'}, "
                                     f"materiality={MAT_NAMES[materiality_argmax[i]]}, "
                                     f"biological={BIO_NAMES[biological_argmax[i]]}")
                    results.append({"language": languages[i], "gt_nature": gt_nature[i], "gt_biotic": gt_biotic[i],
                                    "gt_material": gt_material[i], "pred_nature": pred_nat_batch[i],
                                    "pred_biotic": pred_bio_batch[i], "pred_material": pred_mat_batch[i],
                                    "filename": filenames[i], "no_taxonomy_match": False,
                                    "raw_prediction": raw_pred_name})

    # ==========================================
    # 6. METRICS -- combined + per-language
    # ==========================================
    def metrics_for(subset):
        """Compute all three axis metrics for one subset of results (either
        the full combined set, or just one language's results)."""
        return {
            "nature": compute_binary_metrics([r["gt_nature"] for r in subset], [r["pred_nature"] for r in subset]),
            "biotic": compute_binary_metrics([r["gt_biotic"] for r in subset], [r["pred_biotic"] for r in subset]),
            "material": compute_binary_metrics([r["gt_material"] for r in subset], [r["pred_material"] for r in subset]),
        }

    combined_metrics = metrics_for(results)
    per_language_metrics = {}
    for lang in sorted(set(r["language"] for r in results)):
        per_language_metrics[lang] = metrics_for([r for r in results if r["language"] == lang])

    # ==========================================
    # 6b. DIAGNOSTIC: "no taxonomy match" rate + eyeball examples
    # ==========================================
    # This directly tests the domain-gap hypothesis rather than asserting it:
    # for imagenet/places, "no match" means the predicted class isn't in the
    # taxonomy at all; for coco_q2l, it means nothing was detected above
    # threshold across all 80 classes. multitask_direct always answers, so
    # this concept doesn't apply there (no_taxonomy_match is always False).
    n_no_match = sum(1 for r in results if r["no_taxonomy_match"])
    no_match_rate = n_no_match / len(results) if results else 0.0

    print("\n" + "-" * 60)
    print(f"DIAGNOSTIC: no-taxonomy-match rate = {no_match_rate:.1%} "
          f"({n_no_match}/{len(results)} images)")
    if args.model_family == "multitask_direct":
        print("  (not applicable -- this model always answers directly, no 'unmapped' concept)")
    else:
        print("  High values here support a domain-gap explanation for low metrics: it means")
        print("  the model is frequently landing on a class with no taxonomy signal at all,")
        print("  rather than confidently giving a wrong taxonomy answer.")
    print("-" * 60)

    if args.verbose:
        buckets = stratified_eyeball_examples(results, n_per_bucket=args.eyeball_samples_per_bucket)
        print(f"\n=== QUALITATIVE SAMPLES BY CATEGORY (up to {args.eyeball_samples_per_bucket} each) ===")
        print_eyeball_examples(buckets, args.images_cache_dir)
        print("-" * 60)

    # ==========================================
    # 6c. FIXED CROSS-MODEL DIAGNOSTIC SAMPLE + PERSISTENT COMPARISON FILE
    # ==========================================
    # Unlike the buckets above (which depend on THIS model's confusion-matrix
    # outcomes, so the specific images differ run to run), this is a FIXED set
    # of images -- selected once via stratified sampling over ground truth
    # alone (see select_diagnostic_sample), persisted to --diagnostic_sample_file,
    # and reused identically across every --model_family run. This is what
    # makes different models' predictions on the SAME images directly comparable.
    results_by_filename = {r["filename"]: r for r in results}

    comparison = update_comparison_file(args.comparison_file, diagnostic_sample,
                                        results_by_filename, args.model_id)
    print(f"\n💾 Updated cross-model comparison file: {args.comparison_file} "
          f"(model_id='{args.model_id}', {len(comparison)} images tracked, "
          f"{len(set(m for e in comparison.values() for m in e['predictions']))} models so far)")

    if args.verbose:
        print(f"\n=== FIXED CROSS-MODEL DIAGNOSTIC SAMPLE "
              f"({len(diagnostic_sample)} images, from {args.diagnostic_sample_file}) ===")
        for d in diagnostic_sample:
            r = results_by_filename.get(d["filename"])
            local_path = os.path.join(args.images_cache_dir, d["filename"])
            print(f"  {local_path}")
            if r is None:
                print(f"    [MISSING from this run -- download or image-open failed for this run]")
                continue
            print(f"    raw model output: {r['raw_prediction']}")
            print(f"    gt:              nature={r['gt_nature']} biotic={r['gt_biotic']} material={r['gt_material']}")
            print(f"    mapped pred:     nature={r['pred_nature']} biotic={r['pred_biotic']} "
                  f"material={r['pred_material']}"
                  f"{'  [NO TAXONOMY MATCH]' if r['no_taxonomy_match'] else ''}")
        print("-" * 60)

    # ==========================================
    # 7. TERMINAL SUMMARY & SAVE
    # ==========================================
    print("\n" + "=" * 60)
    print(f"📊 BIG-5 BASELINE: {args.model_family.upper()}")
    print("=" * 60)

    def print_block(title, m):
        print(f"\n--- {title} ---")
        for task in ("nature", "biotic", "material"):
            t = m[task]
            print(f"  {task.capitalize():10s} (Support: {t['support']:5d})  "
                  f"Acc: {t['accuracy']:.4f}  P: {t['precision']:.4f}  "
                  f"R: {t['recall']:.4f}  F1: {t['f1']:.4f}")

    print_block("COMBINED (all languages)", combined_metrics)
    for lang, m in per_language_metrics.items():
        print_block(f"LANGUAGE: {lang}", m)
    print("=" * 60)

    if args.wandb:
        import wandb
        wandb_log_dict = {}
        for task in ("nature", "biotic", "material"):
            for k, v in combined_metrics[task].items():
                wandb_log_dict[f"Combined/{task.capitalize()}/{k}"] = v
            for lang, m in per_language_metrics.items():
                for k, v in m[task].items():
                    wandb_log_dict[f"{lang}/{task.capitalize()}/{k}"] = v
        wandb.log(wandb_log_dict)

    diagnostic_sample_predictions = []
    for d in diagnostic_sample:
        r = results_by_filename.get(d["filename"])
        entry = dict(d)
        if r is not None:
            entry.update({"raw_prediction": r["raw_prediction"], "pred_nature": r["pred_nature"],
                         "pred_biotic": r["pred_biotic"], "pred_material": r["pred_material"],
                         "no_taxonomy_match": r["no_taxonomy_match"]})
        else:
            entry["missing_from_this_run"] = True
        diagnostic_sample_predictions.append(entry)

    summary_results = {
        "model_family": args.model_family,
        "model_id": args.model_id,
        "comparison_file": args.comparison_file,
        "dataset": "big5_twitter",
        "samples_evaluated": len(results),
        "no_taxonomy_match_count": n_no_match,
        "no_taxonomy_match_rate": no_match_rate,
        "combined": combined_metrics,
        "per_language": per_language_metrics,
        "diagnostic_sample_file": args.diagnostic_sample_file,
        "diagnostic_sample_predictions": diagnostic_sample_predictions,
    }
    with open(args.output_file, "w") as f:
        json.dump(summary_results, f, indent=4)
    print(f"💾 Results saved to {args.output_file}")

    if args.wandb:
        import wandb
        wandb.save(args.output_file)
        wandb.finish()


if __name__ == "__main__":
    main()
