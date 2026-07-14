#!/usr/bin/env python3
"""
Evaluate closed-set SOTA classifiers on the Places365 validation split, projected
onto the BIG-5 nature taxonomy (nature / biotic-abiotic / material-immaterial).

This is the Places365 counterpart of evaluate_imagenet.py. The structure, metrics
and W&B logging are intentionally kept identical. The ONLY conceptually different
part is the class->taxonomy mapping, because Places365 gives us category ids and
scene names instead of WordNet ids.

------------------------------------------------------------------------------
IMPORTANT NOTES ON THE PLACES -> WORDNET MAPPING
------------------------------------------------------------------------------
Unlike ImageNet, torchvision does not hand us a WordNet id per Places class. The
original Places->WordNet mapping your team produced during the workshop was
MANUAL and was never stored as an explicit id/name->synset column. It therefore
CANNOT be reconstructed losslessly by any automatic heuristic: several scene
names share a lemma with a synset that was actually mapped from a *different*
Places class (e.g. `forest.n.02` came from a forest road/path variant, while
`forest/broadleaf` was deliberately left unmapped; `stadium/baseball` naively
resolves to `baseball.n.02`, the sport, not the venue).

To stay faithful to the real labels instead of fabricating them, this script:

  1. Reads the 'still missing MIT Places' sheet of your taxonomy workbook as an
     authoritative EXCLUSION SET. Any Places class listed there is treated as
     "unmapped" and excluded from the taxonomic (nature/biotic/material)
     metrics -- exactly like unmapped ImageNet classes are handled.

  2. For every remaining class, it resolves a WordNet synset via nltk and keeps
     it ONLY if that synset is actually present in the 'sourcekey' sheet of the
     same workbook. This guarantees we never invent a synset the taxonomy does
     not contain.

  3. It REPORTS (and, unless --allow-unresolved, hard-fails) any class it cannot
     resolve, so you resolve those by hand rather than trusting a silent guess.

Both sheets live in the SAME workbook passed via --excel_path (the one that
also has the 'data corrected' sheet used for the taxonomy labels themselves),
so no separate CSV files are needed.

You can also bypass all reconstruction by passing --mapping_csv pointing to a
2-column file `places_name,synset` (or `places_id,synset`). If provided, that
file is authoritative and WordNet resolution is skipped. THIS IS THE RECOMMENDED
PATH once you have exported the real mapping.

------------------------------------------------------------------------------
THE IMAGEFOLDER SORTING TRAP (read this)
------------------------------------------------------------------------------
Your val_formatted folder groups images into 365 subfolders whose names are the
category ids "0".."364". torchvision.ImageFolder sorts class directory names
LEXICOGRAPHICALLY as strings, so folder "10" sorts before "2". That means the
integer label the model/dataloader emits is NOT the Places category id -- it is
the position of the (string-sorted) folder. This script recovers the true
Places category id from the folder NAME (int(name)) and maps through that, so the
lexicographic ordering never corrupts the labels.

------------------------------------------------------------------------------
WHY RESOLUTION IS RESTRICTED TO 'MIT'-TAGGED SYNSETS
------------------------------------------------------------------------------
flat_wordnet_tree_fixed.xlsx's 'sourcekey' sheet tags every taxonomy synset with the
dataset(s) it was drawn from (wordnet / imagenet / coco / MIT / combinations).
Only synsets tagged with 'MIT' originate from the Places mapping, so those are
the only legitimate resolution targets for a Places class name. Matching
against the FULL taxonomy is unsafe: e.g. the Places class "house" resolves to
`house.n.01`, but that synset is tagged purely 'imagenet' in your sourcekey --
i.e. it was mapped from an ImageNet class, not from Places' "house" -- so
accepting it would silently borrow the wrong node. Restricting the candidate
pool to MIT-tagged synsets avoids this failure mode (verified: 0 false
positives against the known still_missing exclusion set, vs. 13 when matching
against the unrestricted taxonomy).

------------------------------------------------------------------------------
CATEGORY LIST IS READ FROM DISK, NOT HARDCODED
------------------------------------------------------------------------------
categories_places365.txt (id -> scene name) lives alongside val_formatted in
your dataset folder. This script loads it directly rather than embedding a
transcribed copy, so there is no risk of transcription drift and it stays
correct if you're on Places365-Challenge or any custom variant.
"""

import os
import sys
import argparse
import json
import random

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import wandb

# Get the absolute path of the directory one level up and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from first_tests.evaluation import TaxonomyEvaluationPipeline


# ============================================================================
# MULTITASK DIRECT-TAXONOMY MODEL (Paula Feliu's TFG) -- same as evaluate_imagenet.py
# https://github.com/paulafeliu/TFG-Interpretability-Techniques-in-Social-Media-Images
# ============================================================================
# Inlined rather than imported from a cloned repo (see evaluate_imagenet.py's
# comment for the full rationale). Copied verbatim from models/backbone.py and
# models/multitask_model.py, with the same weights=None change (harmless,
# since load_state_dict(strict=True) overwrites every weight anyway).
class CustomBackbone(nn.Module):
    def __init__(self, model_choice='ResNet18'):
        super(CustomBackbone, self).__init__()
        self.model_choice = model_choice
        if model_choice == 'DenseNet121':
            model_base = models.densenet121(weights=None)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1024
        elif model_choice == 'ResNet18':
            model_base = models.resnet18(weights=None)
            model_base.fc = nn.Identity()
            self.feature_dim = 512
        elif model_choice == 'EfficientNetB0':
            model_base = models.efficientnet_b0(weights=None)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            model_base = models.resnet50(weights=None)
            model_base.fc = nn.Identity()
            self.feature_dim = 2048
        self.backbone = model_base

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feature_dim):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.fc_nature = nn.Linear(feature_dim, 2)
        self.fc_materiality = nn.Linear(feature_dim, 3)
        self.fc_biological = nn.Linear(feature_dim, 3)
        self.fc_landscape = nn.Linear(feature_dim, 8)

    def forward(self, x):
        features = self.backbone(x)
        out_nature = self.fc_nature(features)
        out_materiality = self.fc_materiality(features)
        out_biological = self.fc_biological(features)
        out_landscape = self.fc_landscape(features)
        return out_nature, out_materiality, out_biological, out_landscape


# Verified label encodings (see evaluate_imagenet.py for the exact source citation):
#   nature_visual:          {"Yes": 1, "No": 0}                       -> matches our convention directly
#   nep_materiality_visual: {"material": 0, "immaterial": 1, "nan": 2} -> OPPOSITE of our convention
#   nep_biological_visual:  {"biotic": 0, "abiotic": 1, "nan": 2}      -> OPPOSITE of our convention
MULTITASK_MATERIALITY_TO_OURS = {0: 1, 1: 0}  # their material(0)->our 1, their immaterial(1)->our 0
MULTITASK_BIOLOGICAL_TO_OURS = {0: 1, 1: 0}   # their biotic(0)->our 1, their abiotic(1)->our 0


# ============================================================================
# PLACES365 CATEGORY LIST (id -> scene name), LOADED FROM DISK
# ============================================================================
def load_places365_categories(categories_txt):
    """
    Parse categories_places365.txt into an id-ordered list of 365 scene names,
    e.g. index 0 -> 'airfield', index 8 -> 'apartment_building/outdoor'.

    Expected line format (order in the file does NOT matter, since the id is
    read explicitly from each line):
        /a/airfield 0
        /a/apartment_building/outdoor 8
        ...
    The leading "/x/" dataset-letter prefix is stripped; any remaining slash
    (two-level categories such as "apartment_building/outdoor") is preserved,
    matching the naming convention used in your still_missing CSV.
    """
    if not os.path.isfile(categories_txt):
        raise FileNotFoundError(
            f"Could not find categories_places365.txt at '{categories_txt}'. "
            f"Pass its location explicitly via --places_categories_txt."
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
                raise ValueError(
                    f"{categories_txt}:{line_no}: could not parse line '{raw_line.rstrip()}'. "
                    f"Expected format '/x/name <id>'."
                )
            # strip the leading "/x/" (a single dataset-letter between two slashes)
            if path.startswith("/") and len(path) > 3 and path[2] == "/":
                name = path[3:]
            else:
                name = path.lstrip("/")
            id_to_name[idx] = name

    if set(id_to_name.keys()) != set(range(365)):
        missing_ids = sorted(set(range(365)) - set(id_to_name.keys()))
        extra_ids = sorted(set(id_to_name.keys()) - set(range(365)))
        raise ValueError(
            f"{categories_txt} does not contain exactly ids 0..364. "
            f"Missing ids: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}; "
            f"Unexpected ids: {extra_ids[:10]}{'...' if len(extra_ids) > 10 else ''}."
        )

    return [id_to_name[i] for i in range(365)]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Closed-Set Models on Nature Taxonomy (Places365)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Places365 validation split (val_formatted; subfolders named 0..364).")
    parser.add_argument("--places_categories_txt", type=str, default=None,
                        help="Path to categories_places365.txt (id -> scene name). Defaults to "
                             "'categories_places365.txt' inside the parent directory of --data_dir "
                             "(e.g. --data_dir .../places/val_formatted -> .../places/categories_places365.txt).")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx",
                        help="Path to the taxonomy workbook. Must contain the 'data corrected' sheet "
                             "(same as evaluate_imagenet.py) plus, for Places365 resolution, the "
                             "'sourcekey' and 'still missing MIT Places' sheets (see --sourcekey_sheet "
                             "/ --missing_sheet). No separate CSV files are needed.")
    parser.add_argument("--sourcekey_sheet", type=str, default="sourcekey",
                        help="Sheet in --excel_path listing which synsets exist in the taxonomy and "
                             "their dataset-origin tag (columns: synset, source). Has a header row.")
    parser.add_argument("--missing_sheet", type=str, default="still missing MIT Places",
                        help="Sheet in --excel_path listing Places365 classes that were NOT mapped "
                             "(exclusion set). Single column, NO header row (first row is real data).")
    parser.add_argument("--mapping_csv", type=str, default=None,
                        help="OPTIONAL authoritative mapping file: 2 columns places_name|places_id, synset. "
                             "If given, WordNet reconstruction is skipped entirely.")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Torchvision architecture name (must match --places_weights' architecture). "
                             "NOTE: must be paired with a Places365-trained checkpoint via --places_weights; "
                             "ImageNet-pretrained torchvision weights predict the wrong 1000-class label space.")
    parser.add_argument("--places_weights", type=str, default=None,
                        help="Path to a Places365 checkpoint (.pth.tar/.pth) matching --model_name's architecture. "
                             "For the official CSAIL/MIT checkpoints (resnet18, resnet50, alexnet, densenet161), "
                             "download from e.g. "
                             "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar")
    parser.add_argument("--model_type", type=str, default="torchvision",
                        choices=["torchvision", "multitask_direct"],
                        help="'torchvision' (default): standard Places365 classifier, predictions projected onto "
                             "the taxonomy via the Places->synset mapping (original behavior, unchanged). "
                             "'multitask_direct': a model trained to predict nature/materiality/biological "
                             "directly (e.g. Paula Feliu's TFG multitask model) -- no synset projection needed "
                             "for predictions; --model_name/--places_weights are ignored in this mode, use "
                             "--multitask_checkpoint_path / --multitask_backbone_choice instead.")
    parser.add_argument("--multitask_checkpoint_path", type=str, default=None,
                        help="[multitask_direct mode] Path to the trained multitask checkpoint "
                             "(e.g. trained_DenseNet121_100epochs.pth). A plain state_dict, no wrapper keys.")
    parser.add_argument("--multitask_backbone_choice", type=str, default="DenseNet121",
                        choices=["DenseNet121", "ResNet18", "EfficientNetB0", "ResNet50"],
                        help="[multitask_direct mode] Must match the backbone the checkpoint was trained with.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--output_file", type=str, default="closed_set_places_baseline.json", help="Summary output.")

    # Mapping strictness
    parser.add_argument("--allow-unresolved", dest="allow_unresolved", action="store_true",
                        help="Do not hard-fail on classes that cannot be resolved to a taxonomy synset; "
                             "instead exclude them from taxonomic metrics and continue.")

    # Testing and logging flags
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of images for testing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--wandb", action="store_true", help="Store the results on WandB.")

    return parser.parse_args()


def safe_binary_map(val, positive_str, negative_str):
    """Safely converts string annotations to binary labels. (identical to imagenet script)"""
    if not isinstance(val, str):
        return None
    val = val.strip().lower()
    if val == positive_str.lower():
        return 1
    if val == negative_str.lower():
        return 0
    return None


# ============================================================================
# PLACES -> SYNSET RESOLUTION
# ============================================================================
def load_taxonomy_synsets(excel_path, sheet_name, mit_only=True):
    """
    Read the 'sourcekey' sheet of the taxonomy workbook and return the set of
    synset strings that exist in the taxonomy (strip the ' 1/2' disambiguation
    suffixes). The sheet has a real header row (its two columns hold the raw
    synset string and its dataset-origin tag), so it's read with header=0.

    mit_only=True (default, and what this script uses for Places resolution):
    only keep synsets whose source tag contains 'MIT' -- i.e. synsets that
    were actually drawn from the Places/MIT mapping. This is deliberately
    stricter than using every synset in the taxonomy; see the module
    docstring section "WHY RESOLUTION IS RESTRICTED TO 'MIT'-TAGGED SYNSETS".
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
    except ValueError as e:
        _raise_sheet_not_found(excel_path, sheet_name, e)

    if df.shape[1] < 2:
        raise ValueError(
            f"Sheet '{sheet_name}' in {excel_path} has fewer than 2 columns "
            f"(found {df.shape[1]}). Expected [synset, source_tag]."
        )
    synset_col, source_col = df.columns[0], df.columns[1]

    tax = set()
    for _, row in df.iterrows():
        raw = row[synset_col]
        if pd.isna(raw) or not str(raw).strip():
            continue
        synset = str(raw).strip().split(' ')[0]  # drop ' 1/2'-style disambiguation suffix
        source = "" if pd.isna(row[source_col]) else str(row[source_col]).strip()
        if mit_only and 'MIT' not in source:
            continue
        tax.add(synset)
    return tax


def load_exclusion_set(excel_path, sheet_name):
    """
    Read the 'still missing MIT Places' sheet: a single column of Places class
    names that were never mapped, with NO header row (the first row is real
    data), so it's read with header=None.
    """
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


def _raise_sheet_not_found(excel_path, sheet_name, original_error):
    try:
        available = pd.ExcelFile(excel_path).sheet_names
    except Exception:
        available = ["<could not list sheets>"]
    raise ValueError(
        f"Could not read sheet '{sheet_name}' from {excel_path} ({original_error}). "
        f"Available sheets: {available}. Pass the correct name via --sourcekey_sheet / "
        f"--missing_sheet."
    )


def load_explicit_mapping(mapping_csv):
    """Authoritative override: places_name|places_id -> synset. Returns dict keyed by BOTH
    the name and the str(id) if an id column is detected, so lookup is robust."""
    df = pd.read_csv(mapping_csv)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # find the synset column
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


def resolve_via_wordnet(cls, taxonomy_synsets):
    """Resolve a Places class name to a synset that EXISTS in the taxonomy.
    Returns (synset_str or None, candidate_used or None).
    Conservative: only accepts a synset the taxonomy actually contains."""
    from nltk.corpus import wordnet as wn
    base = cls.replace('/', '_')
    head = cls.split('/')[0]
    candidates = [base, head]
    if '/' in cls:
        candidates.append(cls.split('/')[-1])  # trailing qualifier, e.g. desert/sand -> sand
    seen = set()
    for c in candidates:
        key = c.replace(' ', '_')
        if key in seen:
            continue
        seen.add(key)
        for s in wn.synsets(key, pos='n'):
            if s.name() in taxonomy_synsets:
                return s.name(), c
    return None, None


def build_places_id_to_synset(args, places_categories):
    """
    Build category_id (0..364) -> synset string, honoring:
      * explicit --mapping_csv if provided (authoritative; in this case the
        'sourcekey' sheet is never touched, and the 'still missing MIT Places'
        sheet is only used -- if readable -- to distinguish "deliberately
        excluded" from "unresolved" in the report; it's optional in this
        mode), else
      * WordNet reconstruction constrained to MIT-tagged taxonomy synsets
        only (see module docstring), which DOES require both sheets to be
        present in --excel_path.
    Returns (id_to_synset, report) where report lists excluded/unresolved classes.
    """
    explicit = None
    if args.mapping_csv:
        explicit = load_explicit_mapping(args.mapping_csv)

    # The exclusion sheet is used as a reporting aid in both modes, but only
    # WordNet reconstruction (explicit is None) truly cannot proceed without it.
    try:
        exclusion = load_exclusion_set(args.excel_path, args.missing_sheet)
    except Exception as e:
        if explicit is None:
            raise ValueError(
                f"Could not read the '{args.missing_sheet}' sheet from {args.excel_path} ({e}). "
                f"It's required for WordNet reconstruction (no --mapping_csv was given). Either "
                f"fix --missing_sheet / --excel_path, or supply --mapping_csv with an authoritative "
                f"places->synset mapping."
            )
        print(f"⚠️  Could not read the '{args.missing_sheet}' sheet from {args.excel_path} ({e}). "
              f"Continuing with --mapping_csv only; classes absent from the mapping will be reported "
              f"as 'unresolved' rather than 'excluded' (this only affects the report's bookkeeping, "
              f"not correctness).")
        exclusion = set()

    taxonomy_synsets = None
    if explicit is None:
        # Only the reconstruction path needs the sourcekey taxonomy.
        taxonomy_synsets = load_taxonomy_synsets(args.excel_path, args.sourcekey_sheet, mit_only=True)

    id_to_synset = {}
    excluded = []      # deliberately unmapped (in still_missing)
    unresolved = []    # not excluded, but we still couldn't find a taxonomy synset

    for cid, name in enumerate(places_categories):
        if explicit is not None:
            syn = explicit.get(name) or explicit.get(str(cid))
            if syn:
                id_to_synset[cid] = syn
            else:
                # fall through to exclusion/unresolved bookkeeping
                if name in exclusion:
                    excluded.append((cid, name))
                else:
                    unresolved.append((cid, name))
            continue

        # --- reconstruction path ---
        if name in exclusion:
            excluded.append((cid, name))
            continue
        syn, via = resolve_via_wordnet(name, taxonomy_synsets)
        if syn is not None:
            id_to_synset[cid] = syn
        else:
            unresolved.append((cid, name))

    report = {
        "n_mapped": len(id_to_synset),
        "n_excluded_still_missing": len(excluded),
        "n_unresolved": len(unresolved),
        "excluded": excluded,
        "unresolved": unresolved,
    }
    return id_to_synset, report


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_label = args.model_name if args.model_type == "torchvision" else \
        f"{args.multitask_backbone_choice}-multitask-direct"
    print(f"🚀 Starting Closed-Set Places365 Baseline ({model_label}) on {device.upper()}")

    if args.wandb:
        wandb.init(
            entity="paumonserrat03-universitat-aut-noma-de-barcelona",
            project="TFM_Closed-set",
            config=vars(args),
            name=f"places_baseline_{model_label}",
        )

    # ==========================================
    # 0. LOAD PLACES365 CATEGORY LIST (id -> name) FROM DISK
    # ==========================================
    categories_txt = args.places_categories_txt
    if categories_txt is None:
        data_dir_parent = os.path.dirname(os.path.normpath(args.data_dir))
        categories_txt = os.path.join(data_dir_parent, "categories_places365.txt")
    if args.verbose:
        print(f"[INFO] Loading Places365 category list from {categories_txt} ...")
    places_categories = load_places365_categories(categories_txt)
    if args.verbose:
        print(f"[INFO] Loaded {len(places_categories)} categories "
              f"(id 0 = '{places_categories[0]}', id 364 = '{places_categories[364]}').")

    # ==========================================
    # 1. LOAD MODEL & TRANSFORMS
    # ==========================================
    if args.model_type == "torchvision":
        if args.verbose:
            print(f"[INFO] Fetching architecture/transforms for {args.model_name}...")
        weights = models.get_model_weights(args.model_name).DEFAULT
        # NOTE: for resnet-family architectures this preprocessing (Resize(256) ->
        # CenterCrop(224) -> Normalize(ImageNet mean/std)) is IDENTICAL to the
        # transform CSAIL's own demo code (run_placesCNN_basic.py) applies for the
        # official Places365 checkpoints, so reusing torchvision's default
        # transform here is correct for resnet18/50. If you switch to an
        # architecture whose default torchvision recipe differs (e.g. convnext,
        # swin), you'd need to hardcode the classic 256/224 recipe instead.
        preprocess = weights.transforms()

        if args.places_weights:
            # Build the architecture with a 365-class head and load the Places checkpoint,
            # following CSAIL's own loading recipe (run_placesCNN_basic.py):
            #   model = torchvision.models.__dict__[arch](num_classes=365)
            #   checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            #   state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            #   model.load_state_dict(state_dict)
            model = models.get_model(args.model_name, weights=None)
            _replace_head_365(model, args.model_name)  # equivalent to constructing with num_classes=365

            try:
                checkpoint = torch.load(args.places_weights, map_location="cpu")
            except UnicodeDecodeError:
                # The official .pth.tar files were pickled under Python 2.7 / PyTorch 0.2.
                # Older PyTorch needed encoding='latin1' to unpickle them under Python 3;
                # keep this fallback in case your torch version still needs it.
                if args.verbose:
                    print("[INFO] Plain torch.load hit a UnicodeDecodeError (Python2-pickle artifact); "
                          "retrying with encoding='latin1'...")
                checkpoint = torch.load(args.places_weights, map_location="cpu", encoding="latin1")

            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

            if isinstance(state, torch.nn.Module):
                # Some CSAIL mirrors distribute "whole model" checkpoints (torch.save(model) rather
                # than torch.save(model.state_dict())). In that case there's nothing to load into our
                # freshly-built architecture -- the checkpoint already IS the model.
                if args.verbose:
                    print("[INFO] Checkpoint is a whole saved nn.Module (not a state_dict); using it directly.")
                model = state
            else:
                # The original training used nn.DataParallel, which prefixes every key with 'module.'
                state = {k.replace("module.", ""): v for k, v in state.items()}

                missing_k, unexpected_k = model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded Places weights from {args.places_weights}. "
                      f"missing_keys={len(missing_k)} unexpected_keys={len(unexpected_k)}")
                # A handful of missing keys (e.g. 'num_batches_tracked' buffers absent from the old
                # PyTorch 0.2 checkpoint) is expected and harmless at eval time. Many more than that
                # signals a genuine architecture mismatch between --model_name and the checkpoint.
                harmless = all(k.endswith("num_batches_tracked") for k in missing_k)
                if not harmless or len(unexpected_k) > 0:
                    print(f"⚠️  Non-trivial key mismatch when loading {args.places_weights} into "
                          f"'{args.model_name}'. missing={missing_k[:5]}{'...' if len(missing_k) > 5 else ''} "
                          f"unexpected={unexpected_k[:5]}{'...' if len(unexpected_k) > 5 else ''}\n"
                          f"   This usually means the checkpoint's architecture does not match --model_name. "
                          f"Double-check you downloaded the checkpoint matching '{args.model_name}' from "
                          f"http://places2.csail.mit.edu/models_places365/{args.model_name}_places365.pth.tar")
            model = model.to(device)
        else:
            print("⚠️  No --places_weights given. Loading torchvision DEFAULT weights, whose head is the "
                  "1000-class ImageNet space. Predictions will NOT be Places category ids and the taxonomy "
                  "mapping below will be meaningless.\n"
                  "   Download the official Places365-trained checkpoint for your architecture, e.g. for "
                  "ResNet-50:\n"
                  "     wget http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar\n"
                  "   then re-run with --model_name resnet50 --places_weights resnet50_places365.pth.tar")
            model = models.get_model(args.model_name, weights=weights).to(device)
        model.eval()

    else:  # model_type == "multitask_direct"
        if not args.multitask_checkpoint_path:
            raise ValueError("--multitask_checkpoint_path is required when --model_type multitask_direct.")
        if args.verbose:
            print(f"[INFO] Building {args.multitask_backbone_choice} multitask model...")
        backbone = CustomBackbone(model_choice=args.multitask_backbone_choice)
        model = MultiTaskModel(backbone, backbone.feature_dim).to(device)

        if args.verbose:
            print(f"[INFO] Loading checkpoint from {args.multitask_checkpoint_path}...")
        state = torch.load(args.multitask_checkpoint_path, map_location=device)
        model.load_state_dict(state)  # plain state_dict, no wrapper keys, no cleaning needed (verified)
        model.eval()

        # Deterministic eval-time preprocessing -- same choice as evaluate_imagenet.py's
        # multitask_direct mode (see that script for the full rationale): the repo's own
        # training transform includes random augmentation, so we use the deterministic
        # transform their own repo uses for non-training inference instead.
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


    # ==========================================
    # 2. BUILD PLACES-ID -> TAXONOMY MAPS
    # ==========================================
    if args.verbose:
        print(f"[INFO] Loading Taxonomy from {args.excel_path}...")
    pipeline = TaxonomyEvaluationPipeline()
    df_taxonomy = pd.read_excel(args.excel_path, sheet_name="data corrected")
    pipeline.load_custom_excel_annotations(df_taxonomy, "Biotic/abiotic", "Material/immaterial")

    if args.verbose:
        print("[INFO] Reconstructing Places365 -> WordNet synset mapping (MIT-tagged synsets only)...")
    id_to_synset, report = build_places_id_to_synset(args, places_categories)

    if args.verbose or report["n_unresolved"]:
        print("\n[INFO] --- Places -> Taxonomy mapping report (365 categories) ---")
        print(f"  Mapped to a taxonomy synset:   {report['n_mapped']}")
        print(f"  Excluded (in still_missing):   {report['n_excluded_still_missing']}")
        print(f"  UNRESOLVED (need attention):   {report['n_unresolved']}")
        print("--------------------------------------------------------------")
        if report["unresolved"]:
            print("Unresolved classes (no taxonomy synset found, not in still_missing):")
            for cid, name in report["unresolved"]:
                print(f"   - id {cid:3d}: {name}")
            print("")

    if report["n_unresolved"] and not args.allow_unresolved:
        print("❌ Refusing to produce a baseline with unresolved classes, because guessing their synset "
              "would silently corrupt the nature/biotic/material metrics.\n"
              "   Fix options:\n"
              "     (a) add these classes to the still_missing exclusion CSV if they truly are unmapped, or\n"
              "     (b) supply --mapping_csv with the authoritative places->synset mapping, or\n"
              "     (c) re-run with --allow-unresolved to exclude them from taxonomic metrics and proceed.")
        sys.exit(1)

    # Resolve each mapped Places id -> taxonomy node attributes -> binary labels.
    # Keyed by Places category id (int), NOT by dataloader index.
    map_nature, map_biotic, map_material = {}, {}, {}
    stats = {"nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}

    for cid in range(len(places_categories)):
        synset_str = id_to_synset.get(cid)
        node_attrs = pipeline.get_node_attributes(synset_str) if synset_str else None

        if not node_attrs:
            # Either excluded (still_missing) or unresolved-but-allowed: leave label as None
            # so calculate_binary_metrics ignores it as ground truth, mirroring ImageNet unmapped.
            stats["unmapped"] += 1
            map_nature[cid] = None
            map_biotic[cid] = None
            map_material[cid] = None
            continue

        is_nature = node_attrs.get('is_nature')
        map_nature[cid] = 1 if is_nature else 0
        if is_nature:
            stats["nature"] += 1

        bio_bin = safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
        map_biotic[cid] = bio_bin
        if bio_bin == 1:
            stats["biotic"] += 1
        elif bio_bin == 0:
            stats["abiotic"] += 1

        mat_bin = safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
        map_material[cid] = mat_bin
        if mat_bin == 1:
            stats["material"] += 1
        elif mat_bin == 0:
            stats["immaterial"] += 1
            print(f"Immaterial synset: {synset_str}")

    if args.verbose:
        print("[INFO] --- Taxonomy label statistics (out of 365 classes) ---")
        print(f"  Nature classes:        {stats['nature']}")
        print(f"  Biotic/Abiotic:        {stats['biotic']} / {stats['abiotic']}")
        print(f"  Material/Immaterial:   {stats['material']} / {stats['immaterial']}")
        print(f"  Unmapped (excluded):   {stats['unmapped']}")
        print("--------------------------------------------------------------\n")

    # ==========================================
    # 3. DATASET SETUP  (+ handle the ImageFolder sorting trap)
    # ==========================================
    full_dataset = datasets.ImageFolder(args.data_dir, transform=preprocess)
    # class_to_idx maps folder NAME (a string id "0".."364") -> dataloader index (lexicographic).
    # Build dataloader_idx -> true Places category id (int(name)).
    idx_to_places_id = {}
    for name, dl_idx in full_dataset.class_to_idx.items():
        try:
            idx_to_places_id[dl_idx] = int(name)
        except ValueError:
            raise ValueError(
                f"Subfolder name '{name}' is not an integer category id. This script expects "
                f"val_formatted subfolders named 0..364. Got '{name}'."
            )
    # sanity
    got = sorted(idx_to_places_id.values())
    if got != list(range(len(places_categories))):
        print(f"⚠️  Folder ids do not form a clean 0..364 range (found {len(got)} folders: "
              f"min={got[0]}, max={got[-1]}). Proceeding, but check your val_formatted split.")

    if args.max_samples is not None:
        if args.verbose:
            print(f"⚠️ Restricting to a deterministic random subset of {args.max_samples} samples.")
        random.seed(42)
        subset_indices = random.sample(range(len(full_dataset)), min(args.max_samples, len(full_dataset)))
        dataset_to_use = Subset(full_dataset, subset_indices)
    else:
        dataset_to_use = full_dataset

    dataloader = DataLoader(dataset_to_use, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # ==========================================
    # 4. INFERENCE LOOP
    # ==========================================
    # We collect ground truth as TRUE Places category ids (not dataloader indices).
    all_gt_places = []
    all_pred_places = []                 # torchvision mode: predicted Places category ids
    all_pred_nature_direct = []          # multitask_direct mode: already-final 0/1 nature predictions
    all_pred_material_direct = []        # multitask_direct mode: already-final 1/0/None (material=1)
    all_pred_biotic_direct = []          # multitask_direct mode: already-final 1/0/None (biotic=1)

    print(f"Running Inference over {len(dataset_to_use)} images...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=not args.verbose):
            images = images.to(device)
            # Ground truth: convert dataloader index -> true Places id via folder name.
            gt_ids = [idx_to_places_id[int(l)] for l in labels.cpu().tolist()]
            all_gt_places.extend(gt_ids)

            if args.model_type == "torchvision":
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                # Prediction: a Places365 head outputs logits already in Places id order.
                all_pred_places.extend(preds.cpu().tolist())

            else:  # multitask_direct
                out_nature, out_materiality, out_biological, _out_landscape = model(images)
                pred_nature = out_nature.argmax(dim=1).cpu().tolist()            # 0/1, matches our convention
                pred_materiality = out_materiality.argmax(dim=1).cpu().tolist()  # 0/1/2 (their convention)
                pred_biological = out_biological.argmax(dim=1).cpu().tolist()    # 0/1/2 (their convention)

                all_pred_nature_direct.extend(pred_nature)
                all_pred_material_direct.extend(
                    [MULTITASK_MATERIALITY_TO_OURS.get(p) for p in pred_materiality]  # None if p==2 ("nan")
                )
                all_pred_biotic_direct.extend(
                    [MULTITASK_BIOLOGICAL_TO_OURS.get(p) for p in pred_biological]    # None if p==2 ("nan")
                )

    # ==========================================
    # 5. METRIC CALCULATION
    # ==========================================
    def calculate_binary_metrics(gt_list, pred_list, map_dict, task_name):
        valid_gts, valid_preds = [], []
        for gt_id, pred_id in zip(gt_list, pred_list):
            mapped_gt = map_dict.get(gt_id)
            mapped_pred = map_dict.get(pred_id)
            if mapped_gt is not None:
                valid_gts.append(mapped_gt)
                if mapped_pred is None:
                    valid_preds.append(1 - mapped_gt)  # penalize mapping failure (same as imagenet script)
                else:
                    valid_preds.append(mapped_pred)
        if not valid_gts:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}
        acc = accuracy_score(valid_gts, valid_preds)
        p, r, f1, _ = precision_recall_fscore_support(valid_gts, valid_preds, average='binary', zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
                "f1": float(f1), "support": len(valid_gts)}

    def calculate_binary_metrics_direct(gt_list, pred_direct_list, map_dict):
        """
        Same methodology as calculate_binary_metrics, but for models whose
        predictions are ALREADY final taxonomy labels (0/1/None) rather than
        category ids needing a second lookup through map_dict. A None
        prediction (the model's own "not applicable" class) is penalized as
        wrong, same convention as a mapping failure -- consistent methodology
        across both model types rather than silently excluding hard cases.
        """
        valid_gts, valid_preds = [], []
        for gt_id, pred_direct in zip(gt_list, pred_direct_list):
            mapped_gt = map_dict.get(gt_id)
            if mapped_gt is not None:
                valid_gts.append(mapped_gt)
                if pred_direct is None:
                    valid_preds.append(1 - mapped_gt)
                else:
                    valid_preds.append(pred_direct)
        if not valid_gts:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}
        acc = accuracy_score(valid_gts, valid_preds)
        p, r, f1, _ = precision_recall_fscore_support(valid_gts, valid_preds, average='binary', zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
                "f1": float(f1), "support": len(valid_gts)}

    if args.model_type == "torchvision":
        # Standard Places365 metrics (365-way)
        places_acc = accuracy_score(all_gt_places, all_pred_places)
        places_p, places_r, places_f1, _ = precision_recall_fscore_support(
            all_gt_places, all_pred_places, average='macro', zero_division=0
        )
        nature_metrics = calculate_binary_metrics(all_gt_places, all_pred_places, map_nature, "Nature")
        biotic_metrics = calculate_binary_metrics(all_gt_places, all_pred_places, map_biotic, "Biotic")
        material_metrics = calculate_binary_metrics(all_gt_places, all_pred_places, map_material, "Material")

    else:  # multitask_direct
        # This model was never trained to do 365-class Places classification,
        # so that metric section is not applicable -- reported as None rather
        # than a misleading 0.0.
        places_acc = places_p = places_r = places_f1 = None
        nature_metrics = calculate_binary_metrics_direct(all_gt_places, all_pred_nature_direct, map_nature)
        biotic_metrics = calculate_binary_metrics_direct(all_gt_places, all_pred_biotic_direct, map_biotic)
        material_metrics = calculate_binary_metrics_direct(all_gt_places, all_pred_material_direct, map_material)

    # ==========================================
    # 6. TERMINAL SUMMARY & SAVE
    # ==========================================
    print("\n" + "=" * 55)
    print(f"📊 CLOSED-SET PLACES365 BASELINE: {model_label.upper()}")
    print("=" * 55)
    if args.model_type == "torchvision":
        print("--- 365-Class Places365 ---")
        print(f"Accuracy:  {places_acc:.4f}")
        print(f"Precision: {places_p:.4f} (Macro)")
        print(f"Recall:    {places_r:.4f} (Macro)")
        print(f"F1 Score:  {places_f1:.4f} (Macro)")
    else:
        print("--- 365-Class Places365 ---")
        print("N/A -- this model predicts nature/materiality/biological directly, not Places365 classes.")

    for title, m in [("Nature vs. No Nature", nature_metrics),
                     ("Biotic vs. Abiotic", biotic_metrics),
                     ("Material vs. Immaterial", material_metrics)]:
        print(f"\n--- Binary: {title} (Support: {m['support']}) ---")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1 Score:  {m['f1']:.4f}")
    print("=" * 55)

    if args.wandb:
        print("\n🚀 Uploading baseline metrics to Weights & Biases...")
        wandb.log({
            "Number of unmapped/excluded classes:": stats["unmapped"],
            "Number of unresolved classes:": report["n_unresolved"],

            "Places/Accuracy": places_acc,
            "Places/Precision_Macro": places_p,
            "Places/Recall_Macro": places_r,
            "Places/F1_Macro": places_f1,

            "Nature/Accuracy": nature_metrics['accuracy'],
            "Nature/Precision": nature_metrics['precision'],
            "Nature/Recall": nature_metrics['recall'],
            "Nature/F1": nature_metrics['f1'],
            "Nature/Support": nature_metrics['support'],

            "Biotic/Accuracy": biotic_metrics['accuracy'],
            "Biotic/Precision": biotic_metrics['precision'],
            "Biotic/Recall": biotic_metrics['recall'],
            "Biotic/F1": biotic_metrics['f1'],
            "Biotic/Support": biotic_metrics['support'],

            "Material/Accuracy": material_metrics['accuracy'],
            "Material/Precision": material_metrics['precision'],
            "Material/Recall": material_metrics['recall'],
            "Material/F1": material_metrics['f1'],
            "Material/Support": material_metrics['support'],
        })

    summary_results = {
        "model": model_label,
        "model_type": args.model_type,
        "dataset": "places365",
        "samples_evaluated": len(dataset_to_use),
        "mapping_report": {
            "n_mapped": report["n_mapped"],
            "n_excluded_still_missing": report["n_excluded_still_missing"],
            "n_unresolved": report["n_unresolved"],
            "unresolved_classes": [n for _, n in report["unresolved"]],
        },
        "places_365": {"accuracy": places_acc, "precision_macro": places_p,
                       "recall_macro": places_r, "f1_macro": places_f1},
        "nature": nature_metrics,
        "biotic": biotic_metrics,
        "material": material_metrics,
    }

    with open(args.output_file, "w") as f:
        json.dump(summary_results, f, indent=4)
    print(f"💾 Results saved to {args.output_file}")

    if args.wandb:
        wandb.save(args.output_file)
        wandb.finish()


def _replace_head_365(model, model_name):
    """Swap a torchvision classifier head to 365 outputs, for common architectures
    (convnext_*, swin_*, vit_b_16). Extend as needed for other backbones."""
    import torch.nn as nn
    name = model_name.lower()
    if name.startswith("convnext"):
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, 365)
    elif name.startswith("swin"):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, 365)
    elif name.startswith("vit"):
        # torchvision ViT: heads.head
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, 365)
    elif name.startswith("resnet"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 365)
    else:
        raise ValueError(
            f"Don't know how to swap the head for '{model_name}'. Add a case in _replace_head_365 "
            f"or load a checkpoint whose head is already 365-class."
        )


if __name__ == "__main__":
    main()