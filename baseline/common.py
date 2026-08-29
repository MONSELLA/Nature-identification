#!/usr/bin/env python3
"""
Shared helpers for every baseline/evaluate_*.py closed-set script:
  - TaxonomyLookup: DIRECT-only lookup of the taxonomy Excel's own
    hand-labeled synsets (no ancestor/descendant hop resolution -- that's
    what TaxonomyGraph in src/loaders/excel_loader.py is for, used by the
    VLM pipeline; the closed-set baselines only ever need a class's OWN
    annotated row). Row-parsing logic matches count_classes.py's
    self-contained `_TaxonomyAnnotations` (kept deliberately duplicated
    there -- see that file's module docstring for why it doesn't import
    this module).
  - CustomBackbone / MultiTaskModel: Paula Feliu's TFG multitask model,
    inlined verbatim (see the class docstrings below for provenance).
  - Binary-metric helpers (accuracy + per-polarity precision/recall/F1)
    shared by every script's Nature/Biotic/Material reporting.
  - A shared long-format results CSV writer, so every baseline run appends
    rows in one common shape (dataset, model, category, accuracy,
    precision, recall, f1, support, ...) that can be pivoted straight into
    the thesis tables instead of hand-assembling per-run JSON.
"""

import os
import re

import pandas as pd
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# ============================================================================
# TAXONOMY LOOKUP (direct-annotation only -- see module docstring)
# ============================================================================
# Matches a WordNet synset id string like "golden_retriever.n.01".
_SYNSET_PATTERN = re.compile(r"([\w\-']+\.[nvasr]\.[0-9]+)")


class TaxonomyLookup:
    """DIRECT-only lookup of the Excel's own hand-labeled synsets -- a class
    only counts here if it was annotated on its own row, never inherited from
    an ancestor/descendant (unlike TaxonomyGraph.resolve_labels elsewhere in
    the project, which the baseline/ scripts deliberately do NOT use)."""

    def __init__(self):
        self._nodes = {}  # synset_str -> {"is_nature", "biotic_abiotic", "material_immaterial"}

    def load(self, excel_path, bio_col="Biotic/abiotic", mat_col="Material/immaterial",
             sheet_name="data corrected"):
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        for _, row in df.iterrows():
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


def safe_binary_map(val, positive_str, negative_str):
    """Safely converts string annotations to binary labels."""
    if not isinstance(val, str):
        return None
    val = val.strip().lower()
    if val == positive_str.lower():
        return 1
    if val == negative_str.lower():
        return 0
    return None


# ============================================================================
# COCO CATEGORY ID -> WORDNET SYNSET (copied verbatim from
# src/loaders/dataset_loader.py's COCO_TO_WNSYNSET -- COCO's 80 category ids
# mapped to the synset that best represents each).
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
# MULTITASK DIRECT-TAXONOMY MODEL (Paula Feliu's TFG)
# https://github.com/paulafeliu/TFG-Interpretability-Techniques-in-Social-Media-Images
# ============================================================================
# Inlined rather than imported from a cloned repo: these two classes are ~35
# lines total (verified against the actual repo source), and importing
# external research repos has repeatedly hit dependency/version pain
# (torchvision API drift, missing packages, custom CUDA extensions). Copied
# verbatim from models/backbone.py and models/multitask_model.py, with ONE
# intentional change: the original used `pretrained=True` to initialize the
# backbone with ImageNet weights before THEIR training. We load their
# fully-trained checkpoint afterward with strict=True, which overwrites every
# weight anyway -- so `weights=None` here saves an unnecessary network
# download and gives an identical final result.
class CustomBackbone(nn.Module):
    """A swappable CNN feature-extractor (the "backbone") used inside
    MultiTaskModel below — turns a raw image into a fixed-length feature
    vector, with the ORIGINAL classification head replaced by `nn.Identity()`
    (a no-op layer) since we only want the features, not that model's own
    class predictions."""
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
        # Flatten whatever shape the backbone produces down to a plain
        # [batch_size, feature_dim] 2D tensor.
        return x.view(x.size(0), -1)


class MultiTaskModel(nn.Module):
    """Wraps a CustomBackbone with FOUR separate linear "heads" — one small
    extra layer per task, all sharing the same underlying image features.
    Only the first three heads (nature/materiality/biological) are actually
    used by the baseline scripts; `fc_landscape` exists because the original
    model was trained on a 4th task this evaluation doesn't need."""
    def __init__(self, backbone, feature_dim):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.fc_nature = nn.Linear(feature_dim, 2)       # 2 classes: nature yes/no
        self.fc_materiality = nn.Linear(feature_dim, 3)  # 3 classes: material/immaterial/n-a
        self.fc_biological = nn.Linear(feature_dim, 3)    # 3 classes: biotic/abiotic/n-a
        self.fc_landscape = nn.Linear(feature_dim, 8)     # unused by these scripts

    def forward(self, x):
        features = self.backbone(x)
        # Every head runs on the SAME shared features — this is what "multi-
        # task" means here: one backbone, several independent prediction heads.
        out_nature = self.fc_nature(features)
        out_materiality = self.fc_materiality(features)
        out_biological = self.fc_biological(features)
        out_landscape = self.fc_landscape(features)
        return out_nature, out_materiality, out_biological, out_landscape


# Verified label encodings from utils/main_utils.py's build_dataset():
#   nature_visual:            {"Yes": 1, "No": 0}                       -> matches our convention directly
#   nep_materiality_visual:   {"material": 0, "immaterial": 1, "nan": 2} -> OPPOSITE of our convention (material=1)
#   nep_biological_visual:    {"biotic": 0, "abiotic": 1, "nan": 2}      -> OPPOSITE of our convention (biotic=1)
# "nan" (class 2) means the model itself predicts "not applicable" (their
# convention: undefined when nature=No). We remap 0/1 to our convention and
# treat 2 as "no usable prediction", same as an ImageNet mapping failure.
# These lookup dicts translate the multitask model's own class indices (0/1)
# into THIS project's convention (nature=1/biotic=1/material=1 always means
# "positive"); class index 2 ("n/a") is intentionally NOT a key here, so
# `.get(2)` correctly returns None (no usable prediction) rather than 0.
MULTITASK_MATERIALITY_TO_OURS = {0: 1, 1: 0}  # their material(0)->our 1, their immaterial(1)->our 0
MULTITASK_BIOLOGICAL_TO_OURS = {0: 1, 1: 0}   # their biotic(0)->our 1, their abiotic(1)->our 0


# ============================================================================
# BINARY METRICS (accuracy + per-polarity precision/recall/F1/support)
# ============================================================================
# Positive classes throughout this project: nature=1, biotic=1, material=1
# (see CLAUDE.md "Inherited conventions"). Every metrics dict below has the
# shape {"accuracy": float, "positive": {precision,recall,f1,support}, "negative": {...}}.
def empty_binary_metrics():
    """Placeholder for a binary task that couldn't be computed at all (e.g.
    no positive examples of either class in the eligible pool) -- zeros
    everywhere rather than raising, so a report can still print a row."""
    return {
        "accuracy": 0.0,
        "positive": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        "negative": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    }


def binary_metrics_from_labels(y_true, y_pred):
    """Core metric computation: y_true/y_pred are parallel lists of 0/1
    labels (already resolved -- no None allowed here, callers below handle
    exclusion/penalization before reaching this point)."""
    if not y_true:
        return empty_binary_metrics()
    accuracy = accuracy_score(y_true, y_pred)
    result = {"accuracy": accuracy}
    for polarity, pos_label in (("positive", 1), ("negative", 0)):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
        )
        support = sum(1 for g in y_true if g == pos_label)
        result[polarity] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return result


def calculate_binary_metrics(gt_indices, pred_indices, label_map):
    """torchvision mode: gt_indices/pred_indices are CLASS INDICES (e.g.
    ImageNet/Places category ids); label_map maps a class index to its
    taxonomy label (1/0), or is simply absent/None for an unmapped class.

    Ground-truth-unmapped instances are EXCLUDED (no usable ground truth to
    score against at all). Prediction-unmapped instances are PENALIZED AS
    WRONG (the opposite of ground truth) -- per CLAUDE.md's "Inherited
    conventions": "Prediction-unmapped instances: penalized as wrong (never
    defaulted to 'no nature')."
    """
    y_true, y_pred = [], []
    for g, p in zip(gt_indices, pred_indices):
        gt_label = label_map.get(g)
        if gt_label is None:
            continue
        pred_label = label_map.get(p)
        if pred_label is None:
            pred_label = 1 - gt_label
        y_true.append(gt_label)
        y_pred.append(pred_label)
    return binary_metrics_from_labels(y_true, y_pred)


def calculate_binary_metrics_direct(gt_indices, direct_preds, label_map):
    """multitask_direct mode: gt_indices are CLASS INDICES (resolved via
    label_map, same exclusion rule as calculate_binary_metrics above), but
    direct_preds are ALREADY-FINAL 1/0/None taxonomy predictions (this kind
    of model answers nature/biotic/material directly, no synset projection
    needed). A None prediction (the model's own "not applicable" class) is
    penalized as wrong, same convention as the mapped-prediction case."""
    y_true, y_pred = [], []
    for g, p in zip(gt_indices, direct_preds):
        gt_label = label_map.get(g)
        if gt_label is None:
            continue
        pred_label = p if p is not None else 1 - gt_label
        y_true.append(gt_label)
        y_pred.append(pred_label)
    return binary_metrics_from_labels(y_true, y_pred)


# ============================================================================
# SHARED LONG-FORMAT RESULTS CSV (pivots directly into the thesis tables)
# ============================================================================
# Every baseline run appends rows shaped like:
#   run_id, dataset, model, model_type, category, granularity, accuracy,
#   precision, recall, f1, support, ...
# rather than each script writing its own bespoke per-run JSON as the only
# output. `--results_csv` (every script) overrides this default.
DEFAULT_RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "closed_set_baseline_results.csv")


def single_row(base_row, category, granularity=None, **metrics):
    """One results-CSV row: base_row's own fields (run_id/dataset/model/...),
    plus this row's `category` (e.g. "Nature", "Base (Macro)") and whatever
    metric kwargs the caller supplies (accuracy/precision/recall/f1/support).
    `granularity`, if given, OVERRIDES base_row's own "granularity" entry
    (used by COCO/BIG-5's class-vs-image-level split); left as None (the
    default) it does NOT touch whatever base_row already carries (including
    "no granularity key at all", for ImageNet/Places/BIG-5's single-level rows)."""
    row = {**base_row, "category": category, **metrics}
    if granularity is not None:
        row["granularity"] = granularity
    return row


def binary_metrics_to_rows(base_row, name, metrics):
    """Turn one axis's binary-metrics dict into results-CSV row(s) -- ONE row
    per axis, carrying the POSITIVE-class precision/recall/F1 alongside the
    shared accuracy (matching this project's "positive class = 1" convention
    and the thesis tables' single P/R/F1 column per category). Returns []
    if metrics is None (e.g. multitask_direct's per-class metrics, which
    don't apply to that model family -- see evaluate_coco.py)."""
    if metrics is None:
        return []
    pos = metrics["positive"]
    return [single_row(base_row, name, accuracy=metrics["accuracy"],
                        precision=pos["precision"], recall=pos["recall"],
                        f1=pos["f1"], support=pos["support"])]


def append_results_rows(csv_path, rows):
    """Append `rows` (list of flat dicts) to the shared results CSV, unioning
    columns with whatever's already there (different scripts contribute
    different optional columns, e.g. "granularity" or "training_dataset") --
    re-reading and rewriting the whole file rather than a raw line-append,
    so the header always reflects every column ever written, and a
    column present in old rows but absent from new ones (or vice versa)
    doesn't misalign the CSV."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
        os.makedirs(out_dir, exist_ok=True)
        combined = new_df
    combined.to_csv(csv_path, index=False)
