#!/usr/bin/env python3
"""
Evaluate an official SlongLiu/query2labels (Q2L) checkpoint on the MS-COCO
multi-label classification task, projected onto the BIG-5 nature taxonomy
(nature / biotic-abiotic / material-immaterial).

This is the Q2L counterpart of evaluate_coco.py (ML-Decoder). Same overall
design (multi-label, pycocotools ground truth, per-class + per-image taxonomy
metrics at threshold 0.5), but Q2L's loading/construction is different enough
from ML-Decoder's to be worth documenting separately.

WHAT IS "MULTI-LABEL" AND WHY DOES THAT MATTER HERE? Unlike ImageNet/Places
(where each image belongs to exactly ONE class), a COCO image can contain
MANY different labeled objects at once (a photo with both a "dog" and a
"person" and a "bicycle"). So instead of a single predicted class, the model
outputs an independent yes/no SCORE for EVERY one of the 80 COCO categories —
this script thresholds each of those 80 scores separately (see step 7/8
below) rather than picking one "best" class like the ImageNet/Places scripts do.

WHAT IS "Q2L" (Query2Label)? A specific published multi-label image
classifier architecture built on a transformer: it uses one "query" per class
(80 learned query vectors) that each attend over the image's visual features
to decide "is THIS class present in the image?" — hence the extra transformer
config fields (dim_feedforward, hidden_dim, nheads, etc.) seen below.

------------------------------------------------------------------------------
EVERYTHING BELOW WAS VERIFIED AGAINST YOUR ACTUAL FILES, NOT GUESSED
------------------------------------------------------------------------------
You provided config_new.json, q2l_infer.py, and lib/dataset/get_dataset.py
from your own clone + downloaded checkpoint. This script's model construction
and preprocessing are built directly from those three files.

CHECKPOINT FORMAT (from q2l_infer.py):
    checkpoint = torch.load(resume_path, map_location=...)
    state_dict = clean_state_dict(checkpoint['state_dict'])
    model.load_state_dict(state_dict, strict=True)
(main_mlc.py, the training script, also accepts a 'model' key as a fallback;
this script checks 'state_dict' first, then 'model', for robustness.)

MODEL CONSTRUCTION (from q2l_infer.py's parser_args() + your config_new.json):
    model = build_q2l(args)
args needs ~15 fields. q2l_infer.py's own argparse defaults are used as the
base, then OVERWRITTEN by config_new.json's keys -- exactly replicating
q2l_infer.py's own `for k,v in cfg_dict.items(): setattr(args, k, v)` logic.
This matters: your config_new.json sets dim_feedforward=2048 and
hidden_dim=2048, drastically different from q2l_infer.py's own argparse
defaults (256 and 128) -- using the wrong ones would build a
wrong-shaped transformer and silently fail (or crash) loading the checkpoint.
Some fields (position_embedding, keep_other_self_attn_dec,
keep_first_self_attn_dec, keep_input_proj) aren't in config_new.json at all,
so they're left at q2l_infer.py's argparse defaults, exactly as the original
script would.

PREPROCESSING (from lib/dataset/get_dataset.py's test_data_transform):
    transforms.Resize((img_size, img_size))   # direct square resize, no crop
    transforms.ToTensor()
    transforms.Normalize(mean, std)           # mean/std depend on orid_norm
Your config_new.json has "orid_norm": true, so mean=[0,0,0], std=[1,1,1] --
i.e. no real normalization, same convention as ML-Decoder/TResNet.

WHAT THIS SCRIPT DOES *NOT* REUSE FROM THE OFFICIAL REPO:
q2l_infer.py is built entirely around multi-GPU distributed execution --
it unconditionally calls torch.distributed.init_process_group(backend='nccl')
and wraps the model in DistributedDataParallel, even for single-process use.
Reusing that wholesale would force a NCCL/GPU dependency for no benefit. This
script imports only the pure model-building pieces (build_q2l,
clean_state_dict) and runs its own plain single-process inference loop.

It also does NOT reuse the repo's own CoCoDataset (lib/dataset/cocodataset.py),
which depends on separately-provided train_label_vectors_coco14.npy /
val_label_vectors_coco14.npy files you don't have and don't need -- ground
truth is built directly via pycocotools against your instances_val2017.json,
same as evaluate_coco.py.

------------------------------------------------------------------------------
NOTE: COCO 2014 vs 2017 -- not a leakage risk (same reasoning as ML-Decoder)
------------------------------------------------------------------------------
get_dataset.py's val_dataset is built from 'val2014' / 'instances_val2014.json'
-- confirming Q2L, like ML-Decoder, is trained/validated on standard COCO2014.
Evaluating it here against val2017 is safe: val2017 is the same fixed
5,000-image "minival2014" subset that was always part of the original val2014
and was specifically held out of every standard COCO training convention.
See evaluate_coco.py's docstring for the full argument if needed.

------------------------------------------------------------------------------
IMPORT PATH: point --q2l_repo_path at the repo ROOT, not lib/
------------------------------------------------------------------------------
q2l_infer.py does `import _init_paths` before `from dataset.get_dataset import
get_datasets` -- _init_paths.py's only job is adding <repo_root>/lib to
sys.path, since build_q2l/clean_state_dict/etc. actually live under
<repo_root>/lib/models, <repo_root>/lib/utils, etc. This script adds BOTH
<repo_root> and <repo_root>/lib to sys.path so it works regardless of which
one you pass.
"""

import os
import sys
import json
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models as tv_models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

# Make both the repo root (for the missing `first_tests` module) and this
# script's own directory importable, the same pattern used throughout
# baseline/ — see count_classes.py's comment on why this is needed.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
this_dir = os.path.abspath(os.path.dirname(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from first_tests.evaluation import TaxonomyEvaluationPipeline


# ============================================================================
# MULTITASK DIRECT-TAXONOMY MODEL (Paula Feliu's TFG) -- same as
# evaluate_imagenet.py / evaluate_places.py. See those scripts' comments for
# the full rationale on inlining rather than importing an external repo, and
# for the verified label-encoding source citation.
# ============================================================================
class CustomBackbone(nn.Module):
    """Swappable CNN feature-extractor — see the identical class in
    evaluate_imagenet.py for full comments on its structure."""
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
        out_nature = self.fc_nature(features)
        out_materiality = self.fc_materiality(features)
        out_biological = self.fc_biological(features)
        out_landscape = self.fc_landscape(features)
        return out_nature, out_materiality, out_biological, out_landscape


# Translate the multitask model's own 0/1 class indices into this project's
# convention (1 = positive: nature/biotic/material) — see evaluate_imagenet.py
# for the full citation of where these encodings come from.
MULTITASK_MATERIALITY_TO_OURS = {0: 1, 1: 0}  # their material(0)->our 1, their immaterial(1)->our 0
MULTITASK_BIOLOGICAL_TO_OURS = {0: 1, 1: 0}   # their biotic(0)->our 1, their abiotic(1)->our 0


# ============================================================================
# COCO -> WORDNET SYNSET MAPPING (identical to evaluate_coco.py)
# ============================================================================
# Same hand-built COCO-category-id -> WordNet-synset table used throughout
# this project (see lib/dataset_loader.py's identical COCO_TO_WNSYNSET for the
# general explanation of why COCO needs this manual mapping at all).
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


def load_coco_categories(instances_json_path):
    """Read COCO category id->name straight from the annotation JSON (not hardcoded)."""
    with open(instances_json_path, "r") as f:
        data = json.load(f)
    if "categories" not in data:
        raise ValueError(f"'{instances_json_path}' has no 'categories' field; "
                         f"is this a valid COCO instances annotation file?")
    return {c["id"]: c["name"] for c in data["categories"]}


def build_category_id_to_synset(id_to_name, mapping=None):
    """Split COCO's category ids into (id -> synset) for those we have a
    mapping for, and a list of (id, name) pairs we DON'T (so the caller can
    warn about coverage gaps rather than silently dropping them)."""
    mapping = mapping or COCO_TO_WNSYNSET
    id_to_synset, unmapped_ids = {}, []
    for cid, name in id_to_name.items():
        if cid in mapping:
            id_to_synset[cid] = mapping[cid]
        else:
            unmapped_ids.append((cid, name))
    return id_to_synset, unmapped_ids


# ============================================================================
# Q2L ARGS CONSTRUCTION: q2l_infer.py's own argparse defaults, then
# overwritten by config_new.json -- exactly replicating parser_args()'s
# `for k,v in cfg_dict.items(): setattr(args, k, v)` sequence.
# ============================================================================
def build_q2l_args(config_path, img_size_override=None, num_class_override=None):
    """Build the `args`-like object build_q2l() expects, replicating the
    ORIGINAL Q2L repo's own "argparse defaults, then overridden by whatever
    is in the checkpoint's own config JSON" construction sequence — so the
    model is built with EXACTLY the architecture the checkpoint was trained
    with, not this script's own guesses."""
    # These are q2l_infer.py's own argparse defaults, transcribed verbatim.
    # `SimpleNamespace` is just a plain object whose attributes we can freely
    # get/set (`args.img_size`, etc.) — a lightweight stand-in for a real
    # argparse Namespace without needing to construct an actual parser.
    args = SimpleNamespace(
        dataname='coco14',
        dataset_dir='/comp_robot/liushilong/data/COCO14/',
        img_size=448,
        arch='Q2L-TResL_22k-448',
        output=None,
        loss='asl',
        num_class=80,
        workers=8,
        batch_size=16,
        print_freq=10,
        resume=None,
        pretrained=False,
        eps=1e-5,
        world_size=-1,
        rank=-1,
        dist_url='tcp://127.0.0.1:3451',
        seed=None,
        local_rank=0,
        amp=False,
        orid_norm=False,
        enc_layers=1,
        dec_layers=2,
        dim_feedforward=256,
        hidden_dim=128,
        dropout=0.1,
        nheads=4,
        pre_norm=False,
        position_embedding='sine',
        backbone='resnet101',
        keep_other_self_attn_dec=False,
        keep_first_self_attn_dec=False,
        keep_input_proj=False,
    )
    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)
    # Overwrite the defaults above with whatever this SPECIFIC checkpoint's
    # own config.json actually specifies (e.g. dim_feedforward=2048 instead
    # of the 256 default) — this is what makes the model architecture match
    # the checkpoint's actual trained shape.
    for k, v in cfg_dict.items():
        setattr(args, k, v)

    # Optional CLI overrides (e.g. if you want to force a specific img_size)
    if img_size_override is not None:
        args.img_size = img_size_override
    if num_class_override is not None:
        args.num_class = num_class_override
    return args


def clean_state_dict_fallback(state_dict):
    """Minimal local fallback for utils.misc.clean_state_dict, used only if
    importing the repo's own version fails for some reason. Strips a leading
    'module.' prefix, the standard DataParallel/DDP artifact."""
    # When a model is trained wrapped in `nn.DataParallel` /
    # `DistributedDataParallel`, PyTorch automatically prefixes every
    # parameter name in the checkpoint with "module." — loading that
    # checkpoint into a plain (non-wrapped) model fails unless this prefix is
    # stripped back off first.
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
            for k, v in state_dict.items()}


def install_inplace_abn_shim_if_missing(verbose=False):
    """
    TResNet's backbone code imports inplace_abn.InPlaceABNSync, a package that
    compiles custom CUDA kernels at install time (fuses BatchNorm+activation
    for memory-efficient BACKPROP). It commonly fails to build on clusters
    without a properly configured CUDA dev toolchain.

    This is safe to substitute for eval-only use: the fused kernel computes
    exactly the same math as plain BatchNorm2d followed by an activation --
    the fusion only saves memory during training's backward pass. Since this
    script runs model.eval() under torch.no_grad() (no backward pass ever
    happens), the entire reason the real package exists doesn't apply here,
    and a plain, unfused nn.BatchNorm2d subclass produces numerically
    identical results, with matching state_dict key names (weight, bias,
    running_mean, running_var, num_batches_tracked) so checkpoints load
    correctly.

    Only registers the shim if the real inplace_abn isn't already importable,
    so installing the genuine package (e.g. `pip install inplace-abn`) always
    takes precedence if present.
    """
    try:
        import inplace_abn  # noqa: F401
        if verbose:
            print("[INFO] Real 'inplace_abn' package found; not installing shim.")
        return
    except ImportError:
        pass

    import types
    import torch.nn as nn
    import torch.nn.functional as F

    class InPlaceABNSyncShim(nn.BatchNorm2d):
        """Drop-in replacement matching InPlaceABNSync's constructor
        signature and state_dict key names, but implemented as plain
        (unfused) BatchNorm2d + a manually-applied activation function —
        mathematically identical output for inference, just without the
        memory-efficient fused CUDA kernel (irrelevant here since we never
        run a backward pass)."""
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                     activation="leaky_relu", activation_param=0.01, group=None):
            super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
            self._shim_activation = activation
            self._shim_activation_param = activation_param

        def forward(self, x):
            x = super().forward(x)  # plain BatchNorm2d
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

    # Register a completely FAKE "inplace_abn" module in Python's module
    # cache (sys.modules) — so when the TResNet backbone code later does
    # `import inplace_abn` / `from inplace_abn import InPlaceABNSync`, Python
    # finds and uses OUR shim instead of ever needing the real package to be
    # installed.
    shim_module = types.ModuleType("inplace_abn")
    shim_module.InPlaceABNSync = InPlaceABNSyncShim
    shim_module.InPlaceABN = InPlaceABNSyncShim  # some backbones import this name instead
    sys.modules["inplace_abn"] = shim_module
    if verbose:
        print("[INFO] Real 'inplace_abn' not found; installed an eval-only shim "
              "(numerically identical BatchNorm2d+activation, no CUDA extension needed).")


# ============================================================================
# GROUND TRUTH DATASET (pycocotools-based, same as evaluate_coco.py)
# ============================================================================
class CocoMultiLabelDataset(Dataset):
    """A PyTorch Dataset yielding (preprocessed_image, multi_hot_label_vector,
    image_id) triples built directly from a COCO instances_*.json file via
    pycocotools — the ground truth this whole evaluation is scored against."""
    def __init__(self, images_dir, instances_json, category_ids_sorted, transform):
        from pycocotools.coco import COCO
        self.images_dir = images_dir
        self.coco = COCO(instances_json)
        self.transform = transform
        # Map each COCO category id to a fixed COLUMN INDEX (0..num_classes-1)
        # in the multi-hot label vector below — `category_ids_sorted` fixes
        # this ordering so it stays consistent across the whole run.
        self.cat_id_to_col = {cid: i for i, cid in enumerate(category_ids_sorted)}
        self.num_classes = len(category_ids_sorted)
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.images_dir, info["file_name"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # A "multi-hot" vector: one float per class, 1.0 if that class is
        # annotated as present in this image, 0.0 otherwise — as opposed to a
        # single class INDEX (which only works for single-label datasets).
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            col = self.cat_id_to_col.get(ann["category_id"])
            if col is not None:
                target[col] = 1.0
        return image, target, img_id


def parse_args():
    """Command-line flags: dataset/taxonomy paths, which model to run (Q2L
    checkpoint or the multitask direct model), and evaluation/logging options."""
    parser = argparse.ArgumentParser(description="Evaluate Query2Label on Nature Taxonomy (COCO)")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to COCO validation images (e.g. .../coco/images/val2017).")
    parser.add_argument("--instances_json", type=str, required=True,
                        help="Path to COCO instances annotation file.")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx",
                        help="Path to the taxonomy workbook (needs the 'data corrected' sheet).")
    parser.add_argument("--q2l_repo_path", type=str, default=None,
                        help="[q2l mode] Path to your local clone of https://github.com/SlongLiu/query2labels "
                             "(the repo ROOT; this script also adds <root>/lib to sys.path).")
    parser.add_argument("--q2l_config", type=str, default=None,
                        help="[q2l mode] Path to config_new.json shipped with the checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="[q2l mode] Path to checkpoint.pkl.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="[q2l mode] Sigmoid threshold for 'predicted positive'.")
    parser.add_argument("--model_type", type=str, default="q2l",
                        choices=["q2l", "multitask_direct"],
                        help="'q2l' (default): Query2Label multi-label classifier, predictions projected onto "
                             "the taxonomy via the COCO->synset mapping, reported both per-class and per-image "
                             "(original behavior, unchanged). "
                             "'multitask_direct': a model trained to predict nature/materiality/biological "
                             "directly for a whole image (e.g. Paula Feliu's TFG multitask model). This kind of "
                             "model gives ONE holistic call per image, not a per-object/per-class score, so "
                             "there is no meaningful per-class comparison or mAP for it -- only PER-IMAGE "
                             "metrics are computed in this mode (ground truth: does the image contain any "
                             "nature-tagged COCO object; prediction: the model's own direct call). "
                             "--q2l_repo_path/--q2l_config/--checkpoint_path/--threshold are ignored in this mode; "
                             "use --multitask_checkpoint_path / --multitask_backbone_choice instead.")
    parser.add_argument("--multitask_checkpoint_path", type=str, default=None,
                        help="[multitask_direct mode] Path to the trained multitask checkpoint "
                             "(e.g. trained_DenseNet121_100epochs.pth). A plain state_dict, no wrapper keys.")
    parser.add_argument("--multitask_backbone_choice", type=str, default="DenseNet121",
                        choices=["DenseNet121", "ResNet18", "EfficientNetB0", "ResNet50"],
                        help="[multitask_direct mode] Must match the backbone the checkpoint was trained with.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="q2l_coco_baseline.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # These two blocks emulate "conditionally required" arguments (argparse
    # itself has no clean way to say "required only when --model_type is X"),
    # checked manually after parsing.
    if args.model_type == "q2l":
        missing = [n for n in ("q2l_repo_path", "q2l_config", "checkpoint_path")
                  if getattr(args, n) is None]
        if missing:
            parser.error(f"--model_type q2l requires: {', '.join('--' + m for m in missing)}")
    else:
        if args.multitask_checkpoint_path is None:
            parser.error("--model_type multitask_direct requires --multitask_checkpoint_path")

    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_label = "Q2L" if args.model_type == "q2l" else f"{args.multitask_backbone_choice}-multitask-direct"
    print(f"🚀 Starting COCO Baseline ({model_label}) on {device.upper()}")

    if args.model_type == "q2l":
        print("\n" + "-" * 70)
        print("NOTE: this checkpoint was trained on COCO train2014 (per get_dataset.py's")
        print("'val2014'/instances_val2014.json references) and is evaluated here against")
        print("val2017. This is NOT a leakage risk: val2017 is the same fixed 5,000-image")
        print("'minival2014' subset always held out of standard COCO training conventions.")
        print("-" * 70 + "\n")

    if args.wandb:
        import wandb
        wandb.init(
            entity="paumonserrat03-universitat-aut-noma-de-barcelona",
            project="TFM_Closed-set",
            config=vars(args),
            name=f"coco_baseline_{model_label}",
        )

    if args.model_type == "q2l":
        # ==========================================
        # 1. IMPORT build_q2l / clean_state_dict FROM THE OFFICIAL REPO
        # ==========================================
        repo_root = os.path.abspath(args.q2l_repo_path)
        repo_lib = os.path.join(repo_root, "lib")
        for p in (repo_root, repo_lib):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        install_inplace_abn_shim_if_missing(verbose=args.verbose)

        try:
            from models.query2label import build_q2l
        except ImportError as e:
            raise ImportError(
                f"Could not import build_q2l from 'models.query2label' with sys.path including "
                f"{repo_root} and {repo_lib}. Make sure --q2l_repo_path points at your clone of "
                f"https://github.com/SlongLiu/query2labels. Original error: {e}"
            )
        try:
            from utils.misc import clean_state_dict
        except ImportError:
            if args.verbose:
                print("[INFO] Could not import utils.misc.clean_state_dict; using local fallback "
                      "(strips a leading 'module.' prefix).")
            clean_state_dict = clean_state_dict_fallback

        # ==========================================
        # 2. BUILD ARGS (defaults + config_new.json overlay) & MODEL
        # ==========================================
        if args.verbose:
            print(f"[INFO] Building Q2L args from {args.q2l_config}...")
        q2l_args = build_q2l_args(args.q2l_config)
        if args.verbose:
            print(f"[INFO] backbone={q2l_args.backbone} img_size={q2l_args.img_size} "
                  f"num_class={q2l_args.num_class} orid_norm={q2l_args.orid_norm} "
                  f"dim_feedforward={q2l_args.dim_feedforward} hidden_dim={q2l_args.hidden_dim}")

        model = build_q2l(q2l_args).to(device)

        # ==========================================
        # 3. LOAD CHECKPOINT
        # ==========================================
        if args.verbose:
            print(f"[INFO] Loading checkpoint from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
        elif "model" in checkpoint:
            raw_state = checkpoint["model"]
        else:
            raise ValueError(
                f"Checkpoint at {args.checkpoint_path} has neither 'state_dict' nor 'model' key. "
                f"Found keys: {list(checkpoint.keys())}."
            )
        state_dict = clean_state_dict(raw_state)
        try:
            model.load_state_dict(state_dict, strict=True)
            if args.verbose:
                print("[INFO] Loaded checkpoint with strict=True (exact match, as in the official script).")
        except RuntimeError as e_strict:
            # NOTE: strict=False only tolerates missing/unexpected KEYS, not shape
            # mismatches within a matching key -- so a genuine architecture
            # mismatch (wrong num_class, wrong hidden_dim, etc.) will raise a
            # RuntimeError from strict=False too. Catch both and fail with one
            # clear, actionable message rather than leaking a raw traceback.
            try:
                missing_k, unexpected_k = model.load_state_dict(state_dict, strict=False)
                print(f"⚠️  strict=True load failed ({e_strict}). strict=False succeeded instead: "
                      f"missing_keys={len(missing_k)} unexpected_keys={len(unexpected_k)}.\n"
                      f"   Double-check --q2l_config matches this checkpoint's architecture -- a "
                      f"successful strict=False load with many missing/unexpected keys usually still "
                      f"means something is wrong.")
            except RuntimeError as e_loose:
                raise RuntimeError(
                    f"Could not load {args.checkpoint_path} into the model built from "
                    f"{args.q2l_config}, even with strict=False (this usually means a genuine shape "
                    f"mismatch -- wrong num_class, hidden_dim, or backbone). "
                    f"strict=True error: {e_strict}\nstrict=False error: {e_loose}\n"
                    f"Double-check --q2l_config is the config that actually shipped with this exact "
                    f"checkpoint file."
                ) from e_loose
        model.eval()

        # ==========================================
        # 4. PREPROCESSING (from get_dataset.py's test_data_transform, verified)
        # ==========================================
        if q2l_args.orid_norm:
            # "orid_norm" = "original/raw" normalization: mean 0 / std 1 means
            # this is effectively a NO-OP normalization (pixel values pass
            # through basically unchanged after ToTensor's [0,1] scaling) —
            # this specific checkpoint was trained this way, so evaluation
            # must match it exactly.
            normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((q2l_args.img_size, q2l_args.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

    else:  # model_type == "multitask_direct"
        if args.verbose:
            print(f"[INFO] Building {args.multitask_backbone_choice} multitask model...")
        backbone = CustomBackbone(model_choice=args.multitask_backbone_choice)
        model = MultiTaskModel(backbone, backbone.feature_dim).to(device)

        if args.verbose:
            print(f"[INFO] Loading checkpoint from {args.multitask_checkpoint_path}...")
        state = torch.load(args.multitask_checkpoint_path, map_location=device)
        model.load_state_dict(state)  # plain state_dict, no wrapper keys, no cleaning needed (verified)
        model.eval()

        # Deterministic eval-time preprocessing -- same as evaluate_imagenet.py /
        # evaluate_places.py's multitask_direct mode (see those scripts for the
        # full rationale).
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        q2l_args = None  # not used in this mode; keeps later references safe via model_label instead

    # ==========================================
    # 5. LOAD COCO CATEGORIES & BUILD TAXONOMY MAPPING
    # ==========================================
    if args.verbose:
        print(f"[INFO] Loading COCO categories from {args.instances_json}...")
    id_to_name = load_coco_categories(args.instances_json)
    ordered_cat_ids = sorted(id_to_name.keys())  # Q2L has no per-checkpoint class-order manifest
    if args.model_type == "q2l" and len(ordered_cat_ids) != q2l_args.num_class:
        print(f"⚠️  Annotation file has {len(ordered_cat_ids)} categories but the checkpoint expects "
              f"{q2l_args.num_class}. Standard COCO-80 should match; investigate before trusting results.")

    if args.verbose:
        print(f"[INFO] Loading Taxonomy from {args.excel_path}...")
    pipeline = TaxonomyEvaluationPipeline()
    import pandas as pd
    df_taxonomy = pd.read_excel(args.excel_path, sheet_name="data corrected")
    pipeline.load_custom_excel_annotations(df_taxonomy, "Biotic/abiotic", "Material/immaterial")

    id_to_synset, unmapped_coco_ids = build_category_id_to_synset(id_to_name)
    if unmapped_coco_ids and args.verbose:
        print(f"⚠️  {len(unmapped_coco_ids)} COCO categories have no entry in COCO_TO_WNSYNSET: "
              f"{unmapped_coco_ids}")

    def safe_binary_map(val, positive_str, negative_str):
        if not isinstance(val, str):
            return None
        val = val.strip().lower()
        if val == positive_str.lower(): return 1
        if val == negative_str.lower(): return 0
        return None

    # Build one taxonomy-label COLUMN per COCO category, aligned with the
    # SAME `ordered_cat_ids` column ordering used by CocoMultiLabelDataset
    # above (so column i in col_nature/col_biotic/col_material corresponds
    # exactly to column i of the ground-truth multi-hot vectors).
    col_nature, col_biotic, col_material = [], [], []
    stats = {"nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}
    for cid in ordered_cat_ids:
        synset_str = id_to_synset.get(cid)
        node_attrs = pipeline.get_node_attributes(synset_str) if synset_str else None
        if not node_attrs:
            stats["unmapped"] += 1
            col_nature.append(None); col_biotic.append(None); col_material.append(None)
            continue
        is_nature = node_attrs.get('is_nature')
        col_nature.append(1 if is_nature else 0)
        if is_nature: stats["nature"] += 1
        bio_bin = safe_binary_map(node_attrs.get('biotic_abiotic'), "biotic", "abiotic")
        col_biotic.append(bio_bin)
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1
        mat_bin = safe_binary_map(node_attrs.get('material_immaterial'), "material", "immaterial")
        col_material.append(mat_bin)
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    if args.verbose:
        print("\n[INFO] --- Taxonomy label statistics (out of 80 COCO classes) ---")
        print(f"  Nature classes:        {stats['nature']}")
        print(f"  Biotic/Abiotic:        {stats['biotic']} / {stats['abiotic']}")
        print(f"  Material/Immaterial:   {stats['material']} / {stats['immaterial']}")
        print(f"  Unmapped:              {stats['unmapped']}")
        print("--------------------------------------------------------------\n")

    # ==========================================
    # 6. DATASET / DATALOADER
    # ==========================================
    dataset = CocoMultiLabelDataset(args.images_dir, args.instances_json, ordered_cat_ids, preprocess)
    if args.max_samples is not None:
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), min(args.max_samples, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ==========================================
    # 7. INFERENCE LOOP
    # ==========================================
    all_targets = []
    all_scores = []                   # q2l mode: per-class sigmoid scores
    all_pred_nature_direct = []       # multitask_direct mode: already-final 0/1 per image
    all_pred_material_direct = []     # multitask_direct mode: already-final 1/0/None per image
    all_pred_biotic_direct = []       # multitask_direct mode: already-final 1/0/None per image

    print(f"Running inference over {len(dataset)} images...")
    from tqdm import tqdm
    with torch.no_grad():
        if args.model_type == "q2l":
            # The model outputs raw (unbounded) scores per class ("logits");
            # sigmoid squashes each independently into a [0,1] probability —
            # unlike softmax (used for single-label problems), sigmoid does
            # NOT force the scores to sum to 1, since MULTIPLE classes can be
            # simultaneously "yes" for one image.
            sigmoid = torch.nn.Sigmoid()
            for images, targets, _img_ids in tqdm(dataloader, disable=not args.verbose):
                images = images.to(device)
                scores = sigmoid(model(images)).cpu().numpy()
                all_scores.append(scores)
                all_targets.append(targets.numpy())
            all_scores = np.concatenate(all_scores, axis=0)

        else:  # multitask_direct
            for images, targets, _img_ids in tqdm(dataloader, disable=not args.verbose):
                images = images.to(device)
                out_nature, out_materiality, out_biological, _out_landscape = model(images)
                all_pred_nature_direct.extend(out_nature.argmax(dim=1).cpu().tolist())
                all_pred_material_direct.extend(
                    [MULTITASK_MATERIALITY_TO_OURS.get(p) for p in out_materiality.argmax(dim=1).cpu().tolist()]
                )
                all_pred_biotic_direct.extend(
                    [MULTITASK_BIOLOGICAL_TO_OURS.get(p) for p in out_biological.argmax(dim=1).cpu().tolist()]
                )
                all_targets.append(targets.numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    if args.model_type == "q2l":
        # Apply the threshold to every class score independently — each
        # class becomes an independent binary "present"/"absent" decision.
        all_preds = (all_scores >= args.threshold).astype(int)

    # ==========================================
    # 8. METRICS
    # ==========================================
    def per_image_binary_metrics(col_labels, restrict_pool=None):
        """
        q2l mode: prediction derived by aggregating thresholded per-class scores.

        restrict_pool: optional boolean array over all images, restricting
        which images are ELIGIBLE ground truth for this task. Required for
        biotic/material: those are sub-classifications that only apply once
        something is already nature (a car or laptop has no valid
        biotic/abiotic ground truth at all -- treating "no biotic tag
        present" as ground-truth "abiotic" would silently conflate
        "not nature" with "abiotic nature", which is wrong, not just
        incomplete). Left as None for the top-level nature task itself,
        where every image is a valid ground-truth instance.

        After restricting the pool, this also checks that BOTH classes are
        actually realized in the resulting ground truth (not just that a
        negative-tagged class exists in the taxonomy) -- e.g. COCO's nature
        classes (bird, dog, cat, ... potted_plant) are all biotic; no COCO
        class is tagged abiotic, so even correctly restricted to
        nature-positive images, ground truth would be 100% biotic with zero
        abiotic instances. That's a genuine, structural property of which
        classes COCO happens to contain -- reported honestly as "not
        computable" rather than silently reporting a one-class metric.
        """
        # Which taxonomy-label COLUMNS (COCO categories) are relevant for
        # THIS specific axis — e.g. for nature, every column labeled 1;
        # non-matching/unmapped columns (None/0) are excluded from `cols`.
        cols = [c for c, lab in enumerate(col_labels) if lab == 1]
        if not cols:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        targets_pool = all_targets if restrict_pool is None else all_targets[restrict_pool]
        preds_pool = all_preds if restrict_pool is None else all_preds[restrict_pool]
        if len(targets_pool) == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        # Reduce a per-CLASS multi-hot row down to a single per-IMAGE binary
        # label: "does this image contain ANY of the relevant classes at
        # all?" — done for both ground truth and predictions the same way.
        gt_any_pos = (targets_pool[:, cols].sum(axis=1) > 0).astype(int)
        pred_any_pos = (preds_pool[:, cols].sum(axis=1) > 0).astype(int)
        if len(set(gt_any_pos.tolist())) < 2:
            # Ground truth is all-one-class after restriction -- e.g. no abiotic
            # examples exist among COCO's classes. Not a valid binary task.
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        acc = accuracy_score(gt_any_pos, pred_any_pos)
        p, r, f1, _ = precision_recall_fscore_support(gt_any_pos, pred_any_pos, average='binary', zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
                "f1": float(f1), "support": int(len(gt_any_pos))}

    def per_image_binary_metrics_direct(col_labels, direct_preds, restrict_pool=None):
        """
        multitask_direct mode: SAME ground-truth construction and pool
        restriction as per_image_binary_metrics (see that docstring), but the
        prediction comes directly from a holistic per-image model call
        (Paula's model gives one nature/materiality/biological answer per
        whole image, not a per-object score to threshold and aggregate). A
        None prediction (the model's own "not applicable" class) is
        penalized as wrong, same convention as elsewhere in this project.
        """
        cols = [c for c, lab in enumerate(col_labels) if lab == 1]
        if not cols:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        # dtype=object because this list can contain None values mixed with
        # integers, which a plain numeric numpy array can't represent.
        direct_preds_arr = np.array(direct_preds, dtype=object)
        if restrict_pool is None:
            targets_pool = all_targets
            preds_pool = direct_preds_arr
        else:
            targets_pool = all_targets[restrict_pool]
            preds_pool = direct_preds_arr[restrict_pool]
        if len(targets_pool) == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        gt_any_pos = (targets_pool[:, cols].sum(axis=1) > 0).astype(int)
        if len(set(gt_any_pos.tolist())) < 2:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        # A None prediction (model said "not applicable") is scored as the
        # OPPOSITE of ground truth (`1 - g`), guaranteeing it counts as wrong
        # rather than being silently dropped or defaulted to a fixed class.
        pred_direct = np.array([1 - g if p is None else p for g, p in zip(gt_any_pos, preds_pool)])
        acc = accuracy_score(gt_any_pos, pred_direct)
        p, r, f1, _ = precision_recall_fscore_support(gt_any_pos, pred_direct, average='binary', zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
                "f1": float(f1), "support": int(len(gt_any_pos))}

    # Nature-positive image mask, used to restrict biotic/material ground truth
    # (see per_image_binary_metrics docstring for why this restriction matters).
    nature_pos_cols = [c for c, lab in enumerate(col_nature) if lab == 1]
    is_nature_image = (all_targets[:, nature_pos_cols].sum(axis=1) > 0) if nature_pos_cols \
        else np.zeros(all_targets.shape[0], dtype=bool)

    if args.model_type == "q2l":
        # mAP (mean Average Precision) is COCO's own standard threshold-FREE
        # multi-label metric — computed only over classes that actually have
        # at least one positive example in this evaluation set (a class with
        # zero positives has no meaningful precision-recall curve to average).
        valid_cols = [c for c in range(all_targets.shape[1]) if all_targets[:, c].sum() > 0]
        per_class_ap = [average_precision_score(all_targets[:, c], all_scores[:, c]) for c in valid_cols]
        coco_mAP = float(np.mean(per_class_ap)) if per_class_ap else 0.0

        def per_class_binary_metrics(col_labels):
            """Flatten ALL relevant (image, class) pairs into one long list
            and compute a single binary accuracy/precision/recall/F1 across
            every one — i.e. "out of every (image, nature-class) pair,
            how often did the model correctly say present/absent?" This is a
            DIFFERENT question from per_image_binary_metrics above (which
            collapses each image down to one yes/no first)."""
            cols = [c for c, lab in enumerate(col_labels) if lab is not None]
            if not cols:
                return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}
            # `.ravel()` flattens the 2D (images x classes) array into one
            # long 1D array of individual (image, class) decisions.
            y_true = all_targets[:, cols].astype(int).ravel()
            y_pred = all_preds[:, cols].astype(int).ravel()
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
                    "f1": float(f1), "support": int(len(y_true))}

        nature_per_class = per_class_binary_metrics(col_nature)
        biotic_per_class = per_class_binary_metrics(col_biotic)
        material_per_class = per_class_binary_metrics(col_material)
        nature_per_image = per_image_binary_metrics(col_nature)
        biotic_per_image = per_image_binary_metrics(col_biotic, restrict_pool=is_nature_image)
        material_per_image = per_image_binary_metrics(col_material, restrict_pool=is_nature_image)

    else:  # multitask_direct -- per user instruction: per-class and mAP don't apply,
           # this kind of model gives one holistic call per image, not per-object scores.
        coco_mAP = None
        nature_per_class = biotic_per_class = material_per_class = None
        nature_per_image = per_image_binary_metrics_direct(col_nature, all_pred_nature_direct)
        biotic_per_image = per_image_binary_metrics_direct(
            col_biotic, all_pred_biotic_direct, restrict_pool=is_nature_image)
        material_per_image = per_image_binary_metrics_direct(
            col_material, all_pred_material_direct, restrict_pool=is_nature_image)

    # ==========================================
    # 9. TERMINAL SUMMARY & SAVE
    # ==========================================
    print("\n" + "=" * 60)
    title_suffix = f": {q2l_args.backbone.upper()} @ {q2l_args.img_size}" if args.model_type == "q2l" \
        else f": {model_label.upper()}"
    print(f"📊 COCO BASELINE{title_suffix}")
    print("=" * 60)
    print(f"--- 80-Class COCO Multi-Label (threshold-free mAP) ---")
    if args.model_type == "q2l":
        print(f"mAP: {coco_mAP:.4f}  (evaluated over {len(valid_cols)}/{all_targets.shape[1]} classes with positives)")
    else:
        print("N/A -- this model gives one holistic nature/materiality/biological call per image, "
              "not per-class scores, so neither mAP nor per-class metrics apply.")

    if args.model_type == "q2l":
        for title, m in [("Nature vs. No-Nature", nature_per_class),
                         ("Biotic vs. Abiotic", biotic_per_class),
                         ("Material vs. Immaterial", material_per_class)]:
            print(f"\n--- PER-CLASS Binary: {title} (Support: {m['support']}) ---")
            print(f"Accuracy:  {m['accuracy']:.4f}\nPrecision: {m['precision']:.4f}\n"
                  f"Recall:    {m['recall']:.4f}\nF1 Score:  {m['f1']:.4f}")

    for title, m in [("Nature vs. No-Nature", nature_per_image),
                     ("Biotic vs. Abiotic", biotic_per_image),
                     ("Material vs. Immaterial", material_per_image)]:
        print(f"\n--- PER-IMAGE Binary: {title} (Support: {m['support']}) ---")
        print(f"Accuracy:  {m['accuracy']:.4f}\nPrecision: {m['precision']:.4f}\n"
              f"Recall:    {m['recall']:.4f}\nF1 Score:  {m['f1']:.4f}")
    print("=" * 60)

    if args.wandb:
        import wandb

        def flatten_metrics(prefix, m):
            if m is None:
                return {}
            return {
                f"{prefix}/Accuracy": m['accuracy'],
                f"{prefix}/Precision": m['precision'],
                f"{prefix}/Recall": m['recall'],
                f"{prefix}/F1": m['f1'],
                f"{prefix}/Support": m['support'],
            }

        wandb_log_dict = {"COCO/mAP": coco_mAP}
        wandb_log_dict.update(flatten_metrics("PerClass/Nature", nature_per_class))
        wandb_log_dict.update(flatten_metrics("PerClass/Biotic", biotic_per_class))
        wandb_log_dict.update(flatten_metrics("PerClass/Material", material_per_class))
        wandb_log_dict.update(flatten_metrics("PerImage/Nature", nature_per_image))
        wandb_log_dict.update(flatten_metrics("PerImage/Biotic", biotic_per_image))
        wandb_log_dict.update(flatten_metrics("PerImage/Material", material_per_image))
        wandb.log(wandb_log_dict)

    model_field = f"Q2L-{q2l_args.backbone}-{q2l_args.img_size}" if args.model_type == "q2l" else model_label
    summary_results = {
        "model": model_field,
        "model_type": args.model_type,
        "dataset": "coco",
        "instances_json": args.instances_json,
        "coco_2014_train_val2017_eval_note": (
            "This checkpoint was trained on COCO train2014; evaluated here against "
            "val2017. No train/val leakage: val2017 is the same fixed 5,000-image "
            "'minival2014' subset always held out of standard COCO training conventions."
        ) if args.model_type == "q2l" else None,
        "samples_evaluated": len(dataset),
        "threshold": args.threshold if args.model_type == "q2l" else None,
        "coco_mAP": coco_mAP,
        "per_class": {"nature": nature_per_class, "biotic": biotic_per_class, "material": material_per_class},
        "per_image": {"nature": nature_per_image, "biotic": biotic_per_image, "material": material_per_image},
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
