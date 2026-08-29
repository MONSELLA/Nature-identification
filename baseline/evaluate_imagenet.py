#!/usr/bin/env python3
# WHAT IS THIS SCRIPT? A "closed-set" baseline evaluation — unlike the VLM
# pipeline (which asks a language model open-ended questions), this script
# runs a STANDARD image classifier (e.g. a torchvision ConvNeXt or ResNet
# already trained on the 1000-class ImageNet task) and checks how well its
# predicted class, PROJECTED onto the BIG-5 taxonomy via WordNet, agrees with
# the taxonomy label of the TRUE class. It supports two very different kinds
# of models (see --model_type below): a plain off-the-shelf ImageNet
# classifier, or a custom "multitask" model (from an external student
# project) trained to predict nature/biotic/material labels directly.
import os
import sys
import argparse
import torch
import pandas as pd
import random
from tqdm import tqdm
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import wandb

# Get the absolute path of the directory one level up and add it to sys.path
# so `baseline.common` / `src.loaders...` are importable regardless of cwd.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from baseline.common import (
    TaxonomyLookup, CustomBackbone, MultiTaskModel, safe_binary_map,
    MULTITASK_MATERIALITY_TO_OURS, MULTITASK_BIOLOGICAL_TO_OURS,
    calculate_binary_metrics, calculate_binary_metrics_direct,
    append_results_rows, binary_metrics_to_rows, single_row, DEFAULT_RESULTS_CSV,
)


def parse_args():
    """Command-line flags: dataset/taxonomy paths, which model to run (and
    how — torchvision off-the-shelf, or the multitask checkpoint), and
    generation/testing/logging options."""
    parser = argparse.ArgumentParser(description="Evaluate Closed-Set Models on Nature Taxonomy")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageNet validation split.")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx", help="Path to taxonomy.")
    parser.add_argument("--model_name", type=str, default="convnext_base", help="Torchvision model name.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--output_file", type=str, default="closed_set_baseline.json", help="Summary output.")
    parser.add_argument("--results_csv", type=str, default=DEFAULT_RESULTS_CSV,
                        help="Shared long-format CSV every baseline run appends its rows to "
                             "(pivot this to build the thesis tables instead of the per-run JSON).")

    parser.add_argument("--model_type", type=str, default="torchvision",
                        choices=["torchvision", "multitask_direct"],
                        help="'torchvision' (default): standard ImageNet classifier, predictions projected onto "
                             "the taxonomy via the WordNet synset mapping (original behavior, unchanged). "
                             "'multitask_direct': a model trained to predict nature/materiality/biological "
                             "directly (e.g. Paula Feliu's TFG multitask model) -- no synset projection needed "
                             "for predictions; only --multitask_checkpoint_path and --multitask_backbone_choice "
                             "apply in this mode.")
    parser.add_argument("--multitask_checkpoint_path", type=str, default=None,
                        help="[multitask_direct mode] Path to the trained multitask checkpoint "
                             "(e.g. trained_DenseNet121_100epochs.pth). A plain state_dict, no wrapper keys.")
    parser.add_argument("--multitask_backbone_choice", type=str, default="DenseNet121",
                        choices=["DenseNet121", "ResNet18", "EfficientNetB0", "ResNet50"],
                        help="[multitask_direct mode] Must match the backbone the checkpoint was trained with.")

    # Testing and logging flags
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of images for testing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--wandb", action="store_true", help="Store the results on WandB.")

    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_label = args.model_name if args.model_type == "torchvision" else \
        f"{args.multitask_backbone_choice}-multitask-direct"
    print(f"🚀 Starting Closed-Set Baseline ({model_label}) on {device.upper()}")

    if args.wandb:
        wandb.init(
            entity="paumonserrat03-universitat-aut-noma-de-barcelona",
            project="TFM_Closed-set",
            config=vars(args),
            name=f"baseline_{model_label}"
        )

    # ==========================================
    # 1. LOAD MODEL & TRANSFORMS
    # ==========================================
    if args.model_type == "torchvision":
        if args.verbose: print(f"[INFO] Fetching weights and transforms for {args.model_name}...")
        # `models.get_model_weights(name).DEFAULT` fetches torchvision's own
        # recommended pretrained weights for this architecture, and
        # `weights.transforms()` returns the EXACT image preprocessing
        # (resize/crop/normalize) those specific weights were trained with —
        # using any other preprocessing would silently hurt accuracy.
        weights = models.get_model_weights(args.model_name).DEFAULT
        model = models.get_model(args.model_name, weights=weights).to(device)
        model.eval()
        preprocess = weights.transforms()

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

        # Deterministic eval-time preprocessing. NOTE: the repo's own get_transforms()
        # (used for their train/test split) includes RandomResizedCrop/ColorJitter/
        # HorizontalFlip -- training-time augmentation, not appropriate for
        # reproducible evaluation. Instead we use the SAME deterministic transform
        # their own repo uses for non-training inference (main.py's interp_transform,
        # used for interpretability on held-out images): Resize->CenterCrop->
        # Normalize with standard ImageNet stats.
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ==========================================
    # 2. MAP IMAGENET TO TAXONOMY
    # ==========================================
    if args.verbose: print(f"[INFO] Loading Taxonomy from {args.excel_path}...")
    taxonomy = TaxonomyLookup()
    taxonomy.load(args.excel_path)

    if args.verbose: print(f"[INFO] Mapping ImageNet directories from {args.data_dir}...")
    full_dataset = datasets.ImageFolder(args.data_dir, transform=preprocess)

    # Extract mappings BEFORE applying any subsetting wrappers
    # torchvision's ImageFolder assigns each class folder an integer index;
    # this builds the reverse lookup (index -> WordNet id) needed below,
    # BEFORE we potentially wrap the dataset in a Subset() further down
    # (Subset only changes which SAMPLES are used, not the class indexing).
    idx_to_wnid = {v: k for k, v in full_dataset.class_to_idx.items()}

    # These three dicts map an ImageNet class INDEX directly to its
    # taxonomy label (1/0/None) — built once here, then reused for every
    # image of that class during scoring, rather than re-resolving the
    # taxonomy for every single image.
    map_nature = {}
    map_biotic = {}
    map_material = {}
    missing_classes = []

    stats = {"nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}

    for idx, wnid in idx_to_wnid.items():
        synset_str = taxonomy.get_synset_str_from_wnid(wnid)
        node_attrs = taxonomy.get_node_attributes(synset_str)

        if not node_attrs:
            stats["unmapped"] += 1
            missing_classes.append(synset_str)
            # Unmapped classes are EXCLUDED from map_nature/map_biotic/
            # map_material entirely (matching lib/dataset_loader.py's
            # get_gt_from_graph and count_classes.py's identical loop) —
            # without this `continue`, execution would fall through with
            # node_attrs = {} (empty dict), and `{}.get(...)` returning None
            # for every key would silently set map_nature[idx] = 0, counting
            # an unmapped class as "not nature" instead of excluding it as
            # ground truth (contradicting this project's stated convention).
            continue

        # 1. Nature (Boolean mapped to 1/0)
        is_nature = node_attrs.get('is_nature')
        map_nature[idx] = 1 if is_nature else 0
        if is_nature: stats["nature"] += 1

        # 2. Biotic / Abiotic
        bio_val = node_attrs.get('biotic_abiotic')
        bio_bin = safe_binary_map(bio_val, "biotic", "abiotic")
        map_biotic[idx] = bio_bin
        if bio_bin == 1: stats["biotic"] += 1
        elif bio_bin == 0: stats["abiotic"] += 1

        # 3. Material / Immaterial
        mat_val = node_attrs.get('material_immaterial')
        mat_bin = safe_binary_map(mat_val, "material", "immaterial")
        map_material[idx] = mat_bin
        if mat_bin == 1: stats["material"] += 1
        elif mat_bin == 0: stats["immaterial"] += 1

    if args.verbose:
        print("\n[INFO] --- Taxonomy Mapping Statistics (out of 1000 classes) ---")
        print(f"  Nature mappings:       {stats['nature']}")
        print(f"  Biotic/Abiotic:        {stats['biotic']} / {stats['abiotic']}")
        print(f"  Material/Immaterial:   {stats['material']} / {stats['immaterial']}")
        print(f"  Classes not in Excel:  {stats['unmapped']}")
        print("--------------------------------------------------------------\n")

        if missing_classes:
            print("Missing classes:")
            for c in missing_classes:
                print(f" - {c}")
            print("\n")

    # ==========================================
    # 3. DATALOADER SETUP & SUBSETTING
    # ==========================================
    if args.max_samples is not None:
        if args.verbose: print(f"⚠️ Restricting execution to a deterministic random subset of {args.max_samples} samples.")
        random.seed(42) # Ensure the same subset across multiple model test runs
        # `Subset` wraps the full dataset so the DataLoader below only ever
        # touches these specific (randomly chosen but reproducible) indices,
        # rather than iterating the entire validation set — much faster for
        # quick smoke tests.
        subset_indices = random.sample(range(len(full_dataset)), min(args.max_samples, len(full_dataset)))
        dataset_to_use = Subset(full_dataset, subset_indices)
    else:
        dataset_to_use = full_dataset

    dataloader = DataLoader(dataset_to_use, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ==========================================
    # 4. INFERENCE LOOP
    # ==========================================
    all_gt_imgnet = []
    all_pred_imgnet = []                 # torchvision mode: predicted ImageNet class indices
    all_pred_nature_direct = []          # multitask_direct mode: already-final 0/1 nature predictions
    all_pred_material_direct = []        # multitask_direct mode: already-final 1/0/None (material=1)
    all_pred_biotic_direct = []          # multitask_direct mode: already-final 1/0/None (biotic=1)

    print(f"Running Inference over {len(dataset_to_use)} images...")
    with torch.no_grad():  # no gradients needed — pure inference, not training
        for images, labels in tqdm(dataloader, disable=not args.verbose):
            images, labels = images.to(device), labels.to(device)
            all_gt_imgnet.extend(labels.cpu().tolist())

            if args.model_type == "torchvision":
                outputs = model(images)
                # torch.max(outputs, 1) returns (max_value, max_index) along
                # dimension 1 (the class dimension) — we only need the
                # predicted class INDEX, hence `_, preds`.
                _, preds = torch.max(outputs, 1)
                all_pred_imgnet.extend(preds.cpu().tolist())

            else:  # multitask_direct
                out_nature, out_materiality, out_biological, _out_landscape = model(images)
                # `.argmax(dim=1)` picks, for each image, whichever of this
                # head's output classes scored highest — the model's
                # predicted class index for that specific task.
                pred_nature = out_nature.argmax(dim=1).cpu().tolist()          # 0/1, matches our convention
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
    if args.model_type == "torchvision":
        # Standard ImageNet Metrics
        # These metrics measure raw 1000-way ImageNet classification accuracy
        # (nothing taxonomy-related yet) — a sanity check that the loaded
        # model performs as expected before trusting its taxonomy projection.
        imgnet_acc = accuracy_score(all_gt_imgnet, all_pred_imgnet)
        imgnet_p, imgnet_r, imgnet_f1, _ = precision_recall_fscore_support(
            all_gt_imgnet, all_pred_imgnet, average='macro', zero_division=0
        )
        # Taxonomic Binary Metrics (both classes -- see common.calculate_binary_metrics)
        nature_metrics = calculate_binary_metrics(all_gt_imgnet, all_pred_imgnet, map_nature)
        biotic_metrics = calculate_binary_metrics(all_gt_imgnet, all_pred_imgnet, map_biotic)
        material_metrics = calculate_binary_metrics(all_gt_imgnet, all_pred_imgnet, map_material)

    else:  # multitask_direct
        # This model was never trained to do 1000-class ImageNet classification,
        # so that metric section is not applicable -- reported as None rather
        # than a misleading 0.0.
        imgnet_acc = imgnet_p = imgnet_r = imgnet_f1 = None
        nature_metrics = calculate_binary_metrics_direct(all_gt_imgnet, all_pred_nature_direct, map_nature)
        biotic_metrics = calculate_binary_metrics_direct(all_gt_imgnet, all_pred_biotic_direct, map_biotic)
        material_metrics = calculate_binary_metrics_direct(all_gt_imgnet, all_pred_material_direct, map_material)

    # ==========================================
    # 6. TERMINAL SUMMARY & SAVE
    # ==========================================
    print("\n" + "="*55)
    print(f"📊 CLOSED-SET BASELINE: {model_label.upper()}")
    print("="*55)
    if args.model_type == "torchvision":
        print("--- 1000-Class ImageNet ---")
        print(f"Accuracy:  {imgnet_acc:.4f}")
        print(f"Precision: {imgnet_p:.4f} (Macro)")
        print(f"Recall:    {imgnet_r:.4f} (Macro)")
        print(f"F1 Score:  {imgnet_f1:.4f} (Macro)")
    else:
        print("--- 1000-Class ImageNet ---")
        print("N/A -- this model predicts nature/materiality/biological directly, not ImageNet classes.")

    def print_binary_block(title, m):
        print(f"\n--- Binary: {title} ---")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        for polarity in ("positive", "negative"):
            pm = m[polarity]
            print(f"  [{polarity}] Precision: {pm['precision']:.4f}  Recall: {pm['recall']:.4f}  "
                  f"F1: {pm['f1']:.4f}  (Support: {pm['support']})")

    print_binary_block("Nature vs. No Nature", nature_metrics)
    print_binary_block("Biotic vs. Abiotic", biotic_metrics)
    print_binary_block("Material vs. Immaterial", material_metrics)
    print("="*55)

    if args.wandb:
        print("\n🚀 Uploading baseline metrics to Weights & Biases...")
        wandb_log_dict = {
            "Number of missing classes:": stats["unmapped"],

            "Imagenet/Accuracy": imgnet_acc,
            "Imagenet/Precision_Macro": imgnet_p,
            "Imagenet/Recall_Macro": imgnet_r,
            "Imagenet/F1_Macro": imgnet_f1,
        }
        for name, m in [("Nature", nature_metrics), ("Biotic", biotic_metrics), ("Material", material_metrics)]:
            wandb_log_dict[f"{name}/Accuracy"] = m["accuracy"]
            for polarity in ("positive", "negative"):
                pm = m[polarity]
                wandb_log_dict[f"{name}/{polarity.capitalize()}/Precision"] = pm["precision"]
                wandb_log_dict[f"{name}/{polarity.capitalize()}/Recall"] = pm["recall"]
                wandb_log_dict[f"{name}/{polarity.capitalize()}/F1"] = pm["f1"]
                wandb_log_dict[f"{name}/{polarity.capitalize()}/Support"] = pm["support"]
        wandb.log(wandb_log_dict)

    summary_results = {
        "model": model_label,
        "model_type": args.model_type,
        "samples_evaluated": len(dataset_to_use),
        "imagenet_1000": {"accuracy": imgnet_acc, "precision_macro": imgnet_p, "recall_macro": imgnet_r, "f1_macro": imgnet_f1},
        "nature": nature_metrics,
        "biotic": biotic_metrics,
        "material": material_metrics
    }

    with open(args.output_file, "w") as f:
        json.dump(summary_results, f, indent=4)
    print(f"💾 Results saved to {args.output_file}")

    if args.wandb:
        wandb.save(args.output_file)
        wandb.finish()

    # ==========================================
    # 7. APPEND TO THE SHARED RESULTS CSV (Table 1/2 shape)
    # ==========================================
    base_row = {"run_id": model_label, "dataset": "imagenet", "model": model_label, "model_type": args.model_type}
    rows = []
    if args.model_type == "torchvision":
        rows.append(single_row(base_row, "Base (Macro)", accuracy=imgnet_acc,
                                precision=imgnet_p, recall=imgnet_r, f1=imgnet_f1, support=len(dataset_to_use)))
    rows += binary_metrics_to_rows(base_row, "Nature", nature_metrics)
    rows += binary_metrics_to_rows(base_row, "Biotic", biotic_metrics)
    rows += binary_metrics_to_rows(base_row, "Material", material_metrics)
    append_results_rows(args.results_csv, rows)
    print(f"💾 Appended {len(rows)} rows to {args.results_csv}")

if __name__ == "__main__":
    main()
