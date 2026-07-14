#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from first_tests.evaluation import TaxonomyEvaluationPipeline


# ============================================================================
# MULTITASK DIRECT-TAXONOMY MODEL (Paula Feliu's TFG)
# https://github.com/paulafeliu/TFG-Interpretability-Techniques-in-Social-Media-Images
# ============================================================================
# Inlined rather than imported from a cloned repo: these two classes are ~35
# lines total (verified against the actual repo source), and this session has
# already hit several rounds of dependency/version pain importing external
# research repos (torchvision API drift, missing packages, custom CUDA
# extensions). A one-person student repo with no guarantee of long-term
# availability is exactly the kind of dependency worth avoiding when the
# alternative is this cheap. Copied verbatim from models/backbone.py and
# models/multitask_model.py, with ONE intentional change: the original used
# `pretrained=True` to initialize the backbone with ImageNet weights before
# THEIR training. We load their fully-trained checkpoint afterward with
# strict=True, which overwrites every weight anyway -- so `weights=None` here
# saves an unnecessary network download and gives an identical final result.
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


# Verified label encodings from utils/main_utils.py's build_dataset():
#   nature_visual:            {"Yes": 1, "No": 0}                       -> matches our convention directly
#   nep_materiality_visual:   {"material": 0, "immaterial": 1, "nan": 2} -> OPPOSITE of our convention (material=1)
#   nep_biological_visual:    {"biotic": 0, "abiotic": 1, "nan": 2}      -> OPPOSITE of our convention (biotic=1)
# "nan" (class 2) means the model itself predicts "not applicable" (their
# convention: undefined when nature=No). We remap 0/1 to our convention and
# treat 2 as "no usable prediction", same as an ImageNet mapping failure.
MULTITASK_MATERIALITY_TO_OURS = {0: 1, 1: 0}  # their material(0)->our 1, their immaterial(1)->our 0
MULTITASK_BIOLOGICAL_TO_OURS = {0: 1, 1: 0}   # their biotic(0)->our 1, their abiotic(1)->our 0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Closed-Set Models on Nature Taxonomy")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageNet validation split.")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx", help="Path to taxonomy.")
    parser.add_argument("--model_name", type=str, default="convnext_base", help="Torchvision model name.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--output_file", type=str, default="closed_set_baseline.json", help="Summary output.")

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

def safe_binary_map(val, positive_str, negative_str):
    """Safely converts string annotations to binary labels."""
    if not isinstance(val, str): return None
    val = val.strip().lower()
    if val == positive_str.lower(): return 1
    if val == negative_str.lower(): return 0
    return None

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
    pipeline = TaxonomyEvaluationPipeline()
    df_taxonomy = pd.read_excel(args.excel_path, sheet_name="data corrected")
    pipeline.load_custom_excel_annotations(df_taxonomy, "Biotic/abiotic", "Material/immaterial")

    if args.verbose: print(f"[INFO] Mapping ImageNet directories from {args.data_dir}...")
    full_dataset = datasets.ImageFolder(args.data_dir, transform=preprocess)
    
    # Extract mappings BEFORE applying any subsetting wrappers
    idx_to_wnid = {v: k for k, v in full_dataset.class_to_idx.items()}

    map_nature = {}
    map_biotic = {}
    map_material = {}
    missing_classes = []
    
    stats = {"nature": 0, "biotic": 0, "abiotic": 0, "material": 0, "immaterial": 0, "unmapped": 0}

    for idx, wnid in idx_to_wnid.items():
        synset_str = pipeline.get_synset_str_from_wnid(wnid)
        node_attrs = pipeline.get_node_attributes(synset_str)
        
        if not node_attrs:
            stats["unmapped"] += 1
            missing_classes.append(synset_str)
            
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
    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=not args.verbose):
            images, labels = images.to(device), labels.to(device)
            all_gt_imgnet.extend(labels.cpu().tolist())

            if args.model_type == "torchvision":
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_pred_imgnet.extend(preds.cpu().tolist())

            else:  # multitask_direct
                out_nature, out_materiality, out_biological, _out_landscape = model(images)
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
    def calculate_binary_metrics(gt_list, pred_list, map_dict, task_name):
        valid_gts, valid_preds = [], []
        
        for gt_idx, pred_idx in zip(gt_list, pred_list):
            mapped_gt = map_dict.get(gt_idx)
            mapped_pred = map_dict.get(pred_idx)
            
            if mapped_gt is not None:
                valid_gts.append(mapped_gt)
                if mapped_pred is None:
                    valid_preds.append(1 - mapped_gt) # Penalize mapping failure
                else:
                    valid_preds.append(mapped_pred)

        if not valid_gts:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        acc = accuracy_score(valid_gts, valid_preds)
        p, r, f1, _ = precision_recall_fscore_support(valid_gts, valid_preds, average='binary', zero_division=0)
        
        return {
            "accuracy": float(acc), 
            "precision": float(p), 
            "recall": float(r), 
            "f1": float(f1),
            "support": len(valid_gts)
        }

    def calculate_binary_metrics_direct(gt_list, pred_direct_list, map_dict):
        """
        Same methodology as calculate_binary_metrics, but for models whose
        predictions are ALREADY final taxonomy labels (0/1/None) rather than
        class indices needing a second lookup through map_dict. Ground truth
        is still built the same way (from the dataset's own class label,
        mapped through the taxonomy). A None prediction (e.g. the model's
        own "not applicable" class) is penalized as wrong, same convention
        as an ImageNet mapping failure -- consistent methodology across both
        model types rather than silently excluding hard cases from support.
        """
        valid_gts, valid_preds = [], []
        for gt_idx, pred_direct in zip(gt_list, pred_direct_list):
            mapped_gt = map_dict.get(gt_idx)
            if mapped_gt is not None:
                valid_gts.append(mapped_gt)
                if pred_direct is None:
                    valid_preds.append(1 - mapped_gt)  # Penalize "not applicable" / undefined prediction
                else:
                    valid_preds.append(pred_direct)

        if not valid_gts:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

        acc = accuracy_score(valid_gts, valid_preds)
        p, r, f1, _ = precision_recall_fscore_support(valid_gts, valid_preds, average='binary', zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r),
                "f1": float(f1), "support": len(valid_gts)}

    if args.model_type == "torchvision":
        # Standard ImageNet Metrics
        imgnet_acc = accuracy_score(all_gt_imgnet, all_pred_imgnet)
        imgnet_p, imgnet_r, imgnet_f1, _ = precision_recall_fscore_support(
            all_gt_imgnet, all_pred_imgnet, average='macro', zero_division=0
        )
        # Taxonomic Binary Metrics
        nature_metrics = calculate_binary_metrics(all_gt_imgnet, all_pred_imgnet, map_nature, "Nature")
        biotic_metrics = calculate_binary_metrics(all_gt_imgnet, all_pred_imgnet, map_biotic, "Biotic")
        material_metrics = calculate_binary_metrics(all_gt_imgnet, all_pred_imgnet, map_material, "Material")

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
    
    print(f"\n--- Binary: Nature vs. No Nature (Support: {nature_metrics['support']}) ---")
    print(f"Accuracy:  {nature_metrics['accuracy']:.4f}")
    print(f"Precision: {nature_metrics['precision']:.4f}")
    print(f"Recall:    {nature_metrics['recall']:.4f}")
    print(f"F1 Score:  {nature_metrics['f1']:.4f}")
    
    print(f"\n--- Binary: Biotic vs. Abiotic (Support: {biotic_metrics['support']}) ---")
    print(f"Accuracy:  {biotic_metrics['accuracy']:.4f}")
    print(f"Precision: {biotic_metrics['precision']:.4f}")
    print(f"Recall:    {biotic_metrics['recall']:.4f}")
    print(f"F1 Score:  {biotic_metrics['f1']:.4f}")
    
    print(f"\n--- Binary: Material vs. Immaterial (Support: {material_metrics['support']}) ---")
    print(f"Accuracy:  {material_metrics['accuracy']:.4f}")
    print(f"Precision: {material_metrics['precision']:.4f}")
    print(f"Recall:    {material_metrics['recall']:.4f}")
    print(f"F1 Score:  {material_metrics['f1']:.4f}")
    print("="*55)

    if args.wandb:
        print("\n🚀 Uploading baseline metrics to Weights & Biases...")
        wandb_log_dict = {
            "Number of missing classes:":stats["unmapped"],
            
            "Imagenet/Accuracy": imgnet_acc,
            "Imagenet/Precision_Macro": imgnet_p,
            "Imagenet/Recall_Macro": imgnet_r,
            "Imagenet/F1_Macro": imgnet_f1,
            
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
            "Material/Support": material_metrics['support']
        }
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

if __name__ == "__main__":
    main()