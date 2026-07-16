#!/usr/bin/env python3
"""
evaluate_taxonomy_labeling.py

Evaluates a VLM's TAXONOMIC REASONING CALIBRATION by sending the image alongside 
the Ground Truth (GT) target class name. This mirrors the exact fallback condition 
of the final multi-modal pipeline when processing unmapped objects.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import wandb
from pydantic import BaseModel, Field
from typing import Literal

from lib.excel_loader import TaxonomyGraph
from lib.vlm import MODEL_REGISTRY, create_vlm
from lib.dataset_loader import load_dataset


class TaxonomyResponse(BaseModel):
    """
    Pydantic schema driving `outlines` structured output.
    By defining `reasoning` first, we force the model into a chain-of-thought 
    generation process before it commits to the final taxonomic labels.
    """
    reasoning: str = Field(description="One concise sentence justifying your classification based on the visual evidence.")
    nature: Literal["yes", "no"]
    biotic: Literal["biotic", "abiotic", "n/a"]
    material: Literal["material", "immaterial", "n/a"]


_AXIS_INSTRUCTIONS = {
    "nature": '"nature": either "yes" or "no" — whether this instance counts as nature under the definition above',
    "biotic": '"biotic": either "biotic", "abiotic", or "n/a" — only answer "biotic"/"abiotic" if "nature" is "yes"; use "n/a" if "nature" is "no"',
    "material": '"material": either "material", "immaterial", or "n/a" — only answer "material"/"immaterial" if "nature" is "yes"; use "n/a" if "nature" is "no"'
}


def build_classification_prompt(class_name, axes):
    """
    Constructs the contextualized prompt. The model is forced to evaluate the 
    taxonomic labels based on the specific visual instance depicted in the image.
    """
    unknown_axes = set(axes) - set(_AXIS_INSTRUCTIONS)
    if unknown_axes:
        raise ValueError(f"Unknown axis/axes requested: {unknown_axes}")

    field_lines = "\n".join(f"  - {_AXIS_INSTRUCTIONS[axis]}" for axis in axes)

    return f"""You are analyzing a specific object identified in the provided image. 
The object is classified as: {class_name}

Based on the visual evidence in the image and the definitions provided, classify this specific instance of the object. 

Provide your reasoning first, followed by the specific labels according to these rules:
{field_lines}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM taxonomy classification via image + GT target class pairing."
    )
    
    parser.add_argument("--output_mode", type=str, choices=["structured", "free_form"], default="structured")
    parser.add_argument("--excel_path", type=str, default="../flat_wordnet_tree_fixed.xlsx",
                        help="Path to the BIG-5 WordNet taxonomy Excel file.")
    parser.add_argument("--sheet_name", type=str, default="data corrected")

    # Standard Dataset Arguments
    parser.add_argument("--dataset", type=str, required=True, choices=["coco", "imagenet", "places365", "big5"],
                        help="Dataset to load and evaluate.")
    parser.add_argument("--data_dir", type=str, help="Path to images directory (for COCO/ImageNet/Places).")
    parser.add_argument("--instances_json", type=str, help="Path to instances json (for COCO).")
    parser.add_argument("--places_categories_txt", type=str, help="Path to categories_places365.txt (for Places).")
    
    # BIG-5 Dataset Arguments
    parser.add_argument("--twitter_en_gt_csv", type=str, default=None, help="table_for_pau_twitter-en-6.csv")
    parser.add_argument("--twitter_es_gt_csv", type=str, default=None, help="table_for_pau_twitter-es-6.csv")
    parser.add_argument("--twitter_en_media_csv", type=str, default=None, help="phase-1_twitter-en.csv")
    parser.add_argument("--twitter_es_media_csv", type=str, default=None, help="phase-1_twitter-es.csv")
    parser.add_argument("--images_cache_dir", type=str, default="./big_5_cache", help="Cache for BIG-5 downloads.")

    # Model Arguments
    parser.add_argument("--model_family", type=str, required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")

    # Generation Arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Context Files
    parser.add_argument("--nature_definition_path", type=str, default="docs/big5_nature_definition.txt")
    parser.add_argument("--taxonomy_axes_path", type=str, default="docs/big5_taxonomy_axes.txt")
    
    parser.add_argument("--output_file", type=str, default="taxonomy_calibration_results.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluations.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--wandb", action="store_true", help="Store the results on WandB.")

    return parser.parse_args()


def load_system_prompt(nature_def_path, taxonomy_axes_path):
    nature_def = Path(nature_def_path).read_text()
    taxonomy_axes = Path(taxonomy_axes_path).read_text()
    return f"{nature_def}\n\n{taxonomy_axes}"


def _label_to_bool(value, axis):
    """Standardizes string answers into boolean logic depending on the axis."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    
    if axis == "nature":
        if normalized == "yes": return True
        if normalized == "no": return False
    elif axis == "biotic":
        if normalized == "biotic": return True
        if normalized == "abiotic": return False
        if normalized == "n/a": return None
    elif axis == "material":
        if normalized == "material": return True
        if normalized == "immaterial": return False
        if normalized == "n/a": return None
    return None


def calculate_binary_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if not y_true:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "support": len(y_true)}


def main():
    args = parse_args()
    if args.output_mode == "free_form":
        raise ValueError("This evaluation cannot be run in free_form mode.")

    model_label = f"{args.model_family}-{args.model_name}".replace("/", "_")
    print(f"🚀 Starting Contextualized VLM Taxonomy Calibration ({model_label}) on dataset '{args.dataset}'")

    if args.wandb:
        wandb.init(
            entity="paumonserrat03-universitat-aut-noma-de-barcelona",
            project="TFM_Closed-set",
            config=vars(args),
            name=f"taxonomy_image_calibration_{args.dataset}_{model_label}",
        )

    if args.verbose:
        print(f"[INFO] Loading taxonomy graph from {args.excel_path}...")
    graph = TaxonomyGraph()
    graph.load_excel(args.excel_path, sheet_name=args.sheet_name if not args.sheet_name.isdigit() else int(args.sheet_name))

    if args.verbose:
        print(f"[INFO] Loading {args.dataset} dataset instances...")
        
    dataset = load_dataset(
        args.dataset, 
        taxonomy_graph=graph,
        data_dir=args.data_dir,
        instances_json=args.instances_json,
        places_categories_txt=args.places_categories_txt,
        excel_path=args.excel_path,
        en_gt=args.twitter_en_gt_csv,
        es_gt=args.twitter_es_gt_csv,
        en_media=args.twitter_en_media_csv,
        es_media=args.twitter_es_media_csv,
        cache_dir=args.images_cache_dir
    )

    # Flatten the dataset into independent evaluation instances
    eval_instances = []
    for item in dataset:
        for t in item["targets"]:
            if t["gt_nature"] is not None:
                eval_instances.append({
                    "image_path": item["image_path"],
                    "class_name": t["class_name"],
                    "gt_nature": t["gt_nature"],
                    "gt_biotic": t["gt_biotic"],
                    "gt_material": t["gt_material"]
                })

    if args.max_samples is not None:
        random.seed(42)
        eval_instances = random.sample(eval_instances, min(args.max_samples, len(eval_instances)))

    if not eval_instances:
        print("No mapped evaluation instances found — exiting.")
        sys.exit(1)

    system_prompt = load_system_prompt(args.nature_definition_path, args.taxonomy_axes_path)

    if args.verbose:
        print(f"[INFO] Creating VLM: family='{args.model_family}', model='{args.model_name}'...")
        
    VLLM_FAMILIES = ("qwen", "mistral", "llava")
    if args.model_family in VLLM_FAMILIES:
        vlm_kwargs = {
            "dtype": args.dtype, "gpu_memory_utilization": args.gpu_memory_utilization, "trust_remote_code": args.trust_remote_code
        }
        if args.max_model_len is not None: vlm_kwargs["max_model_len"] = args.max_model_len
    else:
        vlm_kwargs = {"device": args.device}
        
    vlm = create_vlm(args.model_family, args.model_name, **vlm_kwargs)

    print(f"Running contextualized classification over {len(eval_instances)} target objects...")
          
    scored_results = []
    num_batches = (len(eval_instances) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        batch = eval_instances[start:start + args.batch_size]
        
        batch_prompts = [build_classification_prompt(r["class_name"], axes=["nature", "biotic", "material"]) for r in batch]
        batch_images = [r["image_path"] for r in batch]

        t0 = time.time()
        try:
            batch_results = vlm.generate_batch(
                prompts=batch_prompts, images=batch_images, system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                output_mode=args.output_mode, schema=TaxonomyResponse
            )
        except Exception as e: 
            print(f"⚠️ Batch {batch_idx + 1}/{num_batches} FAILED ({e!r}).")
            batch_results = [None] * len(batch)

        for r, result in zip(batch, batch_results):
            if result is None:
                r["prediction"] = {"reasoning": None, "parse_failed": True}
                r["scored_pred_nature"] = not r["gt_nature"]
                r["scored_pred_biotic"] = not r["gt_biotic"] if r["gt_biotic"] is not None else None
                r["scored_pred_material"] = not r["gt_material"] if r["gt_material"] is not None else None
            else:
                result = dict(result)
                result["parse_failed"] = False
                r["prediction"] = result
                
                pred_nature = _label_to_bool(result.get("nature"), "nature")
                pred_biotic = _label_to_bool(result.get("biotic"), "biotic")
                pred_material = _label_to_bool(result.get("material"), "material")
                
                r["scored_pred_nature"] = pred_nature if pred_nature is not None else (not r["gt_nature"])
                r["scored_pred_biotic"] = pred_biotic if pred_biotic is not None else (not r["gt_biotic"] if r["gt_biotic"] is not None else None)
                r["scored_pred_material"] = pred_material if pred_material is not None else (not r["gt_material"] if r["gt_material"] is not None else None)
            
            scored_results.append(r)

        if args.verbose:
            print(f"[INFO] Batch {batch_idx + 1}/{num_batches} done in {time.time() - t0:.1f}s.")

    # Calculate Metrics
    parse_failure_rate = sum(r["prediction"]["parse_failed"] for r in scored_results) / len(scored_results)

    nature_metrics = calculate_binary_metrics(
        [r["gt_nature"] for r in scored_results], [r["scored_pred_nature"] for r in scored_results]
    )
    
    biotic_subset = [r for r in scored_results if r["gt_biotic"] is not None]
    biotic_metrics = calculate_binary_metrics(
        [r["gt_biotic"] for r in biotic_subset], [r["scored_pred_biotic"] for r in biotic_subset]
    )
    
    material_subset = [r for r in scored_results if r["gt_material"] is not None]
    material_metrics = calculate_binary_metrics(
        [r["gt_material"] for r in material_subset], [r["scored_pred_material"] for r in material_subset]
    )

    print("\n" + "=" * 55)
    print(f"📊 CONTEXTUAL CALIBRATION: {model_label.upper()} on {args.dataset.upper()}")
    print("=" * 55)
    print(f"Evaluated Instances: {len(scored_results)}")
    print(f"Parse failure rate: {parse_failure_rate:.1%}")

    print(f"\n--- Binary: Nature vs. No Nature (Support: {nature_metrics['support']}) ---")
    print(f"Accuracy:  {nature_metrics['accuracy']:.4f}\nPrecision: {nature_metrics['precision']:.4f}")
    print(f"Recall:    {nature_metrics['recall']:.4f}\nF1 Score:  {nature_metrics['f1']:.4f}")

    print(f"\n--- Binary: Biotic vs. Abiotic (Support: {biotic_metrics['support']}) ---")
    print(f"Accuracy:  {biotic_metrics['accuracy']:.4f}\nPrecision: {biotic_metrics['precision']:.4f}")
    print(f"Recall:    {biotic_metrics['recall']:.4f}\nF1 Score:  {biotic_metrics['f1']:.4f}")
    
    print(f"\n--- Binary: Material vs. Immaterial (Support: {material_metrics['support']}) ---")
    print(f"Accuracy:  {material_metrics['accuracy']:.4f}\nPrecision: {material_metrics['precision']:.4f}")
    print(f"Recall:    {material_metrics['recall']:.4f}\nF1 Score:  {material_metrics['f1']:.4f}")
    print("=" * 55)

    if args.wandb:
        print("\n🚀 Uploading calibration metrics to Weights & Biases...")
        wandb.log({
            "ParseFailureRate": parse_failure_rate,
            "Nature/Accuracy": nature_metrics["accuracy"], "Nature/F1": nature_metrics["f1"],
            "Biotic/Accuracy": biotic_metrics["accuracy"], "Biotic/F1": biotic_metrics["f1"],
            "Material/Accuracy": material_metrics["accuracy"], "Material/F1": material_metrics["f1"]
        })

    # Prepare outputs
    flat_rows = []
    for r in scored_results:
        flat_rows.append({
            "image_path": r["image_path"],
            "class_name": r["class_name"],
            "gt_nature": r["gt_nature"], "gt_biotic": r["gt_biotic"], "gt_material": r["gt_material"],
            "pred_nature": r["prediction"].get("nature"),
            "pred_biotic": r["prediction"].get("biotic"),
            "pred_material": r["prediction"].get("material"),
            "reasoning": r["prediction"].get("reasoning"),
            "parse_failed": r["prediction"]["parse_failed"]
        })

    summary_results = {
        "model": model_label, "dataset": args.dataset,
        "parse_failure_rate": parse_failure_rate,
        "nature": nature_metrics, "biotic": biotic_metrics, "material": material_metrics,
    }

    output_path = Path(args.output_file)
    with open(output_path, "w") as f: json.dump(summary_results, f, indent=4)
    with open(output_path.with_name(output_path.stem + "_predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    if args.wandb:
        wandb.save(str(output_path))
        wandb.finish()


if __name__ == "__main__":
    main()