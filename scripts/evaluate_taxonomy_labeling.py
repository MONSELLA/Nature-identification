#!/usr/bin/env python3
"""
evaluate_taxonomy_labeling.py

Evaluates a VLM's TAXONOMIC REASONING CALIBRATION by sending the image alongside
the Ground Truth (GT) target class name. This mirrors the exact fallback condition
of the final multi-modal pipeline when processing unmapped objects.

WHAT PROBLEM IS THIS SCRIPT ANSWERING?
The main VLM pipeline (src/vlm_pipeline.py) only asks the VLM to classify
nature/biotic/material for objects it couldn't resolve via WordNet mapping
(the "fallback" path). Before trusting that fallback in the full pipeline, we
want to measure, in isolation, HOW GOOD the VLM actually is at this specific
task: "given an image and the correct class name for something in it, can you
correctly say whether it's nature, biotic/abiotic, material/immaterial?"

This script does exactly that, directly against KNOWN ground-truth classes
(so we already know the right answer), and reports standard accuracy/
precision/recall/F1 for each of the three taxonomy axes. This is a
"calibration" check — it isolates the VLM's raw reasoning ability from
whether it correctly SPOTTED the object in the first place (that's a
different, separate part of the pipeline).
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import wandb

from src.loaders.excel_loader import TaxonomyGraph
from src.models.vlm_models import MODEL_REGISTRY, create_vlm
from src.loaders.dataset_loader import load_dataset
from src.results_store import update_results_store
# TaxonomyResponse + build_classification_prompt live in prompts.py so this
# calibration eval and the VLM pipeline's fallback path share the EXACT same
# prompt and schema (they cannot drift — same imported objects).
from src.models.prompts import TaxonomyResponse, build_classification_prompt


def parse_args():
    """Define and parse all command-line flags this script accepts. Grouped
    into: taxonomy/Excel settings, dataset-specific paths (only the ones
    relevant to --dataset need to actually be supplied), model/generation
    settings, and the three axis-definition text files fed to the VLM."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM taxonomy classification via image + GT target class pairing."
    )

    parser.add_argument("--output_mode", type=str, choices=["structured", "free_form"], default="structured")
    parser.add_argument("--excel_path", type=str, default="/home/pmonserrat/code/data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx",
                        help="Path to the BIG-5 WordNet taxonomy Excel file.")
    parser.add_argument("--sheet_name", type=str, default="data corrected")

    # Standard Dataset Arguments
    parser.add_argument("--dataset", type=str, required=True, choices=["coco", "imagenet", "places365", "big5"],
                        help="Dataset to load and evaluate.")
    parser.add_argument("--data_dir", type=str, help="Path to images directory (for COCO/ImageNet/Places).")
    parser.add_argument("--instances_json", type=str, default="/home/pmonserrat/datasets/coco/annotations/instances_val2017.json", help="Path to instances json (for COCO).")
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
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Context Files
    parser.add_argument("--nature_definition_path", type=str, default="/home/pmonserrat/code/data/big5_taxonomy/big5_nature_definition.txt")
    parser.add_argument("--biotic_definition_path", type=str, default="/home/pmonserrat/code/data/big5_taxonomy/big5_biotic_definition.txt")
    parser.add_argument("--material_definition_path", type=str, default="/home/pmonserrat/code/data/big5_taxonomy/big5_material_definition.txt")

    parser.add_argument("--output_file", type=str, default="taxonomy_calibration_results.json",
                        help="Results store JSON, keyed by dataset then model name (updated in "
                             "place — a rerun of the same model overwrites its entry).")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional subfolder name to write --output_file (and its "
                             "_predictions.csv) into, e.g. --run_name ablation_single_pass. "
                             "Useful for keeping results from different configurations in "
                             "separate, clearly labeled folders. Created if it doesn't exist. "
                             "Default: write into the current directory.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluations.")
    parser.add_argument("--num_preds_to_store", type=int, default=None,
                        help="Number of images whose per-instance predictions get written to the "
                             "_predictions.csv file. Images are chosen deterministically (sorted "
                             "by image_path), so the SAME fixed set of images is stored across "
                             "different models/runs on the same dataset, keeping CSVs comparable. "
                             "Default: store all scored instances.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--wandb", action="store_true", help="Store the results on WandB.")

    return parser.parse_args()


def load_system_prompt(nature_def_path, biotic_def_path, material_def_path):
    """Read the three plain-text axis-definition files and concatenate them
    into ONE system prompt. This is sent as context on EVERY VLM call in this
    script, so the model always has the exact rules for nature/biotic/material
    available when making its judgment — identical to what the main pipeline
    sends during its own fallback labeling calls."""
    nature_def = Path(nature_def_path).read_text()
    biotic_def = Path(biotic_def_path).read_text()
    material_def = Path(material_def_path).read_text()
    return f"{nature_def}\n\n{biotic_def}\n\n{material_def}"


def _label_to_bool(value, axis):
    """Standardizes string answers into boolean logic depending on the axis.
    Mirrors src/vlm_pipeline.py's label_to_bool exactly (kept as a separate
    local copy here rather than importing it, since this script predates that
    refactor — behavior is identical)."""
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
    """Standard binary classification metrics (accuracy/precision/recall/F1)
    via scikit-learn, for one taxonomy axis at a time. Returns all-zero stats
    if there's nothing to score (empty input) rather than letting sklearn
    raise an error on empty arrays."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if not y_true:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "support": 0}

    acc = accuracy_score(y_true, y_pred)
    # average="binary" tells sklearn these are simple True/False labels (not
    # multi-class); zero_division=0 avoids a warning/crash if, say, the model
    # never predicted the positive class at all in this batch.
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "support": len(y_true)}


def main():
    args = parse_args()
    if args.output_mode == "free_form":
        # This calibration is only meaningful with STRUCTURED output — a
        # free-form answer wouldn't give us clean yes/no/biotic/etc. values
        # to score against ground truth.
        raise ValueError("This evaluation cannot be run in free_form mode.")

    model_label = f"{args.model_family}-{args.model_name}".replace("/", "_")
    print(f"🚀 Starting Contextualized VLM Taxonomy Calibration ({model_label}) on dataset '{args.dataset}'")

    if args.wandb:
        # Weights & Biases (wandb) is an experiment-tracking service — this
        # just registers a new "run" so all the metrics logged later end up
        # visible on the project's dashboard, tagged with this run's config.
        wandb.init(
            entity="paumonserrat03-universitat-aut-noma-de-barcelona",
            project="TFM_VLM",
            config=vars(args),
            name=f"taxonomy_image_calibration_{args.dataset}_{model_label}",
        )

    if args.verbose:
        print(f"[INFO] Loading taxonomy graph from {args.excel_path}...")
    graph = TaxonomyGraph()
    # --sheet_name can be given either as a sheet NAME (string) or a sheet
    # INDEX (a number, but still typed as a string from argparse) — this line
    # detects which one was passed and converts to int only if needed.
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
    # `dataset` is a list of IMAGES, each with a list of `targets` (an image
    # can have multiple labeled objects, e.g. COCO). For THIS calibration
    # check we want one independent evaluation row PER (image, target)
    # pair — so we "flatten" that nested structure into a single flat list,
    # keeping only targets that actually have a nature label (i.e. the ones
    # that successfully mapped to the taxonomy — see get_gt_from_graph).
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
        # Fixed random seed (42) so re-running with the same --max_samples
        # always picks the SAME random subset — makes results reproducible
        # and comparable across different models/runs.
        random.seed(42)
        eval_instances = random.sample(eval_instances, min(args.max_samples, len(eval_instances)))

    if not eval_instances:
        print("No mapped evaluation instances found — exiting.")
        sys.exit(1)

    system_prompt = load_system_prompt(
        args.nature_definition_path, args.biotic_definition_path, args.material_definition_path
    )

    if args.verbose:
        print(f"[INFO] Creating VLM: family='{args.model_family}', model='{args.model_name}'...")

    # Different VLM backends need different constructor arguments: vLLM-served
    # models (qwen/mistral/llava) take vLLM-specific settings like GPU memory
    # fraction and max sequence length, while the HuggingFace-served BLIP
    # family just needs a device string.
    VLLM_FAMILIES = ("qwen", "mistral", "llava")
    if args.model_family in VLLM_FAMILIES:
        vlm_kwargs = {
            "dtype": args.dtype, "gpu_memory_utilization": args.gpu_memory_utilization, "trust_remote_code": args.trust_remote_code
        }
        if args.max_model_len is not None: vlm_kwargs["max_model_len"] = args.max_model_len
    else:
        vlm_kwargs = {"device": args.device, "dtype": args.dtype}

    vlm = create_vlm(args.model_family, args.model_name, **vlm_kwargs)

    print(f"Running contextualized classification over {len(eval_instances)} target objects...")

    scored_results = []
    # Ceiling division: how many batches of size batch_size are needed to
    # cover every evaluation instance (the last batch may be smaller).
    num_batches = (len(eval_instances) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        batch = eval_instances[start:start + args.batch_size]

        # Build one taxonomy-classification prompt per instance in this batch
        # (using each instance's own GT class_name), paired with its own image.
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
            # If the ENTIRE batch call fails (e.g. an out-of-memory error),
            # don't crash the whole run — treat every instance in this batch
            # as a parse failure (scored as wrong below) and keep going.
            print(f"⚠️ Batch {batch_idx + 1}/{num_batches} FAILED ({e!r}).")
            batch_results = [None] * len(batch)

        for r, result in zip(batch, batch_results):
            if result is None:
                # This instance's generation failed entirely — per project
                # convention, a prediction failure is PENALIZED AS WRONG
                # (never silently dropped or treated as a "no nature" default).
                # We do this by scoring it as the OPPOSITE of the ground
                # truth, which guarantees it counts as an error in the metrics
                # below rather than accidentally counting as correct.
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

                # Even when the JSON parsed successfully, an individual field
                # might still be an unexpected/missing value (pred_* is None).
                # Same "penalize as wrong" rule applies at the per-field level:
                # fall back to the OPPOSITE of ground truth rather than
                # dropping this instance from the metric entirely.
                r["scored_pred_nature"] = pred_nature if pred_nature is not None else (not r["gt_nature"])
                r["scored_pred_biotic"] = pred_biotic if pred_biotic is not None else (not r["gt_biotic"] if r["gt_biotic"] is not None else None)
                r["scored_pred_material"] = pred_material if pred_material is not None else (not r["gt_material"] if r["gt_material"] is not None else None)

            scored_results.append(r)

        if args.verbose:
            print(f"[INFO] Batch {batch_idx + 1}/{num_batches} done in {time.time() - t0:.1f}s.")

    # Calculate Metrics
    parse_failure_rate = sum(r["prediction"]["parse_failed"] for r in scored_results) / len(scored_results)

    # Nature is scored across EVERY instance (every instance has a gt_nature
    # value, by construction of eval_instances above).
    nature_metrics = calculate_binary_metrics(
        [r["gt_nature"] for r in scored_results], [r["scored_pred_nature"] for r in scored_results]
    )

    # Biotic/material are only meaningful for instances that ARE nature (a
    # non-nature instance has gt_biotic/gt_material = None) — filter down to
    # just the relevant subset before scoring each axis.
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
    # Build one flat, CSV-friendly row per evaluated instance, combining its
    # ground truth, its raw VLM prediction, and the model's reasoning text —
    # useful for manually spot-checking individual right/wrong answers later.
    # Fixed at --num_preds_to_store images, chosen deterministically by
    # sorting on image_path (not on eval order, which can vary run to run due
    # to the random.sample above) so the SAME set of images is stored for
    # every model/dataset run, keeping the CSVs directly comparable.
    if args.num_preds_to_store is not None:
        chosen_paths = sorted({r["image_path"] for r in scored_results})[: args.num_preds_to_store]
        preds_to_store = set(chosen_paths)
    else:
        preds_to_store = {r["image_path"] for r in scored_results}

    flat_rows = []
    for r in scored_results:
        if r["image_path"] not in preds_to_store:
            continue
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
    if args.run_name:
        # Route both the results JSON and the predictions CSV into a
        # user-named subfolder (e.g. one per ablation configuration), so
        # results from different configurations never land in the same place.
        output_path = Path(args.run_name) / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
    update_results_store(output_path, dataset=args.dataset, model=model_label, metrics=summary_results)
    # Include dataset + model in the filename — otherwise every model run
    # writes to the same "<stem>_predictions.csv" and each rerun (e.g. a
    # different VLM on the same dataset) silently overwrites the previous
    # model's predictions.
    if flat_rows:
        csv_path = output_path.with_name(f"{output_path.stem}_{args.dataset}_{model_label}_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"💾 wrote {csv_path} ({len(preds_to_store)} images stored)")

    if args.wandb:
        wandb.save(str(output_path))
        wandb.finish()


if __name__ == "__main__":
    main()
