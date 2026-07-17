#!/usr/bin/env python3
"""
run_vlm_pipeline.py

End-to-end driver for the baseline BIG-5 VLM pipeline, in three stages:

  --stage all   : THE STANDARD WAY TO RUN THIS PIPELINE. Loads the VLM, runs
                  caption -> extraction -> per-object labeling over every image,
                  and writes a JSON-lines artifact of raw responses to
                  --responses_file. The VLM is then explicitly unloaded
                  (src.models.vlm_models.unload_vlm — tears down vLLM's
                  distributed process group / KV-cache, or the HF model's
                  tensors, then empties the CUDA cache) BEFORE open_clip + the
                  TaxonomyGraph are loaded to score the stored artifact. One
                  command, one process, the VLM and CLIP never hold GPU memory
                  at the same time.

  --stage infer : run ONLY the inference half (VLM -> --responses_file) and
                  exit. Useful to run inference on one machine/job and defer
                  scoring to another, or to inspect the raw artifact first.

  --stage score : run ONLY the scoring half, reading a --responses_file written
                  by a previous --stage infer (or --stage all) run. Useful to
                  re-score an existing artifact (e.g. after a metrics-code
                  change) without re-running inference.

Metrics (per CLAUDE.md scoping):
  - accuracy / precision / recall / F1   : ALL datasets. Nature is image-level
    (nature=1 if ANY extracted object is nature). Biotic/material are scored on
    the GT object when it is matched among the extractions (extraction-hit
    subset).
  - F-CLIPScore + Object-CLIPScore       : ALL datasets (reference-free).
  - ClipMatch + hP/hR/hF1                : ImageNet + Places ONLY (fixed vocab).
  - Diagnostics: extraction-hit rate, WordNet-mapping vs VLM-fallback rate,
    objects/image, parse-failure rate.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from src.evaluation import taxonomy_metrics
from src.loaders.excel_loader import TaxonomyGraph
from src.loaders.dataset_loader import load_dataset, get_candidate_vocab, build_mapping_vocab
from src.models.vlm_models import MODEL_REGISTRY, VLLM_FAMILIES, create_vlm, unload_vlm
from src.vlm_pipeline import run_inference, resolve_hybrid_label, normalize_objects, _normalize_object
from src.evaluation import clip_metrics


# =============================================================================
# System prompts (built from the ../data/big5_taxonomy/ definition files)
# =============================================================================
def build_system_prompts(nature_path, biotic_path, material_path):
    """Caption stage sees the NATURE definition only (no axis-priming, per the
    recap). Labeling stage sees all three axis definitions — identical to
    evaluate_taxonomy_labeling.py's system prompt, so the fallback matches the
    calibration eval."""
    nature = Path(nature_path).read_text()
    biotic = Path(biotic_path).read_text()
    material = Path(material_path).read_text()
    caption_system = nature
    label_system = f"{nature}\n\n{biotic}\n\n{material}"
    return caption_system, label_system


# =============================================================================
# GT / extraction matching (pure helpers — unit-testable without models)
# =============================================================================
from functools import lru_cache


@lru_cache(maxsize=None)
def _synset_lemma_terms(synset_id):
    """Normalized WordNet lemma surface forms for a synset (cached — the same
    class recurs across thousands of images, so recomputing per image is waste)."""
    from nltk.corpus import wordnet as wn
    try:
        return frozenset(_normalize_object(l.name().replace("_", " ")) for l in wn.synset(synset_id).lemmas())
    except Exception:
        return frozenset()


def gt_match_terms(target):
    """Normalized surface forms that count as 'the GT object was extracted':
    the class name plus every WordNet lemma of the GT synset (recap §8d — GT
    class OR any WordNet-synset synonym)."""
    terms = set()
    cn = target.get("class_name")
    if cn:
        terms.add(_normalize_object(cn))
    syn = target.get("synset_id")
    if syn:
        terms |= _synset_lemma_terms(syn)
    return {t for t in terms if t}


def find_matching_object(objects, target):
    """Return the index of the first extracted object matching the GT target
    (by normalized class-name / synonym, full phrase or trailing head noun), or
    None. Used for extraction-hit and matched-object axis scoring."""
    terms = gt_match_terms(target)
    if not terms:
        return None
    for i, obj in enumerate(objects):
        norm = _normalize_object(obj)
        if norm in terms:
            return i
        if " " in norm and norm.split()[-1] in terms:
            return i
    return None


def image_gt_nature(targets):
    """Image-level GT nature: True if ANY target is nature, False if all targets
    are explicitly non-nature, None if no target carries a nature label."""
    vals = [t.get("gt_nature") for t in targets if t.get("gt_nature") is not None]
    if not vals:
        return None
    return any(vals)


def image_pred_nature(object_final_labels):
    """Image-level predicted nature: True if ANY extracted object is labeled
    nature. An image with no objects predicts False (no nature found)."""
    return any(bool(o["final_nature"]) for o in object_final_labels)


# =============================================================================
# Metric aggregation
# =============================================================================
def _binary_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "support": len(y_true)}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


# =============================================================================
# PHASE 1 — inference
# =============================================================================
def phase_infer(args):
    print(f"🚀 [infer] dataset='{args.dataset}', model='{args.model_family}/{args.model_name}'")

    graph = TaxonomyGraph()
    sheet = args.sheet_name if not str(args.sheet_name).isdigit() else int(args.sheet_name)
    graph.load_excel(args.excel_path, sheet_name=sheet)

    dataset = load_dataset(
        args.dataset, taxonomy_graph=graph,
        data_dir=args.data_dir, instances_json=args.instances_json,
        places_categories_txt=args.places_categories_txt, excel_path=args.excel_path,
        en_gt=args.twitter_en_gt_csv, es_gt=args.twitter_es_gt_csv,
        en_media=args.twitter_en_media_csv, es_media=args.twitter_es_media_csv,
        cache_dir=args.images_cache_dir,
    )
    if not dataset:
        print("No dataset instances loaded — exiting."); sys.exit(1)

    if args.max_samples is not None:
        random.seed(42)
        dataset = random.sample(dataset, min(args.max_samples, len(dataset)))

    # Authoritative vocabularies (Phase 1 has the data access) — stored in the
    # artifact header so Phase 2 needs no dataset files.
    vocab_kwargs = dict(data_dir=args.data_dir, places_categories_txt=args.places_categories_txt,
                        excel_path=args.excel_path)
    mapping_vocab = build_mapping_vocab(args.dataset, **vocab_kwargs)
    candidate_vocab = get_candidate_vocab(args.dataset, **vocab_kwargs)  # None for coco/big5

    caption_system, label_system = build_system_prompts(
        args.nature_definition_path, args.biotic_definition_path, args.material_definition_path)

    if args.model_family in VLLM_FAMILIES:
        vlm_kwargs = {"dtype": args.dtype, "gpu_memory_utilization": args.gpu_memory_utilization,
                      "trust_remote_code": args.trust_remote_code}
        if args.max_model_len is not None:
            vlm_kwargs["max_model_len"] = args.max_model_len
    else:
        vlm_kwargs = {"device": args.device, "dtype": args.dtype}
    vlm = create_vlm(args.model_family, args.model_name, **vlm_kwargs)

    header = {
        "record_type": "header",
        "dataset": args.dataset,
        "model": f"{args.model_family}/{args.model_name}",
        "mapping_vocab": mapping_vocab,
        "candidate_vocab": candidate_vocab,
    }

    out_path = Path(args.responses_file)
    t0 = time.time()
    n = 0
    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for rec in run_inference(
            vlm, dataset,
            caption_system_prompt=caption_system, label_system_prompt=label_system,
            batch_size=args.batch_size,
            caption_max_new_tokens=args.max_new_tokens_caption,
            label_max_new_tokens=args.max_new_tokens_label,
            temperature=args.temperature, verbose=args.verbose,
        ):
            rec["record_type"] = "image"
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"💾 [infer] wrote {n} image records to {out_path} in {time.time()-t0:.1f}s")
    return vlm


# =============================================================================
# PHASE 2 — scoring
# =============================================================================
def _read_artifact(path):
    header = None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "header":
                header = obj
            else:
                records.append(obj)
    if header is None:
        raise ValueError(f"Artifact {path} has no header line.")
    return header, records


def phase_score(args):
    print(f"📊 [score] reading {args.responses_file}")
    header, records = _read_artifact(args.responses_file)
    dataset = header["dataset"]
    mapping_vocab = header.get("mapping_vocab") or {}
    candidate_vocab = header.get("candidate_vocab")
    run_clipmatch = dataset in clip_metrics.CLIPMATCH_DATASETS and candidate_vocab
    if args.max_samples is not None:
        records = records[: args.max_samples]

    graph = TaxonomyGraph()
    sheet = args.sheet_name if not str(args.sheet_name).isdigit() else int(args.sheet_name)
    graph.load_excel(args.excel_path, sheet_name=sheet)

    scorer = clip_metrics.CLIPScorer(model_name=args.clip_model, pretrained=args.clip_pretrained,
                                     device=args.device, batch_size=args.clip_batch_size)

    # ---- Resolve hybrid labels for every object, up front (needs the graph) ----
    for rec in records:
        finals = []
        for obj, lab in zip(rec["objects"], rec["object_labels"]):
            finals.append(resolve_hybrid_label(obj, lab, graph, mapping_vocab))
        rec["object_finals"] = finals

    # ---- CLIP encodings (batched across ALL images for efficiency) ----
    image_paths = [r["image_path"] for r in records]
    captions = [r["caption"] for r in records]
    image_embs = scorer.encode_images(image_paths)
    caption_embs = scorer.encode_text(captions, warn_truncation=True)

    # Flatten object texts with per-image offsets, encode once.
    flat_texts, offsets = [], []
    for r in records:
        offsets.append(len(flat_texts))
        flat_texts.extend(clip_metrics.object_texts(r["objects"]))
    offsets.append(len(flat_texts))
    obj_embs_all = scorer.encode_text(flat_texts) if flat_texts else None

    candidate_embs = None
    if run_clipmatch:
        candidate_embs = scorer.encode_text(
            [clip_metrics.OBJECT_TEMPLATE.format(c["class_name"]) for c in candidate_vocab])

    # ---- Per-image metric accumulation ----
    nat_true, nat_pred = [], []
    bio_true, bio_pred, mat_true, mat_pred = [], [], [], []
    fclip_vals, objclip_vals = [], []
    hp_vals, hr_vals, hf1_vals = [], [], []
    n_gt_targets = n_extraction_hits = 0
    n_objects_total = n_parse_fail = n_object_records = 0
    n_map_nature = n_vlm_nature = 0
    clipmatch_top1 = 0
    clipmatch_support = 0
    flat_rows = []

    for idx, rec in enumerate(records):
        objs = rec["objects"]
        finals = rec["object_finals"]
        targets = rec.get("targets", [])
        obj_slice = slice(offsets[idx], offsets[idx + 1])
        rec_obj_embs = obj_embs_all[obj_slice] if obj_embs_all is not None else None

        # diagnostics
        n_objects_total += len(objs)
        for lab, fin in zip(rec["object_labels"], finals):
            n_object_records += 1
            if lab.get("parse_failed"):
                n_parse_fail += 1
            if fin["nature_source"] == "wordnet":
                n_map_nature += 1
            else:
                n_vlm_nature += 1

        # --- reference-free CLIP metrics (all datasets) ---
        if rec_obj_embs is not None:
            fclip_vals.append(clip_metrics.f_clipscore(image_embs[idx], caption_embs[idx], rec_obj_embs))
            objclip_vals.append(clip_metrics.object_clipscore(image_embs[idx], rec_obj_embs))

        # --- image-level nature ---
        g_nat = image_gt_nature(targets)
        if g_nat is not None:
            nat_true.append(bool(g_nat))
            nat_pred.append(bool(image_pred_nature(finals)))

        # --- matched-object biotic/material + extraction hit ---
        for t in targets:
            if t.get("gt_nature") is None:
                continue
            n_gt_targets += 1
            mi = find_matching_object(objs, t)
            if mi is None:
                continue
            n_extraction_hits += 1
            fin = finals[mi]
            # A present GT with a failed/absent prediction (final_* is None) is
            # PENALIZED AS WRONG, not dropped (CLAUDE.md: "Prediction-unmapped
            # instances: penalized as wrong (never defaulted)"). We encode a
            # failed prediction as the negation of the GT so it always counts as
            # an error rather than silently inflating the metric.
            if t.get("gt_biotic") is not None:
                gt_b = bool(t["gt_biotic"])
                pred_b = fin["final_biotic"]
                bio_true.append(gt_b)
                bio_pred.append(bool(pred_b) if pred_b is not None else (not gt_b))
            if t.get("gt_material") is not None:
                gt_m = bool(t["gt_material"])
                pred_m = fin["final_material"]
                mat_true.append(gt_m)
                mat_pred.append(bool(pred_m) if pred_m is not None else (not gt_m))

        # --- ClipMatch + hP/hR (imagenet/places, single-label) ---
        # An image with a GT synset ALWAYS counts toward ClipMatch/hP support.
        # If the model extracted no objects (or none map), it cannot predict a
        # class — that is a top-1 miss and hP/hR = 0, NOT an excluded sample
        # (CLAUDE.md: prediction-unmapped penalized as wrong).
        pred_class_synset = pred_node = None
        gt_syn = targets[0].get("synset_id") if (run_clipmatch and targets) else None
        if gt_syn:
            clipmatch_support += 1
            if rec_obj_embs is not None and rec_obj_embs.shape[0] > 0:
                _, pred_idx, per_obj_sim = clip_metrics.clipmatch(rec_obj_embs, candidate_embs)
                if pred_idx >= 0:
                    pred_class_synset = candidate_vocab[pred_idx]["synset_id"]
                    if pred_class_synset == gt_syn:
                        clipmatch_top1 += 1
                    pred_node = taxonomy_metrics.resolve_to_wordnet(
                        list(per_obj_sim), pred_class_synset, objs)
            hier = taxonomy_metrics.compute_hierarchical_metrics(graph, gt_syn, pred_node)
            hp_vals.append(hier["hp"]); hr_vals.append(hier["hr"]); hf1_vals.append(hier["hf1"])

        # per-object CSV rows
        for obj, lab, fin in zip(objs, rec["object_labels"], finals):
            flat_rows.append({
                "image_path": rec["image_path"], "caption": rec["caption"], "object": obj,
                "mapped": fin["mapped"], "mapped_synset": fin["mapped_synset"],
                "final_nature": fin["final_nature"], "final_biotic": fin["final_biotic"],
                "final_material": fin["final_material"],
                "nature_source": fin["nature_source"], "biotic_source": fin["biotic_source"],
                "vlm_nature": lab.get("nature"), "vlm_biotic": lab.get("biotic"),
                "vlm_material": lab.get("material"), "parse_failed": lab.get("parse_failed"),
            })

    # ---- Assemble summary ----
    n_images = len(records)
    summary = {
        "dataset": dataset, "model": header.get("model"), "n_images": n_images,
        "diagnostics": {
            "objects_per_image": (n_objects_total / n_images) if n_images else 0.0,
            "parse_failure_rate": (n_parse_fail / n_object_records) if n_object_records else 0.0,
            "extraction_hit_rate": (n_extraction_hits / n_gt_targets) if n_gt_targets else 0.0,
            "wordnet_mapping_rate": (n_map_nature / n_object_records) if n_object_records else 0.0,
            "vlm_fallback_rate": (n_vlm_nature / n_object_records) if n_object_records else 0.0,
        },
        "reference_free": {
            "f_clipscore": _mean(fclip_vals),
            "object_clipscore": _mean(objclip_vals),
        },
        "nature": _binary_metrics(nat_true, nat_pred),
        "biotic_matched": _binary_metrics(bio_true, bio_pred),
        "material_matched": _binary_metrics(mat_true, mat_pred),
        "material_caveat": ("Material GT for imagenet/coco/places is the heuristic "
                            "gt_material=True default (real photos); only BIG-5 has genuine "
                            "material GT."),
    }
    if run_clipmatch:
        summary["clipmatch"] = {
            "top1_accuracy": (clipmatch_top1 / clipmatch_support) if clipmatch_support else 0.0,
            "support": clipmatch_support,
        }
        summary["hierarchical"] = {"hp": _mean(hp_vals), "hr": _mean(hr_vals),
                                   "hf1": _mean(hf1_vals), "support": len(hf1_vals)}

    _print_summary(summary, run_clipmatch)

    out_path = Path(args.output_file)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    if flat_rows:
        csv_path = out_path.with_name(out_path.stem + "_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader(); w.writerows(flat_rows)
        print(f"💾 [score] wrote {out_path} and {csv_path}")

    if args.wandb:
        _log_wandb(args, summary, run_clipmatch)


def _print_summary(s, run_clipmatch):
    d = s["diagnostics"]
    print("\n" + "=" * 60)
    print(f"📊 VLM PIPELINE: {s['model']} on {s['dataset'].upper()}  ({s['n_images']} images)")
    print("=" * 60)
    print(f"Objects/image: {d['objects_per_image']:.2f} | Parse-fail: {d['parse_failure_rate']:.1%} "
          f"| Extraction-hit: {d['extraction_hit_rate']:.1%}")
    print(f"WordNet-mapping: {d['wordnet_mapping_rate']:.1%} | VLM-fallback: {d['vlm_fallback_rate']:.1%}")
    print(f"F-CLIPScore: {s['reference_free']['f_clipscore']:.4f} | "
          f"Object-CLIPScore: {s['reference_free']['object_clipscore']:.4f}")
    for axis in ("nature", "biotic_matched", "material_matched"):
        m = s[axis]
        print(f"\n--- {axis} (support {m['support']}) ---")
        print(f"Acc {m['accuracy']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f} | F1 {m['f1']:.4f}")
    if run_clipmatch:
        print(f"\n--- ClipMatch (support {s['clipmatch']['support']}) ---")
        print(f"Top-1: {s['clipmatch']['top1_accuracy']:.4f}")
        h = s["hierarchical"]
        print(f"--- Hierarchical (support {h['support']}) ---")
        print(f"hP {h['hp']:.4f} | hR {h['hr']:.4f} | hF1 {h['hf1']:.4f}")
    print("=" * 60)


def _log_wandb(args, summary, run_clipmatch):
    import wandb
    wandb.init(entity="paumonserrat03-universitat-aut-noma-de-barcelona", project="TFM_VLM",
               config=vars(args), name=f"vlm_pipeline_{summary['dataset']}_{summary['model']}".replace("/", "_"))
    log = {
        "ObjectsPerImage": summary["diagnostics"]["objects_per_image"],
        "ExtractionHitRate": summary["diagnostics"]["extraction_hit_rate"],
        "WordNetMappingRate": summary["diagnostics"]["wordnet_mapping_rate"],
        "F-CLIPScore": summary["reference_free"]["f_clipscore"],
        "Object-CLIPScore": summary["reference_free"]["object_clipscore"],
        "Nature/F1": summary["nature"]["f1"], "Nature/Accuracy": summary["nature"]["accuracy"],
        "Biotic/F1": summary["biotic_matched"]["f1"], "Material/F1": summary["material_matched"]["f1"],
    }
    if run_clipmatch:
        log["ClipMatch/Top1"] = summary["clipmatch"]["top1_accuracy"]
        log["Hierarchical/hF1"] = summary["hierarchical"]["hf1"]
    wandb.log(log)
    wandb.finish()


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Two-phase baseline VLM pipeline (infer -> score).")
    p.add_argument("--stage", choices=["all", "infer", "score"], default="all",
                   help="'all' (default) runs the full end-to-end pipeline in one process, "
                        "releasing the VLM's GPU memory before loading CLIP for scoring. "
                        "'infer'/'score' split the two halves across separate invocations.")
    p.add_argument("--dataset", choices=["coco", "imagenet", "places365", "big5"], required=True)
    p.add_argument("--responses_file", type=str, default="vlm_responses.jsonl",
                   help="Intermediate artifact: written by infer, read by score.")

    # taxonomy / context
    p.add_argument("--excel_path", type=str, default="../data/big5_taxonomy/flat_wordnet_tree_fixed.xlsx")
    p.add_argument("--sheet_name", type=str, default="data corrected")
    p.add_argument("--nature_definition_path", type=str, default="../data/big5_taxonomy/big5_nature_definition.txt")
    p.add_argument("--biotic_definition_path", type=str, default="../data/big5_taxonomy/big5_biotic_definition.txt")
    p.add_argument("--material_definition_path", type=str, default="../data/big5_taxonomy/big5_material_definition.txt")

    # dataset paths
    p.add_argument("--data_dir", type=str)
    p.add_argument("--instances_json", type=str)
    p.add_argument("--places_categories_txt", type=str)
    p.add_argument("--twitter_en_gt_csv", type=str, default=None)
    p.add_argument("--twitter_es_gt_csv", type=str, default=None)
    p.add_argument("--twitter_en_media_csv", type=str, default=None)
    p.add_argument("--twitter_es_media_csv", type=str, default=None)
    p.add_argument("--images_cache_dir", type=str, default="./big_5_cache")

    # VLM (infer)
    p.add_argument("--model_family", type=str, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--model_name", type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens_caption", type=int, default=256)
    p.add_argument("--max_new_tokens_label", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)

    # CLIP (score)
    p.add_argument("--clip_model", type=str, default="ViT-L-14")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--clip_batch_size", type=int, default=64)

    # shared
    p.add_argument("--output_file", type=str, default="vlm_pipeline_results.json")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")

    args = p.parse_args()
    if args.stage in ("infer", "all"):
        if not args.model_family or not args.model_name:
            p.error("--model_family and --model_name are required for the infer stage.")
    return args


def main():
    args = parse_args()

    vlm = None
    if args.stage in ("infer", "all"):
        vlm = phase_infer(args)

    if args.stage == "all":
        # Free the VLM's GPU memory BEFORE loading CLIP for scoring — neither
        # vLLM nor plain torch/transformers release CUDA memory on Python GC
        # alone, so without this explicit step the CLIP load below would
        # contend with (or OOM against) the still-resident VLM.
        print("🧹 [all] releasing VLM GPU memory before loading CLIP for scoring...")
        unload_vlm(vlm)
        del vlm

    if args.stage in ("score", "all"):
        phase_score(args)


if __name__ == "__main__":
    main()
