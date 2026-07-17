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

HOW TO READ THIS FILE
This is the biggest, most "top-level" file in the pipeline — it's the actual
script you run from the command line. It doesn't implement any of the deep
logic itself (that all lives in src/vlm_pipeline.py, src/evaluation/*, etc.);
instead it WIRES everything together: parses command-line flags, loads the
right dataset, calls the VLM pipeline, calls the metric functions, and prints/
saves the results. If you're trying to understand "what actually happens when
I run this script", start reading at `main()` near the bottom and work
backward through phase_infer/phase_score.
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
from src.utils import update_results_store


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
    # The caption call only needs to know what "nature" looks like in general
    # (so it can describe the scene without being biased toward a specific
    # axis); the per-object labeling call needs ALL THREE definitions since it
    # has to answer nature/biotic/material questions directly.
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
    class recurs across thousands of images, so recomputing per image is waste).

    A "lemma" here means a specific word/phrase that WordNet lists as a valid
    name for a synset — e.g. the synset "dog.n.01" has lemmas like "dog" and
    "domestic dog" and "Canis familiaris". We collect ALL of them (normalized:
    lowercased, articles stripped) so that later, when checking whether a VLM
    extracted "the GT object", any of these equivalent phrasings counts as a
    match, not just the exact class_name string.

    `@lru_cache` means: the first time this function is called with a
    particular `synset_id`, it does the real work and REMEMBERS the result;
    every subsequent call with that same synset_id just returns the
    remembered answer instantly instead of recomputing it. This matters a lot
    here because the SAME class (e.g. "dog.n.01") appears across potentially
    thousands of different images in a dataset like ImageNet.
    """
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
        terms |= _synset_lemma_terms(syn)  # `|=` merges the lemma set into `terms`
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
        # Also accept a match on just the LAST word of a multi-word extracted
        # phrase, e.g. extracted "brown golden retriever" matching GT term
        # "golden retriever"... actually the reverse: extracted "big dog"
        # matching GT term "dog" via its trailing word.
        if " " in norm and norm.split()[-1] in terms:
            return i
    return None


def image_gt_nature(targets):
    """Image-level GT nature: True if ANY target is nature, False if all targets
    are explicitly non-nature, None if no target carries a nature label."""
    vals = [t.get("gt_nature") for t in targets if t.get("gt_nature") is not None]
    if not vals:
        # No target on this image has a usable nature label at all (shouldn't
        # normally happen, since loaders only keep mapped targets, but this is
        # a defensive fallback).
        return None
    return any(vals)


def image_pred_nature(object_final_labels):
    """Image-level predicted nature: True if ANY extracted object is labeled
    nature. An image with no objects predicts False (no nature found)."""
    # `any(...)` on an empty list correctly returns False in Python, so an
    # image with zero extracted objects naturally predicts "no nature found"
    # without needing a special-case check here.
    return any(bool(o["final_nature"]) for o in object_final_labels)


# =============================================================================
# Metric aggregation
# =============================================================================
def _binary_metrics(y_true, y_pred):
    """Standard accuracy/precision/recall/F1 for one taxonomy axis, given
    matching lists of true and predicted booleans."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "support": len(y_true)}


def _mean(vals):
    """Average of a list, treating None entries as "not applicable" (skipped
    rather than counted as zero). Returns 0.0 for an empty/all-None list."""
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


# =============================================================================
# PHASE 1 — inference
# =============================================================================
def phase_infer(args):
    """
    Runs the ENTIRE inference half of the pipeline: load the dataset and the
    taxonomy graph, create the VLM, run caption -> extraction -> labeling over
    every image (src.vlm_pipeline.run_inference), and stream the results out to
    --responses_file as they're produced (rather than holding them all in
    memory). Returns the created `vlm` object so --stage all can explicitly
    release its GPU memory afterward (see unload_vlm below / main()).
    """
    print(f"🚀 [infer] dataset='{args.dataset}', model='{args.model_family}/{args.model_name}'")

    graph = TaxonomyGraph()
    # --sheet_name may be a sheet NAME (string) or a numeric INDEX (still a
    # string coming from argparse) — detect which and convert accordingly.
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
        # Fixed seed so re-running with the same --max_samples always samples
        # the SAME subset — keeps results comparable across runs/models.
        random.seed(42)
        dataset = random.sample(dataset, min(args.max_samples, len(dataset)))

    # Authoritative vocabularies (Phase 1 has the data access) — stored in the
    # artifact header so Phase 2 needs no dataset files.
    # These two vocabularies need the SAME dataset-specific file paths (e.g.
    # ImageNet's data_dir, Places' categories file) that we just used to load
    # the dataset itself — we compute them here (while we still have those
    # paths handy) and save them straight into the output artifact's header,
    # so Phase 2 (scoring) never needs to re-open any dataset files at all.
    vocab_kwargs = dict(data_dir=args.data_dir, places_categories_txt=args.places_categories_txt,
                        excel_path=args.excel_path)
    mapping_vocab = build_mapping_vocab(args.dataset, **vocab_kwargs)
    candidate_vocab = get_candidate_vocab(args.dataset, **vocab_kwargs)  # None for coco/big5

    caption_system, label_system = build_system_prompts(
        args.nature_definition_path, args.biotic_definition_path, args.material_definition_path)

    # Different VLM backends need different constructor keyword arguments —
    # see src/models/vlm_models.py for the classes behind each family name.
    if args.model_family in VLLM_FAMILIES:
        vlm_kwargs = {"dtype": args.dtype, "gpu_memory_utilization": args.gpu_memory_utilization,
                      "trust_remote_code": args.trust_remote_code}
        if args.max_model_len is not None:
            vlm_kwargs["max_model_len"] = args.max_model_len
    else:
        vlm_kwargs = {"device": args.device, "dtype": args.dtype}
    vlm = create_vlm(args.model_family, args.model_name, **vlm_kwargs)

    # The very first line written to the output file is a special "header"
    # record (distinguished by record_type="header") carrying metadata that
    # applies to the WHOLE run, not any single image — Phase 2 reads this
    # first to know which dataset/model produced the file and to get the
    # vocabularies computed above.
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
    # "JSON Lines" format: one complete, independent JSON object per line of
    # the file (as opposed to one giant JSON array for the whole file). This
    # lets us write results incrementally as they're produced (streaming,
    # rather than building one huge in-memory list and writing it all at the
    # end) and lets Phase 2 read them back one line at a time too.
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
    """Read back a JSON-Lines artifact written by phase_infer: the first
    header-tagged line, plus every per-image record line, as plain Python
    dicts."""
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
    """
    Runs the ENTIRE scoring half of the pipeline: read the artifact written by
    phase_infer, resolve each object's final hybrid label (WordNet + VLM), run
    every CLIP-based metric, aggregate per-axis accuracy/precision/recall/F1,
    print a human-readable summary, and write the results to --output_file
    (JSON summary) plus a per-object CSV.
    """
    print(f"📊 [score] reading {args.responses_file}")
    header, records = _read_artifact(args.responses_file)
    dataset = header["dataset"]
    mapping_vocab = header.get("mapping_vocab") or {}
    candidate_vocab = header.get("candidate_vocab")
    # ClipMatch/hP/hR only make sense for datasets with a fixed candidate
    # class list (ImageNet/Places) — see clip_metrics.CLIPMATCH_DATASETS.
    run_clipmatch = dataset in clip_metrics.CLIPMATCH_DATASETS and candidate_vocab
    if args.max_samples is not None:
        records = records[: args.max_samples]

    graph = TaxonomyGraph()
    sheet = args.sheet_name if not str(args.sheet_name).isdigit() else int(args.sheet_name)
    graph.load_excel(args.excel_path, sheet_name=sheet)

    # This is where CLIP actually gets loaded onto the GPU — by this point in
    # `--stage all`, the VLM has already been unloaded (see main() below), so
    # CLIP has the GPU memory to itself.
    scorer = clip_metrics.CLIPScorer(model_name=args.clip_model, pretrained=args.clip_pretrained,
                                     device=args.device, batch_size=args.clip_batch_size)

    # ---- Resolve hybrid labels for every object, up front (needs the graph) ----
    # `resolve_hybrid_label` combines each object's raw VLM answer with a
    # WordNet lookup (see src/vlm_pipeline.py) to get its FINAL nature/biotic/
    # material labels. We do this once for every object across every image up
    # front, storing the results as `rec["object_finals"]`, so the big scoring
    # loop below can just read them off rather than recomputing per metric.
    for rec in records:
        finals = []
        for obj, lab in zip(rec["objects"], rec["object_labels"]):
            finals.append(resolve_hybrid_label(obj, lab, graph, mapping_vocab))
        rec["object_finals"] = finals

    # ---- CLIP encodings (batched across ALL images for efficiency) ----
    # Rather than encoding one image/caption/object-list at a time inside the
    # loop below, we batch ALL of them together right now — a small number of
    # large batched GPU calls is much faster than thousands of tiny ones.
    image_paths = [r["image_path"] for r in records]
    captions = [r["caption"] for r in records]
    image_embs = scorer.encode_images(image_paths)
    caption_embs = scorer.encode_text(captions, warn_truncation=True)

    # Flatten object texts with per-image offsets, encode once.
    # Every image has a DIFFERENT number of extracted objects, so we can't
    # just make a fixed-size 2D array. Instead we lay every object phrase
    # (across every image) end-to-end into one long flat list, and remember
    # where each image's own objects START in that list (`offsets`) — so
    # `flat_texts[offsets[i]:offsets[i+1]]` gives back exactly image i's
    # object texts, while still letting us encode everything in ONE batched
    # call to the CLIP text encoder.
    flat_texts, offsets = [], []
    for r in records:
        offsets.append(len(flat_texts))
        flat_texts.extend(clip_metrics.object_texts(r["objects"]))
    offsets.append(len(flat_texts))  # sentinel end-offset for the last image
    obj_embs_all = scorer.encode_text(flat_texts) if flat_texts else None

    candidate_embs = None
    if run_clipmatch:
        # Also encode the FIXED candidate-class vocabulary just once (rather
        # than per image) — every image's ClipMatch score is computed against
        # this same set of candidate-class embeddings.
        candidate_embs = scorer.encode_text(
            [clip_metrics.OBJECT_TEMPLATE.format(c["class_name"]) for c in candidate_vocab])

    # ---- Per-image metric accumulation ----
    # These lists/counters accumulate results across every image in the
    # dataset; they get turned into final aggregate numbers (accuracy, mean
    # scores, etc.) after the loop below finishes.
    nat_true, nat_pred = [], []
    bio_true, bio_pred, mat_true, mat_pred = [], [], [], []
    fclip_vals, objclip_vals = [], []
    hp_vals, hr_vals, hf1_vals = [], [], []
    n_gt_targets = n_extraction_hits = 0
    n_objects_total = n_parse_fail = n_object_records = 0
    n_map_nature = n_vlm_nature = 0
    clipmatch_top1 = 0
    clipmatch_support = 0
    flat_rows = []  # per-object rows for the output CSV

    # Which images get their per-object predictions written to the CSV.
    # Fixed at --num_preds_to_store images, chosen deterministically by
    # sorting on image_path (not on inference order, which can vary run to
    # run) so the SAME set of images is stored for every model/dataset run,
    # keeping the CSVs directly comparable across models.
    if args.num_preds_to_store is not None:
        chosen_paths = sorted({r["image_path"] for r in records})[: args.num_preds_to_store]
        preds_to_store = set(chosen_paths)
    else:
        preds_to_store = {r["image_path"] for r in records}

    for idx, rec in enumerate(records):
        objs = rec["objects"]
        finals = rec["object_finals"]
        targets = rec.get("targets", [])
        # Slice this image's own chunk of object embeddings out of the big
        # flat array we built above, using the offsets we remembered.
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
            # Did the model actually EXTRACT (mention) an object matching this
            # ground-truth target at all? If not, we can't score its
            # biotic/material prediction (there IS no prediction to compare).
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
                # ClipMatch picks whichever candidate class has the strongest
                # matching object among everything the model extracted (see
                # clip_metrics.clipmatch for the full algorithm).
                _, pred_idx, per_obj_sim = clip_metrics.clipmatch(rec_obj_embs, candidate_embs)
                if pred_idx >= 0:
                    pred_class_synset = candidate_vocab[pred_idx]["synset_id"]
                    if pred_class_synset == gt_syn:
                        clipmatch_top1 += 1
                    # Turn the predicted CLASS into a specific WordNet synset
                    # NODE by picking whichever extracted object phrase best
                    # represents that prediction (needed for hP/hR below).
                    pred_node = taxonomy_metrics.resolve_to_wordnet(
                        list(per_obj_sim), pred_class_synset, objs)
            # hP/hR compare the GT synset's ancestor chain against the
            # predicted node's ancestor chain — computed even when pred_node
            # is None (a total miss), giving all-zero scores in that case.
            hier = taxonomy_metrics.compute_hierarchical_metrics(graph, gt_syn, pred_node)
            hp_vals.append(hier["hp"]); hr_vals.append(hier["hr"]); hf1_vals.append(hier["hf1"])

        # per-object CSV rows
        # Build one output row per extracted object, capturing both the raw
        # VLM answer and the final resolved label — handy for manually
        # spot-checking individual predictions later.
        if rec["image_path"] in preds_to_store:
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

    # Everything lands under --results_dir ("results/" by default); --run_name
    # further nests it into a per-ablation-configuration subfolder, so results
    # from different pipeline configurations never land in the same place and
    # stay easy to tell apart later.
    out_dir = Path(args.results_dir)
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args.output_file).name
    update_results_store(out_path, dataset=dataset, model=header.get("model"), metrics=summary)
    if flat_rows:
        # Include dataset + model in the filename — otherwise every model run
        # writes to the same "<stem>_predictions.csv" and each rerun (e.g. a
        # different VLM on the same dataset) silently overwrites the previous
        # model's predictions.
        model_slug = header.get("model", "unknown_model").replace("/", "_")
        csv_path = out_path.with_name(f"{out_path.stem}_{dataset}_{model_slug}_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader(); w.writerows(flat_rows)
        print(f"💾 [score] wrote {out_path} and {csv_path} "
              f"({len(preds_to_store)} images stored)")

    if args.wandb:
        _log_wandb(args, summary, run_clipmatch)


def _print_summary(s, run_clipmatch):
    """Pretty-print the final metrics summary to the console."""
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
    """Push the final summary metrics to Weights & Biases for tracking/
    comparison across runs, if --wandb was passed."""
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
    """Define every command-line flag this script accepts. Grouped into:
    stage/dataset selection, taxonomy/context files, per-dataset paths, VLM
    settings (only used by --stage infer/all), CLIP settings (only used by
    --stage score/all), and shared output options."""
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
    p.add_argument("--output_file", type=str, default="vlm_pipeline_results.json",
                   help="Results store JSON, keyed by dataset then model name (updated in place — "
                        "a rerun of the same model overwrites its entry).")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Base directory all results (JSON store + predictions CSV) are written "
                        "under. Created if it doesn't exist.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional subfolder of --results_dir to write --output_file (and its "
                        "_predictions.csv) into, e.g. --run_name ablation_single_pass -> "
                        "results/ablation_single_pass/. Useful for keeping results from "
                        "different pipeline configurations (ablations) in separate, clearly "
                        "labeled folders. Created if it doesn't exist. Default: write directly "
                        "into --results_dir.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_preds_to_store", type=int, default=None,
                   help="Number of images whose per-object predictions get written to the "
                        "_predictions.csv file. The images are chosen deterministically (sorted "
                        "by image_path), so the SAME fixed set of images is stored across "
                        "different models/runs on the same dataset, keeping CSVs comparable. "
                        "Default: store all scored images.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")

    args = p.parse_args()
    if args.stage in ("infer", "all"):
        # These two flags are only conditionally required (argparse can't
        # express "required unless --stage is score" declaratively), so we
        # check manually here and raise the same kind of clean CLI error
        # argparse itself would produce.
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
