#!/usr/bin/env python3
"""
run_vlm_pipeline.py

End-to-end driver for the baseline BIG-5 VLM pipeline, in three stages:

  --stage all   : THE STANDARD WAY TO RUN THIS PIPELINE. Runs inference then
                  scoring, but each in its OWN spawned subprocess (see main()):
                  the infer subprocess loads the VLM, writes the JSON-lines
                  artifact to --responses_file, then EXITS — which makes the OS
                  reclaim 100% of the VLM's VRAM before the score subprocess
                  loads open_clip + the TaxonomyGraph. The VLM and CLIP never
                  hold GPU memory at the same time, and we don't rely on
                  in-process CUDA cleanup (vLLM/torch release it unreliably on
                  GC). 'spawn' is required for CUDA; only the picklable args
                  Namespace crosses the process boundary. --responses_file is
                  PURELY that internal handoff in this mode, so it is deleted
                  once scoring finishes successfully (pass --keep_responses_file
                  to retain it).

  --stage infer : run ONLY the inference half (VLM -> --responses_file) and
                  exit. Useful to run inference on one machine/job and defer
                  scoring to another, or to inspect the raw artifact first.

  --stage score : run ONLY the scoring half, reading a --responses_file written
                  by a previous --stage infer (or --stage all) run. Useful to
                  re-score an existing artifact (e.g. after a metrics-code
                  change) without re-running inference.

Metrics (per CLAUDE.md scoping):
  - accuracy / precision / recall / F1   : ALL datasets, but scored differently
    by dataset type (recap §6):
      * ImageNet/Places (single-label): nature/biotic/material all come from
        the extracted object with the highest CLIP embedding similarity to
        the GT class's own template embedding (embedding-based "exact match",
        not the ClipMatch top-1 class's taxonomy position, and not lexical
        extraction matching) — that object's own hybrid-resolved label is
        used directly for all three axes. material is ALWAYS the VLM's own
        label regardless (CLAUDE.md — never mapped). ClipMatch top-1 + hP/hR
        remain a SEPARATE metric (still the global argmax over the full
        candidate vocabulary — see below).
      * COCO/BIG-5: image-level nature (nature=1 if ANY extracted object is
        nature) + matched-object biotic/material (COCO box-IoU matching is
        future work, gated on the Grounding pipeline — §6.4).
  - F-CLIPScore + Object-CLIPScore       : ALL datasets (reference-free).
  - ClipMatch + hP/hR/hF1                : ImageNet + Places ONLY (fixed vocab).
  - Diagnostics: extraction-hit rate (exact-match, reporting-only — no longer
    gates axis scores), WordNet-mapping vs VLM-fallback rate, objects/image,
    parse-failure rate.

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

import os

# Quiet a couple of low-value third-party STARTUP notices by default — neither
# comes from the pipeline itself. vLLM's own INFO logging (engine config dump,
# torch.compile timings, etc.) and the HF Hub download/load progress bars are
# LEFT ALONE (deliberately not touched here) since they're useful run-progress
# signal on a cluster job. Set here (before torch/vllm/transformers are
# imported, and inherited by the spawned infer/score subprocesses) via
# setdefault, so you can still override either from the shell, e.g.:
#     TRANSFORMERS_VERBOSITY=info python scripts/run_vlm_pipeline.py ...
# To silence the HF-token warning for good (and get faster downloads), export a
# real token instead: `export HF_TOKEN=hf_...`.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # silences the use_fast deprecation etc.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # avoids the fork/parallelism warning

import argparse
import csv
import json
import logging
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

# The HF "You are sending unauthenticated requests to the HF Hub" notice is
# emitted by huggingface_hub at WARNING level; hide it unless a token is truly
# needed. (Setting HF_TOKEN is the real fix and also lifts rate limits.)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from src.evaluation import taxonomy_metrics
from src.loaders.excel_loader import TaxonomyGraph
from src.loaders.dataset_loader import load_dataset, get_candidate_vocab
from src.models.prompts import build_system_prompts
from src.models.vlm_models import MODEL_REGISTRY, VLLM_FAMILIES, create_vlm
from src.vlm_pipeline import run_inference, resolve_hybrid_label, normalize_objects, _normalize_object
from src.evaluation import clip_metrics
from src.utils import update_results_store, update_dataset_class_stats, compute_class_stats, format_duration


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
    matching lists of true and predicted booleans — computed for BOTH the
    positive class (nature/biotic/material, per CLAUDE.md's convention) and the
    negative class (no-nature/abiotic/immaterial, suffixed `_neg`), since
    accuracy/positive-class numbers alone can hide a weak negative-class score
    (e.g. a model that over-predicts "nature" can show high nature recall while
    quietly missing most no-nature/abiotic/immaterial cases)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "precision_neg": 0.0, "recall_neg": 0.0, "f1_neg": 0.0, "support": 0}
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=True, zero_division=0)
    p_neg, r_neg, f1_neg, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=False, zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "precision_neg": float(p_neg), "recall_neg": float(r_neg), "f1_neg": float(f1_neg),
        "support": len(y_true),
    }


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
    memory). Returns the created `vlm` object (unused by --stage all now, which
    reclaims VRAM by exiting the infer subprocess — see main()).
    """
    print(f"🚀 [infer] dataset='{args.dataset}', model='{args.model_family}/{args.model_name}' "
          f"-> responses_file='{args.responses_file}'")
    phase_t0 = time.time()

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

    # Authoritative candidate vocabulary for ClipMatch/hP-hR (Phase 1 has the
    # data access) — stored in the artifact header so Phase 2 needs no dataset
    # files. Needs the SAME dataset-specific file paths (e.g. ImageNet's
    # data_dir, Places' categories file) that we just used to load the dataset
    # itself — computed here (while we still have those paths handy) and saved
    # straight into the output artifact's header.
    vocab_kwargs = dict(data_dir=args.data_dir, places_categories_txt=args.places_categories_txt,
                        excel_path=args.excel_path)
    candidate_vocab = get_candidate_vocab(args.dataset, **vocab_kwargs)  # None for coco/big5

    caption_system, label_system_full, label_system_material = build_system_prompts(
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
        "candidate_vocab": candidate_vocab,
        "max_hops": args.max_hops,
    }

    out_path = Path(args.responses_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loop_t0 = time.time()
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
            caption_system_prompt=caption_system,
            label_system_full=label_system_full,
            label_system_material=label_system_material,
            tax_graph=graph, max_hops=args.max_hops,
            batch_size=args.batch_size,
            caption_max_new_tokens=args.max_new_tokens_caption,
            label_max_new_tokens=args.max_new_tokens_label,
            temperature=args.temperature, verbose=args.verbose,
        ):
            rec["record_type"] = "image"
            f.write(json.dumps(rec) + "\n")
            n += 1
        # The header (written above, before dataset loading finished vs the
        # inference loop) can't carry the total elapsed time since it's
        # written before that time is known — so a "footer" record (last line
        # of the file) carries it instead. Includes dataset-load + VLM-creation
        # time (phase_t0), not just the generation loop, since that's the full
        # wall-clock cost of "this model finishing this run".
        footer = {"record_type": "footer", "inference_time_seconds": time.time() - phase_t0}
        f.write(json.dumps(footer) + "\n")
    print(f"💾 [infer] wrote {n} image records to {out_path} in {time.time()-loop_t0:.1f}s "
          f"(total inference phase: {footer['inference_time_seconds']:.1f}s)")
    return vlm


# =============================================================================
# PHASE 2 — scoring
# =============================================================================
def _read_artifact(path):
    """Read back a JSON-Lines artifact written by phase_infer: the first
    header-tagged line (merged with the footer line's timing info, if
    present — older artifacts written before footers existed simply won't
    have it), plus every per-image record line, as plain Python dicts."""
    header = None
    footer = None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "header":
                header = obj
            elif obj.get("record_type") == "footer":
                footer = obj
            else:
                records.append(obj)
    if header is None:
        raise ValueError(f"Artifact {path} has no header line.")
    if footer:
        header = {**header, **{k: v for k, v in footer.items() if k != "record_type"}}
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
    phase_t0 = time.time()
    header, records = _read_artifact(args.responses_file)
    dataset = header["dataset"]
    candidate_vocab = header.get("candidate_vocab")
    # ClipMatch/hP/hR only make sense for datasets with a fixed candidate
    # class list (ImageNet/Places) — see clip_metrics.CLIPMATCH_DATASETS.
    run_clipmatch = dataset in clip_metrics.CLIPMATCH_DATASETS and candidate_vocab
    # Single-label datasets drive their nature/biotic/material metrics off the
    # forced top-1 ClipMatch class (recap §6.1/§6.2); that requires the fixed
    # candidate vocab, so it coincides exactly with run_clipmatch. COCO/BIG-5
    # keep the image-level-OR + matched-object path instead.
    single_label = bool(run_clipmatch)
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

    # ---- Hybrid labels per object ----
    # Mapping + hybrid resolution now happen in Phase 1 (see
    # src/vlm_pipeline.run_inference), so records written by the current infer
    # stage already carry `object_finals`. We only recompute here for BACKWARD
    # COMPATIBILITY with older artifacts that predate that change.
    max_hops = header.get("max_hops", args.max_hops)
    for rec in records:
        if rec.get("object_finals") is not None:
            continue
        finals = []
        for obj, lab in zip(rec["objects"], rec["object_labels"]):
            finals.append(resolve_hybrid_label(obj, lab, graph, max_hops=max_hops))
        rec["object_finals"] = finals

    # ---- CLIP encodings (batched across ALL images for efficiency) ----
    # Rather than encoding one image/caption/object-list at a time inside the
    # loop below, we batch ALL of them together right now — a small number of
    # large batched GPU calls is much faster than thousands of tiny ones.
    image_paths = [r["image_path"] for r in records]
    captions = [r["caption"] for r in records]
    image_embs = scorer.encode_images(image_paths)
    caption_embs = scorer.encode_text(captions, warn_truncation=False)

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
    # encode_text([]) itself returns a correctly-shaped zero-row array (see
    # CLIPScorer.encode_text), so this is called unconditionally — NOT gated
    # on `if flat_texts` — so a dataset where every image happens to extract
    # zero objects still gets f_clipscore's sentence-only term per image below
    # instead of silently skipping reference-free scoring for the whole run.
    obj_embs_all = scorer.encode_text(flat_texts)

    candidate_embs = None
    synset_to_candidate_idx = {}
    if run_clipmatch:
        # Also encode the FIXED candidate-class vocabulary just once (rather
        # than per image) — every image's ClipMatch score is computed against
        # this same set of candidate-class embeddings.
        candidate_embs = scorer.encode_text(
            [clip_metrics.OBJECT_TEMPLATE.format(c["class_name"]) for c in candidate_vocab])
        # Reverse lookup synset -> its own row in candidate_embs, reused below
        # to fetch a single image's GT-class embedding without re-encoding it
        # (every single-label image's GT synset is one of these candidates,
        # since candidate_vocab is a superset of every taxonomy-labeled class
        # in the dataset — see get_candidate_vocab's docstring).
        synset_to_candidate_idx = {c["synset_id"]: i for i, c in enumerate(candidate_vocab)}

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
    # EXPERIMENTAL caption-based ClipMatch variant (recap §11 open item) —
    # accumulated in parallel with the object-list version above, purely for
    # side-by-side comparison at the end (see clip_metrics.clipmatch_from_caption).
    clipmatch_cap_top1 = 0
    clipmatch_cap_support = 0
    hp_cap_vals, hr_cap_vals, hf1_cap_vals = [], [], []
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
        rec_obj_embs = obj_embs_all[obj_slice]

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
        fclip_vals.append(clip_metrics.f_clipscore(image_embs[idx], caption_embs[idx], rec_obj_embs))
        objclip_vals.append(clip_metrics.object_clipscore(image_embs[idx], rec_obj_embs))

        # --- extraction-hit diagnostic (ALL datasets; reporting-only) ---
        # recap §6.3: keep the exact-match extraction rate as a descriptive
        # diagnostic, but it NO LONGER gates or feeds the nature/biotic/material
        # scores (those come from the forced top-1 ClipMatch class on single-
        # label datasets, and from matched-object finals on COCO/BIG-5).
        for t in targets:
            if t.get("gt_nature") is None:
                continue
            n_gt_targets += 1
            if find_matching_object(objs, t) is not None:
                n_extraction_hits += 1

        if single_label:
            # --- ImageNet/Places (single-label): axes from an embedding-matched
            # extracted object ---
            # The GT class is embedded via CLIP, and the extracted object with
            # the highest cosine similarity to THAT embedding is taken as this
            # image's representative of the (single, already-known) GT class.
            # Its OWN hybrid-resolved final_nature/final_biotic/final_material
            # (see src/vlm_pipeline.resolve_hybrid_label) are used directly as
            # the prediction for all three axes.
            #
            # This replaces the earlier approach of reading nature/biotic off the
            # taxonomy position of the ClipMatch top-1 PREDICTED CLASS (a global
            # argmax over the ENTIRE candidate vocabulary, e.g. all 1000 ImageNet
            # classes). That global argmax is noisy in a busy scene: a handful of
            # extracted objects can push the argmax onto a semantically-unrelated
            # candidate class just because it happened to score marginally higher
            # against one of them, even though the model correctly recognized and
            # correctly labeled the actual GT object. Matching against the GT
            # class's OWN embedding (rather than competing across every other
            # candidate class) only asks "which extracted object represents the
            # thing we already know is in this image", which is far less exposed
            # to that cross-class noise.
            best_obj_idx = None
            gt_syn = targets[0].get("synset_id") if targets else None
            if gt_syn is not None and rec_obj_embs.shape[0] > 0:
                gt_idx = synset_to_candidate_idx.get(gt_syn)
                if gt_idx is not None:
                    sims_to_gt = rec_obj_embs @ candidate_embs[gt_idx]
                    best_obj_idx = int(sims_to_gt.argmax())

            t0 = targets[0]
            # A present GT with no matched object (best_obj_idx None — no objects
            # extracted, or the GT synset isn't in the candidate vocab) is
            # PENALIZED AS WRONG — encoded as the negation of the GT — never
            # dropped (CLAUDE.md: prediction-unmapped penalized).
            if t0.get("gt_nature") is not None:
                gt_n = bool(t0["gt_nature"])
                pn = finals[best_obj_idx]["final_nature"] if best_obj_idx is not None else None
                nat_true.append(gt_n)
                nat_pred.append(bool(pn) if pn is not None else (not gt_n))
            if t0.get("gt_biotic") is not None:
                gt_b = bool(t0["gt_biotic"])
                pb = finals[best_obj_idx]["final_biotic"] if best_obj_idx is not None else None
                bio_true.append(gt_b)
                bio_pred.append(bool(pb) if pb is not None else (not gt_b))
            # material: ALWAYS the VLM's own prediction (CLAUDE.md — never mapped),
            # taken from the same embedding-matched object's final_material. None
            # (object judged non-nature, parse failure, or no match) is penalized
            # as wrong, never taxonomy-defaulted.
            if t0.get("gt_material") is not None:
                gt_m = bool(t0["gt_material"])
                pm = finals[best_obj_idx]["final_material"] if best_obj_idx is not None else None
                mat_true.append(gt_m)
                mat_pred.append(bool(pm) if pm is not None else (not gt_m))

            # ClipMatch top-1 (exact synset) + hP/hR: UNCHANGED — this is a
            # separate reported metric (recap §8e/§8f), a genuine classification-
            # into-the-fixed-candidate-vocabulary question, not the axis scores
            # above. Still needs the global argmax over every candidate class.
            pred_class_synset = pred_node = None
            if rec_obj_embs.shape[0] > 0:
                _, pred_idx, per_obj_sim = clip_metrics.clipmatch(rec_obj_embs, candidate_embs)
                if pred_idx >= 0:
                    pred_class_synset = candidate_vocab[pred_idx]["synset_id"]
                    # Turn the predicted CLASS into a specific WordNet synset
                    # NODE by picking whichever extracted object phrase best
                    # represents that prediction (needed for hP/hR below).
                    pred_node = taxonomy_metrics.resolve_to_wordnet(
                        list(per_obj_sim), pred_class_synset, objs)

            # Every image with a GT synset counts toward support; a total miss
            # (no prediction) scores top-1=0 and hP/hR=0, not excluded.
            if gt_syn:
                clipmatch_support += 1
                if pred_class_synset is not None and pred_class_synset == gt_syn:
                    clipmatch_top1 += 1
                hier = taxonomy_metrics.compute_hierarchical_metrics(graph, gt_syn, pred_node)
                hp_vals.append(hier["hp"]); hr_vals.append(hier["hr"]); hf1_vals.append(hier["hf1"])

            # --- EXPERIMENTAL: caption-based ClipMatch variant (recap §11 open
            # item), computed in PARALLEL with the object-list version above for
            # side-by-side comparison — NOT used for the axis scores, and not
            # (yet) the default. Predicts the class from the WHOLE CAPTION's
            # similarity to each candidate, then asks which extracted object
            # best aligns with THAT predicted class (for hP/hR resolution) —
            # see clip_metrics.clipmatch_from_caption.
            pred_class_synset_cap = pred_node_cap = None
            _, pred_idx_cap = clip_metrics.clipmatch_from_caption(caption_embs[idx], candidate_embs)
            if pred_idx_cap >= 0:
                pred_class_synset_cap = candidate_vocab[pred_idx_cap]["synset_id"]
                if rec_obj_embs.shape[0] > 0:
                    sims_to_pred_cap = rec_obj_embs @ candidate_embs[pred_idx_cap]
                    pred_node_cap = taxonomy_metrics.resolve_to_wordnet(
                        list(sims_to_pred_cap), pred_class_synset_cap, objs)

            if gt_syn:
                clipmatch_cap_support += 1
                if pred_class_synset_cap is not None and pred_class_synset_cap == gt_syn:
                    clipmatch_cap_top1 += 1
                hier_cap = taxonomy_metrics.compute_hierarchical_metrics(graph, gt_syn, pred_node_cap)
                hp_cap_vals.append(hier_cap["hp"]); hr_cap_vals.append(hier_cap["hr"])
                hf1_cap_vals.append(hier_cap["hf1"])
        else:
            # --- COCO (multi-label) + BIG-5 (holistic): image-level nature OR
            #     + matched-object biotic/material ---
            # TODO(grounding-pipeline, recap §6.4): for COCO, replace this
            # lexical find_matching_object matching with Hungarian box-IoU
            # assignment (IoU>=0.5) once Grounding DINO 1.5 provides predicted
            # boxes — matched GT boxes score bio/material/nature as usual;
            # unmatched GT boxes are penalized as wrong; unmatched PREDICTED
            # boxes are excluded, NOT penalized (COCO's 80 classes are a curated
            # subset, so an extra real object is not a hallucination). Not
            # implementable until the Grounding pipeline exists.
            g_nat = image_gt_nature(targets)
            if g_nat is not None:
                nat_true.append(bool(g_nat))
                nat_pred.append(bool(image_pred_nature(finals)))
            for t in targets:
                if t.get("gt_nature") is None:
                    continue
                mi = find_matching_object(objs, t)
                if mi is None:
                    continue
                fin = finals[mi]
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
        "axis_scoring_note": ("imagenet/places: nature/biotic/material all come from the "
                              "extracted object with the highest CLIP embedding similarity to "
                              "the GT class's OWN template embedding (NOT the ClipMatch top-1 "
                              "class's taxonomy position, and NOT lexical extraction matching) "
                              "— that object's hybrid-resolved final_nature/final_biotic/"
                              "final_material are used directly. ClipMatch top-1 + hP/hR (below) "
                              "remain a separate metric, still the global argmax over the full "
                              "candidate vocabulary. coco/big5: image-level nature (OR) + "
                              "matched-object biotic/material (coco box-IoU is future work, "
                              "recap §6.4)."),
        "material_caveat": ("Material GT for imagenet/coco/places is the heuristic "
                            "gt_material=True default (real photos); only BIG-5 has genuine "
                            "material GT. Predicted material is always the VLM's judgment "
                            "(never mapped), so on imagenet/places it is scored against that "
                            "heuristic GT default — BIG-5 is the only genuine material benchmark."),
    }
    if run_clipmatch:
        summary["clipmatch"] = {
            "top1_accuracy": (clipmatch_top1 / clipmatch_support) if clipmatch_support else 0.0,
            "support": clipmatch_support,
        }
        summary["hierarchical"] = {"hp": _mean(hp_vals), "hr": _mean(hr_vals),
                                   "hf1": _mean(hf1_vals), "support": len(hf1_vals)}
        # EXPERIMENTAL caption-based variant (recap §11), printed alongside the
        # object-list version above for comparison — see clipmatch_from_caption.
        summary["clipmatch_caption"] = {
            "top1_accuracy": (clipmatch_cap_top1 / clipmatch_cap_support) if clipmatch_cap_support else 0.0,
            "support": clipmatch_cap_support,
        }
        summary["hierarchical_caption"] = {"hp": _mean(hp_cap_vals), "hr": _mean(hr_cap_vals),
                                           "hf1": _mean(hf1_cap_vals), "support": len(hf1_cap_vals)}

    # Wall-clock time this model took to finish this run, formatted "D-HH:MM:SS"
    # (SLURM-style elapsed time). inference_time_seconds comes from phase_infer's
    # footer record (dataset load + VLM creation + generation loop); it's None
    # on an artifact written before footers existed, or on a --stage
    # score-only rerun of such an artifact — in that case "total" falls back to
    # just the scoring time actually measured here, so it's never silently
    # wrong, just incomplete. scoring_time_seconds always covers this ENTIRE
    # phase_score call (artifact read, CLIP load + encode, metric loop).
    inference_time = header.get("inference_time_seconds")
    scoring_time = time.time() - phase_t0
    total_time = (inference_time or 0.0) + scoring_time
    summary["execution_time"] = {
        "inference": format_duration(inference_time),
        "scoring": format_duration(scoring_time),
        "total": format_duration(total_time),
    }

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
    # Distinct-target-class nature/biotic/material composition of THIS run's
    # sampled dataset (recap: sampling is deterministic — a fixed --max_samples
    # always yields the same subset — so this is stable across reruns of the
    # same config). Keyed by --max_samples so different configurations (e.g.
    # 1000 vs the full dataset) accumulate side by side instead of overwriting.
    all_targets = [t for rec in records for t in rec.get("targets", [])]
    class_stats = compute_class_stats(all_targets)
    config_key = str(args.max_samples) if args.max_samples is not None else "full"
    update_dataset_class_stats(out_path, dataset=dataset, config_key=config_key, stats=class_stats)
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
    neg_labels = {"nature": "no_nature", "biotic_matched": "abiotic", "material_matched": "immaterial"}
    for axis in ("nature", "biotic_matched", "material_matched"):
        m = s[axis]
        print(f"\n--- {axis} (support {m['support']}) ---")
        print(f"Acc {m['accuracy']:.4f}")
        print(f"  {axis.split('_')[0]:<12} (pos) P {m['precision']:.4f} | R {m['recall']:.4f} | F1 {m['f1']:.4f}")
        print(f"  {neg_labels[axis]:<12} (neg) P {m['precision_neg']:.4f} | R {m['recall_neg']:.4f} | F1 {m['f1_neg']:.4f}")
    if run_clipmatch:
        print(f"\n--- ClipMatch [object-list] (support {s['clipmatch']['support']}) ---")
        print(f"Top-1: {s['clipmatch']['top1_accuracy']:.4f}")
        h = s["hierarchical"]
        print(f"--- Hierarchical [object-list] (support {h['support']}) ---")
        print(f"hP {h['hp']:.4f} | hR {h['hr']:.4f} | hF1 {h['hf1']:.4f}")
        # EXPERIMENTAL caption-based variant (recap §11), printed for direct
        # comparison against the object-list version above.
        print(f"\n--- ClipMatch [caption, EXPERIMENTAL] (support {s['clipmatch_caption']['support']}) ---")
        print(f"Top-1: {s['clipmatch_caption']['top1_accuracy']:.4f}")
        hc = s["hierarchical_caption"]
        print(f"--- Hierarchical [caption, EXPERIMENTAL] (support {hc['support']}) ---")
        print(f"hP {hc['hp']:.4f} | hR {hc['hr']:.4f} | hF1 {hc['hf1']:.4f}")
    t = s["execution_time"]
    inf_str = t["inference"] if t["inference"] is not None else "n/a"
    print(f"\nExecution time (D-HH:MM:SS): inference {inf_str} | scoring {t['scoring']} | total {t['total']}")
    print("=" * 60)


def _log_wandb(args, summary, run_clipmatch):
    """Push the final summary metrics to Weights & Biases for tracking/
    comparison across runs, if --wandb was passed."""
    import wandb
    # Under --stage all each stage runs in its OWN subprocess (see main()), so a
    # shared run id (generated once in main, passed through the pickled args) +
    # resume="allow" makes every subprocess log into the SAME W&B run instead of
    # spawning a separate run per process. For a standalone --stage score run no
    # id is set, so this falls back to a fresh auto-id run.
    run_id = getattr(args, "wandb_run_id", None)
    wandb.init(entity="paumonserrat03-universitat-aut-noma-de-barcelona", project="TFM_VLM",
               config=vars(args), name=f"vlm_pipeline_{summary['dataset']}_{summary['model']}".replace("/", "_"),
               id=run_id, resume="allow" if run_id else None)
    log = {
        "ObjectsPerImage": summary["diagnostics"]["objects_per_image"],
        "ExtractionHitRate": summary["diagnostics"]["extraction_hit_rate"],
        "WordNetMappingRate": summary["diagnostics"]["wordnet_mapping_rate"],
        "F-CLIPScore": summary["reference_free"]["f_clipscore"],
        "Object-CLIPScore": summary["reference_free"]["object_clipscore"],
        "Nature/F1": summary["nature"]["f1"], "Nature/Accuracy": summary["nature"]["accuracy"],
        "Nature/F1_NoNature": summary["nature"]["f1_neg"],
        "Biotic/F1": summary["biotic_matched"]["f1"], "Biotic/F1_Abiotic": summary["biotic_matched"]["f1_neg"],
        "Material/F1": summary["material_matched"]["f1"], "Material/F1_Immaterial": summary["material_matched"]["f1_neg"],
    }
    if run_clipmatch:
        log["ClipMatch/Top1"] = summary["clipmatch"]["top1_accuracy"]
        log["Hierarchical/hF1"] = summary["hierarchical"]["hf1"]
        # EXPERIMENTAL caption-based variant (recap §11).
        log["ClipMatchCaption/Top1"] = summary["clipmatch_caption"]["top1_accuracy"]
        log["HierarchicalCaption/hF1"] = summary["hierarchical_caption"]["hf1"]
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
    p.add_argument("--responses_file", type=str, default=None,
                   help="Intermediate artifact: written by infer, read by score. Default: "
                        "'vlm_responses.jsonl' inside --results_dir/--run_name (the SAME "
                        "folder --output_file lands in), so a run's artifact and its results "
                        "JSON/CSV are always co-located and both respect --results_dir/"
                        "--run_name. Pass an explicit path to override (e.g. to write it "
                        "somewhere else, or to point --stage score at a specific prior "
                        "artifact). Under --stage all this file is PURELY an internal handoff "
                        "between the infer and score subprocesses, so it is deleted once "
                        "scoring finishes successfully — see --keep_responses_file to retain "
                        "it. --stage infer/score never delete it (infer's whole point is to "
                        "persist it for a later --stage score; score's whole point is to "
                        "reread an existing artifact, possibly after a metrics-code change).")
    p.add_argument("--keep_responses_file", action="store_true",
                   help="Under --stage all, keep --responses_file on disk after scoring "
                        "finishes instead of deleting it (e.g. to inspect the raw VLM outputs, "
                        "or to re-run --stage score later without re-running inference). "
                        "Ignored for --stage infer/score, which never delete the file regardless.")

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
    p.add_argument("--max_hops", type=int, default=0,
                   help="Maximum WordNet hop distance allowed when mapping an EXTRACTED "
                        "object onto the labeled taxonomy (map_object_to_taxonomy -> "
                        "resolve_labels). 0 = map only when an annotator labeled that exact "
                        "synset (no inherited labels); 1 = also accept a label inherited "
                        "across one hypernym/hyponym hop; etc. Lower values make the VLM "
                        "fallback fire more often (fewer objects map), higher values map "
                        "more aggressively. Default 3 (previous behavior). Stored in the "
                        "artifact header and reused by --stage score.")

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


def _resolve_responses_file(args):
    """Fill in --responses_file's default (None until now) as
    '<results_dir>/<run_name>/vlm_responses.jsonl' — the SAME directory
    --output_file lands in — so the intermediate artifact respects
    --results_dir/--run_name exactly like every other output this script
    writes, instead of always landing at a fixed cwd-relative path regardless
    of those flags. An explicitly-passed --responses_file is left untouched.
    Creates that directory so phase_infer can open the file for writing.
    Mutates and returns `args`."""
    if args.responses_file is None:
        out_dir = Path(args.results_dir)
        if args.run_name:
            out_dir = out_dir / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        args.responses_file = str(out_dir / "vlm_responses.jsonl")
    return args


# =============================================================================
# Subprocess workers (for --stage all)
# =============================================================================
def _infer_worker(args):
    """Subprocess entrypoint for the inference half of --stage all. Runs
    phase_infer and then simply RETURNS/EXITS: letting the whole subprocess die
    is what actually reclaims the VLM's VRAM — the OS tears down the process's
    entire CUDA context, which is more reliable than an in-process unload (vLLM
    and torch do not dependably release CUDA memory on Python GC)."""
    phase_infer(args)


def _score_worker(args):
    """Subprocess entrypoint for the scoring half of --stage all — loads CLIP in
    a FRESH process that starts with zero GPU memory held (the infer subprocess
    that held the VLM has already exited)."""
    phase_score(args)


def main():
    args = parse_args()
    args = _resolve_responses_file(args)

    if args.stage == "infer":
        phase_infer(args)
        return
    if args.stage == "score":
        phase_score(args)
        return

    # --stage all: run each half in its OWN spawned subprocess so GPU memory is
    # fully reclaimed by the OS between stages — the VLM (infer) and CLIP (score)
    # never coexist on the GPU, and we don't rely on in-process CUDA cleanup.
    #   - 'spawn' (NOT 'fork') is required to use CUDA in a child process.
    #   - Only the picklable argparse Namespace crosses the boundary — paths and
    #     plain scalars, never tensors or model handles.
    #   - A shared W&B run id (below) is threaded through so both subprocesses
    #     log into one run (see _log_wandb).
    if args.wandb:
        import wandb
        args.wandb_run_id = wandb.util.generate_id()

    ctx = mp.get_context("spawn")

    p1 = ctx.Process(target=_infer_worker, args=(args,))
    p1.start()
    p1.join()
    if p1.exitcode != 0:
        raise RuntimeError(f"VLM inference stage failed (exit code {p1.exitcode}).")

    p2 = ctx.Process(target=_score_worker, args=(args,))
    p2.start()
    p2.join()
    if p2.exitcode != 0:
        raise RuntimeError(f"Scoring stage failed (exit code {p2.exitcode}).")

    # --stage all's --responses_file is purely the internal handoff between
    # the two subprocesses above — scoring just finished reading it, so
    # nothing downstream needs it anymore. Delete it by default (opt out with
    # --keep_responses_file) so a run doesn't silently leave a
    # potentially-large JSON-Lines artifact behind that the user never
    # explicitly asked to keep. --stage infer/score never reach this code
    # path at all, so a standalone infer run (or a later standalone re-score)
    # is never affected.
    if not args.keep_responses_file:
        try:
            Path(args.responses_file).unlink()
            print(f"🗑️  [all] removed intermediate {args.responses_file} "
                  f"(pass --keep_responses_file to retain it)")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
