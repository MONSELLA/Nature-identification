#!/usr/bin/env python3
"""
run_vlm_pipeline.py

End-to-end driver for the baseline BIG-5 VLM pipeline, in three stages:

  --stage all   : THE STANDARD WAY TO RUN THIS PIPELINE. Runs inference then
                  scoring, but each as its OWN genuinely separate OS process
                  (see main()): a real `subprocess.run([sys.executable,
                  __file__, "--stage", "infer", ...])` child, NOT a
                  `multiprocessing.Process`. The infer child loads the VLM,
                  writes the JSON-lines artifact to --responses_file, then
                  EXITS — which makes the OS reclaim 100% of the VLM's VRAM
                  before the score child loads the CLIP model + the
                  TaxonomyGraph. The VLM and CLIP never hold GPU memory at
                  the same time, and we don't rely on in-process CUDA cleanup
                  (vLLM/torch release it unreliably on GC).
                  WHY subprocess.run AND NOT multiprocessing.Process: vLLM's
                  own V1 engine spawns ITS OWN internal subprocess for the
                  engine core (visible in logs as "(EngineCore pid=...)").
                  Wrapping that in another `multiprocessing.Process` (even
                  with the 'spawn' start method) nests one spawned-process
                  tree inside another and can silently DEADLOCK right at
                  vLLM's async-engine IPC handshake — confirmed in practice:
                  `--stage infer` alone completed engine init and started
                  generating, while the exact same run under the old
                  `--stage all` (multiprocessing.Process wrapper) hung
                  forever at that same point. A plain OS subprocess has no
                  such nesting — it's the same relationship `--stage infer`
                  run by hand has to vLLM's own child process. Args cross the
                  boundary as reconstructed CLI flags (see _args_to_cli),
                  not a pickled Namespace. --responses_file is ALWAYS
                  retained on disk (never deleted, under any --stage) — it
                  carries every raw VLM response plus the resolved per-object
                  labels, needed downstream by the Grounding pipeline.

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
      * ImageNet/Places (single-label): nature/biotic come from the ClipMatch
        TOP-1 PREDICTED CLASS's own taxonomy position — the global argmax over
        candidate_vocab (see src/loaders/dataset_loader.get_candidate_vocab),
        which is restricted to classes MAPPED INTO THE GRAPH only, so the
        prediction can never land on an unannotated class. material is ALWAYS
        the VLM's own label (CLAUDE.md — never mapped), taken from the
        extracted object most representative of that predicted class.
      * COCO/BIG-5: image-level nature (nature=1 if ANY extracted object is
        nature) + matched-object biotic/material (COCO box-IoU matching is
        future work, gated on the Grounding pipeline — §6.4).
  - CLIPScore + F-CLIPScore + Object-CLIPScore : ALL datasets (reference-free),
    scored with --clipscore_model.
  - ClipMatch + hP/hR/hF1                : ImageNet + Places ONLY (fixed vocab,
    restricted to classes mapped into the graph), scored with
    --clipmatch_model (independently selectable from --clipscore_model;
    defaults to the same checkpoint).
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
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# The HF "You are sending unauthenticated requests to the HF Hub" notice is
# emitted by huggingface_hub at WARNING level; hide it unless a token is truly
# needed. (Setting HF_TOKEN is the real fix and also lifts rate limits.)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from src.evaluation import taxonomy_metrics
from src.loaders.excel_loader import TaxonomyGraph
from src.loaders.dataset_loader import load_dataset, get_candidate_vocab, BIG5_DATASETS
from src.models.prompts import build_system_prompts
from src.models.vlm_models import MODEL_REGISTRY, create_vlm
from src.vlm_pipeline import run_inference, resolve_hybrid_label, _normalize_object
from src.evaluation import clip_metrics
from src.utils import (update_results_store, update_dataset_image_stats, compute_image_stats,
                       format_duration, crosses_decile)

# Per-model outputs are split by FILE TYPE into these two subfolders of
# <results_dir>/<run_name>, so a run_name folder full of models stays
# readable at a glance: the shared results JSON sits directly in the folder,
# every model's raw inference artifact sits in RESPONSES_SUBDIR, and every
# model's per-image predictions CSV sits in PREDICTIONS_SUBDIR.
# scripts/run_grounding_pipeline.py mirrors RESPONSES_SUBDIR (it reconstructs
# the same artifact path independently) — keep the two in sync if this ever
# changes.
RESPONSES_SUBDIR = "responses"
PREDICTIONS_SUBDIR = "predictions"


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


def _fmt_big5_axis_cell(vals):
    """Render a BIG-5 gt_biotic/gt_material list (src.loaders.dataset_loader.
    load_big5 — usually one element, [True, False] when coders genuinely
    disagreed) as a single flat CSV cell: None when not applicable, the plain
    bool when there's one value, or "True/False (disagreement)" when both
    directions apply at once. Used for BOTH the GT list and its
    direction-aware predicted-value list (same shape, same ordering), so the
    same helper renders either side of the pair.
    """
    if not vals:
        return None
    if len(vals) == 1:
        return bool(vals[0])
    return "True/False (disagreement)"


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
                "precision_neg": 0.0, "recall_neg": 0.0, "f1_neg": 0.0, "support": 0,
                "n_pos": 0, "n_neg": 0}
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=True, zero_division=0)
    p_neg, r_neg, f1_neg, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=False, zero_division=0)
    n_pos = int(sum(1 for v in y_true if v))
    return {
        "accuracy": float(acc),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "precision_neg": float(p_neg), "recall_neg": float(r_neg), "f1_neg": float(f1_neg),
        "support": len(y_true),
        # GT-label split of this axis's supported images — e.g. how many of the
        # material/immaterial-supported images are actually GT-material vs.
        # GT-immaterial — reported alongside support so a flat support number
        # doesn't hide a heavily skewed axis.
        "n_pos": n_pos, "n_neg": len(y_true) - n_pos,
    }


def _mean(vals):
    """Average of a list, treating None entries as "not applicable" (skipped
    rather than counted as zero). Returns 0.0 for an empty/all-None list."""
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(vals):
    """Population standard deviation of a list (None entries dropped, same
    convention as _mean) — reported alongside the hierarchical metrics' means
    (hP/hR/hF1/Wu-Palmer) so a mean alone doesn't hide whether per-image scores
    cluster tightly around it or are spread widely. Returns 0.0 for a list with
    fewer than 2 values (nothing to measure spread over)."""
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals))


def _hier_bucket_summary(bucket):
    """Turn one hier_by_gt_nature[...] accumulator into the same
    ("failures as error", "mapped only") pair of dicts as the top-level
    summary["hierarchical"]/["hierarchical_mapped"] — same keys, same mean +
    population-std convention, just scoped to this one GT-nature bucket."""
    failures_as_error = {
        "hp": _mean(bucket["hp"]), "hr": _mean(bucket["hr"]),
        "hf1": _mean(bucket["hf1"]), "wup": _mean(bucket["wup"]),
        "hp_std": _std(bucket["hp"]), "hr_std": _std(bucket["hr"]),
        "hf1_std": _std(bucket["hf1"]), "wup_std": _std(bucket["wup"]),
        "support": len(bucket["hf1"]),
        "mapping_failure_rate": (bucket["mapping_fail"] / len(bucket["hf1"]))
                                 if bucket["hf1"] else 0.0,
        "mapping_failures": bucket["mapping_fail"],
    }
    mapped_only = {
        "hp": _mean(bucket["hp_mapped"]), "hr": _mean(bucket["hr_mapped"]),
        "hf1": _mean(bucket["hf1_mapped"]), "wup": _mean(bucket["wup_mapped"]),
        "hp_std": _std(bucket["hp_mapped"]), "hr_std": _std(bucket["hr_mapped"]),
        "hf1_std": _std(bucket["hf1_mapped"]), "wup_std": _std(bucket["wup_mapped"]),
        "support": len(bucket["hf1_mapped"]),
    }
    return failures_as_error, mapped_only


def _clip_token_stats(scorer, texts):
    """Per-text CLIP-tokenizer token counts, mean length, and truncated count
    (>= scorer.context_length) for a list of texts under `scorer`'s own
    tokenizer. Returns (None, None, None) when the scorer can't count tokens
    (Long-CLIP's tokenizer always pads/truncates to a fixed length — see
    CLIPScorer.count_tokens)."""
    try:
        counts = scorer.count_tokens(texts)
    except NotImplementedError:
        return None, None, None
    mean = sum(counts) / len(counts) if counts else 0.0
    truncated = sum(1 for n in counts if n >= scorer.context_length)
    return counts, mean, truncated


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
        twitter_en_gt=args.twitter_en_gt_csv, twitter_es_gt=args.twitter_es_gt_csv,
        twitter_images_dir=args.big_5_twitter_images_dir,
        weibo_ch0_gt=args.weibo_ch0_gt_csv, weibo_ch1_gt=args.weibo_ch1_gt_csv,
        weibo_images_dir=args.big_5_weibo_images_dir,
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
    candidate_vocab = get_candidate_vocab(args.dataset, graph, **vocab_kwargs)  # None for coco/big5

    caption_system, label_system_full, label_system_material = build_system_prompts(
        args.nature_definition_path, args.biotic_definition_path, args.material_definition_path)

    # Summary-caption ClipMatch is now the DEFAULT wherever ClipMatch itself
    # runs (ImageNet/Places — see clip_metrics.CLIPMATCH_DATASETS): measured
    # 0.41 vs 0.34 top-1 against the raw caption, with far less truncation.
    # --no_summarize_clipmatch_caption opts back into the old raw-caption
    # behavior. Never runs on COCO/BIG-5 regardless of the flag — nothing
    # there would ever read the result, so don't spend the extra VLM call.
    summarize_for_clipmatch = (args.dataset in clip_metrics.CLIPMATCH_DATASETS
                                and not args.no_summarize_clipmatch_caption)

    # Every registered family is vLLM-served (see src/models/vlm_models.py's
    # MODEL_REGISTRY) — the HuggingFace-served BLIP family was removed.
    vlm_kwargs = {"dtype": args.dtype, "gpu_memory_utilization": args.gpu_memory_utilization,
                  "trust_remote_code": args.trust_remote_code,
                  # 0 is the CLI's "disabled" spelling (argparse can't default
                  # an int flag to None while still accepting a positive int);
                  # VLLMBackedVLM.__init__ itself expects None for "disabled".
                  "max_image_side": args.max_image_side or None}
    if args.max_model_len is not None:
        vlm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        vlm_kwargs["max_num_seqs"] = args.max_num_seqs
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
        # Kept separately from "model" (the family/name identifier used for
        # results-store keys and the CSV's own "model" column) so Phase 2 can
        # build filenames from JUST the model name — see _model_slug.
        "model_name": args.model_name,
        "candidate_vocab": candidate_vocab,
        "max_hops": args.max_hops,
        "summarize_clipmatch_caption": summarize_for_clipmatch,
        # WHICH summary prompt this artifact was produced with (object-centric
        # for ImageNet vs scene-centric for Places — see
        # prompts.SUMMARY_CAPTION_PROMPTS). Recorded because the two are not
        # interchangeable: comparing ClipMatch numbers across artifacts written
        # with different prompt variants would be comparing different setups.
        "summary_caption_prompt": (f"summary_caption_prompt_{args.dataset}"
                                    if summarize_for_clipmatch else None),
    }

    out_path = Path(args.responses_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- --resume: skip images this artifact already has ---------------------
    # Inference is by far the expensive half (hours on a full dataset), and a
    # crash used to mean redoing all of it. The streamed artifact already
    # contains every completed image, so resuming is just "filter the dataset
    # down to what's missing and APPEND". The header is only written on a
    # fresh run; on a resume the existing one is kept (it carries the
    # candidate_vocab/max_hops/prompt-variant this artifact was built with,
    # and rewriting it from the current args could silently mix two configs).
    resume_mode = False
    if getattr(args, "resume", False) and out_path.exists():
        done, has_footer, n_bad = _already_inferred_paths(out_path)
        if has_footer:
            print(f"✅ [infer] {out_path} already ends in a footer — this run "
                  f"is complete ({len(done)} images). Nothing to resume; "
                  f"delete the file or drop --resume to redo it.")
            return
        before = len(dataset)
        dataset = [d for d in dataset if d["image_path"] not in done]
        resume_mode = True
        print(f"⏩ [infer] resuming {out_path}: {len(done)} image(s) already done, "
              f"{len(dataset)} of {before} remaining"
              + (f" ({n_bad} truncated line(s) dropped — those images redone)" if n_bad else ""))
        if not dataset:
            print("⚠️ [infer] nothing left to infer, but the artifact has no footer — "
                  "appending one so --stage score sees a complete run.")
            with open(out_path, "a") as f:
                f.write(json.dumps({"record_type": "footer",
                                     "inference_time_seconds": None}) + "\n")
            return

    loop_t0 = time.time()
    n = 0
    # "JSON Lines" format: one complete, independent JSON object per line of
    # the file (as opposed to one giant JSON array for the whole file). This
    # lets us write results incrementally as they're produced (streaming,
    # rather than building one huge in-memory list and writing it all at the
    # end) and lets Phase 2 read them back one line at a time too.
    with open(out_path, "a" if resume_mode else "w") as f:
        if not resume_mode:
            f.write(json.dumps(header) + "\n")
        for rec in run_inference(
            vlm, dataset,
            caption_system_prompt=caption_system,
            label_system_full=label_system_full,
            label_system_material=label_system_material,
            tax_graph=graph, max_hops=args.max_hops,
            batch_size=args.batch_size,
            caption_max_new_tokens=args.max_new_tokens_caption,
            extraction_max_new_tokens=args.max_new_tokens_extraction,
            label_max_new_tokens=args.max_new_tokens_label,
            temperature=args.temperature, verbose=args.verbose,
            summarize_for_clipmatch=summarize_for_clipmatch,
            summary_max_new_tokens=args.summary_max_new_tokens,
            # Dataset NAME (not the loaded instances in the positional arg
            # above) — picks the object-centric vs scene-centric summary
            # prompt, see prompts.get_summary_caption_prompt.
            dataset=args.dataset,
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
def _already_inferred_paths(path):
    """Every image_path already present as a completed image record in a
    partially-written artifact, plus whether that file ends in a footer.

    Used by --resume. Because phase_infer STREAMS one JSON line per image as
    it goes (rather than buffering and writing at the end), a run killed
    mid-way — an OOM, a SLURM timeout, the CMYK crash — leaves a file whose
    every complete line is still valid, finished work. Resuming just means
    skipping those images and appending the rest.

    Lines are parsed defensively: a hard kill (SIGKILL / CUDA abort) can
    truncate the FINAL line mid-write, since Python's file buffer isn't
    flushed per record. Such a line is silently dropped, so that one image is
    simply re-inferred rather than corrupting the resume set.

    Returns (set_of_image_paths, has_footer, n_bad_lines).
    """
    seen, has_footer, bad = set(), False, 0
    p = Path(path)
    if not p.exists():
        return seen, has_footer, bad
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1   # truncated tail line — that image gets redone
                continue
            rt = obj.get("record_type")
            if rt == "footer":
                has_footer = True
            elif rt not in ("header", "footer") and obj.get("image_path"):
                seen.add(obj["image_path"])
    return seen, has_footer, bad


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
    scorer = clip_metrics.CLIPScorer(model_name=args.clipscore_model, device=args.device,
                                     batch_size=args.clip_batch_size,
                                     trust_remote_code=args.clipscore_trust_remote_code)

    if args.verbose: print(f"{args.clipscore_model} loaded. Handles a context length of {scorer.context_length} tokens!\n")

    # ClipMatch + hP/hR (ImageNet/Places only) can use a DIFFERENT CLIP
    # checkpoint from the reference-free metrics (CLIPScore/F-CLIPScore/
    # Object-CLIPScore) — the two jobs have different requirements (matching
    # a whole caption against a fixed candidate vocabulary vs. reference-free
    # image-text plausibility). Default is the SAME checkpoint as --clipscore_model;
    # only load a second model if the resolved (model, trust_remote_code) pair
    # actually differs, to avoid holding two copies of the same weights.
    clipmatch_model_name = args.clipmatch_model or args.clipscore_model
    clipmatch_trust_remote_code = (args.clipmatch_clip_trust_remote_code
                                    if args.clipmatch_clip_trust_remote_code is not None
                                    else args.clipscore_trust_remote_code)
    cm_scorer = scorer
    if run_clipmatch and (clipmatch_model_name != args.clipscore_model
                          or clipmatch_trust_remote_code != args.clipscore_trust_remote_code):
        cm_scorer = clip_metrics.CLIPScorer(model_name=clipmatch_model_name, device=args.device,
                                            batch_size=args.clip_batch_size,
                                            trust_remote_code=clipmatch_trust_remote_code)
        if args.verbose:
            print(f"{clipmatch_model_name} (ClipMatch) loaded. Handles a context length of "
                  f"{cm_scorer.context_length} tokens!\n")

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
    image_embs = scorer.encode_images(image_paths, verbose=args.verbose)
    caption_embs = scorer.encode_text(captions, warn_truncation=False,
                                      verbose=args.verbose, desc="captions")

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
        flat_texts.extend(clip_metrics.object_texts(r["objects"], use_inflect=args.use_inflect_for_clipmatch))
    offsets.append(len(flat_texts))  # sentinel end-offset for the last image
    # encode_text([]) itself returns a correctly-shaped zero-row array (see
    # CLIPScorer.encode_text), so this is called unconditionally — NOT gated
    # on `if flat_texts` — so a dataset where every image happens to extract
    # zero objects still gets f_clipscore's sentence-only term per image below
    # instead of silently skipping reference-free scoring for the whole run.
    obj_embs_all = scorer.encode_text(flat_texts, verbose=args.verbose, desc="objects")

    # ClipMatch/hP-hR embeddings — SEPARATE from the ones above whenever
    # cm_scorer is a different model (must stay in one consistent embedding
    # space: the candidate vocab, the caption, and the anchor-object
    # similarity search all need to come from the SAME CLIP checkpoint).
    # When cm_scorer IS scorer (default), reuse the embeddings already
    # computed above instead of re-encoding the same text twice.
    candidate_embs = None
    run_summary_clipmatch = False
    summary_texts = None
    n_summary_truncated = None
    summary_token_mean = None
    summary_token_counts = None
    caption_token_mean = n_caption_truncated = None
    candidate_token_counts = candidate_token_mean = n_candidate_truncated = None
    primary_embs_cm = None
    if run_clipmatch:
        # Which text drives ClipMatch THIS run: the VLM's own summary caption
        # when the artifact has one, else the raw caption (old artifacts /
        # --no_summarize_clipmatch_caption). Decided up front so only the text
        # that is ACTUALLY primary gets encoded — there is no longer a
        # secondary raw-caption comparison to encode the caption for.
        run_summary_clipmatch = bool(header.get("summarize_clipmatch_caption"))
        obj_embs_all_cm = obj_embs_all if cm_scorer is scorer else cm_scorer.encode_text(
            flat_texts, verbose=args.verbose, desc="objects (clipmatch)")
        # Also encode the FIXED candidate-class vocabulary just once (rather
        # than per image) — every image's ClipMatch score is computed against
        # this same set of candidate-class embeddings. candidate_vocab is
        # already restricted to mapped-into-the-graph classes only (see
        # get_candidate_vocab), so every candidate here carries its own
        # gt_nature/gt_biotic and the ClipMatch top-1 prediction can be read
        # straight off its candidate_vocab entry — no separate graph lookup.
        # --use_wordnet_definitions_clipmatch swaps the plain "a photo of a
        # {class}" template for a richer WordNet-definition prose per class
        # (clip_metrics.wordnet_definition_text) — untested hypothesis (per
        # Pau) that giving the CANDIDATE side more semantic content aligns
        # better against a VLM's own descriptive image summary in CLIP's text
        # embedding space than a bare templated noun phrase. Otherwise, the
        # candidate side is the OpenAI CLIP prompt-ensemble mean (80
        # ImageNet-object templates / 15 Places-scene templates — see
        # clip_metrics.CLIPMATCH_CANDIDATE_TEMPLATES), computed once per class
        # since the candidate vocab is fixed for the whole run.
        if args.use_wordnet_definitions_clipmatch:
            candidate_texts = clip_metrics.candidate_vocab_texts(candidate_vocab, use_wordnet_definitions=True)
            candidate_embs = cm_scorer.encode_text(
                candidate_texts, verbose=args.verbose, desc="candidate_vocab")
        else:
            candidate_templates = clip_metrics.CLIPMATCH_CANDIDATE_TEMPLATES[dataset]
            candidate_embs, candidate_texts = clip_metrics.encode_candidate_vocab_ensemble(
                cm_scorer, candidate_vocab, candidate_templates,
                verbose=args.verbose, desc="candidate_vocab",
                use_inflect=args.use_inflect_for_clipmatch)
        candidate_token_counts, candidate_token_mean, n_candidate_truncated = _clip_token_stats(
            cm_scorer, candidate_texts)

        # Summary-caption ClipMatch — PRIMARY when the artifact has one (see
        # --summarize_clipmatch_caption / src.vlm_pipeline.summarize_caption_batch):
        # each image's own short VLM-written summary of its baseline caption.
        # Measured 0.41 vs 0.34 top-1 against the full caption on the
        # spot-check run, with 0% truncation vs. the full caption's silent
        # CLIP-tokenizer cutoff — a real re-summarization beats letting CLIP
        # truncate arbitrary content, so this DRIVES anchor selection / axis
        # scoring / hierarchical metrics whenever available. Falls back to the
        # full caption for any individual record where the summary is
        # missing/empty (e.g. an older partial artifact) so this never
        # silently drops an image from scoring.
        if run_summary_clipmatch:
            summary_texts = [r.get("clipmatch_summary_caption") or r["caption"] for r in records]
            primary_embs_cm = cm_scorer.encode_text(
                summary_texts, verbose=args.verbose, desc="summary captions (clipmatch)")
            summary_token_counts, summary_token_mean, n_summary_truncated = _clip_token_stats(
                cm_scorer, summary_texts)
        else:
            primary_embs_cm = caption_embs if cm_scorer is scorer else cm_scorer.encode_text(
                captions, warn_truncation=False, verbose=args.verbose, desc="captions (clipmatch)")
            _, caption_token_mean, n_caption_truncated = _clip_token_stats(cm_scorer, captions)

    # ---- Per-image metric accumulation ----
    # These lists/counters accumulate results across every image in the
    # dataset; they get turned into final aggregate numbers (accuracy, mean
    # scores, etc.) after the loop below finishes.
    nat_true, nat_pred = [], []
    bio_true, bio_pred, mat_true, mat_pred = [], [], [], []
    fclip_vals, objclip_vals, clipscore_vals = [], [], []
    # Two parallel accumulations of the hierarchical metrics (hP/hR/hF1 + Wu-
    # Palmer), differing ONLY in how they treat images whose ClipMatch-predicted
    # object could NOT be resolved onto a WordNet node (resolve_to_wordnet ->
    # pred_node is None):
    #   * `*_vals`        — INCLUDES those failures, scored as 0.0 (the
    #                       "prediction-unmapped penalized as wrong" convention).
    #   * `*_vals_mapped` — EXCLUDES them entirely, so it reflects hierarchical
    #                       quality ONLY over images where the predicted object
    #                       actually mapped into WordNet.
    # `n_hier_mapping_fail` counts the excluded (mapping-failure) images so the
    # failure RATE can be reported alongside both versions.
    hp_vals, hr_vals, hf1_vals = [], [], []
    wup_vals = []
    hp_vals_mapped, hr_vals_mapped, hf1_vals_mapped = [], [], []
    wup_vals_mapped = []
    n_hier_mapping_fail = 0
    # Same hierarchical metrics as above, additionally bucketed by whether
    # the IMAGE'S OWN GT is nature or no-nature (ImageNet/Places only) — a
    # model can be much better at resolving nature GTs onto a fine-grained
    # WordNet node than no-nature ones (or vice versa), which the one pooled
    # macro-average above cannot show. Keyed by the GT nature bool so the
    # accumulation loop below can do `hier_by_gt_nature[gt_nature_val][...]`
    # directly; each bucket carries both the "failures as error" and
    # "mapped only" views, mirroring hp_vals vs hp_vals_mapped above.
    hier_by_gt_nature = {
        True: {"hp": [], "hr": [], "hf1": [], "wup": [],
               "hp_mapped": [], "hr_mapped": [], "hf1_mapped": [], "wup_mapped": [],
               "mapping_fail": 0},
        False: {"hp": [], "hr": [], "hf1": [], "wup": [],
                "hp_mapped": [], "hr_mapped": [], "hf1_mapped": [], "wup_mapped": [],
                "mapping_fail": 0},
    }
    n_gt_targets = n_extraction_hits = 0
    n_objects_total = n_parse_fail = n_object_records = 0
    # Stage-2 (whole-image object-list) parse failures — DISTINCT from
    # n_parse_fail below, which counts per-OBJECT Stage-3 labeling failures.
    # An extraction failure means the model's structured output didn't parse
    # AT ALL for this image, so it silently got zero extracted objects; left
    # unflagged, that's indistinguishable from "the model genuinely found
    # nothing" (see src.vlm_pipeline.extract_objects_batch).
    n_extraction_parse_fail = 0
    n_map_nature = n_vlm_nature = 0
    # Labeling-stage parse failures, scoped to the objects the VLM was ACTUALLY
    # asked to label (`vlm_called`). This is NOT the same denominator as
    # n_object_records: label_objects_batch routes mapped NON-nature objects to
    # NO VLM call at all (recap §6), so dividing failures by every extracted
    # object silently dilutes the rate by however many objects the mapping
    # already settled. Split by the two call types, which use different schemas
    # and different system prompts and so can fail at different rates:
    #   full     — unmapped object, one TaxonomyResponse call (all three axes)
    #   material — mapped-nature object, MaterialResponse call (material only)
    # `fin["mapped"]` reproduces that routing exactly (unmapped -> full,
    # mapped + vlm_called -> material).
    n_label_calls_full = n_label_calls_material = 0
    n_parse_fail_full = n_parse_fail_material = 0
    n_label_calls = 0
    # Grounding diagnostics (see src/grounding_pipeline.py): only meaningful
    # when this artifact was actually enriched by the Grounding pipeline
    # (run_grounding = header carries a "grounding" key, added by
    # run_grounding_pipeline.py's phase_ground). "attempted" = nature entities
    # SAM3 was asked about (grounded is True or False, never None); "confirmed"
    # = SAM3 found nature pixels for it; the rest were attempted but SAM3's
    # confidence never crossed --mask_threshold (recap §9's "agreement with an
    # independent model", never a ground-truth signal).
    run_grounding = bool(header.get("grounding"))
    n_grounding_attempted = n_grounding_confirmed = 0
    clipmatch_top1 = 0
    clipmatch_support = 0
    flat_rows = []  # one row per stored IMAGE for the output CSV (see below)

    for idx, rec in enumerate(records):
        if args.verbose:
            done = idx + 1
            # Every ~10% of the dataset (see utils.crosses_decile), same
            # cadence as CLIPScorer's own encode_text/encode_images progress
            # prints — not once per image, which floods a redirected log.
            if crosses_decile(idx, done, len(records)):
                print(f"🔎 [CLIP] scoring: {done}/{len(records)} images ({done / len(records):.1%})", flush=True)
        objs = rec["objects"]
        finals = rec["object_finals"]
        targets = rec.get("targets", [])
        extraction_parse_failed = bool(rec.get("extraction_parse_failed"))
        if extraction_parse_failed:
            n_extraction_parse_fail += 1
        # Slice this image's own chunk of object embeddings out of the big
        # flat array we built above, using the offsets we remembered.
        obj_slice = slice(offsets[idx], offsets[idx + 1])
        rec_obj_embs = obj_embs_all[obj_slice]
        # Same slice, but from cm_scorer's embedding space (identical to
        # rec_obj_embs whenever cm_scorer is scorer) — used for the ClipMatch
        # anchor-object search below, which must stay in ONE consistent space
        # with candidate_embs/primary_embs_cm.
        rec_obj_embs_cm = obj_embs_all_cm[obj_slice] if run_clipmatch else None

        # Reset every loop iteration (not just inside `if single_label:`) so a
        # COCO/BIG-5 image never accidentally inherits a stale value left over
        # from a PRECEDING single-label image (Python has no block scoping).
        # These are ALL the per-image values a predictions-CSV row can carry;
        # populated below depending on single_label vs COCO/BIG-5,
        # left None wherever a given dataset type has no such value.
        pred_vocab_entry = None
        best_obj_idx = None
        anchor_similarity = None
        best_final = None
        gt_syn = pred_class_synset = pred_node = None
        hier = {"hp": None, "hr": None, "hf1": None}
        wup_sim = None
        clipmatch_correct = None
        clipmatch_pred_similarity = None
        gt_nature_val = pred_nature_val = None
        gt_biotic_val = pred_biotic_val = None
        gt_material_val = pred_material_val = None
        image_gt_nature_val = image_pred_nature_val = None
        image_gt_biotic_val = image_pred_biotic_val = None
        image_gt_material_val = image_pred_material_val = None
        target_matches = None

        # diagnostics
        n_objects_total += len(objs)
        image_n_map_nature = image_n_vlm_nature = image_n_parse_fail = 0
        image_n_label_calls = 0
        for lab, fin in zip(rec["object_labels"], finals):
            n_object_records += 1
            if lab.get("parse_failed"):
                n_parse_fail += 1
                image_n_parse_fail += 1
            if lab.get("vlm_called"):
                n_label_calls += 1
                image_n_label_calls += 1
                # Which schema this call used. Prefer the explicit label_route
                # written by resolve_hybrid_label; fall back to the old
                # `mapped` heuristic for artifacts written before that field
                # existed (back then mapped + vlm_called could ONLY mean the
                # material-only call, since mapped-non-nature got no call at
                # all — that equivalence no longer holds, which is exactly why
                # the explicit field was added).
                route = fin.get("label_route")
                is_material_call = (route == "mapped_nature_material" if route is not None
                                    else bool(fin["mapped"]))
                if is_material_call:
                    n_label_calls_material += 1
                    if lab.get("parse_failed"):
                        n_parse_fail_material += 1
                else:
                    n_label_calls_full += 1
                    if lab.get("parse_failed"):
                        n_parse_fail_full += 1
            if fin["nature_source"] == "wordnet":
                n_map_nature += 1
                image_n_map_nature += 1
            else:
                n_vlm_nature += 1
                image_n_vlm_nature += 1

        # --- reference-free CLIP metrics (all datasets) ---
        image_clipscore = clip_metrics.clipscore(image_embs[idx], caption_embs[idx])
        image_fclip = clip_metrics.f_clipscore(image_embs[idx], caption_embs[idx], rec_obj_embs)
        image_objclip = clip_metrics.object_clipscore(image_embs[idx], rec_obj_embs)
        clipscore_vals.append(image_clipscore)
        fclip_vals.append(image_fclip)
        objclip_vals.append(image_objclip)

        # --- extraction-hit diagnostic (ALL datasets; reporting-only) ---
        # recap §6.3: keep the exact-match extraction rate as a descriptive
        # diagnostic, but it NO LONGER gates or feeds the nature/biotic/material
        # scores (those come from the forced top-1 ClipMatch class on single-
        # label datasets, and from matched-object finals on COCO/BIG-5).
        image_n_gt_targets = image_n_extraction_hits = 0
        for t in targets:
            if t.get("gt_nature") is None:
                continue
            n_gt_targets += 1
            image_n_gt_targets += 1
            if find_matching_object(objs, t) is not None:
                n_extraction_hits += 1
                image_n_extraction_hits += 1

        if single_label:
            # --- ImageNet/Places (single-label) ---
            gt_syn = targets[0].get("synset_id") if targets else None
            pred_class_synset = pred_node = None
            best_obj_idx = None

            # 1. ClipMatch Top-1 Class (The Anchor) — uses cm_scorer's embedding space.
            # PRIMARY text source is the VLM's short summary caption when the
            # artifact has one (run_summary_clipmatch): measured 0.41 vs 0.34
            # top-1 against the full caption on the spot-check run, with 0%
            # truncation vs. the full caption's silent CLIP-tokenizer cutoff —
            # a real re-summarization beats leaving CLIP to truncate arbitrary
            # content. Falls back to the full caption when the artifact
            # predates --summarize_clipmatch_caption (primary_embs_cm is set
            # once, above the per-image loop).
            per_cand_sim, pred_idx = clip_metrics.clipmatch(primary_embs_cm[idx], candidate_embs)
            if pred_idx >= 0:
                pred_vocab_entry = candidate_vocab[pred_idx]
                pred_class_synset = pred_vocab_entry["synset_id"]
                clipmatch_pred_similarity = float(per_cand_sim[pred_idx])

                # 2. Compare Anchor embedding with ALL extracted objects' embeddings
                #    (argmax) — must use rec_obj_embs_cm (SAME embedding space as
                #    candidate_embs), not rec_obj_embs (the reference-free scorer's space).
                if rec_obj_embs_cm.shape[0] > 0:
                    sims_to_pred = rec_obj_embs_cm @ candidate_embs[pred_idx]
                    best_obj_idx = int(sims_to_pred.argmax())
                    # How strongly the winning entity actually matched the
                    # predicted class — a LOW value here flags an anchor
                    # chosen by weak/topical similarity rather than a real
                    # conceptual match, which is exactly when the axis verdict
                    # read off that anchor is least trustworthy.
                    anchor_similarity = float(sims_to_pred[best_obj_idx])
                    
                    # 3. Resolve ONLY the argmax extracted object (best_obj_idx) to a
                    #    WordNet node for hierarchical metrics — no fallback to the
                    #    next-best object if this one fails to resolve. Passing a
                    #    single-candidate list makes resolve_to_wordnet try exactly
                    #    that one object and return None (a mapping failure, scored
                    #    downstream as 0.0 / excluded — see the hierarchical vs
                    #    hierarchical_mapped split below) if it can't be mapped.
                    pred_node = taxonomy_metrics.resolve_to_wordnet(
                        [sims_to_pred[best_obj_idx]], pred_class_synset, [objs[best_obj_idx]]
                    )

            t0 = targets[0]
            # Pull the VLM hybrid resolved labels for the argmax extracted object
            best_final = finals[best_obj_idx] if best_obj_idx is not None else None

            # --- NATURE axis: from the argmax object's hybrid label ---
            if t0.get("gt_nature") is not None:
                gt_n = bool(t0["gt_nature"])
                pn = best_final["final_nature"] if best_final else None
                gt_nature_val, pred_nature_val = gt_n, bool(pn) if pn is not None else (not gt_n)
                nat_true.append(gt_n)
                nat_pred.append(pred_nature_val)

            # --- BIOTIC axis: from the argmax object's hybrid label ---
            if t0.get("gt_biotic") is not None:
                gt_b = bool(t0["gt_biotic"])
                pb = best_final["final_biotic"] if best_final else None
                gt_biotic_val, pred_biotic_val = gt_b, bool(pb) if pb is not None else (not gt_b)
                bio_true.append(gt_b)
                bio_pred.append(pred_biotic_val)

            # --- MATERIAL axis: from the argmax object's hybrid label ---
            if t0.get("gt_material") is not None:
                gt_m = bool(t0["gt_material"])
                pm = best_final["final_material"] if best_final else None
                gt_material_val, pred_material_val = gt_m, bool(pm) if pm is not None else (not gt_m)
                mat_true.append(gt_m)
                mat_pred.append(pred_material_val)

            # --- HIERARCHICAL & ClipMatch metrics ---
            if gt_syn:
                clipmatch_support += 1
                clipmatch_correct = pred_class_synset is not None and pred_class_synset == gt_syn
                if clipmatch_correct:
                    clipmatch_top1 += 1
                
                # Compare GT synset vs the resolved WordNet node of the argmax
                # object. `pred_node is None` means the ClipMatch-predicted
                # object could not be mapped onto ANY WordNet node (mapping
                # failure) — compute_hierarchical_metrics/compute_wup_similarity
                # both score that as 0.0, which the `*_vals` lists keep (failures
                # counted as error). The `*_vals_mapped` lists only collect the
                # cases where mapping SUCCEEDED (pred_node is not None), and the
                # failures are tallied into n_hier_mapping_fail for the rate.
                hier = taxonomy_metrics.compute_hierarchical_metrics(graph, gt_syn, pred_node)
                hp_vals.append(hier["hp"]); hr_vals.append(hier["hr"]); hf1_vals.append(hier["hf1"])

                wup_sim = taxonomy_metrics.compute_wup_similarity(gt_syn, pred_node)
                wup_vals.append(wup_sim)

                if pred_node is None:
                    n_hier_mapping_fail += 1
                else:
                    hp_vals_mapped.append(hier["hp"]); hr_vals_mapped.append(hier["hr"])
                    hf1_vals_mapped.append(hier["hf1"])
                    wup_vals_mapped.append(wup_sim)

                # Same accumulation again, bucketed by this image's GT nature
                # value — skipped (not just bucketed as False) when GT itself
                # is missing (gt_nature_val is None), since that's "unknown",
                # not "no-nature".
                if gt_nature_val is not None:
                    bucket = hier_by_gt_nature[gt_nature_val]
                    bucket["hp"].append(hier["hp"]); bucket["hr"].append(hier["hr"])
                    bucket["hf1"].append(hier["hf1"]); bucket["wup"].append(wup_sim)
                    if pred_node is None:
                        bucket["mapping_fail"] += 1
                    else:
                        bucket["hp_mapped"].append(hier["hp"]); bucket["hr_mapped"].append(hier["hr"])
                        bucket["hf1_mapped"].append(hier["hf1"]); bucket["wup_mapped"].append(wup_sim)

        elif dataset in BIG5_DATASETS:
            # --- BIG-5 (holistic image-level annotation) ---
            # Covers every BIG-5 variant (pooled "big5" and the per-platform
            # "big5_twitter"/"big5_weibo") — they share one GT schema and one
            # scoring rule; only which CSVs get loaded differs.
            # GT here is ONE label per axis for the WHOLE scene (see
            # src.loaders.dataset_loader.load_big5 — a single "scene" target,
            # not a named object to find lexically), so unlike COCO there is
            # no specific object to match against. nature is scored the same
            # image-level OR as always (image_pred_nature: True iff ANY
            # extracted entity is nature). biotic/material use a
            # DIRECTION-AWARE "at least one matching entity" rule instead:
            # whichever value the GT actually is (biotic or abiotic; material
            # or immaterial), correctness means the model output at least one
            # nature-positive entity carrying THAT specific label — an image
            # can contain both a biotic and an abiotic entity at once (e.g. a
            # dog next to a rock) and still be scored correctly against a
            # GT of just "biotic", since only the GT's own direction needs a
            # hit. This deliberately looks at GT to decide which existence
            # check to apply (has_biotic vs has_abiotic), rather than a single
            # fixed-positive-class OR — confirmed as the intended semantics,
            # not a mistake, unlike the nature axis, where "no nature
            # entity output at all" is sufficient for a no-nature GT (there is
            # no equivalent meaningful "found an explicit non-nature entity"
            # signal to require there).
            g_nat = image_gt_nature(targets)
            if g_nat is not None:
                image_gt_nature_val = bool(g_nat)
                image_pred_nature_val = bool(image_pred_nature(finals))
                nat_true.append(image_gt_nature_val)
                nat_pred.append(image_pred_nature_val)

            nature_entities = [fin for fin in finals if fin["final_nature"] is True]
            has_biotic = any(fin["final_biotic"] is True for fin in nature_entities)
            has_abiotic = any(fin["final_biotic"] is False for fin in nature_entities)
            has_material = any(fin["final_material"] is True for fin in nature_entities)
            has_immaterial = any(fin["final_material"] is False for fin in nature_entities)

            # gt_biotic/gt_material are LISTS (src.loaders.dataset_loader.load_big5),
            # not a plain bool|None: usually one element, but BOTH [True, False]
            # when the human coders genuinely disagreed (e.g. "material;
            # immaterial") — per Pau, that image counts as GENUINELY BOTH labels
            # at once, not "neither"/excluded, so EVERY element contributes its
            # own separate GT instance against these SAME extracted entities
            # (one image can therefore add up to two rows to bio_true/bio_pred).
            t0 = targets[0] if targets else {}
            target_matches = [{"class_name": t0.get("class_name")}]
            gt_biotic_vals = t0.get("gt_biotic")
            if gt_biotic_vals is not None:
                pred_biotic_vals = []
                for gt_b in gt_biotic_vals:
                    gt_b = bool(gt_b)
                    pred_b = has_biotic if gt_b else (not has_abiotic)
                    bio_true.append(gt_b)
                    bio_pred.append(pred_b)
                    pred_biotic_vals.append(pred_b)
                target_matches[0].update({"gt_biotic": gt_biotic_vals, "pred_biotic": pred_biotic_vals,
                                          "has_biotic_entity": has_biotic, "has_abiotic_entity": has_abiotic})
                # Flat, top-level mirror of the above (see "image_gt_nature"
                # just below) — the qualitative-review CSV must carry biotic/
                # material the same way it already carries nature, not just
                # buried in the gt_targets JSON blob (see the visualization
                # app fix in the same change: it only ever rendered gt_nature/
                # gt_biotic/gt_material as top-level columns, which BIG-5 never
                # populated, so biotic/material silently never showed up for
                # BIG-5 rows even though the data existed one level down).
                image_gt_biotic_val = _fmt_big5_axis_cell(gt_biotic_vals)
                image_pred_biotic_val = _fmt_big5_axis_cell(pred_biotic_vals)
            gt_material_vals = t0.get("gt_material")
            if gt_material_vals is not None:
                pred_material_vals = []
                for gt_m in gt_material_vals:
                    gt_m = bool(gt_m)
                    pred_m = has_material if gt_m else (not has_immaterial)
                    mat_true.append(gt_m)
                    mat_pred.append(pred_m)
                    pred_material_vals.append(pred_m)
                target_matches[0].update({"gt_material": gt_material_vals, "pred_material": pred_material_vals,
                                          "has_material_entity": has_material, "has_immaterial_entity": has_immaterial})
                image_gt_material_val = _fmt_big5_axis_cell(gt_material_vals)
                image_pred_material_val = _fmt_big5_axis_cell(pred_material_vals)
        else:
            # --- COCO (multi-label): image-level nature OR + matched-object
            #     biotic/material, via lexical GT matching (find_matching_object) ---
            # TODO(grounding-pipeline, recap §6.4): replace this lexical
            # matching with Hungarian box-IoU assignment (IoU>=0.5) once
            # Grounding DINO 1.5 provides predicted boxes — matched GT boxes
            # score bio/material/nature as usual; unmatched GT boxes are
            # penalized as wrong; unmatched PREDICTED boxes are excluded, NOT
            # penalized (COCO's 80 classes are a curated subset, so an extra
            # real object is not a hallucination). Not implementable until the
            # Grounding pipeline exists.
            g_nat = image_gt_nature(targets)
            if g_nat is not None:
                image_gt_nature_val = bool(g_nat)
                image_pred_nature_val = bool(image_pred_nature(finals))
                nat_true.append(image_gt_nature_val)
                nat_pred.append(image_pred_nature_val)
            # Per-target matched-object detail (multi-label datasets can have
            # several GT targets per image), kept for the CSV's enriched
            # "gt_targets" field so each target's own matched object + scored
            # biotic/material prediction is visible, not just the aggregate.
            target_matches = []
            for t in targets:
                if t.get("gt_nature") is None:
                    continue
                mi = find_matching_object(objs, t)
                match_info = {"class_name": t.get("class_name"), "matched_object_idx": mi,
                             "matched_object_text": objs[mi] if mi is not None else None}
                if mi is None:
                    target_matches.append(match_info)
                    continue
                fin = finals[mi]
                if t.get("gt_biotic") is not None:
                    gt_b = bool(t["gt_biotic"])
                    pred_b = fin["final_biotic"]
                    pred_b = bool(pred_b) if pred_b is not None else (not gt_b)
                    bio_true.append(gt_b)
                    bio_pred.append(pred_b)
                    match_info["gt_biotic"], match_info["pred_biotic"] = gt_b, pred_b
                if t.get("gt_material") is not None:
                    gt_m = bool(t["gt_material"])
                    pred_m = fin["final_material"]
                    pred_m = bool(pred_m) if pred_m is not None else (not gt_m)
                    mat_true.append(gt_m)
                    mat_pred.append(pred_m)
                    match_info["gt_material"], match_info["pred_material"] = gt_m, pred_m
                target_matches.append(match_info)

        # One CSV row PER IMAGE (not per object) — everything needed to
        # spot-check a single image's whole prediction lives on one line:
        # the caption, every extracted object with its own classification and
        # mapping status, the GT target(s), and (single-label datasets only)
        # the ClipMatch top-1 predicted object and its classification.
        # "objects" and "gt_targets" are JSON-encoded since a CSV cell can't
        # hold a nested list directly — json.loads() them back when reading.
        # Every image gets a row (no subsampling) — the full per-image record
        # is a first-class output the Grounding pipeline consumes downstream.
        objects_json = json.dumps([
            {
                "text": obj,
                "mapped": fin["mapped"], "mapped_synset": fin["mapped_synset"],
                "nature": fin["final_nature"], "biotic": fin["final_biotic"],
                "material": fin["final_material"],
                "nature_source": fin["nature_source"], "biotic_source": fin["biotic_source"],
                # Which labeling route this object took — "human_exclusion"
                # (no VLM call), "mapped_nature_material" (MaterialResponse),
                # or "vlm_full" (TaxonomyResponse). None on older artifacts.
                "label_route": fin.get("label_route"),
                "parse_failed": lab.get("parse_failed"),
                # Stage-3 labeling justification (TaxonomyResponse's combined
                # nature_reasoning+sub_axes_reasoning, or MaterialResponse's own
                # reasoning) and whether this object even got its own VLM call
                # at all (False = mapped non-nature, skipped the VLM entirely —
                # see label_objects_batch) — both already stored in the .jsonl
                # artifact's object_labels, previously not surfaced here.
                "reasoning": lab.get("reasoning"),
                "vlm_called": lab.get("vlm_called"),
            }
            for obj, lab, fin in zip(objs, rec["object_labels"], finals)
        ])
        # "gt_targets" carries the raw target dicts, PLUS (multi-label
        # datasets only) "target_matches" — which object matched each
        # target and what it scored on biotic/material (see the coco/big5
        # branch above). None on single-label datasets, where there's only
        # ever one target and it's covered by the clipmatch_* columns.
        gt_targets_json = json.dumps({"targets": targets, "target_matches": target_matches})
        # Grounding-pipeline results, present only when src/grounding_pipeline.py
        # has enriched this artifact (see run_pipeline.py / run_grounding_pipeline.py)
        # — None/blank on a VLM-only artifact, so scoring one still works fine.
        # mask_rle IS included here (pycocotools RLE — a small JSON dict, not
        # the decoded pixel array) so the predictions CSV is self-sufficient
        # for qualitative review (decode_rle in grounding_pipeline.py turns it
        # back into a boolean mask for overlay plotting) without needing the
        # raw .jsonl artifact open alongside it.
        object_groundings = rec.get("object_groundings")
        groundings_json = json.dumps([
            {"object": g["object"], "prompt": g["prompt"], "is_nature": g["is_nature"],
             "grounded": g["grounded"], "pixel_count": g["pixel_count"], "mask_rle": g["mask_rle"]}
            for g in object_groundings
        ]) if object_groundings is not None else None
        # Nature entities SAM3 actually ATTEMPTED (is_nature True implies an
        # attempt was made — see _nature_object_indices) but whose predicted
        # mask never crossed --mask_threshold: `grounded is False` distinguishes
        # this from `grounded is None` (never attempted at all, not nature).
        # Surfaced per image for qualitative review (which specific objects the
        # VLM claimed but SAM3 couldn't confirm in pixels) and accumulated below
        # into a run-level rate (recap §9: agreement with an independent model,
        # never ground truth — a low confirmation rate flags VLM hallucination
        # OR an under-sensitive --mask_threshold, not "ground truth says wrong").
        ungrounded_objects = ([g["object"] for g in object_groundings
                               if g["is_nature"] and g["grounded"] is False]
                              if object_groundings is not None else None)
        image_n_attempted = image_n_confirmed = None
        if object_groundings is not None:
            image_n_attempted = sum(1 for g in object_groundings if g["is_nature"])
            image_n_confirmed = sum(1 for g in object_groundings if g["grounded"] is True)
            n_grounding_attempted += image_n_attempted
            n_grounding_confirmed += image_n_confirmed
        flat_rows.append({
            "image_path": rec["image_path"],
            "dataset": dataset,
            "model": header.get("model"),
            "caption": rec["caption"],
            "extraction_reasoning": rec.get("extraction_reasoning"),
            # True iff Stage 2's whole-image object-list call failed to parse
            # at all (objects/n_objects above are then simply [] / 0 as a
            # fallback — NOT a genuine "the model found nothing"). Distinct
            # from any per-object parse_failed below, which is Stage 3
            # (per-object taxonomy labeling), a completely separate call.
            "extraction_parse_failed": extraction_parse_failed,
            "objects": objects_json,
            "gt_targets": gt_targets_json,
            "n_objects": len(objs),
            "clipscore": image_clipscore,
            "f_clipscore": image_fclip,
            "object_clipscore": image_objclip,
            "object_groundings": groundings_json,
            "nature_relevance_score_coverage_ratio": rec.get("nature_relevance_score_coverage_ratio"),
            "nature_relevance_score_center_weighted": rec.get("nature_relevance_score_center_weighted"),
            "grounding_ungrounded_objects": json.dumps(ungrounded_objects) if ungrounded_objects is not None else None,
            "grounding_nature_entities_attempted_image": image_n_attempted,
            "grounding_confirmed_image": image_n_confirmed,
            "grounding_confirmation_rate_image": (image_n_confirmed / image_n_attempted)
                                                  if image_n_attempted else None,
            "wordnet_mapping_rate_image": (image_n_map_nature / (image_n_map_nature + image_n_vlm_nature))
                                           if (image_n_map_nature + image_n_vlm_nature) else None,
            "parse_failure_count_image": image_n_parse_fail,
            # Labeling calls actually issued for THIS image's objects, so the
            # count above can be read as a rate per image (mapped non-nature
            # objects never reach the VLM — see the diagnostics block).
            "label_vlm_calls_image": image_n_label_calls,
            "label_parse_failure_rate_image": (image_n_parse_fail / image_n_label_calls)
                                               if image_n_label_calls else None,
            "extraction_hit_rate_image": (image_n_extraction_hits / image_n_gt_targets)
                                          if image_n_gt_targets else None,
            # image-level nature (COCO/BIG-5 only; single-label's nature
            # verdict is the clipmatch_pred_nature column below instead)
            "image_gt_nature": image_gt_nature_val,
            "image_pred_nature": image_pred_nature_val,
            # image-level biotic/material (BIG-5 only — its GT is one
            # holistic scene label, never a per-object match; see
            # _fmt_big5_axis_cell). COCO has no equivalent single scalar here
            # since its GT is multiple distinct objects with independently
            # scored biotic/material (see the "gt_targets" JSON's
            # per-target target_matches instead).
            "image_gt_biotic": image_gt_biotic_val,
            "image_pred_biotic": image_pred_biotic_val,
            "image_gt_material": image_gt_material_val,
            "image_pred_material": image_pred_material_val,
            # single-label (ImageNet/Places) axis verdicts
            "gt_nature": gt_nature_val, "pred_nature": pred_nature_val,
            "gt_biotic": gt_biotic_val, "pred_biotic": pred_biotic_val,
            "gt_material": gt_material_val, "pred_material": pred_material_val,
            # ClipMatch (PRIMARY) — the metric actually used for scoring: the
            # VLM's summary caption when available (run_summary_clipmatch),
            # else the full caption. See "clip_models"/"clipmatch_primary_source"
            # in the run summary for which one this run used.
            "clipmatch_pred_class": pred_vocab_entry["class_name"] if pred_vocab_entry else None,
            "clipmatch_pred_synset": pred_vocab_entry["synset_id"] if pred_vocab_entry else None,
            "clipmatch_pred_similarity": clipmatch_pred_similarity,
            "clipmatch_pred_nature": best_final["final_nature"] if best_final else None,
            "clipmatch_pred_biotic": best_final["final_biotic"] if best_final else None,
            "clipmatch_pred_material": best_final["final_material"] if best_final else None,
            "clipmatch_top1_correct": clipmatch_correct,
            "clipmatch_summary_caption": summary_texts[idx] if summary_texts is not None else None,
            "clipmatch_summary_caption_token_length": summary_token_counts[idx] if summary_token_counts is not None else None,
            "gt_synset": gt_syn,
            # The ANCHOR: which extracted entity won the CLIP-similarity
            # argmax against the ClipMatch-predicted class. This is the object
            # whose own hybrid label supplies nature/biotic/material on
            # single-label datasets, AND the only phrase resolve_to_wordnet is
            # given for hP/hR — so `resolved_pred_synset` alone is not
            # interpretable without it (a null there could mean either "no
            # anchor was found" or "this specific phrase didn't resolve", and
            # you can't tell which, or which phrase was blamed).
            "anchor_object": objs[best_obj_idx] if best_obj_idx is not None else None,
            "anchor_object_idx": best_obj_idx,
            "anchor_object_similarity": anchor_similarity,
            "resolved_pred_synset": pred_node,
            "hierarchical_precision": hier["hp"], "hierarchical_recall": hier["hr"],
            "hierarchical_f1": hier["hf1"],
            "wup_similarity": wup_sim,
        })

    # ---- Assemble summary ----
    n_images = len(records)
    summary = {
        "dataset": dataset, "model": header.get("model"), "n_images": n_images,
        "diagnostics": {
            "objects_per_image": (n_objects_total / n_images) if n_images else 0.0,
            "extraction_hit_rate": (n_extraction_hits / n_gt_targets) if n_gt_targets else 0.0,
            # Stage-2 (whole-image object-list) parse failures — an image
            # whose structured extraction call never parsed at all, silently
            # left with an empty object list. Distinct from the label_parse_*
            # numbers below, which are Stage-3 per-object failures.
            "extraction_parse_failures": n_extraction_parse_fail,
            "extraction_parse_failure_rate": (n_extraction_parse_fail / n_images) if n_images else 0.0,
            "wordnet_mapping_rate": (n_map_nature / n_object_records) if n_object_records else 0.0,
            "vlm_fallback_rate": (n_vlm_nature / n_object_records) if n_object_records else 0.0,
            # LABELING-stage schema-parse failures, over the objects the VLM was
            # actually asked about — NOT over every extracted object (mapped
            # non-nature objects never reach the VLM at all, and counting them
            # in the denominator would understate the real failure rate). The
            # full/material split matters because the two calls use different
            # schemas: TaxonomyResponse (three axes at once, unmapped objects)
            # vs MaterialResponse (one axis, mapped-nature objects).
            "label_vlm_calls": n_label_calls,
            "label_parse_failures": n_parse_fail,
            "label_parse_failure_rate": (n_parse_fail / n_label_calls) if n_label_calls else 0.0,
            "label_calls_full": n_label_calls_full,
            "label_parse_failures_full": n_parse_fail_full,
            "label_parse_failure_rate_full": (n_parse_fail_full / n_label_calls_full)
                                              if n_label_calls_full else 0.0,
            "label_calls_material": n_label_calls_material,
            "label_parse_failures_material": n_parse_fail_material,
            "label_parse_failure_rate_material": (n_parse_fail_material / n_label_calls_material)
                                                  if n_label_calls_material else 0.0,
            # How much of the extracted-object set the mapping answered outright
            # (i.e. never needed a labeling call) — context for the rates above.
            "label_vlm_call_rate": (n_label_calls / n_object_records) if n_object_records else 0.0,
        },
        "reference_free": {
            # Mean AND population std (same convention as the hierarchical
            # metrics): a mean alone can't distinguish "every image scores
            # close to this" from a widely-spread or bimodal distribution.
            "clipscore": _mean(clipscore_vals),
            "clipscore_std": _std(clipscore_vals),
            "f_clipscore": _mean(fclip_vals),
            "f_clipscore_std": _std(fclip_vals),
            "object_clipscore": _mean(objclip_vals),
            "object_clipscore_std": _std(objclip_vals),
        },
        "clip_models": {
            "reference_free": args.clipscore_model,
            "clipmatch": clipmatch_model_name if run_clipmatch else None,
            "clipmatch_primary_source": ("summary_caption" if run_summary_clipmatch else "caption")
                                         if run_clipmatch else None,
            "clipmatch_candidate_text": ("wordnet_definition" if args.use_wordnet_definitions_clipmatch
                                         else f"prompt_ensemble_{dataset}") if run_clipmatch else None,
            "use_inflect_for_clipmatch": args.use_inflect_for_clipmatch,
        },
        "nature": _binary_metrics(nat_true, nat_pred),
        "biotic_matched": _binary_metrics(bio_true, bio_pred),
        "material_matched": _binary_metrics(mat_true, mat_pred),
        "axis_scoring_note": ("imagenet/places: nature/biotic come from the ClipMatch TOP-1 "
                              "predicted class's OWN taxonomy position — ClipMatch predicts the "
                              "class from a text embedding's CLIP similarity to candidate_vocab "
                              "(which is restricted to classes mapped into the graph — see "
                              "get_candidate_vocab); that text is the VLM's own short summary "
                              "caption when the artifact was produced with "
                              "--summarize_clipmatch_caption (measured 0.41 vs 0.34 top-1 against "
                              "the full caption, with far less CLIP-tokenizer truncation — see "
                              "clip_models.clipmatch_primary_source), else the WHOLE raw caption; "
                              "material is ALWAYS the VLM's own judgment "
                              "(never mapped), taken from the extracted object most representative "
                              "of that predicted class. big5 (holistic, one GT label per image): "
                              "nature is the same image-level OR as always; biotic/material are "
                              "scored as 'did the model output at least one nature-positive entity "
                              "matching the GT's own direction' (has_biotic when GT=biotic, "
                              "has_abiotic when GT=abiotic, symmetrically for material) — an image "
                              "can contain both a biotic AND an abiotic entity and still score "
                              "correctly against a GT of just one of them. coco: image-level nature "
                              "(OR) + matched-object biotic/material via lexical GT matching (box-IoU "
                              "is future work, recap §6.4)."),
        "material_caveat": ("Material GT for imagenet/coco/places is the heuristic "
                            "gt_material=True default (real photos); only BIG-5 has genuine "
                            "material GT. Predicted material is always the VLM's judgment "
                            "(never mapped), so on imagenet/places it is scored against that "
                            "heuristic GT default — BIG-5 is the only genuine material benchmark."),
    }
    if run_grounding:
        # recap §9: this is agreement with an INDEPENDENT model (SAM3), never
        # ground truth — a low confirmation_rate flags either VLM hallucination
        # (claimed an entity SAM3 can't find in pixels) or an under-sensitive
        # --mask_threshold, not "the model was wrong" in a GT sense.
        summary["grounding"] = {
            "nature_entities_attempted": n_grounding_attempted,
            "confirmed": n_grounding_confirmed,
            "confirmation_rate": (n_grounding_confirmed / n_grounding_attempted)
                                  if n_grounding_attempted else 0.0,
            "sam3_model": header["grounding"].get("sam3_model"),
            "mask_threshold": header["grounding"].get("mask_threshold"),
        }
    if run_clipmatch:
        # Token stats for whichever text is PRIMARY this run: the summary
        # caption when the artifact has one, else the raw caption itself
        # (only the primary text is encoded/measured at all — there is no
        # secondary raw-caption comparison).
        primary_truncated = n_summary_truncated if run_summary_clipmatch else n_caption_truncated
        primary_mean_tokens = summary_token_mean if run_summary_clipmatch else caption_token_mean
        summary["clipmatch"] = {
            "top1_accuracy": (clipmatch_top1 / clipmatch_support) if clipmatch_support else 0.0,
            "support": clipmatch_support,
            "truncation_rate": (primary_truncated / n_images)
                                if (primary_truncated is not None and n_images) else None,
            "truncated_count": primary_truncated,
            "context_length": cm_scorer.context_length,
            "mean_token_length": primary_mean_tokens,
        }
        # Candidate-vocab text diagnostic: token stats for whichever text was
        # used to build candidate_embs (the 80/15-template prompt ensemble per
        # class by default, or WordNet-definition prose under
        # --use_wordnet_definitions_clipmatch — see
        # clip_models.clipmatch_candidate_text above). candidate_texts is the
        # FLAT list actually encoded (n_candidates * n_templates under the
        # ensemble path, just n_candidates under wordnet-definitions), so the
        # truncation rate is reported over that same flat count, not the class
        # count, while n_candidates itself still reports the class count.
        n_candidates = len(candidate_vocab)
        n_candidate_texts = len(candidate_texts)
        summary["clipmatch_candidates"] = {
            "n_candidates": n_candidates,
            "n_candidate_texts": n_candidate_texts,
            "mean_token_length": candidate_token_mean,
            "truncation_rate": (n_candidate_truncated / n_candidate_texts)
                                if (n_candidate_truncated is not None and n_candidate_texts) else None,
            "truncated_count": n_candidate_truncated,
            "context_length": cm_scorer.context_length,
        }
        # Two hierarchical views over the SAME images (see the accumulation
        # comment above): "hierarchical" penalizes WordNet-mapping failures as
        # 0.0 (support = every ClipMatch-scored image); "hierarchical_mapped"
        # drops those failures (support = only images whose predicted object
        # mapped into WordNet). mapping_failure_rate is the fraction of scored
        # images where the predicted object could not be mapped at all.
        summary["hierarchical"] = {"hp": _mean(hp_vals), "hr": _mean(hr_vals),
                                   "hf1": _mean(hf1_vals), "wup": _mean(wup_vals),
                                   "hp_std": _std(hp_vals), "hr_std": _std(hr_vals),
                                   "hf1_std": _std(hf1_vals), "wup_std": _std(wup_vals),
                                   "support": len(hf1_vals),
                                   "mapping_failure_rate": (n_hier_mapping_fail / len(hf1_vals))
                                                           if hf1_vals else 0.0,
                                   "mapping_failures": n_hier_mapping_fail}
        summary["hierarchical_mapped"] = {"hp": _mean(hp_vals_mapped), "hr": _mean(hr_vals_mapped),
                                          "hf1": _mean(hf1_vals_mapped), "wup": _mean(wup_vals_mapped),
                                          "hp_std": _std(hp_vals_mapped), "hr_std": _std(hr_vals_mapped),
                                          "hf1_std": _std(hf1_vals_mapped), "wup_std": _std(wup_vals_mapped),
                                          "support": len(hf1_vals_mapped)}
        # Same two views, additionally split by whether THIS IMAGE's own GT is
        # nature or no-nature — see hier_by_gt_nature's accumulation comment
        # above for why (one pooled macro-average can hide a model that's
        # much better at one GT direction than the other).
        hier_nat, hier_nat_mapped = _hier_bucket_summary(hier_by_gt_nature[True])
        hier_nonat, hier_nonat_mapped = _hier_bucket_summary(hier_by_gt_nature[False])
        summary["hierarchical_by_gt_nature"] = {"nature": hier_nat, "no_nature": hier_nonat}
        summary["hierarchical_mapped_by_gt_nature"] = {"nature": hier_nat_mapped,
                                                        "no_nature": hier_nonat_mapped}

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
    # stay easy to tell apart later. The shared results JSON (--output_file)
    # sits directly in that folder; per-model raw outputs are split by file
    # type one level down — the predictions CSV here goes in
    # PREDICTIONS_SUBDIR, the .jsonl artifact from _resolve_responses_file
    # goes in RESPONSES_SUBDIR.
    out_dir = Path(args.results_dir)
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args.output_file).name
    update_results_store(out_path, dataset=dataset, model=header.get("model"), metrics=summary)
    # Per-IMAGE nature/biotic/material GT composition of THIS run's sampled
    # dataset (recap: sampling is deterministic — a fixed --max_samples always
    # yields the same subset — so this is stable across reruns of the same
    # config). Keyed by --max_samples so different configurations (e.g. 1000 vs
    # the full dataset) accumulate side by side instead of overwriting.
    # IMAGE-level, not distinct-class: BIG-5 has no class vocabulary (every
    # image shares one holistic "scene" target), so the old class breakdown
    # collapsed all of BIG-5 to `total_classes: 1`. See compute_image_stats.
    image_stats = compute_image_stats(records)
    config_key = str(args.max_samples) if args.max_samples is not None else "full"
    update_dataset_image_stats(out_path, dataset=dataset, config_key=config_key, stats=image_stats)
    if flat_rows:
        # Include dataset + model in the filename — otherwise every model run
        # writes to the same "<stem>_predictions.csv" and each rerun (e.g. a
        # different VLM on the same dataset) silently overwrites the previous
        # model's predictions. Grouped into PREDICTIONS_SUBDIR (separate from
        # RESPONSES_SUBDIR, where that model's .jsonl artifact lives — see
        # _resolve_responses_file), not next to the shared results JSON.
        # model_slug uses ONLY the model NAME (see _model_slug's rationale),
        # falling back to the full family/name "model" string for older
        # artifacts written before the header carried "model_name" separately.
        predictions_dir = out_dir / PREDICTIONS_SUBDIR
        predictions_dir.mkdir(parents=True, exist_ok=True)
        model_slug = (header.get("model_name") or header.get("model", "unknown_model")).replace("/", "_")
        csv_path = predictions_dir / f"{out_path.stem}_{dataset}_{model_slug}_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader(); w.writerows(flat_rows)
        print(f"💾 [score] wrote {out_path} and {csv_path} ({len(flat_rows)} images stored)")

    if args.wandb:
        _log_wandb(args, summary, run_clipmatch)


def _print_summary(s, run_clipmatch):
    """Pretty-print the final metrics summary to the console."""
    d = s["diagnostics"]
    print("\n" + "=" * 60)
    print(f"📊 VLM PIPELINE: {s['model']} on {s['dataset'].upper()}  ({s['n_images']} images)")
    print("=" * 60)
    print(f"Objects/image: {d['objects_per_image']:.2f} | Extraction-hit: {d['extraction_hit_rate']:.1%}")
    print(f"Extraction parse-fail: {d['extraction_parse_failure_rate']:.1%} "
          f"({d['extraction_parse_failures']}/{s['n_images']} images)")
    print(f"WordNet-mapping: {d['wordnet_mapping_rate']:.1%} | VLM-fallback: {d['vlm_fallback_rate']:.1%}")
    print(f"Labeling parse-fail: {d['label_parse_failure_rate']:.1%} "
          f"({d['label_parse_failures']}/{d['label_vlm_calls']} VLM calls) — "
          f"material-only {d['label_parse_failure_rate_material']:.1%} "
          f"({d['label_parse_failures_material']}/{d['label_calls_material']})")
    rf = s["reference_free"]
    print(f"CLIPScore: {rf['clipscore']:.4f}±{rf.get('clipscore_std', 0.0):.4f} | "
          f"F-CLIPScore: {rf['f_clipscore']:.4f}±{rf.get('f_clipscore_std', 0.0):.4f} | "
          f"Object-CLIPScore: {rf['object_clipscore']:.4f}±{rf.get('object_clipscore_std', 0.0):.4f}")
    neg_labels = {"nature": "no_nature", "biotic_matched": "abiotic", "material_matched": "immaterial"}
    axis_names = {"nature": "nature", "biotic_matched": "life category", "material_matched": "tangibility"}
    for axis in ("nature", "biotic_matched", "material_matched"):
        m = s[axis]
        pos_label = axis.split('_')[0]
        print(f"\n--- {axis_names[axis]} (support {m['support']} | "
              f"{m['n_pos']} {pos_label} | {m['n_neg']} {neg_labels[axis]}) ---")
        print(f"Acc {m['accuracy']:.4f}")
        print(f"  {axis.split('_')[0]:<12} (pos) P {m['precision']:.4f} | R {m['recall']:.4f} | F1 {m['f1']:.4f}")
        print(f"  {neg_labels[axis]:<12} (neg) P {m['precision_neg']:.4f} | R {m['recall_neg']:.4f} | F1 {m['f1_neg']:.4f}")
    g = s.get("grounding")
    if g is not None:
        print(f"\n--- Grounding [{g['sam3_model']}, threshold {g['mask_threshold']}] "
              f"(nature entities attempted {g['nature_entities_attempted']}) ---")
        print(f"Confirmation rate: {g['confirmation_rate']:.1%} "
              f"({g['confirmed']}/{g['nature_entities_attempted']}) — agreement with SAM3, not ground truth")
    if run_clipmatch:
        primary_label = "summary-caption" if s["clip_models"]["clipmatch_primary_source"] == "summary_caption" else "caption-based"
        cm = s["clipmatch"]
        trunc_str = f"{cm['truncation_rate']:.1%}" if cm["truncation_rate"] is not None else "n/a"
        mean_tok_str = f"{cm['mean_token_length']:.1f}" if cm["mean_token_length"] is not None else "n/a"
        print(f"\n--- ClipMatch [{primary_label}, PRIMARY] (support {cm['support']}) ---")
        print(f"Top-1: {cm['top1_accuracy']:.4f} | Mean token length: {mean_tok_str} "
              f"| Truncation rate (>= {cm['context_length']} tok): "
              f"{trunc_str} ({cm['truncated_count'] if cm['truncated_count'] is not None else 'n/a'}/{s['n_images']})")
        cv = s["clipmatch_candidates"]
        cv_trunc_str = f"{cv['truncation_rate']:.1%}" if cv["truncation_rate"] is not None else "n/a"
        cv_mean_tok_str = f"{cv['mean_token_length']:.1f}" if cv["mean_token_length"] is not None else "n/a"
        print(f"\n--- ClipMatch candidate vocab [{s['clip_models']['clipmatch_candidate_text']}] "
              f"({cv['n_candidates']} classes) ---")
        print(f"Mean token length: {cv_mean_tok_str} | Truncation rate (>= {cv['context_length']} tok): "
              f"{cv_trunc_str} ({cv['truncated_count'] if cv['truncated_count'] is not None else 'n/a'}/{cv['n_candidate_texts']})")
        h = s["hierarchical"]
        hm = s["hierarchical_mapped"]
        print(f"--- Hierarchical (support {h['support']}) ---")
        print(f"[failures as error]  hP {h['hp']:.4f}±{h['hp_std']:.4f} | hR {h['hr']:.4f}±{h['hr_std']:.4f} "
              f"| hF1 {h['hf1']:.4f}±{h['hf1_std']:.4f} | Wu-Palmer {h['wup']:.4f}±{h['wup_std']:.4f}")
        print(f"WordNet mapping-failure rate: {h['mapping_failure_rate']:.1%} "
              f"({h['mapping_failures']}/{h['support']})")
        print(f"[mapped only, support {hm['support']}]  hP {hm['hp']:.4f}±{hm['hp_std']:.4f} "
              f"| hR {hm['hr']:.4f}±{hm['hr_std']:.4f} | hF1 {hm['hf1']:.4f}±{hm['hf1_std']:.4f} "
              f"| Wu-Palmer {hm['wup']:.4f}±{hm['wup_std']:.4f}")
        for split_label, key in (("GT=nature", "nature"), ("GT=no-nature", "no_nature")):
            hn = s["hierarchical_by_gt_nature"][key]
            hnm = s["hierarchical_mapped_by_gt_nature"][key]
            print(f"--- Hierarchical [{split_label}] (support {hn['support']}) ---")
            print(f"  [failures as error]  hP {hn['hp']:.4f}±{hn['hp_std']:.4f} "
                  f"| hR {hn['hr']:.4f}±{hn['hr_std']:.4f} | hF1 {hn['hf1']:.4f}±{hn['hf1_std']:.4f} "
                  f"| Wu-Palmer {hn['wup']:.4f}±{hn['wup_std']:.4f}")
            print(f"  [mapped only, support {hnm['support']}]  hP {hnm['hp']:.4f}±{hnm['hp_std']:.4f} "
                  f"| hR {hnm['hr']:.4f}±{hnm['hr_std']:.4f} | hF1 {hnm['hf1']:.4f}±{hnm['hf1_std']:.4f} "
                  f"| Wu-Palmer {hnm['wup']:.4f}±{hnm['wup_std']:.4f}")
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
        "ExtractionParseFailureRate": summary["diagnostics"]["extraction_parse_failure_rate"],
        "WordNetMappingRate": summary["diagnostics"]["wordnet_mapping_rate"],
        "LabelParseFailureRate": summary["diagnostics"]["label_parse_failure_rate"],
        "LabelParseFailureRate/MaterialOnly": summary["diagnostics"]["label_parse_failure_rate_material"],
        "CLIPScore": summary["reference_free"]["clipscore"],
        "F-CLIPScore": summary["reference_free"]["f_clipscore"],
        "Object-CLIPScore": summary["reference_free"]["object_clipscore"],
        "Nature/F1": summary["nature"]["f1"], "Nature/Accuracy": summary["nature"]["accuracy"],
        "Nature/F1_NoNature": summary["nature"]["f1_neg"],
        "Nature/Support": summary["nature"]["support"],
        "Nature/N_Nature": summary["nature"]["n_pos"], "Nature/N_NoNature": summary["nature"]["n_neg"],
        "Biotic/F1": summary["biotic_matched"]["f1"], "Biotic/F1_Abiotic": summary["biotic_matched"]["f1_neg"],
        "Biotic/Support": summary["biotic_matched"]["support"],
        "Biotic/N_Biotic": summary["biotic_matched"]["n_pos"], "Biotic/N_Abiotic": summary["biotic_matched"]["n_neg"],
        "Material/F1": summary["material_matched"]["f1"], "Material/F1_Immaterial": summary["material_matched"]["f1_neg"],
        "Material/Support": summary["material_matched"]["support"],
        "Material/N_Material": summary["material_matched"]["n_pos"], "Material/N_Immaterial": summary["material_matched"]["n_neg"],
    }
    if run_clipmatch:
        log["ClipMatch/Top1"] = summary["clipmatch"]["top1_accuracy"]
        if summary["clipmatch"]["truncation_rate"] is not None:
            log["ClipMatch/TruncationRate"] = summary["clipmatch"]["truncation_rate"]
        if summary["clipmatch"]["mean_token_length"] is not None:
            log["ClipMatch/MeanTokenLength"] = summary["clipmatch"]["mean_token_length"]
        cv = summary["clipmatch_candidates"]
        if cv["mean_token_length"] is not None:
            log["ClipMatch_Candidates/MeanTokenLength"] = cv["mean_token_length"]
        if cv["truncation_rate"] is not None:
            log["ClipMatch_Candidates/TruncationRate"] = cv["truncation_rate"]
        log["Hierarchical/hF1"] = summary["hierarchical"]["hf1"]
        log["Hierarchical/hF1_Std"] = summary["hierarchical"]["hf1_std"]
        log["Hierarchical/WuPalmer"] = summary["hierarchical"]["wup"]
        log["Hierarchical/WuPalmer_Std"] = summary["hierarchical"]["wup_std"]
        log["Hierarchical/MappingFailureRate"] = summary["hierarchical"]["mapping_failure_rate"]
        log["Hierarchical/hF1_MappedOnly"] = summary["hierarchical_mapped"]["hf1"]
        log["Hierarchical/hF1_MappedOnly_Std"] = summary["hierarchical_mapped"]["hf1_std"]
        log["Hierarchical/WuPalmer_MappedOnly"] = summary["hierarchical_mapped"]["wup"]
        log["Hierarchical/WuPalmer_MappedOnly_Std"] = summary["hierarchical_mapped"]["wup_std"]
        for key, wandb_key in (("nature", "GTNature"), ("no_nature", "GTNoNature")):
            log[f"Hierarchical/hF1_{wandb_key}"] = summary["hierarchical_by_gt_nature"][key]["hf1"]
            log[f"Hierarchical/WuPalmer_{wandb_key}"] = summary["hierarchical_by_gt_nature"][key]["wup"]
            log[f"Hierarchical/hF1_{wandb_key}_MappedOnly"] = summary["hierarchical_mapped_by_gt_nature"][key]["hf1"]
            log[f"Hierarchical/WuPalmer_{wandb_key}_MappedOnly"] = summary["hierarchical_mapped_by_gt_nature"][key]["wup"]
    if summary.get("grounding") is not None:
        log["Grounding/ConfirmationRate"] = summary["grounding"]["confirmation_rate"]
        log["Grounding/NatureEntitiesAttempted"] = summary["grounding"]["nature_entities_attempted"]
        log["Grounding/Confirmed"] = summary["grounding"]["confirmed"]
    wandb.log(log)
    wandb.finish()


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser():
    """Define every command-line flag this script accepts. Grouped into:
    stage/dataset selection, taxonomy/context files, per-dataset paths, VLM
    settings (only used by --stage infer/all), CLIP settings (only used by
    --stage score/all), and shared output options.

    Split out from parse_args() (which just calls this + .parse_args()) so
    main()'s --stage all can also build a parser purely to INTROSPECT its
    defaults via _args_to_cli, when reconstructing CLI flags to re-invoke
    this same script as a real subprocess for --stage infer/score."""
    p = argparse.ArgumentParser(description="Two-phase baseline VLM pipeline (infer -> score).")
    p.add_argument("--stage", choices=["all", "infer", "score"], default="all",
                   help="'all' (default) runs the full end-to-end pipeline in one process, "
                        "releasing the VLM's GPU memory before loading CLIP for scoring. "
                        "'infer'/'score' split the two halves across separate invocations.")
    # "big5" pools every configured BIG-5 platform into one run; "big5_twitter"
    # / "big5_weibo" restrict it to one. Since the results store and the
    # predictions CSV are keyed by dataset name, the per-platform names are
    # what keep Twitter and Weibo numbers separate rather than pooled.
    p.add_argument("--dataset", required=True,
                   choices=["coco", "imagenet", "places365", *BIG5_DATASETS])
    p.add_argument("--responses_file", type=str, default=None,
                   help="Intermediate artifact: written by infer, read by score. Default: "
                        f"'vlm_responses_<model_slug>.jsonl' inside --results_dir/--run_name/"
                        f"{RESPONSES_SUBDIR} (all models' .jsonl artifacts grouped together "
                        f"there; predictions CSVs go in --results_dir/--run_name/"
                        f"{PREDICTIONS_SUBDIR} instead; --output_file's shared results JSON "
                        "stays directly in --results_dir/--run_name). Pass an explicit path to "
                        "override (e.g. to write it somewhere else, or to point --stage score "
                        "at a specific prior artifact). ALWAYS retained on disk (never deleted, "
                        "under any --stage) — it carries every raw VLM response plus the "
                        "resolved per-object labels, "
                        "which the Grounding pipeline needs downstream, so this artifact is a "
                        "first-class output, not just an internal handoff.")

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
    # BIG-5 — one majority-vote GT CSV per platform split, plus one images
    # folder per PLATFORM (Twitter and Weibo live in separate folders on the
    # cluster; each platform's own splits share its folder). All splits use
    # the same column schema and differ only in how many per-image annotation
    # slots they carry (Twitter 4, Weibo 9 — auto-detected, see
    # dataset_loader._big5_slot_count).
    p.add_argument("--twitter_en_gt_csv", type=str, default=None,
                   help="BIG-5 Twitter ENGLISH majority-vote GT CSV (platform_id, n_images, "
                        "nature_visual_<idx>/nep_materiality_visual_<idx>/"
                        "nep_biological_visual_<idx> per up-to-4 images).")
    p.add_argument("--twitter_es_gt_csv", type=str, default=None,
                   help="BIG-5 Twitter SPANISH majority-vote GT CSV, same schema as "
                        "--twitter_en_gt_csv.")
    p.add_argument("--weibo_ch0_gt_csv", type=str, default=None,
                   help="BIG-5 WEIBO majority-vote GT CSV, split ch6B0 (e.g. "
                        "weiboch6B0_majority.csv). Same column schema as the Twitter CSVs "
                        "but with NINE image slots per row (nature_visual_0..8) instead of "
                        "four — the slot count is read off the CSV's own columns, so nothing "
                        "needs to be told which platform it is.")
    p.add_argument("--weibo_ch1_gt_csv", type=str, default=None,
                   help="BIG-5 WEIBO majority-vote GT CSV, split ch6B1. Same schema as "
                        "--weibo_ch0_gt_csv.")
    p.add_argument("--big_5_twitter_images_dir", "--big5_images_dir", type=str,
                   default="./big_5_images", dest="big_5_twitter_images_dir",
                   help="Local folder (shared by the English and Spanish splits) already "
                        "containing every BIG-5 TWITTER image, named "
                        "'<platform_id>_<idx>.<ext>'. Images are located by globbing for that "
                        "stem (the extension isn't fixed — jpg/jpeg/png all appear) rather "
                        "than downloaded — there is no remote fetch step. "
                        "--big5_images_dir is kept as an alias for backwards compatibility.")
    p.add_argument("--big_5_weibo_images_dir", type=str, default="./big_5_weibo_images",
                   help="Local folder (shared by both Weibo splits) already containing every "
                        "BIG-5 WEIBO image, named '<platform_id>_<idx>.<ext>' — same flat "
                        "layout and same glob-by-stem lookup as the Twitter folder.")

    # VLM (infer)
    p.add_argument("--model_family", type=str, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--model_name", type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--max_num_seqs", type=int, default=None,
                   help="Passed straight to vLLM's own EngineArgs (LLM(max_num_seqs=...)) — "
                        "caps how many sequences the ENGINE runs concurrently, independent of "
                        "--batch_size (which only controls how many prompts THIS SCRIPT submits "
                        "per generate_batch call; vLLM's own scheduler still decides how many of "
                        "those it runs — and vision-encodes — at once, currently uncapped). This "
                        "is the lever for a vision-encoder OOM that only shows up on "
                        "large-image datasets (BIG-5) at a --batch_size that's fine for "
                        "ImageNet/Places: fewer images processed by the vision encoder "
                        "SIMULTANEOUSLY, without lowering --batch_size (still submitted/queued "
                        "at full size) or --max_image_side (image quality untouched). Default "
                        "None leaves vLLM's own default (unset — no change from prior behavior).")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_image_side", type=int, default=1024,
                   help="Downscale (preserving aspect ratio) any image whose longest side "
                        "exceeds this many pixels before it reaches the VLM (see "
                        "vlm_models.VLLMBackedVLM._encode_image). Nothing in this pipeline "
                        "resizes images otherwise: ImageNet/COCO/Places images are typically "
                        "already modest, but raw social-media images (BIG-5 Twitter/Weibo) can "
                        "be arbitrary phone-camera/screenshot resolutions with no cap, and a "
                        "vision encoder's patch count (and attention memory) scales with input "
                        "resolution, not a fixed square — one oversized image in a batch can "
                        "OOM the vision encoder even at a batch_size/max_model_len that's "
                        "comfortable for ImageNet/Places at the same settings. Pass 0 to "
                        "disable resizing entirely (reproduces old behavior).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens_caption", type=int, default=248) # 248 tokens is the maximum length that LongCLIP can handle
    p.add_argument("--max_new_tokens_extraction", type=int, default=None,
                   help="Max new tokens for the Stage-2 object-extraction call (structured, "
                        "ObjectExtractionResponse). Defaults to --max_new_tokens_caption if not set "
                        "(the old shared-budget behavior), but is worth setting HIGHER and "
                        "independently once EXTRACTION_PROMPT's schema includes a 'reasoning' field "
                        "(current prompts.py does): under guided decoding the model must write that "
                        "full reasoning paragraph BEFORE it can emit the objects array, and "
                        "--max_new_tokens_caption's default (248) was sized for captioning alone, "
                        "not for a reasoning-plus-list schema — a too-small budget here can silently "
                        "truncate the objects list (or the whole structured output) without any "
                        "parse-failure signal, showing up only as a suspiciously low "
                        "objects-per-image diagnostic.")
    p.add_argument("--max_new_tokens_label", type=int, default=248)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no_summarize_clipmatch_caption", action="store_true",
                   help="On ImageNet/Places, --stage infer asks the VLM for an extra short "
                        "(<=~20-word) summary of its own caption, grounded in the image, and "
                        "--stage score uses THAT (not the full ~248-token caption) to drive "
                        "ClipMatch/nature/biotic/material/hP-hR — measured 0.41 vs 0.34 top-1 "
                        "against the raw caption, with far less CLIP-tokenizer truncation. This "
                        "is now the DEFAULT for those two datasets (ignored on COCO/BIG-5, where "
                        "ClipMatch never runs). Pass this flag to opt back into the raw-caption "
                        "behavior (e.g. to reproduce old results, or skip the extra VLM call) — "
                        "the raw caption then becomes the one and only ClipMatch text.")
    p.add_argument("--summary_max_new_tokens", type=int, default=77,
                   help="Max new tokens for the summary-caption VLM call (default 64 — generous "
                        "headroom over the ~50-word target so it isn't itself cut off "
                        "mid-sentence). No effect if --no_summarize_clipmatch_caption is passed.")
    p.add_argument("--use_wordnet_definitions_clipmatch", action="store_true",
                   help="--stage score only: build each ClipMatch candidate class's text from its "
                        "WordNet lemma(s) + gloss in natural prose (clip_metrics."
                        "wordnet_definition_text), e.g. 'A golden retriever, also known as golden "
                        "retriever dog. It is defined as: an English breed having a long silky "
                        "golden coat.' instead of the plain 'a photo of a golden retriever' "
                        "template. Hypothesis (untested until you run it): richer candidate-side "
                        "semantic content aligns better against a VLM's own descriptive image "
                        "summary than a bare noun phrase. Needs the `inflect` package.")
    p.add_argument("--use_inflect_for_clipmatch", action="store_true",
                   help="--stage score only: grammar-correct the indefinite article in front of "
                        "every class/entity name filled into a CLIP template (clip_metrics."
                        "fill_template), e.g. 'a photo of a apple' -> 'a photo of an apple', "
                        "'a photo of a cars' -> 'a photo of cars'. Applies to OBJECT_TEMPLATE "
                        "(extracted-object text, ClipMatch's anchor-object search) AND the "
                        "80/15-template candidate-vocab ensemble on ImageNet/Places. OFF by "
                        "default — an inflect-driven determiner was tried project-wide once "
                        "before, reverted on suspicion of a ClipMatch drop, and that suspicion "
                        "was never actually isolated from a concurrent CLIP-backend swap at the "
                        "time (see the recap), so it's an explicit opt-in rather than the default "
                        "this time, to keep any future comparison unambiguous. Needs the "
                        "`inflect` package. Has no effect together with "
                        "--use_wordnet_definitions_clipmatch (that path always inflects its own "
                        "article regardless, as its own separate pre-existing behavior).")
    p.add_argument("--max_hops", type=int, default=0,
                   help="Maximum WordNet hop distance allowed when mapping an EXTRACTED "
                        "object onto the labeled taxonomy (map_object_to_taxonomy -> "
                        "resolve_labels). 0 = map only when an annotator labeled that exact "
                        "synset (no inherited labels); 1 = also accept a label inherited "
                        "across one hypernym/hyponym hop; etc. Lower values make the VLM "
                        "fallback fire more often (fewer objects map), higher values map "
                        "more aggressively. Default 3 (previous behavior). Stored in the "
                        "artifact header and reused by --stage score.")

    # CLIP (score) — loaded via transformers (src/evaluation/clip_metrics.py's
    # CLIPScorer), NOT open_clip. --clipscore_model accepts either a short alias
    # from clip_metrics.CLIP_PRESETS ("original", "eva-clip", "siglip2",
    # "jina-clip-v2") or any raw HuggingFace repo id directly (e.g. to
    # override a preset's default checkpoint, or use a variant not in the
    # preset table). llm2clip is deliberately NOT supported — its text tower
    # needs the `llm2vec` package, which hard-pins transformers<=4.44.2 and
    # is incompatible with vLLM's transformers>=5.5.3 in this project's
    # single environment (see clip_metrics.CLIP_PRESETS's module comment).
    # FG-CLIP2 was tried and abandoned (meta-tensor crash in its
    # trust_remote_code __init__, incompatible with this transformers
    # version — see clip_metrics.CLIP_PRESETS's comment for the full story).
    p.add_argument("--clipscore_model", type=str, default="original",
                   help="CLIP checkpoint used for the REFERENCE-FREE metrics (CLIPScore, "
                        "F-CLIPScore, Object-CLIPScore — all datasets): a "
                        "clip_metrics.CLIP_PRESETS alias ('original', 'metaclip', 'metaclip2', "
                        "'siglip2', 'jina-clip-v2') or a raw HuggingFace repo id. Also the default for "
                        "--clipmatch_model when that isn't set separately.")
    p.add_argument("--clipscore_trust_remote_code", type=lambda s: s.lower() != "false", default=True,
                   help="Passed to transformers' from_pretrained calls for --clipscore_model "
                        "(default True). Several CLIP variants (EVA-CLIP, Jina-CLIP-v2) ship "
                        "custom modeling code on the Hub that requires this; it's a no-op for "
                        "checkpoints that don't need it (e.g. the original OpenAI CLIP, "
                        "SigLIP2). Pass --clipscore_trust_remote_code false to disable.")
    p.add_argument("--clip_batch_size", type=int, default=77)
    p.add_argument("--clipmatch_model", type=str, default=None,
                   help="CLIP checkpoint used for ClipMatch + hP/hR/hF1 (ImageNet + Places "
                        "only) — kept independently selectable from --clipscore_model since the two "
                        "jobs have different needs (ClipMatch matches a whole caption against a "
                        "fixed candidate vocabulary; F-CLIPScore/Object-CLIPScore/CLIPScore are "
                        "reference-free plausibility scores). Default: same value as "
                        "--clipscore_model. When it resolves to the SAME checkpoint + "
                        "--clipscore_trust_remote_code setting as --clipscore_model, the already-loaded "
                        "scorer is reused instead of loading a second model.")
    p.add_argument("--clipmatch_clip_trust_remote_code", type=lambda s: s.lower() != "false", default=None,
                   help="Passed to transformers' from_pretrained calls for "
                        "--clipmatch_model (default: same as --clipscore_trust_remote_code). "
                        "Pass --clipmatch_clip_trust_remote_code false/true to override "
                        "independently of --clipscore_trust_remote_code.")

    # shared
    p.add_argument("--output_file", type=str, default="vlm_pipeline_results.json",
                   help="Shared results store JSON, keyed by dataset then model name (updated "
                        "in place — a rerun of the same model overwrites its own entry; other "
                        "models' entries are untouched). Lands directly in --results_dir/"
                        f"--run_name — NOT inside {RESPONSES_SUBDIR}/{PREDICTIONS_SUBDIR}, since "
                        "it's a single shared file every model in that folder reports into, "
                        "unlike the per-model artifacts.")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Base directory all results are written under. Created if it doesn't "
                        "exist.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional subfolder of --results_dir (may itself be a multi-level path, "
                        "e.g. --run_name vlm_pipeline/imagenet), useful for keeping results from "
                        "different pipeline configurations (ablations) in separate, clearly "
                        "labeled folders. Layout inside it: --output_file's shared results JSON "
                        f"directly in this folder; every model's --responses_file .jsonl "
                        f"artifact in a '{RESPONSES_SUBDIR}' subfolder; every model's "
                        f"_predictions.csv in a '{PREDICTIONS_SUBDIR}' subfolder. Created if it "
                        "doesn't exist. Default: write directly into --results_dir.")
    p.add_argument("--resume", action="store_true",
                   help="--stage infer/all only. If --responses_file already exists and has "
                        "no footer (i.e. a previous run died part-way), skip every image "
                        "already recorded in it and APPEND the rest, keeping the existing "
                        "header. Inference streams one JSON line per image, so everything "
                        "written before the crash is complete, valid work — this turns a "
                        "fatal OOM/timeout hours into a run that picks up where it left off "
                        "instead of restarting from image 0. A truncated final line (hard "
                        "kill mid-write) is dropped and that one image redone. If the file "
                        "already ends in a footer the run is complete and this exits "
                        "immediately rather than redoing anything. NOTE: pair with the SAME "
                        "--max_samples as the original run — the sampled subset is "
                        "seed-42 deterministic, so a different value resumes against a "
                        "different subset.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_run_id", type=str, default=None, help=argparse.SUPPRESS)
    # ^ Internal plumbing, not meant to be passed by hand: under --stage all,
    # main() generates ONE shared W&B run id and this flag is how it's threaded
    # into the infer/score subprocess CLI (see _args_to_cli + _log_wandb's
    # resume="allow" if run_id else None), so both halves log into the same
    # W&B run instead of each starting its own. Hidden from --help.

    return p


def parse_args():
    p = build_arg_parser()
    args = p.parse_args()
    if args.stage in ("infer", "all"):
        # These two flags are only conditionally required (argparse can't
        # express "required unless --stage is score" declaratively), so we
        # check manually here and raise the same kind of clean CLI error
        # argparse itself would produce.
        if not args.model_family or not args.model_name:
            p.error("--model_family and --model_name are required for the infer stage.")
    return args


def _model_slug(args):
    """Filesystem-safe slug identifying the VLM for this run, e.g.
    --model_name 'Qwen/Qwen3.5-0.8B' -> 'Qwen_Qwen3.5-0.8B'. Deliberately built
    from ONLY --model_name, NOT --model_family: model_name already carries the
    org/model identity, so folding model_family in too just repeats that
    identity a second time under a different name (e.g. the old
    'qwen2_5_vl_Qwen_Qwen3.5-0.8B'). The full family/name pair remains the
    canonical model identifier everywhere else (the header's own "model"
    field, results-store keys, the CSV's "model" column) — this slug is
    filename-only. Returns None when the model isn't set (e.g. a standalone
    --stage score run that relies on an explicit --responses_file)."""
    if not args.model_name:
        return None
    return args.model_name.replace("/", "_")


def _resolve_responses_file(args):
    """Fill in --responses_file's default (None until now) as
    '<results_dir>/<run_name>/RESPONSES_SUBDIR/vlm_responses_<model_slug>.jsonl'
    — grouped into RESPONSES_SUBDIR (separate from PREDICTIONS_SUBDIR, where
    phase_score writes the predictions CSV), so a --run_name folder with
    several models stays easy to scan: the shared results JSON directly
    inside it, every model's raw .jsonl artifacts together in one subfolder,
    every model's predictions CSVs together in another. The filename is
    suffixed with the MODEL slug (see _model_slug) so each model's infer
    artifact is uniquely named (the Grounding pipeline consumes these
    per-model JSON-Lines files, and it keeps a rerun on a different VLM from
    overwriting a previous model's artifact). Falls back to a plain
    'vlm_responses.jsonl' when the model isn't known (a standalone --stage
    score run). An explicitly-passed --responses_file is left untouched.
    Creates that directory so phase_infer can open the file for writing.
    Mutates and returns `args`."""
    if args.responses_file is None:
        out_dir = Path(args.results_dir)
        if args.run_name:
            out_dir = out_dir / args.run_name
        responses_dir = out_dir / RESPONSES_SUBDIR
        responses_dir.mkdir(parents=True, exist_ok=True)
        slug = _model_slug(args)
        fname = f"vlm_responses_{slug}.jsonl" if slug else "vlm_responses.jsonl"
        args.responses_file = str(responses_dir / fname)
    return args


# =============================================================================
# Subprocess re-invocation (for --stage all)
# =============================================================================
def _args_to_cli(args, parser, stage=None):
    """Rebuild a `sys.argv`-style flag list from a parsed args Namespace, to
    re-invoke THIS SAME SCRIPT as a genuinely separate OS process (see
    main()'s --stage all for why — a real subprocess, not
    multiprocessing.Process, to avoid nesting inside vLLM's own internal
    engine-core subprocess).

    `stage` is prepended as `--stage <stage>` when given. It's optional so
    scripts/run_pipeline.py can reuse this same reconstruction for the GROUNDING
    stage's parser, which has no --stage flag of its own.

    Only emits a flag when its value differs from that action's own default
    (so the subprocess falls back to the SAME defaults for anything the user
    didn't explicitly set, rather than this function needing to know what
    every default is). Introspects `parser`'s actions rather than
    hardcoding each flag's name/type, so it stays correct as flags are
    added/changed in build_arg_parser() without needing a parallel update
    here.
    """
    argv = ["--stage", stage] if stage is not None else []
    for action in parser._actions:
        if not action.option_strings or action.dest in ("help", "stage"):
            continue
        value = getattr(args, action.dest, None)
        if value == action.default:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def _run_stage_subprocess(args, parser, stage):
    """Re-invoke this script (`python run_vlm_pipeline.py --stage <stage> ...`)
    as a genuine OS subprocess and wait for it to finish, raising if it
    failed. Inherits this process's stdout/stderr/env/cwd, so logs interleave
    live exactly as they would running that --stage by hand."""
    cmd = [sys.executable, str(Path(__file__).resolve())] + _args_to_cli(args, parser, stage)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"--stage {stage} subprocess failed (exit code {result.returncode}).")


def main():
    args = parse_args()
    args = _resolve_responses_file(args)

    if args.stage == "infer":
        phase_infer(args)
        return
    if args.stage == "score":
        phase_score(args)
        return

    # --stage all: run each half as its OWN separate OS subprocess (see the
    # module docstring's --stage all section for why this is subprocess.run,
    # NOT multiprocessing.Process — the latter nests inside vLLM's own
    # internal engine-core subprocess and can deadlock). GPU memory is fully
    # reclaimed by the OS when the infer subprocess exits, before the score
    # subprocess starts — the VLM and CLIP never coexist on the GPU, and we
    # don't rely on in-process CUDA cleanup.
    #   - A shared W&B run id (below) is threaded through via --wandb_run_id
    #     so both subprocesses log into one run (see _log_wandb).
    if args.wandb:
        import wandb
        args.wandb_run_id = wandb.util.generate_id()

    parser = build_arg_parser()
    _run_stage_subprocess(args, parser, "infer")
    _run_stage_subprocess(args, parser, "score")

    # --responses_file is ALWAYS retained (never deleted here, under any
    # --stage) — it carries every raw VLM response plus the resolved
    # per-object labels, which the Grounding pipeline consumes downstream, so
    # it's a first-class output of this run, not a disposable handoff.
    print(f"💾 [all] retained inference artifact at {args.responses_file}")


if __name__ == "__main__":
    main()
