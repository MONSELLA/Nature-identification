# BIG-5 Nature Identification — Project Context

## Overview
Master's thesis (TFM): evaluating VLMs and closed-set CV models for detecting nature
representations in social media images, for the BIG-5 project. Taxonomy: three binary
axes — nature/non-nature, biotic/abiotic, material/immaterial. Scope: evaluation and
benchmarking, NOT fine-tuning. We have a mapping for almost all the target classes
from ImageNet, COCO and Places365. Therefore, these datasets can be used as well for 
evaluating the models. 

## Reference files (load these, don't ask me to re-explain)
@data/big5_taxonomy/big5_nature_definition.txt
@data/big5_taxonomy/big5_material_definition.txt
@data/big5_taxonomy/big5_biotic_definition.txt
@data/llm_reference/vlm_pipeline_recap.txt

## Two pipelines — do not conflate
- **VLM pipeline** (language-based, IMPLEMENTED — `scripts/run_vlm_pipeline.py`,
  `src/vlm_pipeline.py`): caption → object extraction → mapping → taxonomy
  labeling. Produces per-image predictions scoreable with standard
  accuracy/precision/recall/F1.
- **Grounding pipeline** (pixel-based, IMPLEMENTED — `src/grounding_pipeline.py`,
  `scripts/run_grounding_pipeline.py`): SAM3 semantic segmentation of every
  nature-labeled entity → nature relevance score. Enriches the VLM pipeline's
  own artifact in place; does NOT write a separate output file.
  SUPERSEDED DESIGN — do not reintroduce: the thing/stuff split (Grounding DINO
  + SAM for things, an open-vocab segmenter for stuff) and its WordNet-lexname /
  COCO-Panoptic / LLM routing step are DROPPED. One model, SAM3, handles both
  through a single output head. The FG-CLIP2 hierarchy-margin verification step
  is likewise not part of the shipped pipeline (FG-CLIP2 was abandoned as a CLIP
  backend — see clip_metrics.CLIP_PRESETS's comment).

## VLM pipeline — code layout & how to run
- Entrypoint: `scripts/run_vlm_pipeline.py` (`--stage all|infer|score`).
- `--stage all` runs infer then score in SEPARATE spawned subprocesses so the
  VLM's VRAM is fully released (OS reclaim on infer-process exit) before CLIP
  loads for scoring — more reliable than in-process unload. A shared W&B run id
  is threaded through so both subprocesses log into one run.
- Core modules: `src/vlm_pipeline.py` (caption/extract/map/label + hybrid
  resolution, all in Phase-1 inference), `src/models/prompts.py` (prompts +
  schemas: `TaxonomyResponse`, `MaterialResponse`, `ObjectExtractionResponse`),
  `src/models/vlm_models.py` (VLM backends), `src/loaders/dataset_loader.py`
  (`COCO_LABELS`, `COCO_TO_WNSYNSET`, `build_mapping_vocab`),
  `src/loaders/excel_loader.py` (`TaxonomyGraph.resolve_labels`, `max_hops`),
  `src/evaluation/clip_metrics.py` + `taxonomy_metrics.py` (metrics).

## VLM pipeline — hard conventions
- Baseline is TWO-PASS: open-ended caption (no schema) → separate structured
  extraction call. Never collapse into single-pass-with-schema without it being
  an explicit, labeled ablation.
- Caption prompt (baseline, neutral, no nature-priming):
    "Describe this image in 3-4 sentences, covering the main subject, the 
    background, the setting, and any secondary elements present. Be specific but concise."
  Do NOT add "pay attention to nature" to this prompt unless running the
  nature-priming ablation explicitly.
- Context files go in the `system` role, never `user`. Read once at startup,
  not per-call (keeps the string stable for vLLM prefix caching).
- Taxonomy labeling is a HYBRID, resolved during PHASE-1 INFERENCE (mapping is
  done BEFORE the VLM labeling calls, not deferred to scoring):
  - nature/no-nature AND biotic/abiotic → WordNet mapping first (when object is
    in ImageNet/COCO/Places mapped vocab), VLM fallback when unmapped.
  - material/immaterial → ALWAYS VLM, NEVER mapping. Always pass the image to
    the model for this judgment, not just the object string as text.
- Labeling calls are ROUTED per object (map first, then ask the VLM only what
  mapping could not answer — saves compute):
  - unmapped object → ONE full VLM call (nature+biotic+material,
    `TaxonomyResponse`; system prompt = all three definitions).
  - mapped-nature object → material-only VLM call (`MaterialResponse`; system
    prompt = material definition only); nature/biotic come from the mapping.
  - mapped non-nature object → NO VLM call (nature=False, biotic/material n/a).
- EVERY extracted object is labeled on all three axes regardless of GT matching.
  "Best-matching" selection happens only at SCORING time (to reduce to one
  prediction per image on single-label datasets); it never restricts labeling.
- `--max_hops` controls how far an extracted object may resolve onto the taxonomy
  (0 = only annotator-labeled nodes; default 3). Stored in the artifact header.
- Always track and log: WordNet-mapping-rate vs. VLM-fallback-rate, and total
  objects extracted per image (diagnostic, not just the taxonomy scores).

## Grounding pipeline — code layout & hard conventions
- Entrypoints: `scripts/run_grounding_pipeline.py` (grounding only, over an
  existing VLM artifact) and `scripts/run_pipeline.py` (VLM infer → grounding
  end-to-end, each stage its own OS subprocess so VRAM is fully reclaimed
  between the VLM and SAM3 — same `subprocess.run` rationale as `--stage all`,
  never `multiprocessing.Process`). Core module: `src/grounding_pipeline.py`.
- ONE artifact per run: grounding reads the VLM's JSON-Lines artifact
  (`vlm_responses_<model>.jsonl`) and adds its own fields to each image record —
  never a parallel per-stage output file. Added: `object_groundings` (aligned
  index-for-index with `objects`/`object_labels`/`object_finals`, each entry
  `{object, prompt, is_nature, grounded, mask_rle, pixel_count}`),
  plus top-level `nature_relevance_score` and `relevance_score_method`.
- Only entities with `final_nature is True` (the HYBRID label) are grounded.
  Everything else gets `grounded: null` = never attempted; `grounded: false` =
  SAM3 looked and found nothing. Entities are NEVER deleted from the record —
  the full VLM output stays auditable.
- SAM3 (`facebook/sam3`) via plain transformers AutoModel/AutoProcessor. Read
  `outputs.semantic_seg` (concept-level pixel coverage), NOT the instance-level
  `pred_masks`/`pred_boxes`/`pred_logits`.
- `semantic_seg` is RAW LOGITS (unbounded), verified in the transformers source
  — sigmoid is required. Use `processor.post_process_semantic_segmentation(...)`,
  which does sigmoid → resize to original size → binarize at 0.5 (SAM3's own
  default). `--debug_semantic_range` prints the raw min/max to re-confirm.
- SAM3 resolves ONE text prompt per forward pass (`Sam3Model.forward` shares one
  batch dim between `pixel_values` and `input_ids`; `semantic_seg` is
  `(batch,1,H,W)`). There is no ragged per-image prompt list. Batching is
  therefore over flattened (image, entity) PAIRS: `--batch_size` counts images,
  `--max_pairs_per_forward` bounds actual GPU work.
- Masks stored as `pycocotools` RLE (`counts` decoded to str for JSON).
- Nature relevance score is computed over the RLE-MERGED UNION of surviving
  masks (overlapping pixels counted once), `--relevance_method`:
  `coverage_ratio` (default; nature px / total px) or `center_weighted`
  (Gaussian in normalized distance-from-center, `--center_sigma`, aspect-ratio
  independent). Both in [0,1]; no grounded nature entity → 0.0.
- Report `grounding_confirmation_rate` (grounded / nature entities). Per recap
  §9 this is AGREEMENT WITH AN INDEPENDENT MODEL, never ground truth.

## Metrics — exact definitions, do not rename or merge
- **F-CLIPScore** (faithful, cite Oh & Hwang exactly):
  `F-CLIPScore(S) = [CLIPScore(S) + sum_i CLIPScore(n_i)] / (N+1)`
  S = caption sentence, n_i = extracted objects.
- **Object-CLIPScore** (ours, F-CLIPScore-INSPIRED — never call this
  "F-CLIPScore"): mean of `CLIPScore("a photo of a {object}")` over extracted
  objects only. No sentence term.
- CLIP text encoder truncates at 77 tokens — long captions risk truncating the
  sentence-level term. Check which CLIP variant is in use before assuming the
  full caption is encoded; vanilla CLIP, SigLIP2, and EVA-CLIP all truncate
  around this range, Jina-CLIP-v2 handles much longer text. FG-CLIP2 was
  tried as a long-context option and abandoned — see
  src/evaluation/clip_metrics.py's `CLIP_PRESETS` comment.
- **ClipMatch** (ImageNet + Places only — not COCO, not BIG-5): score the
  WHOLE CAPTION's CLIP embedding against each GT candidate class; argmax =
  predicted class. SUPERSEDES the earlier object-list variant (max similarity
  across independently-embedded extracted objects) — the caption-based version
  empirically performs better, so the object-list implementation has been
  removed from the codebase (see data/llm_reference/vlm_pipeline_recap.txt for
  the history). Known caveat carried over: long captions risk CLIP's 77-token
  truncation — accepted given the empirical gain.
- **hP/hR/hF1** (hierarchical precision/recall/F1): ImageNet + Places only. Map
  the ClipMatch-predicted class onto a WordNet node via the extracted-object list
  (`resolve_to_wordnet`: rank objects by CLIP sim to the predicted class,
  Wu-Palmer disambiguation for polysemy), then score ancestral-closure overlap
  of the GT node vs. the predicted node.

## Axis scoring (nature/biotic/material accuracy) — per dataset
- **ImageNet/Places (single-label)**: ClipMatch (whole-caption CLIP embedding
  vs. candidate_vocab, global argmax — no lexical matching, no similarity
  threshold) picks the top-1 predicted class, restricted to classes mapped into
  the graph. That predicted class is then used ONLY to pick an ANCHOR among the
  extracted objects: the object whose own CLIP embedding is most similar to the
  predicted class's embedding (`best_obj_idx`/`best_final` in
  `run_vlm_pipeline.py`'s single-label branch). nature/biotic/material are all
  read off that ANCHOR OBJECT's own hybrid-resolved label
  (`final_nature`/`final_biotic`/`final_material`) — NOT off the predicted
  class's own stored taxonomy position. This means an incidental-but-correct
  object never counts against the single GT label, and also means the axis
  verdict can in principle diverge from the ClipMatch-predicted class itself
  (e.g. if the anchor object's hybrid label came from the VLM fallback rather
  than the mapping). material is always the VLM's own label (never mapped).
  No anchor object (empty extraction or failed ClipMatch) → prediction-unmapped
  → penalized as wrong.
- **COCO/BIG-5**: image-level nature = OR over extracted objects; biotic/material
  scored on the matched GT object. COCO box-IoU matching (Hungarian, IoU≥0.5) is
  FUTURE WORK gated on the Grounding pipeline — for now COCO uses the same
  lexical GT matching as BIG-5.
- **Extraction-hit rate** (exact-match: was the GT object mentioned) is a
  REPORTING-ONLY diagnostic; it no longer gates or feeds the axis scores.

## Inherited conventions (from the closed-set baseline work)
- Positive classes: nature=1, biotic=1, material=1.
- Ground-truth-unmapped instances: excluded from taxonomy metrics.
- Prediction-unmapped instances: penalized as wrong (never defaulted to "no
  nature").
- Report mapped-subset and unmapped-subset metrics separately — never pool
  them into one number without saying so.

## Environment
- W&B project: `TFM_VLM`, entity `paumonserrat03-universitat-aut-noma-de-barcelona`
- Taxonomy Excel: `/home/pmonserrat/code/flat_wordnet_tree_fixed.xlsx`
- BIG-5 data: `/home/pmonserrat/datasets/big_5/`
- Imagenet data: `/home/pmonserrat/datasets/imagenet/`
- COCO data: `/home/pmonserrat/datasets/coco/`
- Places365 data: `/home/pmonserrat/datasets/places/`
- Dev-loop model: Qwen/Qwen3.5-0.8B (architecture-search proxy — not a
  performance benchmark; spot-check final config on larger models before
  locking in)

## Current focus
Baseline VLM pipeline is IMPLEMENTED end-to-end (caption → extraction →
mapping-routed labeling → hybrid resolution → metrics: F-CLIPScore,
Object-CLIPScore, per-axis acc/P/R/F1, ClipMatch + hP/hR on ImageNet/Places).
Grounding pipeline is IMPLEMENTED end-to-end (SAM3 semantic segmentation of
nature entities → RLE masks → nature relevance score), enriching the same
artifact. Next: spot-check both pipelines on Qwen3.5-0.8B + a real SAM3 run
(confirm the semantic_seg range empirically with --debug_semantic_range and
tune --max_pairs_per_forward to the GPU), hand off to Ramin for the BSC infra
check, then the sequential ablations (recap §7).