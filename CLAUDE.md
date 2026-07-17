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
- **VLM pipeline** (language-based, in progress): caption → object extraction →
  taxonomy labeling. Produces per-image predictions scoreable with standard
  accuracy/precision/recall/F1.
- **Grounding pipeline** (geometric/embedding-based, designed but not built):
  Grounding DINO + SAM (thing/stuff routing) → FG-CLIP2 hierarchy-margin
  verification → nature importance score.

## VLM pipeline — hard conventions
- Baseline is TWO-PASS: open-ended caption (no schema) → separate structured
  extraction call. Never collapse into single-pass-with-schema without it being
  an explicit, labeled ablation.
- Caption prompt (baseline, neutral, no nature-priming):
  "Describe this image in detail, including the main subject, the background,
  the setting, and any secondary elements present."
  Do NOT add "pay attention to nature" to this prompt unless running the
  nature-priming ablation explicitly.
- Context files go in the `system` role, never `user`. Read once at startup,
  not per-call (keeps the string stable for vLLM prefix caching).
- Taxonomy labeling is a HYBRID:
  - nature/no-nature AND biotic/abiotic → WordNet mapping first (when object is
    in ImageNet/COCO/Places mapped vocab), VLM fallback when unmapped.
  - material/immaterial → ALWAYS VLM, NEVER mapping. Always pass the image to
    the model for this judgment, not just the object string as text.
- Always track and log: WordNet-mapping-rate vs. VLM-fallback-rate, and total
  objects extracted per image (diagnostic, not just the taxonomy scores).

## Metrics — exact definitions, do not rename or merge
- **F-CLIPScore** (faithful, cite Oh & Hwang exactly):
  `F-CLIPScore(S) = [CLIPScore(S) + sum_i CLIPScore(n_i)] / (N+1)`
  S = caption sentence, n_i = extracted objects.
- **Object-CLIPScore** (ours, F-CLIPScore-INSPIRED — never call this
  "F-CLIPScore"): mean of `CLIPScore("a photo of a {object}")` over extracted
  objects only. No sentence term.
- CLIP text encoder truncates at 77 tokens — long captions risk truncating the
  sentence-level term. Check which CLIP variant is in use before assuming the
  full caption is encoded (FG-CLIP2 / Long-CLIP handle longer text; vanilla
  CLIP does not).
- **ClipMatch** (ImageNet + Places only — not COCO, not BIG-5): score each
  extracted object independently (`"a photo of a {object}"`), take the MAX
  similarity across objects per GT candidate class. Do NOT run against the raw
  caption or a concatenated object-list sentence — both were tried and
  rejected (token limit + semantic dilution — see data/llm_reference/vlm_pipeline_recap.txt
  for why).
- **hP/hR/hF1** (hierarchical precision/recall/F1): ImageNet + Places only, via BGE
  cross-encoder mapping the ClipMatch-predicted class onto WordNet.

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
Week 1 of 8 (deadline Sept 10): VLM pipeline baseline + F-CLIPScore +
Object-CLIPScore on Qwen3.5-0.8B, to hand off to Ramin for BSC infra check.