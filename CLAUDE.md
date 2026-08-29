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

## Two output files — distinct purposes, do not conflate
- **`.jsonl` artifact** (`vlm_responses_<model>.jsonl`, enriched in place by
  grounding): the RAW PREDICTION RECORD, model-facing. Everything either
  pipeline actually predicted — caption, extracted objects, per-object
  labels + reasoning, hybrid finals, ClipMatch summary caption, SAM3
  groundings + relevance scores — lives here, complete and unflattened, so
  it can feed metrics not yet invented. Nothing here is a computed metric.
- **predictions `.csv`** (written by `--stage score`, one row per image):
  the QUALITATIVE-REVIEW file — everything from the `.jsonl` PLUS every
  per-image metric/diagnostic computed at scoring time, flattened into one
  browsable row. The goal is that this file alone is sufficient for
  spot-checking a run — nothing in the `.jsonl` should require going back
  to it. When adding a new prediction field (raw model output) or a new
  per-image metric, it belongs in BOTH files: the `.jsonl` because it's a
  prediction that might feed a metric not yet written, the `.csv` because
  it's needed for qualitative review right now.

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
- `--max_image_side` (default 1024): every image is downscaled to this many
  pixels on its longest side before reaching the VLM
  (`vlm_models.VLLMBackedVLM._encode_image`) — pass 0 to disable. Nothing
  else in this pipeline resizes images. ImageNet/COCO/Places images are
  typically already modest (pre-resized benchmark images), but raw
  social-media images (BIG-5 Twitter/Weibo) can be arbitrary phone-camera/
  screenshot resolutions with no cap, and a vision encoder's patch count
  (and attention memory) scales with input resolution rather than a fixed
  square — one oversized image in a batch can OOM the vision encoder at a
  `--batch_size`/`--max_model_len` that's comfortable for ImageNet/Places at
  the exact same settings (confirmed: a BIG-5 OOM traceback failed inside
  the vision encoder's own attention, not the text KV cache). Fast path for
  already-small images (`Image.open()` only reads the header to check size,
  no decode/re-encode) so this costs ~nothing at the project's 2M-image
  scale; oversized images are downscaled and re-encoded as JPEG
  (quality=90) rather than the original format.
- `--max_num_seqs` (default: unset, vLLM's own default): passed straight to
  vLLM's `EngineArgs`, capping how many sequences the ENGINE runs — and
  vision-encodes — concurrently. Distinct from `--batch_size`, which only
  controls how many prompts THIS SCRIPT submits per `generate_batch` call;
  vLLM's own scheduler still decides how many of those it actually runs
  together, currently uncapped. Use this (not `--batch_size` or
  `--max_image_side`) when a large-image dataset (BIG-5) OOMs the vision
  encoder at a `--batch_size` that's fine for ImageNet/Places and you don't
  want to sacrifice submission throughput or image resolution to fix it —
  it reduces peak *concurrent* vision-encoder memory without touching
  either. `--gpu_memory_utilization` (already existed) is the other
  zero-cost lever for the same OOM: lowering it (e.g. 0.9 -> 0.85) shrinks
  vLLM's upfront KV-cache reservation, leaving more real headroom for the
  vision encoder's activation memory, again without touching batch size or
  image quality.

## VLM pipeline — hard conventions
- Baseline is TWO-PASS: open-ended caption (no schema) → separate structured
  extraction call. Never collapse into single-pass-with-schema without it being
  an explicit, labeled ablation.
- CAPTION ABLATION (`--no_caption`, `scripts/job_vlm_pipeline_no_caption.sh`):
  the ONE labeled exception to the two-pass rule. Skips Stage 1 entirely (no
  caption VLM call) and prompts extraction from the image alone with
  `prompts.get_extraction_prompt(no_caption=True)` — byte-identical to
  `EXTRACTION_PROMPT` minus the `{caption}` preamble, so the only variable vs.
  the baseline is the caption stage. Records still carry `caption: ""`; the
  artifact header carries `caption_stage: false` +
  `extraction_prompt_variant`, and `--stage score` reads it to report
  CLIPScore/F-CLIPScore as N/A (means over an empty set, `caption_stage:
  false` in `summary["reference_free"]`, `None` in W&B) instead of a
  meaningless number from an empty-string embedding. Object-CLIPScore and every
  taxonomy/axis metric are unaffected. Give it its own `--run_name` so
  baseline artifacts are never overwritten.
- ON IMAGENET `--no_caption` COMPOSES with the ClipMatch summary call instead
  of being refused: ClipMatch needs SOME caption-derived text, so that call
  survives and switches to `prompts.SUMMARY_CAPTION_PROMPT_IMAGENET_NO_CAPTION`
  — no `{caption}` placeholder, used verbatim, becoming the run's ONE short
  direct image description rather than a re-summarization. It adds a
  NAMING-SPECIFICITY clause (ImageNet-1k is fine-grained and no stage produces
  a specific name once the caption is gone) and explicit background
  suppression, keeps the 20-word cap, the conditional plural and the
  "Output ONLY" guard. Header records
  `summary_caption_prompt: summary_caption_prompt_imagenet_no_caption`
  (`prompts.summary_caption_prompt_name`). CONSEQUENCE FOR REPORTING: the
  ImageNet arm compares "long caption + summary" vs "short caption", NOT
  "caption vs no caption" — the clean caption ablation is BIG-5's. Two
  combinations are refused up front by
  `run_vlm_pipeline._validate_no_caption_config` (called before any slow
  load): `--no_caption` on Places365 (no validated caption-free scene prompt
  exists — add one to `SUMMARY_CAPTION_PROMPTS_NO_CAPTION` if that run is
  wanted), and `--no_caption --no_summarize_clipmatch_caption` on any
  ClipMatch dataset (leaves no text at all).
- Caption prompt (baseline, neutral, no nature-priming), `src/models/prompts.py`'s
  `CAPTION_PROMPT`:
    "Describe this image, including any text."
  The "including any text" clause covers BIG-5's text-heavy/meme/screenshot
  images, where a bare "describe this image" would otherwise skip on-image
  text entirely. Do NOT add "pay attention to nature" to this prompt unless
  running the nature-priming ablation explicitly.
- The CAPTION CALL ITSELF gets NO system prompt (deliberate, settled): unlike
  extraction and labeling, `caption_batch` (via `run_inference`,
  `src/vlm_pipeline.py`) is called without the nature-definition system
  prompt, so this first free-form look at the image is uninfluenced by ANY
  nature-related context — neither an explicit priming instruction nor the
  passive definition file. Extraction (the "second look") and both labeling
  calls still receive it as normal.
- Extraction prompt (`src/models/prompts.py`'s `EXTRACTION_PROMPT`) explicitly
  asks for the overall SETTING/SCENE/ENVIRONMENT as its own extractable
  entity, not just discrete objects — needed because this one prompt is
  shared across ImageNet/COCO (object-centric) AND Places365 (whose candidate
  classes ARE scenes, e.g. "kitchen") AND BIG-5 (whose nature taxonomy lists
  "Ecosystems & Environments" as a first-class inclusion category, not just
  the flora/fauna within them). Without this, extraction only ever lists
  discrete objects inside a scene, leaving no extracted phrase for ClipMatch's
  anchor-selection or hP/hR's WordNet resolution to match against a
  scene-level GT class — same failure mode the summary-caption prompt had before
  its "subject or setting" fix.
- Context files go in the `system` role, never `user`. Read once at startup,
  not per-call (keeps the string stable for vLLM prefix caching).
- Taxonomy labeling is a HYBRID, resolved during PHASE-1 INFERENCE (mapping is
  done BEFORE the VLM labeling calls, not deferred to scoring):
  - nature/no-nature AND biotic/abiotic → WordNet mapping is trusted in ONE
    DIRECTION ONLY: when the mapped node IS nature. A "not nature" mapping is
    NEVER authoritative — the VLM decides instead (see below).
  - material/immaterial → ALWAYS VLM, NEVER mapping. Always pass the image to
    the model for this judgment, not just the object string as text.
- Labeling calls are ROUTED per object (`vlm_pipeline.label_objects_batch`;
  the route taken is recorded per object as `label_route` in BOTH the `.jsonl`
  and the predictions CSV):
  - HUMAN term (`vlm_pipeline.HUMAN_TERMS`/`is_human_term` — whole normalized
    phrase or its head noun) → NO VLM call, nature=False outright
    (`label_route: "human_exclusion"`). The nature definition excludes human
    beings explicitly, and unlike every other non-nature concept this really
    is instance-independent. "person" is also the single most-extracted
    entity, so this is where the saved compute is. Body-part words are
    DELIBERATELY excluded from the set (a "hand"/"hair"/"eye" may belong to an
    extracted animal, and a false positive here is uncorrectable).
  - mapped-NATURE object → material-only VLM call (`MaterialResponse`; system
    prompt = material definition only); nature/biotic come from the mapping
    (`label_route: "mapped_nature_material"`).
  - EVERYTHING ELSE — unmapped OR mapped-NON-nature → ONE full VLM call
    (nature+biotic+material, `TaxonomyResponse`; system prompt = all three
    definitions) (`label_route: "vlm_full"`).
- WHY mapped-non-nature goes to the VLM (CHANGED — it used to get no call at
  all; do not revert without re-reading this): "is nature" is concept-
  determined but "is NOT nature" is not. A depicted or stylized tree is still
  nature (the taxonomy counts "Representations of Nature"), so a nature
  mapping can't be undone by instance variation — but the "Nature-Based
  Artefacts" clause admits ANY manufactured object "where the natural material
  of origin remains visually identifiable by its structure, grain, or
  texture", and names wooden tables and stone/wood houses as its own examples.
  A `spatula`/`desk`/`house` node cannot encode whether THIS image shows a
  plastic one or one with visible wood grain — the same instance-vs-concept
  argument already accepted for material/immaterial. Under the old routing the
  mapping could produce only FALSE NEGATIVES on nature, and unfalsifiable ones
  (every spatula came back no-nature whether or not the model would have seen
  wood). Measured: 9.9% of extracted object records took that no-call path.
  Costs ~10% more full `TaxonomyResponse` calls, minus the humans.
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
- `run_vlm_pipeline.py --stage score`'s predictions CSV mirrors the grounding
  fields for qualitative review WITHOUT needing the raw `.jsonl` open
  alongside it: `object_groundings` (per-object JSON, now including
  `mask_rle`), `grounding_ungrounded_objects` (nature entities SAM3 attempted
  but whose mask never crossed `--mask_threshold` — `grounded: false`, not
  `null`), and `grounding_confirmation_rate_image`. The run summary/console/
  W&B also report a dataset-wide `confirmation_rate` (recap §9: agreement
  with SAM3, an INDEPENDENT model, never ground truth). On COCO the CSV also
  mirrors the per-image ENTITY-level detection results — counts, per-image
  precision/recall, and the actual `detection_matches` (GT class vs predicted
  entity, IoU, exact-match AND hierarchical verdicts), plus
  `detection_false_positives` / `detection_excluded_predictions` /
  `detection_missed_gt` / `detection_nature_on_non_nature`, and
  `detection_tp_by_iou_image` (this image's TP count at each rung of the IoU
  ladder, so the strictness curve is inspectable per image and not only
  dataset-wide) — so a disagreement is reviewable without opening the `.jsonl`.
- SAM3 (`facebook/sam3`) via `Sam3Model`/`Sam3Processor`, loaded EXPLICITLY —
  NEVER `AutoModel`/`AutoProcessor`. Confirmed a real production regression:
  on a newer transformers release, `AutoModel.from_pretrained("facebook/sam3")`
  resolves to `Sam3VideoModel` (the video/tracking head, whose `forward()`
  needs an `inference_session` this pipeline never constructs) instead of the
  image model this file needs — silently failing every single forward pass.
  The official facebook/sam3 model card and transformers' own SAM3 doc page
  both load it via the concrete classes for exactly this image+text use case;
  there is no AutoModel example anywhere in the documented usage. Read
  `outputs.semantic_seg` (concept-level pixel coverage) for the relevance
  score on EVERY dataset. The instance-level `pred_masks`/`pred_boxes`/
  `pred_logits` are read ONLY for COCO's box-IoU evaluation (see below) —
  never for the relevance score, which is about pixel coverage of a concept,
  not how many instances of it there are. Every tensor read off a SAM3 output
  goes through `_to_numpy` (`.float()` before `.numpy()`), NEVER a bare
  `.detach().cpu().numpy()` — confirmed a second production regression:
  NumPy has no bfloat16 representation, so `.numpy()` on one raises `Got
  unsupported ScalarType BFloat16`. This bites specifically because
  `run_pipeline.py`'s `--dtype` is a flag BOTH the VLM and grounding stages
  declare, so `--dtype bfloat16` (meant for the VLM) silently also loads SAM3
  in bfloat16 unless `--grounding_dtype` overrides it — `_to_numpy` makes the
  read-out correct either way rather than depending on that override being
  remembered on every invocation.
- INSTANCE GROUNDING (COCO only; `--instance_grounding auto|on|off`, auto =
  on for COCO artifacts, read from the artifact's own header). Adds
  `object_instances` — aligned index-for-index with `objects` like every other
  parallel list — each `{object, prompt, is_nature, attempted, instances:[
  {score, bbox, sam3_bbox, mask_rle, pixel_count}]}`. `attempted: false` is
  the instance-side twin of `grounded: null` (SAM3 never ran for this entity).
  `bbox` is the TIGHT box of the instance mask (`grounding_pipeline.mask_to_bbox`,
  xyxy, far edge EXCLUSIVE so a 1px mask has area 1, not 0) — that is the box
  the metrics score, because it is guaranteed consistent with the mask stored
  beside it; `sam3_bbox` carries SAM3's own regressed box unchanged for
  comparison. NOT a second forward pass: `semantic_seg` and the instance
  tensors are fields of the SAME `Sam3ImageSegmentationOutput`
  (`SAM3Grounder.segment_pairs_full`), so this costs post-processing only and
  leaves `object_groundings` and both relevance scores bit-for-bit unchanged.
  `--instance_score_threshold` (default 0.3, SAM3's own) gates instance
  CONFIDENCE — a separate knob from `--mask_threshold`, which binarizes pixels.
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
  masks (overlapping pixels counted once). BOTH methods are always computed
  and stored, never a choice between them:
  `nature_relevance_score_coverage_ratio` (nature px / total px) and
  `nature_relevance_score_center_weighted` (Gaussian in normalized
  distance-from-center, `--center_sigma`, aspect-ratio independent). Both in
  [0,1]; no grounded nature entity → 0.0 under both. Overlap between two
  DIFFERENT entities' masks (e.g. "tree" and "leaves" covering the same
  pixels) is resolved by the union itself — a pixel counts as nature if ANY
  grounded entity covers it, never a confidence-based per-pixel winner; each
  entity's own mask stays intact and unclipped in `object_groundings`.
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
- **ClipMatch** (ImageNet + Places only — not COCO, not BIG-5): score a text
  embedding against each GT candidate class; argmax = predicted class.
  SUPERSEDES the earlier object-list variant (max similarity across
  independently-embedded extracted objects) — tried TWICE (v5 and again in
  2026-07) and beaten both times by scoring a caption-derived text instead;
  removed from the codebase both times (see
  data/llm_reference/vlm_pipeline_recap.txt for the history).
  PRIMARY text (DEFAULT on ImageNet/Places): the VLM's own short (<=~20-word)
  SUMMARY of its baseline caption, grounded in the image
  (`src.vlm_pipeline.summarize_caption_batch`) — measured 0.41 vs. 0.34 top-1
  against the raw caption on the spot-check run, with ~0% CLIP-tokenizer
  truncation vs. the raw caption's frequent silent cutoff. A real
  re-summarization is more defensible than letting CLIP truncate arbitrary
  content from the ~248-token caption. `--stage infer`'s
  `--no_summarize_clipmatch_caption` opts back into scoring the raw caption
  instead. There is exactly ONE ClipMatch text per run — the old SECONDARY
  raw-caption comparison (`summary["clipmatch_caption"]` + its five
  `clipmatch_caption_*` CSV columns) is REMOVED; do not reintroduce it.
  The summary prompt is FORKED PER DATASET (v15,
  `prompts.SUMMARY_CAPTION_PROMPTS` / `get_summary_caption_prompt`), because
  the two candidate vocabularies are different KINDS of label: ImageNet gets
  an OBJECT-centric prompt ("the key objects and prominent entities in the
  scene, along with their identifying details"), Places365 a SCENE-centric
  one ("the overall scene, environment, or setting"). Both cap at 30 words
  (down from the shared prompt's 50) and both end with "Output ONLY the
  summary text." so no conversational preamble gets embedded as if it were
  image content. Unknown dataset raises rather than silently defaulting.
  Which variant a run used is recorded in the artifact header's
  `summary_caption_prompt`.
  CANDIDATE text (the GT-class side, not the image-description side) is a
  SEPARATE, independent choice. DEFAULT (v14): each candidate class's
  embedding is the MEAN of a fixed prompt-template ensemble for that class,
  L2-renormalized, computed ONCE per class (constant across the whole run) —
  `clip_metrics.encode_candidate_vocab_ensemble` /
  `clip_metrics.CLIPMATCH_CANDIDATE_TEMPLATES`. Two DIFFERENT template sets,
  never mixed: `OPENAI_TEMPLATES` (the official 80-template ImageNet ensemble
  from Radford et al. 2021 — object-centric, e.g. "a sculpture of a {}.",
  "a {} in a video game.") for ImageNet; a smaller 15-template
  `SCENE_TEMPLATES` for Places365 — the ImageNet set doesn't fit scene classes
  ("kitchen", "airport terminal"; a "tattoo of a kitchen" is nonsensical), and
  the original CLIP paper only used 2 bare templates for SUN397 scene
  recognition, which this deliberately improves on with contextual-anchoring
  and lighting/quality variants while staying scene-appropriate (no
  material/medium templates like sculpture/origami/embroidery/tattoo). Per-
  image extracted objects (Object-CLIPScore, ClipMatch's anchor-object search)
  still use the single `OBJECT_TEMPLATE` — ensembling 80 embeddings per
  extracted entity per image was never validated and would be needlessly
  expensive at 2M-image scale.
  EVERY template fill — `OBJECT_TEMPLATE`, all 80 ImageNet, all 15 scene —
  goes through `clip_metrics.fill_template`. DEFAULT is a plain
  `template.format(name)`: whatever article the template hardcodes goes in
  as written (e.g. "a photo of a apple"), ungrammatical or not. `--stage
  score`'s `--use_inflect_for_clipmatch` (v16, opt-in) instead fixes the
  determiner in front of the entity with `inflect`: "a photo of an apple",
  "a photo of a university" (a/an by SOUND, not spelling), "a photo of cars"
  (article DROPPED for a plural). Only an article that actually GOVERNS the
  slot is touched — "the plastic {}" and "itap of my {}" are left alone, and
  the "a" in "a photo of many {}" belongs to "photo", not the class, so it
  survives. Where an adjective intervenes ("a photo of a clean {}") the
  article agrees with the adjective and is kept, but still dropped for a
  plural. NOTE the history: v8 added an inflect determiner project-wide, v10
  reverted it on suspicion (never isolated; confounded with a concurrent
  MetaCLIP switch), v15 reinstated it as the hard default, v16 makes it
  opt-in via this flag instead of reopening that same v10 ambiguity — if a
  ClipMatch change is observed with the flag on, isolate this from any
  concurrent backend swap before attributing it.
  `--stage score`'s `--use_wordnet_definitions_clipmatch` is a SEPARATE,
  non-default opt-in that swaps the whole ensemble for a single richer
  WordNet lemma(s)+gloss prose per class (`clip_metrics.wordnet_definition_text`)
  — MEASURED (v12, back when the default was the single `OBJECT_TEMPLATE`
  phrase): no meaningful ClipMatch top-1 difference, so the "richer candidate
  text aligns better" hypothesis is not supported; kept as a flag rather than
  removed. Recorded in
  `summary["clip_models"]["clipmatch_candidate_text"]`
  (`"prompt_ensemble_<dataset>"` or `"wordnet_definition"`) and
  `summary["clipmatch_candidates"]` (token-length/truncation diagnostic, now
  over the flat per-template text count — `n_candidate_texts` — not just the
  class count `n_candidates`).
- **hP/hR/hF1** (hierarchical precision/recall/F1): ImageNet + Places only. Map
  the ClipMatch-predicted class onto a WordNet node via the extracted-object list
  (`resolve_to_wordnet`: rank objects by CLIP sim to the predicted class,
  Wu-Palmer disambiguation for polysemy), then score ancestral-closure overlap
  of the GT node vs. the predicted node. Reported as mean ± population std
  (`summary["hierarchical"]`/`["hierarchical_mapped"]`'s `*_std` keys) for hP,
  hR, hF1, AND Wu-Palmer — the mean alone doesn't say whether per-image scores
  cluster tightly around it or are widely spread.
  The phrase→synset step (`resolve_to_wordnet`, hardened v15) does, in order:
  (1) SPLIT alternation — "insect/larva" is two candidate names for one
  entity, not a lemma (split on `/` and ` or ` ONLY; never `and`/`,`, which
  would wreck "black and white"); (2) NORMALIZE — lowercase + strip diacritics,
  so "café" reaches its actual WordNet lemma `cafe` instead of resolving to
  nothing; (3) SEARCH SPANS — a whole-phrase lemma is AUTHORITATIVE if it
  exists ("swimming pool", "golden retriever" stay intact), otherwise try
  every contiguous span and single word and keep the MOST SPECIFIC (greatest
  `min_depth`). Specificity, NOT Wu-Palmer, ranks step 3 on purpose: Wu-Palmer
  structurally favours generic nodes (vs. a "brown bear" prediction, bare
  `bear` scores 0.963 but `polar bear` only 0.929), so ranking by it picks
  `retriever` over `golden retriever` and — the originally reported bug —
  `area.n.05` over `cafe.n.01` for "café seating area". Wu-Palmer is still
  used for word-SENSE disambiguation and to choose between step-1
  alternatives. The same diacritic strip is in `_normalize_object`
  (`src/vlm_pipeline.py`) so the MAPPING path doesn't miss "café" either —
  that one is inference-time, so it needs a re-run to take effect.
  ALSO reported split by whether the IMAGE'S OWN GT is nature or no-nature
  (`summary["hierarchical_by_gt_nature"]`/`["hierarchical_mapped_by_gt_nature"]`,
  each `{"nature": {...}, "no_nature": {...}}` with the same keys as the
  pooled `"hierarchical"`/`"hierarchical_mapped"` dicts) — the one pooled
  macro-average can hide a model that resolves nature GTs onto a fine-grained
  WordNet node far better (or worse) than no-nature ones.
- **ImageNet class-name collisions**: some ImageNet-1k WNIDs share a naive
  class name (`synset_name.split('.')[0]`) despite being ENTIRELY DIFFERENT
  WordNet synsets — e.g. n02963159 "cardigan" (the sweater, `cardigan.n.01`)
  vs n02113186 "Cardigan Welsh corgi" (a dog breed, `cardigan.n.02`); also
  n02012849 "crane" (the bird, `crane.n.05`) vs n03126707 "crane" (the
  machine, `crane.n.04`). Since `class_name` is what gets embedded as
  ClipMatch candidate text, two DIFFERENT classes with IDENTICAL text produce
  near-identical embeddings, so ClipMatch's argmax between them degenerates
  into noise — a genuinely correct prediction can land on the wrong sense's
  synset and get scored as flat-out wrong even though the model named the
  right thing. `dataset_loader._imagenet_class_names` disambiguates any
  colliding WNIDs (used by both `load_imagenet` and
  `get_candidate_vocab`'s imagenet branch, so GT and candidate text always
  agree): try each colliding synset's own LONGEST lemma name first (fixes
  cardigan — `cardigan.n.02`'s lemmas are `['Cardigan',
  'Cardigan_Welsh_corgi']`); if that still collides (e.g. both "crane"
  senses have only the single lemma "crane"), qualify with the synset's own
  immediate hypernym instead ("crane, a type of wading bird" vs "crane, a
  type of lifting device"); a synset-id suffix is the last-resort fallback,
  logged with a warning if it's ever actually reached. A unique naive name is
  returned unchanged — this only touches the WNIDs that actually collide.
- **Labeling parse-failure rate**: reported over the objects the VLM was
  ACTUALLY asked about (`vlm_called`), never over all extracted objects —
  human-term objects get no labeling call at all, so including them silently
  dilutes the rate. Split `_full` (unmapped OR mapped-non-nature →
  `TaxonomyResponse`, three axes) vs `_material` (mapped-nature →
  `MaterialResponse`, one axis) since the two use different schemas and can
  fail differently. Which call an object made is read off `label_route`, NOT
  off `mapped` — those stopped being equivalent when mapped-non-nature started
  going to the full call (`mapped` is still True for a node that resolved to
  non-nature). Scoring falls back to the old `mapped` heuristic only for
  artifacts written before `label_route` existed. Only `_material` is printed
  to the console; `_full` stays in the results JSON.

## Axis scoring (nature/biotic/material accuracy) — per dataset
- **ImageNet/Places (single-label)**: ClipMatch (PRIMARY text's CLIP embedding
  — the VLM's summary caption by default, raw caption if
  `--no_summarize_clipmatch_caption` — vs. candidate_vocab, global argmax — no
  lexical matching, no similarity threshold) picks the top-1 predicted class,
  restricted to classes mapped into the graph. That predicted class is then
  used ONLY to pick an ANCHOR among the
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
- **BIG-5 datasets** (`dataset_loader.BIG5_DATASETS`): `big5` pools every
  configured platform; `big5_twitter` / `big5_weibo` restrict to one. Use the
  per-platform names when comparing platforms — the results store and the
  predictions CSV are keyed by dataset name, so pooling would average them.
  GT CSVs: `--twitter_en_gt_csv` / `--twitter_es_gt_csv` (4 image slots per
  row) and `--weibo_ch0_gt_csv` / `--weibo_ch1_gt_csv` (NINE slots,
  `nature_visual_0..8`). The slot count is AUTO-DETECTED from each CSV's own
  `nature_visual_<idx>` columns (`_big5_slot_count`) — never hardcode it; the
  old hardcoded `min(4, ...)` silently dropped 47% of Weibo's annotated
  images. A slot whose `nature_visual_<idx>` is `-` is unannotated and skipped
  (660 of 7364 across the four CSVs). `nep_immaterial_specific_visual_<idx>`
  (illustration/infographic/videogame/plain_text/other) is deliberately
  IGNORED — those format subcategories are not a classification target.
  `platform_id` has a leading apostrophe STRIPPED on load (`load_big5`) —
  Excel's own "force text" marker (added so an id like `-3NEKN7YEcCmPzGy`
  isn't reinterpreted as a formula/number), which survives as a literal `'`
  character once exported to CSV; seen on the production Weibo annotation
  CSVs specifically (not every export has it — the earlier sample CSVs used
  to build this loader didn't). Left un-stripped, every `glob.glob()` image
  lookup misses (the marker isn't part of the real filename) and the whole
  source silently contributes zero images — caught in practice by
  `load_big5`'s own zero-match warning (see `n_annotated`/`n_matched`),
  which is what surfaced this bug.
- **BIG-5 (holistic, one GT label per image)**: nature = OR over extracted
  objects, same as always. biotic/material use a DIRECTION-AWARE "at least one
  matching entity" rule instead of matching a specific object (there is no
  named object to match — BIG-5's GT is one label for the whole scene):
  whichever value the GT actually is, correctness means the model output at
  least one nature-positive entity carrying THAT specific label
  (`has_biotic`/`has_abiotic` for biotic, symmetrically for material). An
  image can contain BOTH a biotic and an abiotic entity at once (e.g. a dog
  next to a rock) and still score correctly against a GT of just one of
  them — this deliberately looks at GT to pick which existence check
  (`has_biotic` vs `has_abiotic`) applies, unlike the nature axis, where "no
  nature entity output at all" is sufficient for a no-nature GT (there's no
  meaningful "found an explicit non-nature entity" signal to require there).
  GT ITSELF can also be BOTH directions at once: the majority-vote GT CSVs
  (`src.loaders.dataset_loader.load_big5`) carry coder-disagreement cells
  like "material; immaterial" — per Pau, that image genuinely counts as BOTH
  labels, not excluded, so `gt_biotic`/`gt_material` are LISTS (`[True]`,
  `[False]`, or `[True, False]` for disagreement), and EVERY element
  contributes its own separate GT instance scored against the same extracted
  entities — a disagreement image can add two rows (one per direction) to
  that axis's accuracy/precision/recall table instead of one.
- **COCO**: image-level nature = OR over extracted objects; biotic/material
  scored on the matched GT object via lexical matching (`find_matching_object`).

## COCO mask-IoU detection evaluation (`src/evaluation/detection_metrics.py`)
IMPLEMENTED. Runs in `--stage score` when the artifact is COCO **and** was
grounded with instance grounding; it runs ALONGSIDE the axis metrics above,
never replacing them.
- **PREDICTIONS COME FROM SAM3's SEMANTIC HEAD** (`semantic_seg`, read from
  `object_groundings[k]["mask_rle"]`) — already one region per extracted
  concept, and the SAME mask the nature relevance score uses, so "the region
  predicted for this concept" means one thing project-wide. The instance head
  is NOT read by detection at all. It previously was (unioned per entity to
  rebuild a concept mask), which was reconstructing something `semantic_seg`
  provides directly AND silently dropped every entity SAM3 confirmed
  semantically but produced no instance for — measured at 10.4% of confirmed
  groundings, each one charging its GT region as a false negative.
  CONSEQUENCES: `--instance_score_threshold` has NO effect on detection (the
  only gate is `--mask_threshold`); detection no longer requires
  `--instance_grounding`, only that grounding ran at all.
- **ENTITY (concept) GRANULARITY ONLY** (`score_image_entities`). The
  instance-level block was REMOVED — do not reintroduce it. This pipeline
  predicts CONCEPTS (the VLM extracts "orange slice", SAM3 grounds that
  concept) while COCO annotates individual objects, so instance matching
  punished two things that are not model errors: SAM3's undeduplicated
  duplicate queries on one object, and COCO's non-exhaustive annotation of
  crowded scenes (a tray of ~24 donuts carries 12 boxes, capping TPs at 12 and
  charging the correct extras as FPs). ACCEPTED COST: no per-object counting
  ("found 8 of 12 cows").
- Summary dicts, deliberately never merged: `summary["detection"]`
  (localization at the headline threshold), `summary["detection_iou_sweep"]`
  (the strictness curve — see below), `summary["detection_by_size"]` (COCO's
  small/medium/large split — see below), `summary["detection_labels"]` (naming),
  `summary["detection_axis_agreement"]` (biotic/material), and
  `summary["detection_nature_on_non_nature"]` (taxonomy disagreement). Console
  + W&B (`Detection/*`, `DetectionSweep/*`, `DetectionSize/*`,
  `DetectionLabels/*`) report all.
- GT is per-INSTANCE (`load_coco` now stores `gt_boxes` alongside the
  class-collapsed `targets`; `dataset_loader.coco_gt_boxes` back-fills them at
  scoring time from `--instances_json` so pre-existing artifacts need NO
  re-inference). Boxes are xyxy, converted from COCO's native xywh once, in
  `_coco_box_xyxy`.
- MATCHING IS ON **MASKS ONLY** — COCO's own `segm` task, not `bbox`. There is
  NO box-matching mode; it was tried and removed (see the recap) because it
  measures the wrong thing for a pipeline whose predictions ARE masks: two
  crossing diagonal strokes have box IoU **1.000** and mask IoU **0.000**
  (verified in the test suite). REQUIRES `--instances_json` — GT segmentation
  is read from the annotation file at scoring time, never stored in the
  artifact (it would bloat every record for a scoring-only use); without it,
  detection is skipped entirely rather than silently degrading to boxes.
  `--instances_json` GT WINS over the record's own `gt_boxes` whenever loaded,
  since only the former carries masks. `summary["detection"]["iou_type"]` is
  fixed provenance (`"mask"`) so a saved results JSON is self-describing
  against any older box-matched run. Every per-pair/FP/excluded/missed-GT
  entry in the predictions CSV carries `gt_mask_rle`/`pred_mask_rle` alongside
  the box (display-only) — `scripts/diagnose_detection_image.py` recomputes
  the real mask-IoU matrix from these to answer "why didn't X match Y"
  without needing the raw `.jsonl`.
- MATCHING IS CLASS-AGNOSTIC — Hungarian (not greedy), one-to-one, maximizing
  total IoU, threshold `--detection_iou_threshold` (default 0.5). This is the
  whole point: matching on class first (the standard detection protocol) would
  discard every cow/bull pair as FP+FN before the hierarchical metrics could
  see it. CONSEQUENCE: precision/recall/F1/AP here measure LOCALIZATION only;
  the naming side is `detection_labels`. Do not describe them as "detection
  accuracy" without that qualifier.
- GT restricted to NATURE-mapped COCO classes, because only nature-labeled
  entities are grounded — a `car` box could never be matched, so counting it
  as a miss would measure the protocol, not the model.
- UNMATCHED PREDICTION → FALSE POSITIVE when its own phrase names a class in
  the evaluated (nature-mapped) vocabulary; EXCLUDED otherwise and reported as
  `excluded_predictions` (COCO annotates 80 curated classes, so a correctly
  detected tree is not a hallucination). `build_coco_eval_vocab` MUST filter to
  nature exactly as the GT filter does — the two agreeing is what makes the
  rule fair. The old GEOMETRIC half of this test (`fp_reason ==
  entity_matched_elsewhere_in_image`) is GONE with the instance block: it
  existed to close a perverse incentive where misnaming a duplicate-heavy
  object improved precision, and at entity granularity each entity appears
  exactly once, so it could never fire.
- **ALWAYS quote `excluded_predictions` next to precision.** It is typically
  the MAJORITY of predictions (measured: 5727 of 7539 on the gemma COCO run,
  76%), because most nature the pipeline finds — tree, sky, grass, water — is
  not in COCO's 80 classes. Precision therefore describes only the minority
  COCO can adjudicate; it is NOT "x% of our masks are good". The console
  prints this ratio explicitly for exactly this reason. Recall has no such
  caveat.
- `iscrowd` regions: not detection targets, but still suppress FPs (IoA over
  the PREDICTION's area ≥ 0.5, not IoU — a crowd box dwarfs any one
  prediction). COCO's own convention.
- TAXONOMY-DISAGREEMENT DIAGNOSTIC (`summary["detection_nature_on_non_nature"]`,
  its own dict, never merged into the others): a grounded NATURE entity region
  overlapping (IoU ≥ threshold) a GT region whose COCO class maps to NON-nature.
  Non-nature GT is merged per class exactly as nature GT is, so the diagnostic's
  granularity matches the evaluation's.
  Can never be a TP (non-nature GT is filtered before matching) and is NOT
  charged as an FP either — these predictions stay in whichever bucket the
  normal rules gave them (nearly always `excluded`), so precision/recall are
  identical with or without this diagnostic. Exists because otherwise the case
  vanishes into `excluded`, indistinguishable from a mask over empty
  background. It is the Nature-Based Artefacts clause made measurable: COCO's
  `dining table` node is non-nature, but a wooden table with visible grain IS
  nature under the taxonomy, and a class node can't encode which one THIS
  image shows — so a hit is often the concept-vs-instance gap, not an error.
  Read `by_gt_class` to tell them apart (wood/stone furniture dominating =
  that clause firing; a flat spread of unrelated classes = loose masks).
- No NMS anywhere, and none is needed: SAM3 applies ONLY a score threshold (no
  dedup, verified in the transformers source), so several queries firing on the
  SAME object all survive — but the entity union collapses them by
  construction. The old `--instance_nms_iou` flag was REMOVED as dead (a union
  is identical with or without suppression); don't re-add it.
- **IoU SWEEP — THE HEADLINE** (`summary["detection_iou_sweep"]`,
  `detection_metrics.sweep_summary`). Precision/recall/F1 recomputed at EVERY
  rung of COCO's ladder (0.50, 0.55 … 0.95) off ONE cached IoU matrix, each
  rung getting its own independent matching pass. Reported at `@0.50`, `@0.75`,
  and `@[.50:.95]` (mean over the ladder) as `precision_50`/`_75`/`_50_95` etc.,
  plus the full `per_iou` table. This curve is a direct readout of **MASK
  TIGHTNESS**: an F1 that holds from 0.50 to 0.75 means the masks genuinely
  trace their objects; one that collapses means blobs that only just cleared
  the permissive threshold.
- **OBJECT-SIZE SPLIT** (`summary["detection_by_size"]`,
  `detection_metrics.size_summary` / `COCO_AREA_RANGES` / `area_bucket`):
  P/R/F1 split by COCO's OWN size buckets — small < 32² px, medium < 96² px,
  large beyond — taken verbatim from `pycocotools`'
  `Params(iouType="segm").areaRng`. This is NOT a project invention: it is the
  area stratification behind the AP^small/AP^medium/AP^large every COCO
  submission reports, and cocoeval applies these exact cut-offs to the `segm`
  task using MASK area. Bucket assignment follows `cocoeval.evaluateImg`: a
  TP/FN takes its **GT** region's bucket, an FP takes its **own** predicted
  region's bucket. Consequence worth relying on: the three buckets PARTITION
  the `detection_iou_sweep` totals and sum back to them exactly (verified in
  tests) — cocoeval itself doesn't have that property, since it re-matches per
  area range with ignore flags.
  **GT'S BUCKET USES THE MEAN OF ITS PRE-MERGE INSTANCE AREAS, not the merged
  region's area.** This pipeline scores CONCEPT regions, so a GT entry is
  every annotated instance of one class merged into one region for MATCHING —
  but bucketing by the merged area would put a bowl of ten small oranges in
  "large" because the union spans the whole bowl, even though every orange is
  "small". Averaging the ORIGINAL, pre-merge instance areas (still available
  before the union — `gt_by_class` retains each instance's own RLE) keeps the
  bucket representative of a typical object of that class. **PREDICTIONS
  CANNOT GET THE SAME FIX**: an FP is bucketed by its own predicted area,
  necessarily the whole blob's, because `semantic_seg` produces one dense mask
  with no instance boundaries to decompose — there is no "individual predicted
  instance" to average. This is a real, unavoidable asymmetry, not an
  oversight: a model that over-generates one big blob for a flock of small
  birds is charged an FP against "large" even though the underlying objects
  were small. `n_gt` per bucket is reported so a weak bucket reads as weak
  rather than low-support.
- **NO AP — deliberate, do not add one back.** AP ranks predictions by
  confidence and integrates the PR curve, so it needs a per-prediction
  confidence. `semantic_seg` is a dense logit map with no scalar score per
  concept mask. An earlier version scored the instance head and ranked by each
  entity's strongest instance — an engineering convenience, not a measurement
  ("the strongest instance that happened to clear the score gate", not a
  calibrated confidence for the region being matched). Manufacturing a
  substitute (mean sigmoid in the mask, pixel count, …) would be worse than
  reporting nothing. The IoU sweep is the headline and needs no confidence.
  `average_precision` and `mask_nms` were deleted from `detection_metrics.py`
  along with this.
- NO TRUE NEGATIVES, so no accuracy is reported — unlike the axis metrics,
  detection has no finite negative class. Don't add one.
- PIXEL COVERAGE (`summary["detection_pixels"]`, `pixel_stats`/`pixel_summary`)
  was REMOVED — do not reintroduce. Two reasons: (1) its `by_gt_class` view
  pooled only TP pairs + missed GT and never the unmatched/excluded
  predictions, so its per-class `pixel_precision`/`pixel_iou` structurally
  excluded the dominant source of imprecision and could not be reconciled with
  the whole-image `mean_image_iou` (observed: worst class 0.51 while mean image
  IoU was 0.32); (2) the IoU sweep answers "how well do the masks trace the
  objects" more cleanly, without mixing in every unmatched prediction's pixels.
- LABEL SCORING, per matched pair only: `exact_match` (same
  `phrase_matches_terms`/`gt_match_terms` test the extraction-hit diagnostic
  uses) AND hP/hR/hF1 + Wu-Palmer against the GT synset, so "bull" for "cow"
  gets partial credit (~0.94 hF1) instead of a flat zero. Reported pooled
  (resolution failures as 0.0) and `_resolved`-only, both mean ± population
  std — same convention as the ImageNet/Places hierarchical block.
  `phrase_matches_terms` matches the whole normalized phrase OR any GT term
  as a TRAILING SPAN — not just the single trailing word — so a modifier in
  front of a multi-word term ("huge potted plant" vs "potted plant") still
  counts as exact; requiring the match at the very end (not anywhere in the
  phrase) is what keeps "cow shed" from wrongly matching "cow". The same
  span-based surface-form set (`_pred_label_terms`) feeds the curated-
  vocabulary FP-vs-excluded test below, for the identical reason.
- AXIS AGREEMENT (`summary["detection_axis_agreement"]`, biotic/material
  only): for the SAME matched pairs, compares the predicted entity's own
  hybrid label (`object_finals`) against the GT box's taxonomy position —
  read off the geometric box correspondence, a tighter binding than the
  lexical `find_matching_object` the plain axis metrics use (matters with
  several same-class instances in one image). NATURE has no entry: every
  matched pair is nature-vs-nature by construction (only nature entities are
  grounded, GT here is nature-restricted), so an agreement rate for it would
  misreport a tautology as a measurement. A `None` on either side (unmapped
  GT, unresolved VLM label) is dropped from that axis's support, not counted
  as disagreement.
- NO GT LEAKAGE: the predicted phrase resolves via
  `taxonomy_metrics.resolve_phrase_to_wordnet(phrase, anchor_synset_id=None)`.
  The anchor is deliberately withheld — the only one available is the GT class,
  and steering sense disambiguation with it would inflate the very hP/hR it
  feeds. (`resolve_to_wordnet`'s own anchored use is safe: there the anchor is
  ClipMatch's PREDICTION, so a wrong anchor drags the score down.) Anchorless
  falls back to most-frequent-sense, with WordNet INSTANCE synsets (proper
  nouns) deprioritized — bare MFS resolves "crane" to Stephen Crane the
  writer. That filter is GT-independent, so it introduces no leakage.
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
- BIG-5 data: `/home/pmonserrat/datasets/big_5/` — TWO platforms, each a flat
  image folder: `.../big_5/twitter` (`--big_5_twitter_images_dir`) and
  `.../big_5/weibo` (`--big_5_weibo_images_dir`). No intra-folder split by
  language or channel; every image is named `<platform_id>_<idx>.<ext>` and
  found by globbing that stem.
- Imagenet data: `/home/pmonserrat/datasets/imagenet/`
- COCO data: `/home/pmonserrat/datasets/coco/`
- Places365 data: `/home/pmonserrat/datasets/places/`
- Dev-loop model: Qwen/Qwen3.5-0.8B (architecture-search proxy — not a
  performance benchmark; spot-check final config on larger models before
  locking in)

## Fine-tuning (rejection sampling) — `fine_tuning/`
ALL fine-tuning code lives in that folder; its `README.md` carries the full
rationale. Hard conventions:
- SCOPE: LoRA (`peft`) on the LANGUAGE DECODER only. The vision tower and
  multimodal projector stay frozen — `train_lora.py` ASSERTS this (aborts if
  any trainable parameter is outside the decoder), because the
  visual-embedding cache is only valid while they are.
- PLAIN LoRA IS THE DEFAULT, `--use_dora` IS OPT-IN — a serving decision, not
  a training one. vLLM's native LoRA adapter serving does not support DoRA
  weights (`vllm-project/vllm#10849`, open; the one PR that tried, #14389, is
  closed/unmerged), while it fully supports plain LoRA with per-request
  hot-swap at ~ms overhead. This project needs exactly that: the adapter
  applied to SOME calls in a run (extraction + labeling) while captioning
  always runs untouched base weights, on ONE resident vLLM engine, in a
  SINGLE pass over the dataset. Passing a DoRA-trained adapter to that path
  would not error — it would silently apply plain-LoRA math to DoRA weights.
  Use `--use_dora` only if you intend to `merge_adapter.py` it and serve
  every call adapted (including captioning).
- SELECTIVE ADAPTER SERVING: `src.models.vlm_models.VLLMBackedVLM` accepts
  `lora_adapter_path` (constructs the engine with `enable_lora=True` and a
  `LoRARequest`) and every `generate`/`generate_batch` call takes a
  `use_lora` flag routed through vLLM's `lora_request=` per call — NOT a
  second engine load. `src.vlm_pipeline.caption_batch` and
  `summarize_caption_batch` take NO `use_lora` parameter at all (structurally
  cannot apply the adapter, mirroring the existing "no system prompt on the
  caption call" convention); `extract_objects_batch`/`label_objects_batch`/
  `run_inference` do, defaulting False. `run_vlm_pipeline.py
  --lora_adapter_path <adapter dir>` wires this up end-to-end and records
  `lora_adapter_path`/`lora_adapter_name` in the artifact header for
  provenance.
- SPLITS (`make_splits.py` → `/home/pmonserrat/datasets/big_5/rft/splits/`, and
  the RFT training sets `build_rft_dataset.py` builds from them,
  `/home/pmonserrat/datasets/big_5/rft/rft_<model>_<balance>/`): 70/10/20,
  Twitter+Weibo POOLED, grouped by POST (`<platform_id>_<slot>` filename
  stem) — never by image, since images within a post are near-duplicates.
  Both live under `datasets/big_5/` alongside the raw images/annotations,
  NOT under the code repo's `data/` — these are data, not code. Derived from
  image paths + GT only, never from model output, so ONE split file serves
  the self-training run and every future distillation run. WRITE-ONCE:
  regenerating with a different `--seed` invalidates every number reported
  against it.
- ACCEPTANCE (`rft_common.image_verdict`): GT non-nature → accepted iff ZERO
  nature entities predicted. GT nature → accepted iff at least ONE nature
  entity carries both a matching life_category and tangibility (`strict`,
  default); `lenient` satisfies the two axes independently and is the rule
  `run_vlm_pipeline.py`'s BIG-5 branch reports with.
- REJECTION SAMPLING IS CLASS-BIASED and the bias runs the wrong way: measured
  93.5% of nature images accepted vs 55.2% of non-nature ones, because
  over-predicting nature is the model's own failure mode. ALWAYS state which
  `--balance` mode (`none` / `downsample_nature` / `loss_weight`) produced a
  number; the first run uses `downsample_nature`.
- TRAINING EXAMPLES are reconstructed EXACTLY from the artifact, importing the
  prompt builders from `src.models.prompts` rather than copying them. The
  extraction example follows the SOURCE ARTIFACT's own `caption_stage` header:
  a `--no_caption` artifact rebuilds from `get_extraction_prompt(no_caption=
  True)`, never `EXTRACTION_PROMPT.format(caption="")` — otherwise training
  sees a prompt shape (empty quoted description) that never occurs at
  inference. Threaded per artifact, since one build can merge several. Stages:
  extraction + both labeling calls; the free-form CAPTION stage is deliberately
  excluded (and errors if requested, rather than being silently dropped).
- `TaxonomyResponse`'s two reasoning fields are now stored SEPARATELY in the
  artifact (`nature_reasoning`/`sub_axes_reasoning`) alongside the space-joined
  `reasoning`. `rft_common.split_taxonomy_reasoning` recovers the split from
  older artifacts and DROPS anything it cannot split rather than fabricating a
  boundary.
- `run_vlm_pipeline.py --split_file <file>` restricts a run to the image paths
  listed in a file (basename-matched), in BOTH `--stage infer` and
  `--stage score` — how the fine-tuned model is scored on the held-out test
  split, and how the baseline artifact is re-scored on that same split for the
  comparison.
- The fine-tuned model is EVALUATED THROUGH THE NORMAL PIPELINE:
  `--model_name <base>` + `--lora_adapter_path <adapter dir>` for the default
  LoRA case (no merge needed); `merge_adapter.py` + `--model_name <merged>`
  only for a `--use_dora` adapter or a standalone-checkpoint need.

## Current focus
Baseline VLM pipeline is IMPLEMENTED end-to-end (caption → extraction →
mapping-routed labeling → hybrid resolution → metrics: F-CLIPScore,
Object-CLIPScore, per-axis acc/P/R/F1, ClipMatch + hP/hR on ImageNet/Places).
Grounding pipeline is IMPLEMENTED end-to-end (SAM3 semantic segmentation of
nature entities → RLE masks → nature relevance score), enriching the same
artifact. COCO's box-IoU detection evaluation is IMPLEMENTED (SAM3 instance
boxes vs COCO GT boxes, class-agnostic Hungarian matching, then exact-match +
hierarchical label scoring on the matched pairs) — NOT yet run on real data:
the numbers it produces are unvalidated until a real SAM3 COCO run exists.
Next: spot-check both pipelines on Qwen3.5-0.8B + a real SAM3 run
(confirm the semantic_seg range empirically with --debug_semantic_range and
tune --max_pairs_per_forward to the GPU), hand off to Ramin for the BSC infra
check, then the sequential ablations (recap §7).