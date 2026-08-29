"""
src/grounding_pipeline.py

Grounding pipeline (geometric/pixel-based), the second half of the BIG-5
nature-detection design: it takes the VLM pipeline's per-image entity list and
answers a question the language pipeline cannot — *how much of the image is
actually nature*.

    caption -> extraction -> labeling      (src/vlm_pipeline.py, done already)
        |
        |  for each entity with final_nature == True
        v
    SAM3 semantic segmentation             (this file, stage 1)
        |
        v
    nature relevance score                 (this file, stage 2)

Stage 1 — SAM3 GROUNDING
  One model for everything: `facebook/sam3` via `transformers`. There is NO
  thing/stuff routing, NO separate detector, NO separate segmentation model —
  the earlier Grounding-DINO-for-things / open-vocab-segmenter-for-stuff split
  (and the WordNet-lexname/COCO-Panoptic/LLM routing step that fed it) is
  dropped entirely; SAM3's single text-prompted head covers both cases.

  We read `outputs.semantic_seg` (the CONCEPT-level map), NOT the instance-level
  `pred_masks`/`pred_boxes`/`pred_logits`: the relevance score is about total
  pixel coverage of a concept, not how many instances of it there are.

  IMPORTANT — `semantic_seg` IS RAW LOGITS, NOT PROBABILITIES. Verified against
  the installed `transformers` source, not assumed:
    * `Sam3MaskDecoder.forward` produces it as a bare
      `self.semantic_projection(pixel_embed)` — no activation applied
      (transformers/models/sam3/modeling_sam3.py).
    * `Sam3ImageProcessor.post_process_semantic_segmentation` opens with
      `semantic_probs = semantic_logits.sigmoid()`
      (transformers/models/sam3/image_processing_sam3.py).
  So the range is UNBOUNDED and `torch.sigmoid()` is required before
  thresholding. Rather than re-implement that, we call the processor's own
  `post_process_semantic_segmentation(...)`, which does exactly the documented
  contract: sigmoid -> bilinear-resize to the ORIGINAL image size -> binarize at
  `threshold` (SAM3's stated default, 0.5). Pass `debug_semantic_range=True` to
  SAM3Grounder to print the raw pre-sigmoid `min()`/`max()` for the first few
  forward passes and re-confirm this empirically on real data.

  BATCHING — what SAM3 actually supports (checked in the installed
  transformers, since the shape matters and guessing it would be wrong):
    * `Sam3Processor.__call__` tokenizes `text` with
      `padding="max_length", max_length=32`, and `Sam3Model.forward` shares ONE
      batch dimension between `pixel_values` and `input_ids`. `semantic_seg`
      comes back as `(batch_size, 1, H, W)` — a single concept map per batch
      element.
    * Therefore SAM3 CANNOT take a ragged per-image list of text prompts and
      segment all of an image's entities in one forward pass. One text prompt
      resolves to one semantic map, full stop. There is no padding/ragged
      convention to exploit here — the limit is the model head, not the
      processor's input handling.
    * What we do instead: flatten to (image, entity) PAIRS, repeating the image
      once per entity, and batch those pairs. `--batch_size` still counts
      IMAGES (so it means the same thing as in the VLM pipeline), with
      `max_pairs_per_forward` capping how many pairs go to the GPU at once so a
      busy image with many nature entities can't blow up VRAM.
    * Possible future optimization (not done — kept simple and obviously
      correct): `Sam3Model.forward` accepts precomputed `vision_embeds` instead
      of `pixel_values`, so the vision backbone could run ONCE per image and be
      expanded across that image's entities, instead of re-encoding the same
      image once per entity.

Stage 2 — NATURE RELEVANCE SCORE
  Union of every surviving nature mask (RLE-merged, so pixels shared by two
  entities are counted once), scored by BOTH methods at once (not a
  selectable either/or — cheap to compute together off the same union mask,
  and storing both lets downstream analysis/evaluation compare them without
  re-running grounding):
    * coverage_ratio  — nature pixels / total pixels, in [0, 1].
    * center_weighted — each pixel weighted by a Gaussian in its distance from
      the image center before summing, normalized by the total weight so the
      result stays in [0, 1]. Rationale: centrally-placed content is more likely
      to be the compositional focus of the image.

Storage convention: this pipeline does NOT write its own separate output file.
It reads the VLM pipeline's JSON-Lines artifact and ENRICHES each image record
in place with its own fields, so one artifact holds everything — entities,
labels, masks and the relevance scores — after both stages have run. Per image
record it adds:

    "object_groundings": [ per-object dict aligned with record["objects"] ],
    "nature_relevance_score_coverage_ratio": float,
    "nature_relevance_score_center_weighted": float,

`object_groundings[k]` aligns index-for-index with `objects[k]` /
`object_labels[k]` / `object_finals[k]`, exactly like the VLM pipeline's own
parallel lists. Each entry is:

    {
      "object": str,             # the extracted phrase, as stored
      "prompt": str | None,      # normalized text prompt actually sent to SAM3
      "is_nature": bool,         # final_nature — only these are grounded
      "grounded": bool | None,   # None = never attempted (not nature)
      "mask_rle": {...} | None,  # pycocotools RLE, only when grounded
      "pixel_count": int,        # 0 when not grounded
    }

Entities that fail SAM3's threshold are MARKED (`grounded: false`), never
deleted — the full VLM output stays auditable even where grounding could not
confirm the entity's presence.

Stage 1b — INSTANCE GROUNDING (opt-in, `instance_grounding=True`; COCO)
  Everything above is CONCEPT-level: one mask per entity, however many
  individual cows it covers. COCO's ground truth is INSTANCE-level (one box
  per annotated object), so evaluating against it needs the other read-out —
  SAM3's `pred_masks`/`pred_boxes`/`pred_logits`, thresholded per instance.
  Every COCO target class is a "thing" (countable, boundable), which is what
  makes the instance head the right head there; the semantic head remains the
  right one for the relevance score on every dataset, including COCO.

  This is NOT a second model run. Both read-outs come off the SAME
  `Sam3ImageSegmentationOutput` from the SAME forward pass — see
  `SAM3Grounder.segment_pairs_full` — so turning it on adds post-processing
  cost only, never a second forward pass, and leaves `object_groundings` and
  both relevance scores bit-for-bit what they would have been without it.

  Adds one more per-image field, again aligned index-for-index with `objects`:

    "object_instances": [
      {
        "object": str,
        "prompt": str | None,
        "is_nature": bool,
        "attempted": bool,          # False = SAM3 never ran for this entity
        "instances": [
          {
            "score": float,         # sigmoid(pred_logits) * presence
            "bbox": [x1,y1,x2,y2],  # TIGHT box of the instance mask (scored)
            "sam3_bbox": [...],     # SAM3's own regressed box, for comparison
            "mask_rle": {...},
            "pixel_count": int,
          }, ...
        ],
      }, ...
    ]

  Metrics over these boxes live in src/evaluation/detection_metrics.py, not
  here — this file stays free of metric computation.

NO metric computation lives here, and no argparse/CLI — see
scripts/run_grounding_pipeline.py (grounding only) and scripts/run_pipeline.py
(VLM inference -> grounding, end to end).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import BatchProgress
from src.vlm_pipeline import _normalize_object

# SAM3's own stated default binarization threshold for the semantic map (also
# the default of Sam3ImageProcessor.post_process_semantic_segmentation).
DEFAULT_MASK_THRESHOLD = 0.5

# SAM3's own stated default confidence threshold for the INSTANCE head
# (Sam3ImageProcessor.post_process_instance_segmentation's `threshold`). Note
# this gates instance CONFIDENCE, not pixels — DEFAULT_MASK_THRESHOLD above
# still binarizes each kept instance's mask.
DEFAULT_INSTANCE_SCORE_THRESHOLD = 0.3

# HuggingFace repo for the single model this whole stage runs on.
SAM3_MODEL_ID = "facebook/sam3"

# The two relevance-score methods, both ALWAYS computed together (see
# nature_relevance_scores) — this tuple just names the dict keys/CSV suffixes
# they land under, it's no longer a CLI "choose one" selector.
RELEVANCE_METHODS = ("coverage_ratio", "center_weighted")

# Default width of the center_weighted Gaussian, in units where the image center
# is 0 and each edge midpoint is 1 (see center_weighted_score). 0.5 means the
# weight has fallen to exp(-2) ≈ 0.135 by the edge midpoint — clearly favoring
# the middle of the frame without zeroing the borders out entirely.
DEFAULT_CENTER_SIGMA = 0.5


# =============================================================================
# JSON-Lines artifact I/O (the SHARED per-image file, enriched in place)
# =============================================================================
def stream_artifact(path):
    """Read a VLM-pipeline JSON-Lines artifact lazily: returns
    `(header, footer_holder, record_iterator)`.

    The artifact's shape (written by run_vlm_pipeline.phase_infer) is one JSON
    object per line: a `record_type="header"` line first, then one
    `record_type="image"` line per image, then a `record_type="footer"` line
    carrying timings. Streaming rather than loading the whole file keeps memory
    flat over a 2M-image run, matching how phase_infer wrote it in the first
    place.

    `footer_holder` is a one-element list that gets filled in once the iterator
    reaches the footer line (a plain return value can't work — the footer is the
    LAST line, so it isn't known until the iterator is exhausted).
    """
    footer_holder: List[Optional[Dict[str, Any]]] = [None]
    f = open(path)
    header = None
    # Find the header first, so the caller has the run metadata (dataset, model,
    # max_hops, candidate_vocab) before consuming any image record.
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("record_type") == "header":
            header = obj
            break
        # An artifact whose first non-empty line isn't a header is malformed —
        # bail loudly rather than silently grounding a file we can't identify.
        f.close()
        raise ValueError(f"Artifact {path} does not start with a header line.")
    if header is None:
        f.close()
        raise ValueError(f"Artifact {path} has no header line.")

    def _records():
        try:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rtype = rec.get("record_type")
                if rtype == "footer":
                    footer_holder[0] = rec
                    continue
                if rtype == "header":
                    continue  # defensive: a second header line is ignored
                yield rec
        finally:
            f.close()

    return header, footer_holder, _records()


# =============================================================================
# RLE helpers (pycocotools)
# =============================================================================
def _mask_utils():
    """Import pycocotools' mask module lazily, so this file stays importable
    (and unit-testable for the pure-math helpers) without the C extension."""
    from pycocotools import mask as mask_utils
    return mask_utils


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """RLE-encode a boolean HxW mask via pycocotools, JSON-safely.

    pycocotools returns `counts` as raw BYTES, which `json.dumps` cannot
    serialize — decode it to an ASCII string here so the RLE can be written
    straight into the artifact. `decode_rle` re-encodes it on the way back, so
    the round trip is lossless.
    """
    mask_utils = _mask_utils()
    # asfortranarray: pycocotools requires column-major (Fortran-order) input.
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
    }


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Inverse of encode_rle: JSON-shaped RLE -> boolean HxW mask."""
    mask_utils = _mask_utils()
    counts = rle["counts"]
    native = {"size": list(rle["size"]),
              "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    return mask_utils.decode(native).astype(bool)


def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """`tensor.detach().cpu().numpy()`, made safe for a bfloat16 tensor.

    NumPy has no bfloat16 representation at all, so `.numpy()` on one raises
    `TypeError: Got unsupported ScalarType BFloat16` — hit in production when
    `--dtype bfloat16` (meant for the VLM) also got inherited by SAM3Grounder
    (run_pipeline.py's `--dtype` is a flag BOTH stages declare; without an
    explicit `--grounding_dtype` override, the shared value applies to SAM3
    too — see the module's `--grounding_dtype` CLI help). `.float()` first is
    a no-op for a tensor that's already float32, bool, or long (the dtypes
    every one of this file's four `.numpy()` call sites actually expects), so
    this is a strict safety net, not a behavior change for the common case —
    and it fixes the crash regardless of which exact op in a given
    transformers version happens to preserve the model's low-precision dtype
    through to the tensor this file reads.
    """
    return tensor.detach().cpu().float().numpy()


def mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    """Tight axis-aligned bounding box of a boolean mask, as
    `[x1, y1, x2, y2]` — the same convention COCO GT is converted to
    (`dataset_loader._coco_box_xyxy`) and the same one SAM3's own `pred_boxes`
    uses, so all three are directly IoU-comparable.

    Returns None for an all-False mask (no box exists).

    Coordinates are EXCLUSIVE on the far edge: a mask whose only set pixel is
    column 5 yields x1=5, x2=6, i.e. a width of 1 pixel rather than 0. Getting
    this wrong collapses single-pixel-wide masks to zero area and silently
    zeroes their IoU.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def union_mask(rles: Sequence[Dict[str, Any]], height: int, width: int) -> np.ndarray:
    """OR every entity's mask together into one nature mask.

    Uses pycocotools' own `merge` (union of RLEs) rather than decoding each mask
    and OR-ing dense arrays, so overlapping pixels between two entities (e.g.
    "tree" and "leaves" covering the same region) are counted exactly ONCE — the
    whole point of taking a union rather than summing per-entity pixel counts.
    """
    if not rles:
        return np.zeros((height, width), dtype=bool)
    mask_utils = _mask_utils()
    native = []
    for rle in rles:
        counts = rle["counts"]
        native.append({"size": list(rle["size"]),
                       "counts": counts.encode("ascii") if isinstance(counts, str) else counts})
    merged = mask_utils.merge(native, intersect=False)
    return mask_utils.decode(merged).astype(bool)


# =============================================================================
# Stage 2 — nature relevance score
# =============================================================================
def coverage_ratio_score(mask: np.ndarray) -> float:
    """Fraction of the image's pixels covered by the nature-mask union, in
    [0, 1]. The simplest reading of "how much of this image is nature"."""
    if mask.size == 0:
        return 0.0
    return float(mask.sum() / mask.size)


def center_weighted_score(mask: np.ndarray, sigma: float = DEFAULT_CENTER_SIGMA) -> float:
    """Gaussian center-weighted nature coverage, in [0, 1].

    Each pixel is weighted by `exp(-d² / 2σ²)`, where d is its distance from the
    image center in NORMALIZED coordinates: the center is 0 and each edge
    midpoint is 1 on its own axis, so the weighting is identical for portrait,
    landscape and square images instead of being distorted by aspect ratio. The
    score is the weighted nature area divided by the TOTAL weight of the image,
    which keeps it bounded in [0, 1] and makes it directly comparable to
    coverage_ratio_score (an all-nature image scores 1.0 under both).

    Motivation (recap §1.B.4): centrally-placed content is more likely to be the
    compositional focus of the image, so nature filling the middle of the frame
    should count for more than nature in a corner.
    """
    if mask.size == 0:
        return 0.0
    height, width = mask.shape
    # Pixel CENTERS (the +0.5) mapped onto [-1, 1] per axis.
    ys = (np.arange(height) + 0.5) / height * 2.0 - 1.0
    xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    d2 = ys[:, None] ** 2 + xs[None, :] ** 2
    weights = np.exp(-d2 / (2.0 * sigma ** 2))
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[mask].sum() / total)


def nature_relevance_scores(
    rles: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    center_sigma: float = DEFAULT_CENTER_SIGMA,
) -> Dict[str, float]:
    """Nature relevance for one image, from its surviving entities' RLE masks,
    under BOTH scoring methods at once — {"coverage_ratio": ..., "center_weighted": ...}.
    Both are computed off the SAME union mask (one extra O(H*W) reduction is
    negligible next to the union/RLE-merge cost), so there's no reason to make
    a caller choose only one and re-run grounding to get the other.

    An image with no grounded nature entity scores 0.0 under both — a genuine
    "no nature found here" verdict, not a missing value.
    """
    if not rles:
        return {"coverage_ratio": 0.0, "center_weighted": 0.0}
    merged = union_mask(rles, height, width)
    return {
        "coverage_ratio": coverage_ratio_score(merged),
        "center_weighted": center_weighted_score(merged, sigma=center_sigma),
    }


# =============================================================================
# Stage 1 — SAM3 grounding
# =============================================================================
def entity_prompt(object_str: str) -> str:
    """Turn an extracted object phrase into the short noun phrase SAM3 wants as
    a text prompt.

    Reuses the VLM pipeline's own `_normalize_object` (lowercase + strip ONE
    leading article) so the phrase SAM3 is asked about is normalized exactly the
    same way the phrase that was mapped to WordNet was — "A Wooden Table" is
    prompted as "wooden table". Falls back to the original string if
    normalization empties it.
    """
    norm = _normalize_object(object_str)
    return norm or object_str.strip()


class SAM3Grounder:
    """Thin wrapper around `facebook/sam3` (`Sam3Model` + `Sam3Processor`,
    loaded explicitly — see `__init__`'s comment on why not AutoModel/
    AutoProcessor) exposing
    exactly one operation: segment a batch of (image, text-prompt) pairs into
    binary masks at each image's ORIGINAL resolution.

    Kept deliberately separate from the record-enrichment logic in
    `ground_records` (same split as clip_metrics.CLIPScorer vs. the metric math),
    so the pure-numpy scoring helpers above stay unit-testable without loading
    torch or downloading a multi-GB checkpoint.
    """

    def __init__(
        self,
        model_name: str = SAM3_MODEL_ID,
        device: str = "cuda",
        dtype: str = "auto",
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        instance_score_threshold: float = DEFAULT_INSTANCE_SCORE_THRESHOLD,
        debug_semantic_range: bool = False,
        hf_token: Optional[str] = None,
    ) -> None:
        import os

        import torch
        # EXPLICIT classes, never AutoModel/AutoProcessor — confirmed a real
        # regression in production: on a newer transformers release,
        # AutoModel.from_pretrained("facebook/sam3") resolves to
        # Sam3VideoModel (SAM3's video/tracking head — its forward() requires
        # an `inference_session` we never construct), not the image model this
        # file actually needs. facebook/sam3's own model card and the
        # transformers SAM3 doc page both load it as Sam3Model/Sam3Processor
        # explicitly for exactly this image+text-prompt use case — there is no
        # AutoModel example anywhere in the official usage. Importing the
        # concrete classes removes the ambiguity outright rather than hoping
        # AutoModel's mapping stays pointed at the image variant.
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.device = device
        self.mask_threshold = mask_threshold
        # Confidence gate for the INSTANCE head only — a separate knob from
        # mask_threshold (which binarizes pixels). See segment_pairs_full.
        self.instance_score_threshold = instance_score_threshold
        # When True, print the RAW (pre-sigmoid) semantic_seg min/max for the
        # first few forward passes. The sigmoid question is already settled from
        # the transformers source (see the module docstring — semantic_seg is
        # logits), but this makes it verifiable on real images/checkpoint rather
        # than taken on faith, and costs nothing when left off.
        self.debug_semantic_range = debug_semantic_range
        self._debug_batches_left = 2 if debug_semantic_range else 0

        dtype_kwargs = {}
        if dtype and dtype != "auto":
            dtype_kwargs["torch_dtype"] = getattr(torch, dtype)
        # facebook/sam3 is a gated repo — needs an authenticated HF token with
        # accepted license terms. NEVER hardcode the token: pass it explicitly
        # (--hf_token, plumbed from run_grounding_pipeline.py) or, by default,
        # pick it up from the HF_TOKEN env var / `huggingface-cli login`'s
        # cached token (huggingface_hub's own resolution order — passing
        # token=None here still lets that automatic lookup happen).
        # `or None` is load-bearing: an EMPTY HF_TOKEN (e.g. a job script doing
        # `export HF_TOKEN="${HF_TOKEN:-}"` with nothing set) would otherwise be
        # passed through as "", and huggingface_hub sends a literal
        # "Authorization: Bearer " header for it — httpx rejects that with
        # `LocalProtocolError: Illegal header value b'Bearer '`. Worse, an empty
        # string SUPPRESSES the automatic fallback below. Collapsing "" to None
        # restores it, so an unset-or-empty token means "use the cached
        # `hf auth login` credential", which is what a caller always intends.
        token = hf_token or os.environ.get("HF_TOKEN") or None
        self.processor = Sam3Processor.from_pretrained(model_name, token=token)
        self.model = Sam3Model.from_pretrained(model_name, token=token, **dtype_kwargs)
        self.model.eval().to(device)

    def _load_image(self, path: str):
        from PIL import Image
        return Image.open(path).convert("RGB")

    def _encode(self, images, texts):
        """Build SAM3 model inputs from a batch of images + text prompts,
        WITHOUT going through `Sam3Processor.__call__`'s own dispatch.

        `Sam3Processor.__call__(images=..., text=..., ...)` is the documented
        one-call entry point, and its OWN implementation (verified against
        transformers 5.14.1's source) is exactly the two calls below: the
        image processor on `images`, the tokenizer on `text` (padded to
        `max_length=32`, SAM3's own convention), merged into one encoding.
        Calling it that way ourselves instead of through the top-level
        `__call__` sidesteps a real regression hit in production: a newer
        transformers release changed how `Sam3Processor.__call__` routes its
        `**kwargs` internally, and `text` started leaking into the IMAGE
        processor's own kwargs validation — `Sam3ImageProcessorKwargs.
        __init__() got an unexpected keyword argument 'text'` — failing every
        single grounding forward pass while looking, from this file's own
        try/except, like an ordinary per-batch failure (see `_ground_batch`).
        Doing the two calls explicitly means this file no longer depends on
        that internal routing at all, so it can't regress the same way again
        even if `Sam3Processor.__call__` itself keeps changing.
        """
        image_inputs = self.processor.image_processor(images, return_tensors="pt")
        text_inputs = self.processor.tokenizer(texts, return_tensors="pt",
                                               padding="max_length", max_length=32)
        image_inputs.update(text_inputs)
        return image_inputs

    def segment_pairs(self, pairs: Sequence[Tuple[Any, str]]) -> List[np.ndarray]:
        """Segment a flat list of (PIL image, prompt) pairs in ONE forward pass.

        Returns one boolean HxW mask per pair, each at that pair's own original
        image size. The image is repeated in `pairs` once per entity — see the
        module docstring on why SAM3 cannot take several prompts per image.
        """
        masks, _ = self.segment_pairs_full(pairs, want_instances=False)
        return masks

    def segment_pairs_full(
        self,
        pairs: Sequence[Tuple[Any, str]],
        want_instances: bool = False,
    ) -> Tuple[List[np.ndarray], Optional[List[List[Dict[str, Any]]]]]:
        """One forward pass, BOTH read-outs: `(semantic_masks, instances)`.

        `instances` is None when `want_instances` is False, and otherwise a
        per-pair list of instance dicts (see `_postprocess_instances`).

        THE POINT OF THIS METHOD: `semantic_seg` and the instance-level
        `pred_masks`/`pred_boxes`/`pred_logits` are all fields of the SAME
        `Sam3ImageSegmentationOutput`, produced by the SAME forward pass. So
        instance grounding is a second post-processing call over tensors that
        already exist — it costs no additional GPU forward work, only the
        post-processing (a sigmoid, a bilinear resize of the kept masks, and a
        threshold). That is why COCO's detection evaluation can be added
        without doubling the cost of a grounding run.
        """
        if not pairs:
            return [], ([] if want_instances else None)
        torch = self._torch
        images = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        # PIL's .size is (width, height); post_process wants (height, width).
        target_sizes = [(img.height, img.width) for img in images]

        inputs = self._encode(images, texts).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        if self._debug_batches_left > 0:
            raw = outputs.semantic_seg
            print(f"[grounding][debug] raw semantic_seg range: min={raw.min().item():.4f} "
                  f"max={raw.max().item():.4f} shape={tuple(raw.shape)} — unbounded => LOGITS, "
                  f"sigmoid applied by post_process_semantic_segmentation", flush=True)
            self._debug_batches_left -= 1

        # Does sigmoid -> bilinear resize to target_sizes -> (> threshold), and
        # returns one long-typed (H, W) map per batch element.
        maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes, threshold=self.mask_threshold
        )
        masks = [_to_numpy(m).astype(bool) for m in maps]

        instances = None
        if want_instances:
            # Same `outputs` object — no second forward pass. `threshold`
            # gates INSTANCE CONFIDENCE (sigmoid(pred_logits) * presence),
            # `mask_threshold` binarizes each kept instance's mask; the two
            # are separate knobs in SAM3's own signature and are kept separate
            # here (self.instance_score_threshold vs self.mask_threshold)
            # rather than collapsed into one number.
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.instance_score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )
            instances = [self._postprocess_instances(r) for r in results]

        return masks, instances

    def _postprocess_instances(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Turn one image's raw `post_process_instance_segmentation` result into
        plain-Python instance dicts.

        Each instance gets BOTH boxes, deliberately:
          * "bbox"      — the tight axis-aligned box of the binarized instance
                          MASK, computed here. This is the box the detection
                          metrics use, because it is guaranteed consistent with
                          the mask stored next to it: a reviewer overlaying the
                          mask sees exactly the region that was scored.
          * "sam3_bbox" — SAM3's own regressed `pred_boxes` entry for the same
                          query, carried through unchanged so the two can be
                          compared (they usually agree closely; they diverge
                          where the mask is fragmented or the box regressor
                          over-reaches).
        An instance whose binarized mask is EMPTY is dropped entirely — it has
        no derivable box, and a score above threshold with no surviving pixels
        is not a detection.
        """
        out: List[Dict[str, Any]] = []
        scores = _to_numpy(result["scores"])
        boxes = _to_numpy(result["boxes"])
        masks = _to_numpy(result["masks"]).astype(bool)
        for score, sam_box, mask in zip(scores, boxes, masks):
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            out.append({
                "score": float(score),
                "bbox": bbox,
                "sam3_bbox": [float(v) for v in sam_box],
                "mask_rle": encode_rle(mask),
                "pixel_count": int(mask.sum()),
            })
        return out


# =============================================================================
# Grounding driver (enriches VLM records in place)
# =============================================================================
def _nature_object_indices(rec: Dict[str, Any]) -> List[int]:
    """Indices of the entities we actually ground: only those whose HYBRID
    resolved label says nature (recap §6 — `final_nature`, i.e. WordNet where
    the object mapped and the VLM's own judgment where it didn't).

    Falls back to the raw VLM label for artifacts old enough to predate
    `object_finals` (the VLM pipeline used to resolve labels at scoring time);
    those store nature as the literal string "yes".
    """
    finals = rec.get("object_finals")
    if finals:
        return [i for i, fin in enumerate(finals) if fin.get("final_nature") is True]
    labels = rec.get("object_labels") or []
    return [i for i, lab in enumerate(labels)
            if str(lab.get("nature", "")).strip().lower() == "yes"]


def _empty_grounding(obj: str, is_nature: bool) -> Dict[str, Any]:
    """Grounding entry for an entity SAM3 was never asked about (not nature).
    `grounded` is None — meaningfully different from False, which means "SAM3
    looked and found nothing"."""
    return {"object": obj, "prompt": None, "is_nature": is_nature,
            "grounded": None, "mask_rle": None, "pixel_count": 0}


def _empty_instances(obj: str, is_nature: bool) -> Dict[str, Any]:
    """Instance-grounding entry for an entity SAM3 was never asked about.

    `attempted` mirrors the `grounded is None` distinction in
    `_empty_grounding`: False means SAM3 was never run for this entity (not
    nature), while True with an empty `instances` list means SAM3 ran and
    returned no instance above threshold. Collapsing the two would make an
    un-attempted entity indistinguishable from a genuine non-detection, which
    is exactly the ambiguity the semantic path already avoids.
    """
    return {"object": obj, "prompt": None, "is_nature": is_nature,
            "attempted": False, "instances": []}


def ground_records(
    grounder: SAM3Grounder,
    records: Iterable[Dict[str, Any]],
    batch_size: int = 8,
    max_pairs_per_forward: int = 16,
    center_sigma: float = DEFAULT_CENTER_SIGMA,
    instance_grounding: bool = False,
    n_total: Optional[int] = None,
    verbose: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Ground every nature entity of every record and yield the SAME record dicts,
    enriched with `object_groundings`, `nature_relevance_score_coverage_ratio`
    and `nature_relevance_score_center_weighted` (see the module docstring for
    the exact shape) — BOTH scores are always computed, never just one.

    With `instance_grounding=True` each record ALSO gets `object_instances`
    (see the module docstring), read off the same forward pass. Nothing about
    `object_groundings` or either relevance score changes when it is on — the
    instance read-out is purely additive, so a COCO run and a BIG-5 run still
    produce the same semantic fields computed the same way.

    Records are consumed and yielded in order, `batch_size` IMAGES at a time, so
    this composes with `stream_artifact` for constant memory over a large run.
    Within a batch, the (image, entity) pairs are flattened and pushed through
    SAM3 in chunks of at most `max_pairs_per_forward` — `batch_size` counts
    images (same meaning as in the VLM pipeline) while this second knob bounds
    actual GPU work, since a busy image can carry many nature entities.

    An image that fails to LOAD is still yielded, with every entity marked
    `grounded: false` and both relevance scores 0.0, so one unreadable file
    never aborts a long run and the failure stays visible in the artifact.
    """
    batch: List[Dict[str, Any]] = []
    progress = BatchProgress(
        # Only an estimate when n_total is unknown (streaming) — BatchProgress
        # just needs a denominator for its ETA.
        num_batches=max(1, ((n_total or 0) + batch_size - 1) // batch_size),
        label="[grounding] batch", verbose=verbose,
    )
    batch_idx = 0
    n_done = 0

    def _flush(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _ground_batch(grounder, chunk, max_pairs_per_forward, center_sigma,
                             instance_grounding, verbose)

    for rec in records:
        batch.append(rec)
        if len(batch) < batch_size:
            continue
        for out in _flush(batch):
            yield out
        n_done += len(batch)
        n_grounded = sum(1 for r in batch for g in r["object_groundings"] if g["grounded"])
        progress.tick(batch_idx, n_done=n_done, n_total=n_total,
                      extra=f"grounded entities {n_grounded}")
        batch_idx += 1
        batch = []

    if batch:
        for out in _flush(batch):
            yield out
        n_done += len(batch)
        n_grounded = sum(1 for r in batch for g in r["object_groundings"] if g["grounded"])
        progress.tick(batch_idx, n_done=n_done, n_total=n_total,
                      extra=f"grounded entities {n_grounded}")


def _ground_batch(
    grounder: SAM3Grounder,
    batch: List[Dict[str, Any]],
    max_pairs_per_forward: int,
    center_sigma: float,
    instance_grounding: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """Ground one batch of image records. Split out from `ground_records` purely
    to keep the streaming/progress bookkeeping there readable."""
    # 1. Pre-fill every entity's grounding entry (non-nature entities keep this
    #    "never attempted" record), and open each image ONCE for the whole batch.
    images: List[Any] = []
    sizes: List[Optional[Tuple[int, int]]] = []
    for rec in batch:
        objs = rec.get("objects", [])
        nature_idx = set(_nature_object_indices(rec))
        rec["object_groundings"] = [_empty_grounding(obj, i in nature_idx)
                                    for i, obj in enumerate(objs)]
        if instance_grounding:
            rec["object_instances"] = [_empty_instances(obj, i in nature_idx)
                                       for i, obj in enumerate(objs)]
        try:
            img = grounder._load_image(rec["image_path"])
            images.append(img)
            sizes.append((img.height, img.width))
        except Exception as exc:  # unreadable/missing file — see docstring
            if verbose:
                print(f"[grounding] failed to open {rec['image_path']}: {exc}", flush=True)
            images.append(None)
            sizes.append(None)

    # 2. Flatten to (image, prompt) pairs — one pair per nature entity, image
    #    repeated (SAM3 resolves ONE prompt per forward pass; module docstring).
    pairs: List[Tuple[Any, str]] = []
    owners: List[Tuple[int, int]] = []  # (record index, object index)
    for ri, rec in enumerate(batch):
        if images[ri] is None:
            # Image unreadable: every nature entity is a confirmed failure, not
            # an un-attempted one, so it reads as grounded=False below.
            for gi, g in enumerate(rec["object_groundings"]):
                if g["is_nature"]:
                    g["prompt"] = entity_prompt(g["object"])
                    g["grounded"] = False
                    if instance_grounding:
                        rec["object_instances"][gi]["prompt"] = g["prompt"]
            continue
        for gi, g in enumerate(rec["object_groundings"]):
            if not g["is_nature"]:
                continue
            g["prompt"] = entity_prompt(g["object"])
            if instance_grounding:
                rec["object_instances"][gi]["prompt"] = g["prompt"]
            pairs.append((images[ri], g["prompt"]))
            owners.append((ri, gi))

    # 3. Forward pass(es), capped at max_pairs_per_forward pairs each.
    for start in range(0, len(pairs), max_pairs_per_forward):
        chunk = pairs[start : start + max_pairs_per_forward]
        chunk_owners = owners[start : start + max_pairs_per_forward]
        try:
            masks, instances = grounder.segment_pairs_full(
                chunk, want_instances=instance_grounding)
        except Exception as exc:
            # Never let one bad forward pass kill a long run — mark this chunk's
            # entities as not grounded and carry on.
            if verbose:
                print(f"[grounding] SAM3 forward failed on {len(chunk)} pairs: {exc}", flush=True)
            for ri, gi in chunk_owners:
                batch[ri]["object_groundings"][gi]["grounded"] = False
            continue
        if instances is not None:
            for (ri, gi), inst in zip(chunk_owners, instances):
                entry = batch[ri]["object_instances"][gi]
                entry["attempted"] = True
                entry["instances"] = inst
        for (ri, gi), mask in zip(chunk_owners, masks):
            g = batch[ri]["object_groundings"][gi]
            # PRESENCE/ABSENCE: any pixel over threshold at all. `mask` is
            # already binarized at grounder.mask_threshold, so `.any()` is
            # exactly "some pixel's probability exceeded the threshold".
            if not mask.any():
                g["grounded"] = False
                g["pixel_count"] = 0
                continue
            g["grounded"] = True
            g["mask_rle"] = encode_rle(mask)
            g["pixel_count"] = int(mask.sum())

    # 4. Per-image relevance scores (BOTH methods) over the UNION of surviving masks.
    for ri, rec in enumerate(batch):
        rles = [g["mask_rle"] for g in rec["object_groundings"]
                if g["grounded"] and g["mask_rle"] is not None]
        size = sizes[ri]
        if size is None:
            scores = {"coverage_ratio": 0.0, "center_weighted": 0.0}
        else:
            scores = nature_relevance_scores(rles, size[0], size[1], center_sigma=center_sigma)
        rec["nature_relevance_score_coverage_ratio"] = scores["coverage_ratio"]
        rec["nature_relevance_score_center_weighted"] = scores["center_weighted"]

    # Release the decoded images promptly rather than waiting on GC — a batch of
    # full-resolution PIL images is not small.
    for img in images:
        if img is not None:
            img.close()

    return batch


# =============================================================================
# Diagnostics
# =============================================================================
def grounding_diagnostics(counts: Dict[str, int], n_images: int) -> Dict[str, Any]:
    """Turn the running counters kept by the driver script into the summary
    numbers worth reporting: how many nature entities SAM3 was asked about, and
    what fraction it actually confirmed in pixels.

    The confirmation RATE is the number that matters for auditing: a low rate
    means the VLM is claiming nature entities the segmenter cannot find, which
    is precisely the hallucination failure mode the Grounding pipeline exists to
    check (recap §9 — agreement with an independent model, NOT ground truth).
    """
    n_nature = counts.get("nature_entities", 0)
    n_grounded = counts.get("grounded_entities", 0)
    n_instances = counts.get("instances_total", 0)
    return {
        "n_images": n_images,
        "objects_total": counts.get("objects_total", 0),
        "nature_entities": n_nature,
        "grounded_entities": n_grounded,
        # Instance-grounding only (0 when it was off): how many individual
        # objects SAM3 resolved the confirmed concepts into. Reported next to
        # the entity counts because the ratio of the two is itself telling —
        # one "sheep" concept mask becoming twelve instances is the whole
        # reason COCO needs the instance head.
        "instances_total": n_instances,
        "instances_per_grounded_entity": (n_instances / n_grounded) if n_grounded else 0.0,
        "grounding_confirmation_rate": (counts.get("grounded_entities", 0) / n_nature) if n_nature else 0.0,
        "nature_entities_per_image": (n_nature / n_images) if n_images else 0.0,
    }
