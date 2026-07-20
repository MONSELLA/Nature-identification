"""
src/evaluation/clip_metrics.py

CLIP-based metrics for the BIG-5 VLM pipeline:

  - F-CLIPScore      (faithful, Oh & Hwang):
        F-CLIPScore(S) = [CLIPScore(S) + Σ_i CLIPScore(n_i)] / (N+1)
    S = caption sentence, n_i = extracted objects.
  - Object-CLIPScore (OURS, F-CLIPScore-INSPIRED — never call this "F-CLIPScore"):
        mean of CLIPScore("a photo of a {object}") over extracted objects only,
        no sentence term.
  - ClipMatch        (Ging et al.; ImageNet + Places only):
        score each extracted object independently via "a photo of a {object}",
        take the MAX similarity across objects per GT candidate class; argmax
        over candidates = predicted class.

Design: the CLIP model wrapper (`CLIPScorer`) is kept separate from the metric
math (pure NumPy on L2-normalized embeddings). This lets the aggregation logic
be unit-tested without loading torch/open_clip, and lets Phase-2 scoring cache
each image embedding once and reuse it across all three metrics.

CLIPScore convention (Hessel et al.): CLIPScore = w · max(cos, 0), w = 2.5 by
default. The scale w is a constant across every term, so it does not change the
relative ordering used for model comparison; it is exposed for reproducibility.

KNOWN LIMITATION: vanilla CLIP text encoders truncate at 77 tokens (the
original OpenAI CLIP checkpoint included — see CLIP_PRESETS). Only the
F-CLIPScore SENTENCE term (the full caption) is at risk; short "a photo of a
{object}" templates are unaffected. `CLIPScorer` warns when a caption exceeds
the encoder's context length.

BACKGROUND — WHAT IS CLIP, AND WHY "COSINE SIMILARITY"?
CLIP is a neural network trained on (image, caption) pairs so that it can turn
BOTH images and text into vectors ("embeddings") living in the SAME
high-dimensional space, where semantically related image/text pairs end up
close together and unrelated ones end up far apart. "Close together" is
measured with cosine similarity: the cosine of the angle between two vectors,
which is 1.0 when they point in exactly the same direction (maximally
similar), 0.0 when they're perpendicular (unrelated), and negative when they
point in opposite directions.

If a vector is "L2-normalized" (rescaled to have length exactly 1), then the
cosine similarity between two normalized vectors is just their dot product —
no extra division needed. That's why every embedding this file produces is
normalized right after encoding: it lets every metric below use a simple `@`
(matrix multiply) or `·` (dot product) instead of a slower explicit cosine
formula.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

# The fixed phrase every extracted object gets wrapped in before being sent to
# CLIP's text encoder, e.g. object="oak tree" -> "a photo of a oak tree".
# This specific template is the one the ClipMatch/CLIPScore literature uses.
OBJECT_TEMPLATE = "a photo of a {}"
# The "w" scale factor from Hessel et al.'s CLIPScore paper (see clip_score()
# below) — a fixed multiplier applied to every score so the numbers land in a
# more human-readable range. It doesn't change which model/caption scores
# higher relative to another, since it multiplies every score equally.
DEFAULT_CLIPSCORE_SCALE = 2.5
# ClipMatch (and the hierarchical hP/hR metrics that build on it) need a FIXED
# list of candidate classes to choose from — only ImageNet and Places365 have
# one (closed, single-label class vocabularies). COCO is multi-label and BIG-5
# has no fixed class list at all, so those two datasets skip this metric.
CLIPMATCH_DATASETS = ("imagenet", "places365")


# =============================================================================
# CLIP model wrapper (transformers, any CLIP-family checkpoint) — returns
# L2-normalized numpy arrays
# =============================================================================
# --clip_model accepts either one of these short aliases or a raw HuggingFace
# repo id directly. These are BEST-EFFORT default checkpoints for each variant
# — verify the exact repo id yourself before relying on results from one of
# them; if a default here turns out wrong/moved, just pass the correct repo id
# straight to --clip_model, no code change needed.
#
# TWO variants are deliberately NOT included, both for the same reason —
# everything here loads through plain `transformers` alone, no extra
# environment required:
#   - LongCLIP: not packaged as a HF trust_remote_code AutoModel; using it
#     would require cloning its own repo.
#   - LLM2CLIP: its text tower isn't a CLIP transformer at all — it's a
#     SEPARATE LLM, encoded via the `llm2vec` package. `llm2vec` hard-pins
#     transformers<=4.44.2, which is incompatible with vLLM's
#     transformers>=5.5.3 in the SAME environment (verified: installing it
#     force-downgraded transformers and broke vLLM here). Since this project
#     runs the VLM (vLLM) and CLIP scoring in one env, llm2clip isn't usable
#     without a second env just for it — not worth the operational overhead.
CLIP_PRESETS = {
    "original": "openai/clip-vit-large-patch14",
    "clip": "openai/clip-vit-large-patch14",  # alias kept for back-compat
    "eva-clip": "BAAI/EVA-CLIP-8B",
    "fg-clip2": "qihoo360/fg-clip2-large",
}


class CLIPScorer:
    """HuggingFace `transformers` wrapper around a CLIP-family model. Encodes
    images and text to L2-normalized float32 numpy arrays so all downstream
    metric math is backend-free numpy.

    Loads ANY CLIP-like checkpoint via `AutoModel`/`AutoProcessor` (falling
    back to separate `AutoTokenizer`/`AutoImageProcessor` when a checkpoint
    has no combined processor), so the same class covers the original OpenAI
    CLIP as well as third-party variants (EVA-CLIP, FG-CLIP2, ...) — see
    CLIP_PRESETS. `trust_remote_code=True` by default since several of these
    variants ship custom modeling code on the Hub; it's a no-op for
    checkpoints (like the original CLIP) that don't need it.

    The model's own feature-extraction method is duck-typed rather than
    hardcoded per checkpoint: `get_image_features`/`get_text_features` (the
    standard `transformers` CLIP API) is tried first, `encode_image`/
    `encode_text` (the convention several trust_remote_code CLIP variants
    carried over from open_clip's API) second — so a new --clip_model swap
    only needs a correct repo id, not new code, as long as it exposes one of
    these two API shapes.
    """

    def __init__(
        self,
        model_name: str = "original",
        device: str = "cuda",
        batch_size: int = 64,
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
    ) -> None:
        # Imported lazily (inside the method, not at module load time) so that
        # simply IMPORTING this file never requires torch/transformers to be
        # installed — only actually creating a CLIPScorer does. This is what
        # lets the pure-math functions further down be unit-tested without a
        # GPU or these heavy dependencies present.
        import torch
        from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        self.repo_id = CLIP_PRESETS.get(model_name, model_name)

        self.model = self._load_model(self.repo_id, trust_remote_code, torch_dtype)
        self.model.eval().to(self.device)
        # Image/text processors always emit float32 tensors regardless of the
        # model's own dtype, so a checkpoint loaded in fp16/bf16 (e.g. via
        # torch_dtype="auto" on EVA-CLIP-8B) would otherwise hit a dtype
        # mismatch on the first forward pass. Recorded once so _to_device()
        # can cast floating inputs to match.
        self._model_dtype = next(self.model.parameters()).dtype

        # Most CLIP-family checkpoints publish one combined AutoProcessor
        # (tokenizer + image processor together); fall back to loading each
        # piece separately for the few that only publish one of the two.
        try:
            processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code)
            self.tokenizer = getattr(processor, "tokenizer", processor)
            self.image_processor = getattr(processor, "image_processor", processor)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code)
            self.image_processor = AutoImageProcessor.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code)

        self.context_length = getattr(self.tokenizer, "model_max_length", 77)
        # Some tokenizers report a sentinel "no limit" value (e.g. 1e30) when
        # they don't actually define model_max_length — falling back to 77
        # (CLIP's original context length) keeps the truncation warning below
        # meaningful instead of silently never firing.
        if not isinstance(self.context_length, int) or self.context_length > 100_000:
            self.context_length = 77

        # Embedding dimensionality varies by checkpoint and isn't exposed
        # uniformly across configs — determine it once via a throwaway encode
        # instead of guessing a config attribute name per backend.
        self._embed_dim = self._encode_text_batch(["a photo of a photo"]).shape[1]

    @staticmethod
    def _load_model(repo_id: str, trust_remote_code: bool, torch_dtype: str = "auto"):
        """Load a checkpoint's model class, working around a real gap in
        `AutoModel.from_pretrained(..., trust_remote_code=True)`: it only
        auto-dispatches to a custom repo's model class if that repo's
        `config.json` declares an `"AutoModel"` key in its `auto_map` — if a
        repo instead only declares e.g. `"AutoModelForImageTextToText"`, or
        no key at all matching a family `AutoModel` recognizes, the plain
        `AutoModel.from_pretrained` call raises `ValueError: Unrecognized
        configuration class ... for this kind of AutoModel: AutoModel` even
        though the repo's custom config/model code loaded and is perfectly
        usable — it's a routing miss, not a real incompatibility.

        Tries plain `AutoModel` first (covers the original CLIP and any
        checkpoint that DOES declare an `"AutoModel"` auto_map entry). On
        that specific "Unrecognized configuration class" failure, falls back
        to reading `auto_map` off the checkpoint's own config directly and
        dynamically loading whichever `AutoModel*`-family class IS declared
        there, via `transformers`' own dynamic-module loader (the same
        mechanism `AutoModel.from_pretrained` uses internally when the
        `"AutoModel"` key IS present — this just widens which auto_map key
        we'll accept).

        `torch_dtype="auto"` (the default) loads each checkpoint in whatever
        dtype its own config declares, rather than transformers' unconditional
        fp32 fallback — load_in fp32 is fine for the small original CLIP, but
        would try to allocate ~32GB for the 8B-param EVA-CLIP-8B preset.
        """
        from transformers import AutoConfig, AutoModel

        try:
            return AutoModel.from_pretrained(
                repo_id, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise

        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
        auto_map = getattr(config, "auto_map", None) or {}
        model_keys = [k for k in auto_map if k.startswith("AutoModel")]
        if not model_keys:
            raise ValueError(
                f"{repo_id}: AutoModel.from_pretrained couldn't find a usable model "
                f"class, and config.auto_map has no AutoModel*-family entry either "
                f"(auto_map keys: {list(auto_map)}). Check this checkpoint's model "
                f"card on HuggingFace for the exact class it expects to be loaded with."
            )
        # Prefer a bare "AutoModel" entry if present (most general-purpose);
        # otherwise take whichever AutoModelFor* entry is declared.
        key = "AutoModel" if "AutoModel" in model_keys else model_keys[0]
        class_ref = auto_map[key]
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        model_cls = get_class_from_dynamic_module(class_ref, repo_id)
        return model_cls.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
        )

    def _unwrap_embedding(self, output, projection_attr: str):
        """`get_text_features`/`get_image_features` are meant to return the
        already-projected embedding tensor. On some `transformers` versions
        (observed with plain `openai/clip-vit-large-patch14` under a
        transformers>=5.5.3-family install — see requirements.txt's vLLM
        pin) that convenience method regresses and instead returns the raw
        encoder `ModelOutput` (e.g. `BaseModelOutputWithPooling`), which has
        no `.norm()`/is not a tensor. Recover manually: pull the pooled
        output and run it through the model's own projection layer, i.e.
        exactly what the convenience method was supposed to do.
        """
        if isinstance(output, self._torch.Tensor):
            return output
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0, :]
        projection = getattr(self.model, projection_attr, None)
        return projection(pooled) if projection is not None else pooled

    def _text_features(self, inputs: dict):
        if hasattr(self.model, "get_text_features"):
            out = self.model.get_text_features(**inputs)
            return self._unwrap_embedding(out, "text_projection")
        if hasattr(self.model, "encode_text"):
            return self.model.encode_text(inputs["input_ids"])
        raise AttributeError(
            f"{self.repo_id}: model exposes neither get_text_features() nor "
            f"encode_text() — CLIPScorer doesn't know how to run text encoding "
            f"for this checkpoint's API."
        )

    def _image_features(self, inputs: dict):
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(**inputs)
            return self._unwrap_embedding(out, "visual_projection")
        if hasattr(self.model, "encode_image"):
            return self.model.encode_image(inputs["pixel_values"])
        raise AttributeError(
            f"{self.repo_id}: model exposes neither get_image_features() nor "
            f"encode_image() — CLIPScorer doesn't know how to run image "
            f"encoding for this checkpoint's API."
        )

    def _to_device(self, inputs: dict) -> dict:
        """Move every tensor to `self.device`, additionally casting floating
        tensors (e.g. `pixel_values`) to the model's own dtype — processors
        always emit float32 regardless of what dtype the model was loaded in
        (see `torch_dtype="auto"` in `_load_model`), so a half-precision
        checkpoint needs its inputs downcast to match or the forward pass
        raises a dtype-mismatch error. Integer tensors (`input_ids`,
        `attention_mask`) are left alone."""
        return {
            k: v.to(self.device, dtype=self._model_dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.context_length, return_tensors="pt")
        inputs = self._to_device(inputs)
        with torch.no_grad():
            feats = self._text_features(inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().float().numpy()

    def _encode_image_batch(self, images) -> np.ndarray:
        torch = self._torch
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = self._to_device(inputs)
        with torch.no_grad():
            feats = self._image_features(inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().float().numpy()

    def encode_text(self, texts: List[str], warn_truncation: bool = False,
                     verbose: bool = False, desc: str = "text") -> np.ndarray:
        """Encode a list of strings → [len(texts), dim] L2-normalized float32."""
        if not texts:
            # No text to encode — return an empty array with the RIGHT number
            # of embedding dimensions (so callers can still safely concatenate
            # or shape-check it) rather than a completely empty/ambiguous array.
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        if warn_truncation:
            # Tokenize WITHOUT truncation to get the real sequence length; if
            # that's at/above the model's context length, the text got cut
            # off and the embedding won't reflect the whole caption — surface
            # a warning so results can be interpreted correctly (see the
            # module docstring's 77-token caveat).
            for t in texts:
                n_tok = len(self.tokenizer(t, truncation=False)["input_ids"])
                if n_tok >= self.context_length:
                    warnings.warn(
                        f"Caption reaches/exceeds CLIP context length "
                        f"({self.context_length} tokens) and will be truncated — "
                        f"the F-CLIPScore sentence term is affected.",
                        stacklevel=2,
                    )
                    break

        out = []
        n_total = len(texts)
        # Process in chunks of `batch_size` rather than all at once, so we
        # don't try to fit an arbitrarily large number of texts into GPU
        # memory in one forward pass.
        for i in range(0, n_total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.append(self._encode_text_batch(batch))
            if verbose:
                done = min(i + self.batch_size, n_total)
                print(f"🔎 [CLIP] {desc}: {done}/{n_total} ({done / n_total:.1%})", flush=True)
        return np.concatenate(out, axis=0)

    def encode_images(self, image_paths: List[str], verbose: bool = False) -> np.ndarray:
        """Encode a list of image paths → [len(paths), dim] L2-normalized float32.
        An unreadable/corrupt image yields a zero row (all its CLIPScores become
        0) rather than aborting the whole scoring run — one bad file must not
        waste an expensive pass over thousands of images. A warning is emitted
        per failure."""
        from PIL import Image

        if not image_paths:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        out = []
        n_total = len(image_paths)
        for i in range(0, n_total, self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            failed = []  # positions (within this batch) that could not be loaded
            for j, p in enumerate(batch_paths):
                try:
                    # Force standard RGB (some images are grayscale/CMYK/have
                    # an alpha channel — CLIP expects RGB); the checkpoint's
                    # own image_processor handles resize/crop/normalize.
                    images.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    # A single corrupt/missing file must not crash a run that
                    # might be scoring thousands of images — log a warning,
                    # remember this position as "failed", and move on.
                    warnings.warn(f"CLIP: could not read image '{p}' ({e!r}); using a zero embedding.", stacklevel=2)
                    failed.append(j)

            if images:
                feats = self._encode_image_batch(images)
                d = feats.shape[1]
            else:
                # Every single image in this batch failed to load.
                d = self._embed_dim
                feats = np.zeros((0, d), dtype=np.float32)

            # Re-insert zero rows for failed positions to keep alignment with input.
            # We only computed embeddings for the images that loaded successfully
            # (`feats`), but the caller expects one row per ORIGINAL path,
            # including the failed ones — so we rebuild the full-size batch
            # here, filling in a zero vector wherever loading failed and the
            # real embedding everywhere else, preserving the original order.
            batch_out = np.zeros((len(batch_paths), d), dtype=np.float32)
            good_iter = iter(feats)
            for j in range(len(batch_paths)):
                if j not in failed:
                    batch_out[j] = next(good_iter)
            out.append(batch_out)
            if verbose:
                done = min(i + self.batch_size, n_total)
                print(f"🔎 [CLIP] images: {done}/{n_total} ({done / n_total:.1%})", flush=True)
        return np.concatenate(out, axis=0)


# =============================================================================
# Metric math (pure numpy on L2-normalized embeddings)
# =============================================================================
def _cos(image_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity of one image embedding [d] against text embeddings
    [n, d] (both assumed L2-normalized) → [n]."""
    if text_embs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    # Because both `image_emb` and every row of `text_embs` are L2-normalized,
    # a plain matrix-vector product IS the cosine similarity — see the module
    # docstring's CLIP background note. `text_embs @ image_emb` computes, for
    # every row (text) in text_embs, its dot product with image_emb, all in
    # one vectorized numpy operation instead of a Python loop.
    return text_embs @ image_emb


def clip_score(sim: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE) -> np.ndarray:
    """CLIPScore = w · max(cos, 0), elementwise."""
    # `np.clip(sim, 0.0, None)` clamps any NEGATIVE similarity up to 0 (a
    # negative cosine means "pointing in opposite directions" — CLIPScore
    # treats that the same as "no similarity" rather than penalizing further),
    # then scales by w. Works elementwise whether `sim` is a single number or
    # a whole array of similarities.
    return w * np.clip(sim, 0.0, None)


def object_clipscore(
    image_emb: np.ndarray,
    object_embs: np.ndarray,
    w: float = DEFAULT_CLIPSCORE_SCALE,
) -> float:
    """Object-CLIPScore (ours): mean CLIPScore over the extracted objects only.
    Returns 0.0 for an image with no extracted objects."""
    if object_embs.shape[0] == 0:
        return 0.0
    # 1. _cos(...) gives one similarity score per extracted object.
    # 2. clip_score(...) turns each into a CLIPScore.
    # 3. np.mean(...) averages them into a single number for this image.
    return float(np.mean(clip_score(_cos(image_emb, object_embs), w)))


def f_clipscore(
    image_emb: np.ndarray,
    caption_emb: np.ndarray,
    object_embs: np.ndarray,
    w: float = DEFAULT_CLIPSCORE_SCALE,
) -> float:
    """F-CLIPScore (Oh & Hwang): [CLIPScore(S) + Σ_i CLIPScore(n_i)] / (N+1).
    `caption_emb` is [d] (single caption). With no objects (N=0) this reduces
    to CLIPScore(S)."""
    # The "sentence term": how well does the WHOLE caption match the image?
    # caption_emb[None, :] just reshapes the single [d] vector into a [1, d]
    # "batch of one" so it works with `_cos`'s expected shape, then we pull
    # the single resulting score back out with [0].
    s_term = float(clip_score(_cos(image_emb, caption_emb[None, :]), w)[0])
    # The "object terms": one CLIPScore per extracted object.
    obj_terms = clip_score(_cos(image_emb, object_embs), w)  # [N]
    n = object_embs.shape[0]
    # Average the sentence term and every object term together (N objects +
    # 1 sentence = N+1 terms total), exactly matching the paper's formula.
    return (s_term + float(np.sum(obj_terms))) / (n + 1)


def clipmatch(
    object_embs: np.ndarray,
    candidate_embs: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    ClipMatch (Ging et al.): score each candidate class as the MAX cosine
    similarity of any extracted object's "a photo of a {object}" embedding to
    that candidate's text embedding. The predicted class is the argmax candidate.

    Args:
        object_embs:    [N_obj, d] L2-normalized object-phrase embeddings.
        candidate_embs: [N_cand, d] L2-normalized candidate-class embeddings.

    Returns:
        per_candidate_max: [N_cand] max-over-objects similarity per candidate.
        pred_index:        argmax candidate index (or -1 if no objects).
        per_object_sim_to_pred: [N_obj] each object's similarity to the
            predicted class — the ranking scores handed to
            taxonomy_metrics.resolve_to_wordnet for hP/hR.
    """
    n_cand = candidate_embs.shape[0]
    if object_embs.shape[0] == 0 or n_cand == 0:
        # No extracted objects (or no candidate classes at all) means there's
        # nothing to compare — return an all-zero score row and pred_index=-1
        # as a sentinel meaning "could not make a prediction" (checked by
        # callers, e.g. scripts/run_vlm_pipeline.py's ClipMatch scoring).
        return np.zeros((n_cand,), dtype=np.float32), -1, np.zeros((object_embs.shape[0],), dtype=np.float32)

    # `object_embs @ candidate_embs.T` computes EVERY object-vs-candidate
    # cosine similarity in a single matrix multiply: row i, column j is how
    # similar extracted object i is to candidate class j. Shape [N_obj, N_cand].
    sim = object_embs @ candidate_embs.T          # [N_obj, N_cand]
    # For each candidate class (each COLUMN), take the single highest
    # similarity across all extracted objects — i.e. "the best-matching
    # object this image has for this candidate class".
    per_candidate_max = sim.max(axis=0)           # [N_cand]
    # The predicted class is whichever candidate got the highest best-match
    # score overall.
    pred_index = int(np.argmax(per_candidate_max))
    # Pull out, for the winning candidate class specifically, how similar
    # EACH extracted object was to it — this lets downstream code (hP/hR) know
    # which extracted object phrase was most responsible for the prediction.
    per_object_sim_to_pred = sim[:, pred_index]   # [N_obj]
    return per_candidate_max, pred_index, per_object_sim_to_pred


def clipmatch_from_caption(
    caption_emb: np.ndarray,
    candidate_embs: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    EXPERIMENTAL variant of ClipMatch (recap §11 open item: "test ClipMatch/
    hP/hR against the raw caption... despite the concerns raised") — predicts
    the class from the single WHOLE-CAPTION embedding's similarity to each
    candidate class, instead of clipmatch()'s max-over-extracted-objects.
    Printed alongside the object-list version for direct comparison, NOT used
    to feed the axis (nature/biotic/material) scores.

    Known caveat this variant reintroduces (the reason the object-list version
    was chosen as the default — see clip_metrics.py's module docstring and
    CLAUDE.md): a long caption risks CLIP's 77-token truncation, and ClipMatch
    was designed/validated on short single-answer responses, not multi-sentence
    captions.

    Args:
        caption_emb:    [d] L2-normalized single caption embedding.
        candidate_embs: [N_cand, d] L2-normalized candidate-class embeddings.

    Returns:
        per_candidate_sim: [N_cand] caption-to-candidate similarity.
        pred_index:        argmax candidate index (or -1 if no candidates).
    """
    n_cand = candidate_embs.shape[0]
    if n_cand == 0:
        return np.zeros((0,), dtype=np.float32), -1
    per_candidate_sim = candidate_embs @ caption_emb   # [N_cand]
    pred_index = int(np.argmax(per_candidate_sim))
    return per_candidate_sim, pred_index


def object_texts(objects: List[str]) -> List[str]:
    """Wrap raw object phrases in the ClipMatch/Object-CLIPScore template."""
    # e.g. ["oak tree", "river"] -> ["a photo of a oak tree", "a photo of a river"]
    return [OBJECT_TEMPLATE.format(o) for o in objects]
