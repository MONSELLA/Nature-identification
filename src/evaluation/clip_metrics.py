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

KNOWN LIMITATION: vanilla CLIP text encoders truncate at 77 tokens (open_clip
ViT-L/14 included). Only the F-CLIPScore SENTENCE term (the full caption) is at
risk; short "a photo of a {object}" templates are unaffected. `CLIPScorer`
warns when a caption exceeds the encoder's context length.

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
from typing import Dict, List, Optional, Tuple

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
# CLIP model wrapper (open_clip ViT-L/14) — returns L2-normalized numpy arrays
# =============================================================================
class CLIPScorer:
    """Thin open_clip wrapper. Encodes images and text to L2-normalized float32
    numpy arrays so all downstream metric math is backend-free numpy."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        # Imported lazily (inside the method, not at module load time) so that
        # simply IMPORTING this file never requires torch/open_clip to be
        # installed — only actually creating a CLIPScorer does. This is what
        # lets the pure-math functions further down be unit-tested without a
        # GPU or these heavy dependencies present.
        import torch
        import open_clip

        self.device = device
        self.batch_size = batch_size
        # open_clip.create_model_and_transforms loads the pretrained CLIP
        # weights and also returns the exact image-preprocessing pipeline
        # (resizing/cropping/normalizing) this specific model was trained
        # with — we must reuse that preprocessing exactly, so images match
        # what the model expects.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # .eval() disables training-only behaviors (like dropout); .to(device)
        # moves all model weights onto the target device (e.g. "cuda" for GPU).
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # open_clip stores the trained context length on the text tower.
        self.context_length = getattr(self.model, "context_length", 77)
        self._torch = torch  # stashed so other methods can use torch without re-importing

    def encode_text(self, texts: List[str], warn_truncation: bool = False) -> np.ndarray:
        """Encode a list of strings → [len(texts), dim] L2-normalized float32."""
        torch = self._torch
        if not texts:
            # No text to encode — return an empty array with the RIGHT number
            # of embedding dimensions (so callers can still safely concatenate
            # or shape-check it) rather than a completely empty/ambiguous array.
            return np.zeros((0, self.model.text_projection.shape[1]), dtype=np.float32)

        if warn_truncation:
            # CLIP's tokenizer pads every sequence to a fixed length with
            # zeros; counting non-zero tokens tells us the REAL sequence
            # length. If that's at/above the model's context length, the
            # text got cut off and the embedding won't reflect the whole
            # caption — surface a warning so results can be interpreted
            # correctly (see the module docstring's 77-token caveat).
            for t in texts:
                n_tok = int((self.tokenizer([t]) != 0).sum().item())
                if n_tok >= self.context_length:
                    warnings.warn(
                        f"Caption reaches/exceeds CLIP context length "
                        f"({self.context_length} tokens) and will be truncated — "
                        f"the F-CLIPScore sentence term is affected.",
                        stacklevel=2,
                    )
                    break

        out = []
        # Process in chunks of `batch_size` rather than all at once, so we
        # don't try to fit an arbitrarily large number of texts into GPU
        # memory in one forward pass.
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            # torch.no_grad(): we're only doing inference here, not training,
            # so there's no need to track gradients — this saves memory/time.
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                # Rescale every embedding vector to unit length (L2 norm = 1),
                # so a plain dot product later equals cosine similarity (see
                # the module docstring's CLIP background section).
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            # Move back to CPU, plain float32, and convert to a numpy array —
            # everything downstream of this class works in numpy, not torch.
            out.append(feats.cpu().float().numpy())
        return np.concatenate(out, axis=0)

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode a list of image paths → [len(paths), dim] L2-normalized float32.
        An unreadable/corrupt image yields a zero row (all its CLIPScores become
        0) rather than aborting the whole scoring run — one bad file must not
        waste an expensive pass over thousands of images. A warning is emitted
        per failure."""
        import torch
        from PIL import Image

        dim = getattr(self.model.visual, "output_dim", None)
        if not image_paths:
            return np.zeros((0, dim or 0), dtype=np.float32)

        out = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            tensors = []
            failed = []  # positions (within this batch) that could not be loaded
            for j, p in enumerate(batch_paths):
                try:
                    # Open the file, force it to standard RGB (some images are
                    # grayscale/CMYK/have an alpha channel — CLIP expects RGB),
                    # then run it through this model's own preprocessing
                    # pipeline (resize/crop/normalize) to get a model-ready tensor.
                    tensors.append(self.preprocess(Image.open(p).convert("RGB")))
                except Exception as e:
                    # A single corrupt/missing file must not crash a run that
                    # might be scoring thousands of images — log a warning,
                    # remember this position as "failed", and move on.
                    warnings.warn(f"CLIP: could not read image '{p}' ({e!r}); using a zero embedding.", stacklevel=2)
                    failed.append(j)

            if tensors:
                # torch.stack bundles the individual preprocessed image
                # tensors into one batch tensor for a single forward pass.
                imgs = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    feats = self.model.encode_image(imgs)
                    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                feats = feats.cpu().float().numpy()
                d = feats.shape[1]
            else:
                # Every single image in this batch failed to load.
                d = dim or 0
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
