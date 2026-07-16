"""
lib/clip_metrics.py

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
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

OBJECT_TEMPLATE = "a photo of a {}"
DEFAULT_CLIPSCORE_SCALE = 2.5


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
        import torch
        import open_clip

        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # open_clip stores the trained context length on the text tower.
        self.context_length = getattr(self.model, "context_length", 77)
        self._torch = torch

    def encode_text(self, texts: List[str], warn_truncation: bool = False) -> np.ndarray:
        """Encode a list of strings → [len(texts), dim] L2-normalized float32."""
        torch = self._torch
        if not texts:
            return np.zeros((0, self.model.text_projection.shape[1]), dtype=np.float32)

        if warn_truncation:
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
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
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
                    tensors.append(self.preprocess(Image.open(p).convert("RGB")))
                except Exception as e:
                    warnings.warn(f"CLIP: could not read image '{p}' ({e!r}); using a zero embedding.", stacklevel=2)
                    failed.append(j)

            if tensors:
                imgs = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    feats = self.model.encode_image(imgs)
                    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                feats = feats.cpu().float().numpy()
                d = feats.shape[1]
            else:
                d = dim or 0
                feats = np.zeros((0, d), dtype=np.float32)

            # Re-insert zero rows for failed positions to keep alignment with input.
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
    return text_embs @ image_emb


def clip_score(sim: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE) -> np.ndarray:
    """CLIPScore = w · max(cos, 0), elementwise."""
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
    s_term = float(clip_score(_cos(image_emb, caption_emb[None, :]), w)[0])
    obj_terms = clip_score(_cos(image_emb, object_embs), w)  # [N]
    n = object_embs.shape[0]
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
        return np.zeros((n_cand,), dtype=np.float32), -1, np.zeros((object_embs.shape[0],), dtype=np.float32)

    sim = object_embs @ candidate_embs.T          # [N_obj, N_cand]
    per_candidate_max = sim.max(axis=0)           # [N_cand]
    pred_index = int(np.argmax(per_candidate_max))
    per_object_sim_to_pred = sim[:, pred_index]   # [N_obj]
    return per_candidate_max, pred_index, per_object_sim_to_pred


def object_texts(objects: List[str]) -> List[str]:
    """Wrap raw object phrases in the ClipMatch/Object-CLIPScore template."""
    return [OBJECT_TEMPLATE.format(o) for o in objects]
