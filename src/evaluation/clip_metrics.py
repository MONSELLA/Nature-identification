"""
src/evaluation/clip_metrics.py

CLIP-based metrics for the BIG-5 VLM pipeline:

  - F-CLIPScore      (faithful, Oh & Hwang):
        F-CLIPScore(S) = [CLIPScore(S) + Σ_i CLIPScore(n_i)] / (N+1)
    S = caption sentence, n_i = extracted objects.
  - Object-CLIPScore (OURS, F-CLIPScore-INSPIRED — never call this "F-CLIPScore"):
        mean of CLIPScore("a photo of a {object}") over extracted objects only,
        no sentence term.
  - ClipMatch        (Ging et al.-INSPIRED, caption-based; ImageNet + Places
    only): score the WHOLE CAPTION's CLIP embedding against each GT candidate
    class; argmax over candidates = predicted class.

Design: the CLIP model wrapper (`CLIPScorer`) is kept separate from the metric
math (pure NumPy on L2-normalized embeddings). This lets the aggregation logic
be unit-tested without loading torch/open_clip, and lets Phase-2 scoring cache
each image embedding once and reuse it across all three metrics.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple
import numpy as np

from src.utils import crosses_decile

OBJECT_TEMPLATE = "a photo of a {}"
DEFAULT_CLIPSCORE_SCALE = 2.5
CLIPMATCH_DATASETS = ("imagenet", "places365")
LONG_CLIP_REPO_PATH = "/home/pmonserrat/Long-CLIP"

# Official OpenAI CLIP 80 prompt templates for ImageNet (Radford et al., 2021
# — "Learning Transferable Visual Models From Natural Language Supervision",
# Appendix A.6 / the repo's notebooks/Prompt_Engineering_for_ImageNet.ipynb).
# Object-centric ("a photo of a {}", "a sculpture of a {}", ...) — designed
# for object-recognition classes, NOT scenes. Used ONLY for the ClipMatch
# CANDIDATE (GT-class) side on ImageNet: each class's embedding is the mean of
# its 80 templated-text embeddings, L2-renormalized (the paper's "prompt
# ensembling" recipe) — computed ONCE per class since it's constant across the
# whole evaluation run. Extracted objects (Object-CLIPScore, per-image, one
# per entity) still use the single OBJECT_TEMPLATE — encoding 80 embeddings
# per extracted entity per image would be needlessly expensive and these were
# never validated for that use case.
OPENAI_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.',
    'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.',
    'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.',
    'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.',
    'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.',
    'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.',
    'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]

# Scene-recognition prompt ensemble for Places365's ClipMatch candidate side
# (per Pau). Original CLIP used only 2 bare templates for SUN397 scene
# recognition; OPENAI_TEMPLATES above is object-centric (tattoos, origami,
# video-game renders, plushies, ...) and doesn't fit scene classes ("kitchen",
# "airport terminal") — a "tattoo of a kitchen" or "the plushie kitchen" is
# nonsensical, so the ImageNet set is not reused here. Kept deliberately
# smaller than the 80-template ImageNet set: scenes don't have the same
# medium/material variety (no sculptures, embroidery, origami, tattoos), so
# most ImageNet templates have no scene-recognition analogue in the first
# place — this list only keeps variations that plausibly still describe a
# PLACE (contextual anchoring + camera-quality/lighting variation).
SCENE_TEMPLATES = [
    # Base structural
    "a photo of a {}.",
    "a photo of the {}.",
    "a scene of a {}.",
    "a view of the {}.",
    "a scenery of the {}.",
    # Contextual anchoring
    "a photo of a {}, a type of place.",
    "a photo of the {}, a place.",
    "a photo taken in a {}.",
    "a scene taken in a {}.",
    # Quality & lighting variations (safe for scenes)
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a black and white photo of a {}.",
    "a blurry photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
]

# ClipMatch candidate-side prompt ensemble, per dataset (see OPENAI_TEMPLATES/
# SCENE_TEMPLATES above). Only imagenet/places365 run ClipMatch at all
# (CLIPMATCH_DATASETS), so only those two need an entry.
CLIPMATCH_CANDIDATE_TEMPLATES = {
    "imagenet": OPENAI_TEMPLATES,
    "places365": SCENE_TEMPLATES,
}

# =============================================================================
# CLIP model wrapper (HuggingFace Native OR Local Pure-PyTorch)
# =============================================================================
CLIP_PRESETS = {
    "original": "openai/clip-vit-large-patch14",
    "metaclip": "facebook/metaclip-h14-fullcc2.5b",
    # MetaCLIP 2 (worldwide, ViT-H-14-quickgelu-worldwide) — a from-scratch
    # multilingual (300+ languages) successor to MetaCLIP 1 above, not a
    # fine-tune of it. Native transformers `MetaClip2Model`/`MetaClip2Processor`
    # (auto-dispatched via AutoModel/AutoProcessor), NO trust_remote_code —
    # same hassle-free category as SigLIP2/the LAION checkpoints below, not
    # FG-CLIP2's custom-init risk (see that preset's comment / recap §11).
    # Needs transformers>=4.56.0 (when MetaClip2 was merged upstream); already
    # covered by this project's transformers>=5.14.1 floor (requirements.txt),
    # so no extra dependency bump. MetaClip2Model mirrors CLIPModel's own
    # submodule shape (text_model+text_projection, vision_model+
    # visual_projection), so CLIPScorer._text_features/_image_features's
    # existing branch handles it with no special-casing.
    "metaclip2": "facebook/metaclip-2-worldwide-huge-quickgelu",
    "altclip": "BAAI/AltCLIP",
    # SigLIP2: native transformers AutoModel/AutoProcessor, NO trust_remote_code
    # — unlike FG-CLIP2 (tried and abandoned as a CLIP backend: its
    # trust_remote_code __init__ crashed with a meta-tensor error under this
    # project's transformers version; see data/llm_reference/vlm_pipeline_recap.txt
    # §11 for the full history), SigLIP2's whole forward path is plain library
    # code, so there's no custom-model-class __init__ to crash. SiglipModel has
    # no separate text_projection/visual_projection submodule (the pooled
    # output from text_model/vision_model IS already the projected embedding),
    # so CLIPScorer._text_features/_image_features fall through to their
    # get_text_features/get_image_features branch, which already returns a
    # plain Tensor — no special-casing needed here.
    "siglip2": "google/siglip2-base-patch16-224",
    # EVA-CLIP itself needs open_clip or a trust_remote_code wrapper — exactly
    # the dependency risk this project avoids (see FG-CLIP2 above). These two
    # LAION checkpoints are the hassle-free equivalent: they're OpenCLIP
    # weights converted to the SAME "original" CLIPModel/CLIPProcessor classes
    # (plain transformers, no trust_remote_code, identical code path to
    # "original" — literally just a different repo id), but trained on
    # LAION-2B (much larger/broader data than OpenAI's original CLIP), landing
    # in the same performance tier as EVA-CLIP's stronger variants.
    "laion_h14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",       # ~78% zero-shot IN1k, faster
    "laion_bigg": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",   # ~80% zero-shot IN1k, EVA-CLIP-tier, slower/bigger
    "longclip": "longclip",  # Routed directly to local GitHub clone
}

class CLIPScorer:
    """Wrapper that seamlessly bridges Native Hugging Face models and the 
    isolated pure PyTorch Long-CLIP codebase.
    """
    def __init__(
        self,
        model_name: str = "original",
        device: str = "cuda",
        batch_size: int = 64,
        torch_dtype: Optional[str] = "auto",
        trust_remote_code: bool = False,
        **kwargs
    ) -> None:
        import torch
        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        self.repo_id = CLIP_PRESETS.get(model_name, model_name)
        
        # ---------------------------------------------------------------------
        # PATH 1: LOCAL PURE-PYTORCH ROUTE (Long-CLIP ECCV 2024)
        # ---------------------------------------------------------------------
        if self.repo_id == "longclip":
            import sys
            import os
            
            # Dynamically inject the external repo into Python's path without 
            # polluting the local project folder.
            if LONG_CLIP_REPO_PATH not in sys.path:
                sys.path.insert(0, LONG_CLIP_REPO_PATH)
            
            try:
                from model import longclip
            except ImportError:
                raise ImportError(
                    f"Could not import Long-CLIP. Ensure the repo is cloned at "
                    f"'{LONG_CLIP_REPO_PATH}' and dependencies (ftfy, regex) are installed."
                )

            ckpt_path = os.path.join(LONG_CLIP_REPO_PATH, "checkpoints", "longclip-L.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing weights: Download longclip-L.pt into {ckpt_path}")

            # Load the official PyTorch model
            self.model, self.image_processor = longclip.load(ckpt_path, device=self.device)
            self.model.eval()
            self.tokenizer = longclip.tokenize
            self.context_length = 248
            self._is_native_hf = False
            
            # Find embedding dimension via throwaway encode
            dummy_text = self.tokenizer(["a photo"]).to(self.device)
            with self._torch.no_grad():
                self._embed_dim = self.model.encode_text(dummy_text).shape[1]

        # ---------------------------------------------------------------------
        # PATH 2: NATIVE HUGGING FACE ROUTE (MetaCLIP, AltCLIP, Original)
        # ---------------------------------------------------------------------
        else:
            from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, AutoModel
            self._is_native_hf = True
            dtype_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}

            self.model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code, **dtype_kwargs)
            self.model.eval().to(self.device)
            self._model_dtype = next(self.model.parameters()).dtype

            self._text_vocab_size = None
            text_model = getattr(self.model, "text_model", None)
            if text_model is not None and hasattr(text_model, "get_input_embeddings"):
                emb = text_model.get_input_embeddings()
                if emb is not None:
                    self._text_vocab_size = emb.num_embeddings

            try:
                processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code)
                self.tokenizer = getattr(processor, "tokenizer", processor)
                self.image_processor = getattr(processor, "image_processor", processor)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code)
                self.image_processor = AutoImageProcessor.from_pretrained(self.repo_id, trust_remote_code=trust_remote_code)

            config = getattr(self.model, "config", None)
            text_config = getattr(config, "text_config", config)
            max_pos = getattr(text_config, "max_position_embeddings", None)
            
            if isinstance(max_pos, int):
                self.context_length = max_pos
            else:
                self.context_length = getattr(self.tokenizer, "model_max_length", 77)
                if not isinstance(self.context_length, int) or self.context_length > 100_000:
                    self.context_length = 77

            dummy_inputs = self.tokenizer(["a photo"], return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                self._embed_dim = self._text_features(dummy_inputs).shape[1]

    # --- Hugging Face Submodule Extraction ---
    @staticmethod
    def _pool(output):
        pooled = getattr(output, "pooler_output", None)
        return pooled if pooled is not None else output.last_hidden_state[:, 0, :]

    def _unwrap_embedding(self, output, projection_attr: str):
        if isinstance(output, self._torch.Tensor):
            return output
        projection = getattr(self.model, projection_attr, None)
        pooled = self._pool(output)
        return projection(pooled) if projection is not None else pooled

    def _text_features(self, inputs: dict):
        if hasattr(self.model, "text_model") and hasattr(self.model, "text_projection"):
            pooled = self._pool(self.model.text_model(**inputs))
            return self.model.text_projection(pooled)
        if hasattr(self.model, "get_text_features"):
            out = self.model.get_text_features(**inputs)
            return self._unwrap_embedding(out, "text_projection")
        raise AttributeError(f"{self.repo_id}: Unrecognized text API.")

    def _image_features(self, inputs: dict):
        if hasattr(self.model, "vision_model") and hasattr(self.model, "visual_projection"):
            pooled = self._pool(self.model.vision_model(**inputs))
            return self.model.visual_projection(pooled)
        if hasattr(self.model, "get_image_features"):
            return self._unwrap_embedding(self.model.get_image_features(**inputs), "visual_projection")
        raise AttributeError(f"{self.repo_id}: Unrecognized vision API.")

    def _to_device(self, inputs: dict) -> dict:
        return {
            k: v.to(self.device, dtype=self._model_dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

    # --- Batch Encoding (Bridges HF and Pure PyTorch) ---
    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        
        if not self._is_native_hf:
            # Pure PyTorch Long-CLIP flow
            tokens = self.tokenizer(texts, truncate=True).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            return feats.cpu().float().numpy()

        # Hugging Face flow
        inputs = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.context_length, return_tensors="pt"
        )
        inputs = self._to_device(inputs)
        with torch.no_grad():
            feats = self._text_features(inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().float().numpy()

    def _encode_image_batch(self, images) -> np.ndarray:
        torch = self._torch
        
        if not self._is_native_hf:
            # Pure PyTorch Long-CLIP flow (image_processor is a torchvision transform)
            tensors = [self.image_processor(img) for img in images]
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_tensor)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            return feats.cpu().float().numpy()

        # Hugging Face flow
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = self._to_device(inputs)
        with torch.no_grad():
            feats = self._image_features(inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().float().numpy()

    # --- Public API ---
    def encode_text(self, texts: List[str], warn_truncation: bool = False,
                     verbose: bool = False, desc: str = "text") -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        if warn_truncation and self._is_native_hf and self.tokenizer is not None:
            for t in texts:
                n_tok = len(self.tokenizer(t, truncation=False)["input_ids"])
                if n_tok >= self.context_length:
                    warnings.warn(
                        f"Caption reaches/exceeds context length ({self.context_length} tokens) "
                        f"and will be truncated. F-CLIPScore sentence term is affected.",
                        stacklevel=2,
                    )
                    break

        out = []
        n_total = len(texts)
        for i in range(0, n_total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.append(self._encode_text_batch(batch))
            if verbose:
                done = min(i + self.batch_size, n_total)
                # Only print every ~10% of n_total (see utils.crosses_decile)
                # — one line per batch is unreadable spam once n_total is
                # large relative to batch_size (e.g. an 80k-text candidate
                # vocab encoded self.batch_size at a time).
                if crosses_decile(i, done, n_total):
                    print(f"🔎 [CLIP] {desc}: {done}/{n_total} ({done / n_total:.1%})", flush=True)
        return np.concatenate(out, axis=0)

    def count_tokens(self, texts: List[str]) -> List[int]:
        """Untruncated token count per text under this model's OWN tokenizer —
        used for a truncation-rate diagnostic (e.g. run_vlm_pipeline.py's
        summary-caption ClipMatch ablation), independent of `context_length`.

        Native-HF only: a plain HF tokenizer call with truncation=False gives
        the real length directly. Long-CLIP's tokenizer is a bespoke function
        (`longclip.tokenize`) that always returns a fixed `context_length`-wide
        padded tensor regardless of truncate=True/False, so it can't report an
        untruncated count the same way — raise rather than silently return a
        misleading number.
        """
        if not self._is_native_hf:
            raise NotImplementedError(
                "count_tokens needs a native-HF tokenizer (Long-CLIP's tokenize() always "
                "pads/truncates to a fixed context_length and can't report a raw token count)."
            )
        return [len(self.tokenizer(t, truncation=False)["input_ids"]) for t in texts]

    def encode_images(self, image_paths: List[str], verbose: bool = False) -> np.ndarray:
        from PIL import Image

        if not image_paths:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        out = []
        n_total = len(image_paths)
        for i in range(0, n_total, self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            failed = []  
            for j, p in enumerate(batch_paths):
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    warnings.warn(f"CLIP: could not read image '{p}' ({e!r}); using a zero embedding.", stacklevel=2)
                    failed.append(j)

            if images:
                feats = self._encode_image_batch(images)
                d = feats.shape[1]
            else:
                d = self._embed_dim
                feats = np.zeros((0, d), dtype=np.float32)

            batch_out = np.zeros((len(batch_paths), d), dtype=np.float32)
            good_iter = iter(feats)
            for j in range(len(batch_paths)):
                if j not in failed:
                    batch_out[j] = next(good_iter)
            out.append(batch_out)
            
            if verbose:
                done = min(i + self.batch_size, n_total)
                if crosses_decile(i, done, n_total):
                    print(f"🔎 [CLIP] images: {done}/{n_total} ({done / n_total:.1%})", flush=True)
        return np.concatenate(out, axis=0)


# =============================================================================
# Metric math (pure numpy on L2-normalized embeddings)
# =============================================================================
def _cos(image_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    if text_embs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return text_embs @ image_emb

def clip_score(sim: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE) -> np.ndarray:
    return w * np.clip(sim, 0.0, None)

def object_clipscore(
    image_emb: np.ndarray, object_embs: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE,
) -> float:
    if object_embs.shape[0] == 0: return 0.0
    return float(np.mean(clip_score(_cos(image_emb, object_embs), w)))

def clipscore(
    image_emb: np.ndarray, caption_emb: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE,
) -> float:
    """Plain CLIPScore(caption): the standard reference-free image-caption
    score (Hessel et al.), no object terms at all. This is the S_term that
    F-CLIPScore also folds in, exposed here as its own metric so it can be
    reported alongside (and compared against) F-CLIPScore/Object-CLIPScore."""
    return float(clip_score(_cos(image_emb, caption_emb[None, :]), w)[0])

def f_clipscore(
    image_emb: np.ndarray, caption_emb: np.ndarray, object_embs: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE,
) -> float:
    s_term = clipscore(image_emb, caption_emb, w)
    obj_terms = clip_score(_cos(image_emb, object_embs), w)
    return (s_term + float(np.sum(obj_terms))) / (object_embs.shape[0] + 1)

def clipmatch(
    caption_emb: np.ndarray, candidate_embs: np.ndarray,
) -> Tuple[np.ndarray, int]:
    n_cand = candidate_embs.shape[0]
    if n_cand == 0:
        return np.zeros((0,), dtype=np.float32), -1
    per_candidate_sim = candidate_embs @ caption_emb
    return per_candidate_sim, int(np.argmax(per_candidate_sim))

_INFLECT_ENGINE = None


def _inflect_engine():
    """Lazy `inflect.engine()` singleton — imported on first use rather than at
    module import so a caller that never fills a template (e.g. a unit test of
    the pure-numpy metric math) doesn't need the dependency."""
    global _INFLECT_ENGINE
    if _INFLECT_ENGINE is None:
        import inflect
        _INFLECT_ENGINE = inflect.engine()
    return _INFLECT_ENGINE


# Prepositions/conjunctions that end a noun phrase. If one of these sits
# between a candidate article and the "{}" slot, that article belongs to some
# EARLIER noun, not to the class name — see fill_template.
_NP_BOUNDARY_WORDS = frozenset({"of", "in", "on", "at", "with", "from", "for", "by"})


def _is_plural(phrase: str) -> bool:
    """Whether `phrase`'s head noun (its LAST word) is plural — "cars",
    "sunglasses", "french fries". inflect's singular_noun() returns the
    singular form for a plural input and False otherwise, which is exactly
    this test. Only the last word is checked because that's the head of an
    English noun phrase ("mountain bikes" is plural, "bike wheel" is not)."""
    words = phrase.strip().split()
    if not words:
        return False
    return bool(_inflect_engine().singular_noun(words[-1]))


def fill_template(template: str, name: str) -> str:
    """Fill a CLIP prompt template's "{}" slot with `name`, fixing the
    indefinite article in front of it so the result is grammatical:

        "a photo of a {}"  + "apple"  -> "a photo of an apple"
        "a photo of a {}"  + "cars"   -> "a photo of cars"
        "a photo of a {}"  + "forest" -> "a photo of a forest"

    Only an article that actually GOVERNS the class name is touched, which is
    what makes this safe to run over all 80 ImageNet + 15 scene templates
    rather than just the plain "a photo of a {}" one. Three cases:

      1. No indefinite article governs the slot — "the plastic {}.",
         "a photo of many {}.", "itap of my {}." Nothing to fix; "the"/"many"/
         "my" are already correct for singular and plural alike. Note the "a"
         in "a photo of many {}" belongs to "photo", NOT to the class — that
         is what _NP_BOUNDARY_WORDS detects, and dropping it would corrupt the
         template into "photo of many trees".
      2. The article sits directly before the slot — "a photo of a {}.",
         "art of a {}.", "a tattoo of a {}." Here it agrees with the class
         name itself, so inflect picks "a" vs "an" by the name's actual
         spelling/sound ("an hour", "a university"), or drops it for a plural.
      3. An adjective intervenes — "a photo of a clean {}.", "a photo of a
         hard to see {}." The article agrees with the ADJECTIVE ("a clean",
         "a hard to see"), which the template already got right, so a/an is
         left alone — but it is still dropped for a plural class, since
         "a clean trees" is wrong while "clean trees" is right.
    """
    slot = template.find("{}")
    if slot == -1:
        return template.format(name)
    prefix = template[:slot]
    tokens = prefix.split()

    # Walk backwards for the nearest "a"/"an", stopping at a noun-phrase
    # boundary (case 1) — anything found past one governs an earlier noun.
    article_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        word = tokens[i].lower()
        if word in _NP_BOUNDARY_WORDS:
            break
        if word in ("a", "an"):
            article_idx = i
            break

    if article_idx is None:
        return template.format(name)

    plural = _is_plural(name)
    intervening = tokens[article_idx + 1:]

    if plural:
        # Case 2/3 with a plural head noun: drop the article outright.
        kept = tokens[:article_idx] + intervening
        rebuilt = " ".join(kept)
        return (rebuilt + (" " if kept else "")) + template[slot:].format(name)

    if intervening:
        return template.format(name)  # case 3: article agrees with the adjective

    # Case 2: the article agrees with the class name itself.
    article = _inflect_engine().a(name).split(maxsplit=1)[0]
    kept = tokens[:article_idx] + [article]
    return " ".join(kept) + " " + template[slot:].format(name)


def object_texts(objects: List[str]) -> List[str]:
    """Per-extracted-object text for Object-CLIPScore / F-CLIPScore's object
    terms and ClipMatch's anchor-object search: the single OBJECT_TEMPLATE
    (never the 80/15-template ensemble — that is candidate-side only)."""
    return [fill_template(OBJECT_TEMPLATE, o) for o in objects]


def wordnet_definition_text(synset_id: str) -> str:
    """Richer ClipMatch candidate text for one GT class: its canonical
    lemma(s) plus WordNet's own gloss, in natural prose —

        "A golden retriever, also known as golden retriever dog. It is
        defined as: an English breed having a long silky golden coat."

    ...instead of the plain OBJECT_TEMPLATE ("a photo of a golden
    retriever"). Motivation (Pau): a VLM's own image descriptions are
    often definitional/explanatory in style, so giving the CANDIDATE side
    more semantic content (not just a bare noun phrase) may align better
    in CLIP's text embedding space than a templated phrase built the same
    way for every class regardless of how distinctive or obscure it is.

    Opt-in via --use_wordnet_definitions_clipmatch (ImageNet/Places only —
    get_candidate_vocab guarantees every candidate's synset_id resolves via
    `wn.synset()`, per its own docstring, so this doesn't need a mapping
    fallback the way object-side WordNet lookups do).
    """
    from nltk.corpus import wordnet as wn

    synset = wn.synset(synset_id)
    lemmas = [lemma.name().replace("_", " ").lower() for lemma in synset.lemmas()]
    primary_name = lemmas[0]
    alt_names = f", also known as {', '.join(lemmas[1:])}" if len(lemmas) > 1 else ""

    # inflect prepends the correct "a"/"an" for the primary lemma's own spelling.
    name_with_article = _inflect_engine().a(primary_name)

    definition = synset.definition()
    return f"{name_with_article.capitalize()}{alt_names}. It is defined as: {definition}."


def candidate_vocab_texts(candidate_vocab: List[dict], use_wordnet_definitions: bool = False) -> List[str]:
    """Text for each ClipMatch candidate class: the plain OBJECT_TEMPLATE
    phrase by default, or wordnet_definition_text(synset_id) per class when
    `use_wordnet_definitions` is set (--use_wordnet_definitions_clipmatch).

    Single text per class — used for the WordNet-definition variant (a
    template ensemble doesn't apply to prose) and as the one-template
    fallback for any dataset without an entry in CLIPMATCH_CANDIDATE_TEMPLATES.
    For the default templated path on ImageNet/Places, prefer
    encode_candidate_vocab_ensemble below (mean of 80/15 prompt-template
    embeddings, not a single text)."""
    if use_wordnet_definitions:
        return [wordnet_definition_text(c["synset_id"]) for c in candidate_vocab]
    return [fill_template(OBJECT_TEMPLATE, c["class_name"]) for c in candidate_vocab]


def candidate_template_texts(candidate_vocab: List[dict], templates: List[str]) -> List[List[str]]:
    """Per-class list of templated texts (one row per candidate class, one
    column per template) — the raw material for encode_candidate_vocab_ensemble.
    Every fill goes through fill_template, so the article in front of the class
    name is corrected per template AND per class ("a photo of an airfield",
    "a photo of badlands")."""
    return [[fill_template(t, c["class_name"]) for t in templates] for c in candidate_vocab]


def encode_candidate_vocab_ensemble(
    scorer, candidate_vocab: List[dict], templates: List[str],
    verbose: bool = False, desc: str = "candidate_vocab",
) -> Tuple[np.ndarray, List[str]]:
    """Encode each ClipMatch candidate class as the MEAN of its `len(templates)`
    prompt-template embeddings, L2-renormalized afterwards — the OpenAI CLIP
    "prompt ensembling" recipe (Radford et al., 2021, and the zero-shot
    notebooks distributed with it). One embedding per class, computed ONCE
    (candidate classes are constant across the whole evaluation run, unlike
    the per-image caption/object embeddings) — never recomputed per image.

    Returns (candidate_embs [n_classes, dim], flat_texts [n_classes *
    len(templates)]) — flat_texts is exposed so callers can still run their
    existing token-length/truncation diagnostics over the actual encoded text.
    """
    per_class_texts = candidate_template_texts(candidate_vocab, templates)
    flat_texts = [t for row in per_class_texts for t in row]
    flat_embs = scorer.encode_text(flat_texts, verbose=verbose, desc=desc)
    n_templates = len(templates)
    embs = flat_embs.reshape(len(candidate_vocab), n_templates, -1).mean(axis=1)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = (embs / norms).astype(np.float32)
    return embs, flat_texts