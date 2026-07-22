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

OBJECT_TEMPLATE = "a photo of a {}"
DEFAULT_CLIPSCORE_SCALE = 2.5
CLIPMATCH_DATASETS = ("imagenet", "places365")
LONG_CLIP_REPO_PATH = "/home/pmonserrat/Long-CLIP"

# =============================================================================
# CLIP model wrapper (HuggingFace Native OR Local Pure-PyTorch)
# =============================================================================
CLIP_PRESETS = {
    "original": "openai/clip-vit-large-patch14",
    "metaclip": "facebook/metaclip-h14-fullcc2.5b",
    "altclip": "BAAI/AltCLIP",
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

            self.model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=False, **dtype_kwargs)
            self.model.eval().to(self.device)
            self._model_dtype = next(self.model.parameters()).dtype

            self._text_vocab_size = None
            text_model = getattr(self.model, "text_model", None)
            if text_model is not None and hasattr(text_model, "get_input_embeddings"):
                emb = text_model.get_input_embeddings()
                if emb is not None:
                    self._text_vocab_size = emb.num_embeddings

            try:
                processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=False)
                self.tokenizer = getattr(processor, "tokenizer", processor)
                self.image_processor = getattr(processor, "image_processor", processor)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=False)
                self.image_processor = AutoImageProcessor.from_pretrained(self.repo_id, trust_remote_code=False)

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
                print(f"🔎 [CLIP] {desc}: {done}/{n_total} ({done / n_total:.1%})", flush=True)
        return np.concatenate(out, axis=0)

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

def f_clipscore(
    image_emb: np.ndarray, caption_emb: np.ndarray, object_embs: np.ndarray, w: float = DEFAULT_CLIPSCORE_SCALE,
) -> float:
    s_term = float(clip_score(_cos(image_emb, caption_emb[None, :]), w)[0])
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

def object_texts(objects: List[str]) -> List[str]:
    return [OBJECT_TEMPLATE.format(o) for o in objects]