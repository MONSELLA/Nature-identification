"""
src/models/vlm_models.py

Unified interface for running VLMs across different serving backends (vLLM /
HuggingFace transformers) and model families, for the BIG-5 VLM pipeline.

NO PROMPTS LIVE IN THIS FILE.

WHAT IS A "VLM" AND WHY DOES THIS FILE NEED TO BE SO ABSTRACTED?
"VLM" = Vision-Language Model: a neural network that can take BOTH an image
and text as input and produce text as output (e.g. "describe this image",
"is there a dog in this picture?"). This project runs and compares SEVERAL
different VLMs (Qwen, Mistral, LLaVA, InternVL, Gemma), all served through
vLLM, a high-performance serving engine (talks to the model via a chat-style
API, handles batching/GPU memory internally).

Rather than making every OTHER file in this project (dataset loaders, the
pipeline, the scripts) know about vLLM's own API shape, this file defines ONE
common interface (`BaseVLM.generate` / `generate_batch`) that every model
family implements, so calling code just does
`vlm.generate_batch(prompts=..., images=..., ...)` and gets consistent
results back no matter which underlying model is actually running.

TWO KEY CONCEPTS USED THROUGHOUT THIS FILE:
  - "structured output" / "guided decoding": normally a language model can
    generate ANY text it wants. When we pass `output_mode="structured"` and a
    `schema` (a pydantic class — see src/models/prompts.py), vLLM constrains
    generation via `StructuredOutputsParams` so the model is FORCED to produce
    valid JSON matching that schema — it becomes literally impossible for the
    model to output something that doesn't parse.
  - "batch": processing many (prompt, image) pairs in a SINGLE model call
    instead of looping one at a time. This matters a lot for speed — GPUs are
    much more efficient when given a big batch of work at once.

NOTE: this file previously also supported a second, HuggingFace-`transformers`-
served backend (the BLIP/BLIP-2/InstructBLIP/BLIP-3 family, via a
HuggingFaceBackedVLM base class and its own outlines-based structured-decoding
path). That backend and all four BLIP model classes have been REMOVED — this
project only evaluates vLLM-served models now. Do not re-add BLIP support
without re-adding `outlines` to requirements.txt first (it was removed
alongside this backend, since nothing else in the codebase imports it).
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# The type of a single "image" argument accepted throughout this file: either
# a file path / URL string, an already-loaded PIL Image object, or None (no
# image for this call). `"PIL.Image.Image"` is written as a STRING here
# (rather than actually importing PIL) so this module doesn't require PIL to
# be installed just to be imported — only actually using an image does.
ImageInput = Union[str, "PIL.Image.Image", None]  # noqa: F821


# =============================================================================
# Abstract base
# =============================================================================

class BaseVLM(ABC):
    """
    The common interface every VLM backend/family must implement. `ABC`
    (Abstract Base Class) + `@abstractmethod` below means: you cannot create a
    plain `BaseVLM()` directly, and every subclass MUST provide its own
    `generate()` implementation — Python will raise an error at class-creation
    time if a subclass forgets to.
    """

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image: ImageInput = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        output_mode: str = "free_form",
        schema: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:
        """Run ONE (prompt, image) pair through the model and return either a
        plain string (output_mode="free_form") or a parsed dict
        (output_mode="structured"), or None if structured parsing failed."""
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: List[str],
        images: Optional[List[ImageInput]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        output_mode: str = "free_form",
        schema: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Union[str, Dict[str, Any], None]]:
        # Fallback sequential loop for edge-case models that don't override this method.
        # This DEFAULT implementation just calls generate() once per item in a
        # plain Python loop — correct, but NOT actually batched/parallelized
        # on the GPU. VLLMBackedVLM overrides this with a real batched
        # implementation; this fallback only kicks in for a subclass that
        # hasn't bothered to.
        if images is None:
            images = [None] * len(prompts)
        if len(images) != len(prompts):
            raise ValueError("`images` and `prompts` must be the same length.")

        return [
            self.generate(
                prompt=p, image=img, system_prompt=system_prompt,
                max_new_tokens=max_new_tokens, temperature=temperature,
                output_mode=output_mode, schema=schema, **kwargs,
            )
            for p, img in zip(prompts, images)
        ]

    def _is_recoverable_overflow(self, exc: Exception) -> bool:
        """Hook for subclasses: does `exc` mean "one prompt in this batch was
        too long for the model's context window" (as opposed to some other
        failure, e.g. OOM)? Base default is False — a backend that never
        raises this kind of error (or hasn't been taught to recognize its own
        error shape yet) gets a safe no-op: `generate_batch_safe` below just
        re-raises everything, identical to plain `generate_batch`."""
        return False

    def generate_batch_safe(
        self,
        prompts: List[str],
        images: Optional[List[ImageInput]] = None,
        label: str = "batch",
        item_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Union[str, Dict[str, Any], None]]:
        """Like `generate_batch`, but tolerant of a single-prompt context-
        window overflow WITHOUT sacrificing batching for the rest of the
        batch. A batched call (e.g. vLLM's single `.chat()` covering every
        conversation) fails ENTIRELY if even one prompt overflows, with no
        indication of which one — so on a `_is_recoverable_overflow` failure
        we BISECT the batch in half and recurse, each half still generated as
        one batched call. This isolates the oversized prompt(s) down to
        single-item granularity (returned as None, aligned with input order)
        while every other sample stays batched together, instead of
        degrading the whole batch to one-at-a-time generation.

        `item_labels` (optional, same length as prompts) is used purely for
        the warning message identifying which item was skipped; falls back
        to a generic "item" if not provided.
        """
        n = len(prompts)
        if n == 0:
            return []
        if images is None:
            images = [None] * n
        try:
            return self.generate_batch(prompts=prompts, images=images, **kwargs)
        except Exception as e:
            if not self._is_recoverable_overflow(e):
                raise
            if n == 1:
                name = item_labels[0] if item_labels else "item"
                print(f"⚠️ Skipping {name} ({label}): prompt too long for max_model_len ({e}).")
                return [None]
            mid = n // 2
            print(f"⚠️ {label}: a prompt exceeded max_model_len — bisecting "
                  f"{n} instances into two sub-batches to isolate it.")
            left_labels = item_labels[:mid] if item_labels else None
            right_labels = item_labels[mid:] if item_labels else None
            left = self.generate_batch_safe(
                prompts[:mid], images[:mid], f"{label}/left", left_labels, **kwargs)
            right = self.generate_batch_safe(
                prompts[mid:], images[mid:], f"{label}/right", right_labels, **kwargs)
            return left + right


# =============================================================================
# Backend base classes
# =============================================================================

def _patch_transformers_config_registration() -> None:
    """Work around a real vLLM bug that breaks `from vllm import LLM` outright
    on newer `transformers` releases: vLLM's Ovis config shim
    (vllm/transformers_utils/configs/ovis.py) calls
    `AutoConfig.register("aimv2", AIMv2Config)` WITHOUT `exist_ok=True`. That
    was fine when `transformers` had no native "aimv2" model type — but once
    a `transformers` release adds native support for it (Apple's AIMv2 vision
    encoder), the SAME registration collides and `CONFIG_MAPPING.register`
    raises `ValueError: 'aimv2' is already used by a Transformers config,
    pick another name.` — which happens at vLLM's own import time, so it's
    not something --clipscore_model/--model_family choices, or any `transformers`/
    `vllm` version PIN, can dodge: it depends on which side (vLLM's shim vs.
    transformers' own native support) landed the "aimv2" name first, which
    isn't expressed as a version constraint pip can resolve around.

    Patches the actual `CONFIG_MAPPING` singleton instance's bound
    `register()` (the exact object vLLM's shim calls into per the traceback)
    to always pass `exist_ok=True`, so a name that's already registered is
    silently skipped — keeping transformers' own native config, which is what
    we want anyway — instead of crashing the whole `from vllm import LLM`
    import chain. Idempotent (checked via an attribute on the mapping
    instance) and a no-op if `transformers` isn't installed or this exact
    collision doesn't occur on your version pair.
    """
    # Defensive: this pokes at a third-party library's internals, which can
    # shift across transformers versions. Any failure here should silently
    # fall back to unpatched (still-broken-on-that-collision) behavior rather
    # than introduce a NEW crash on top of the one we're trying to work around.
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if getattr(CONFIG_MAPPING, "_big5_register_patched", False):
            return
        _orig_register = CONFIG_MAPPING.register

        def _safe_register(model_type, config, exist_ok=False):
            return _orig_register(model_type, config, exist_ok=True)

        CONFIG_MAPPING.register = _safe_register
        CONFIG_MAPPING._big5_register_patched = True
    except Exception:
        return


class VLLMBackedVLM(BaseVLM):
    """VLM family served through vLLM (used for qwen/mistral/llava — see
    MODEL_REGISTRY at the bottom of this file). vLLM exposes an OpenAI-style
    chat API (`self.llm.chat(...)`) that internally handles GPU scheduling,
    KV-cache management, and batching for us."""

    def __init__(self, model_name: str, max_image_side: Optional[int] = 1024, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        # `max_image_side` is an EXPLICIT named parameter (not swept into
        # **kwargs) specifically so it never gets forwarded to LLM(...)
        # below — vLLM's own constructor has no such argument and would
        # raise on an unexpected kwarg. See _encode_image for what it does:
        # downscale any image whose longest side exceeds this many pixels
        # before it ever reaches vLLM. This exists because nothing in this
        # codebase resizes images at all otherwise — ImageNet/COCO/Places
        # images are typically already modest (pre-resized benchmark images),
        # but raw social-media images (BIG-5 Twitter/Weibo) can be arbitrary
        # phone-camera/screenshot resolutions with no cap. A vision encoder
        # like Pixtral scales its patch count (and therefore its attention
        # memory, quadratically) with input resolution, not a fixed small
        # square — one oversized raw image in a batch can OOM the vision
        # encoder even when batch_size and max_model_len are both far lower
        # than what ImageNet/Places comfortably handle at the same settings.
        # Confirmed directly: a BIG-5 OOM traceback failed inside
        # pixtral.py's vision_encoder -> scaled_dot_product_attention, not
        # in the text KV cache, with no image resizing anywhere upstream.
        # None disables resizing entirely (opt out, e.g. to reproduce old
        # behavior or if you've pre-resized images yourself).
        self.max_image_side = max_image_side
        _patch_transformers_config_registration()
        try:
            from vllm import LLM
        except ImportError as e:
            # A clearer error message than the raw ImportError, so it's
            # obvious WHY this failed (missing an optional heavy dependency)
            # rather than looking like a bug.
            raise ImportError("VLLMBackedVLM requires the `vllm` package.") from e
        # This actually loads the model weights onto the GPU (this line can
        # take a while and use a lot of VRAM — see unload_vlm() further down
        # for how we later release this memory).
        self.llm = LLM(model=model_name, **kwargs)

    def _is_recoverable_overflow(self, exc: Exception) -> bool:
        """vLLM's `.chat()` raises a ValueError with this exact substring
        when a conversation's prompt (text + image tokens) exceeds
        max_model_len — the specific, bisectable overflow case
        `generate_batch_safe` knows how to recover from."""
        return isinstance(exc, ValueError) and "longer than the maximum model length" in str(exc)

    def _encode_image(self, image: ImageInput) -> str:
        """Convert an image (file path or PIL Image) into a "data URL" string
        (base64-encoded bytes embedded directly in the string) — the format
        vLLM's chat API expects for the `image_url` message field.

        DOWNSCALES first if the image's longest side exceeds
        `self.max_image_side` (see __init__'s comment for why this exists —
        raw social-media images with no resolution cap can OOM the vision
        encoder even at a modest batch size). Uses a FAST PATH for images
        already under the cap (the common case for ImageNet/COCO/Places,
        and most BIG-5 images too): `Image.open()` only reads the header to
        get `.size`, not the full pixel data, so checking the size is cheap,
        and if no resize is needed the original file bytes are read and
        base64-encoded directly — no PIL decode/re-encode round-trip, which
        matters at this project's 2M-image scale.
        """
        if isinstance(image, str) and image.startswith("data:image"):
            return image  # already a data URL — nothing to do

        from PIL import Image as PILImage

        pil_image = PILImage.open(image) if isinstance(image, str) else image
        width, height = pil_image.size
        needs_resize = self.max_image_side is not None and max(width, height) > self.max_image_side

        if not needs_resize:
            if isinstance(image, str):
                with open(image, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                # Preserve the original file extension (jpg/png/etc.) in the
                # data URL's MIME type, defaulting to png if there's none.
                suffix = Path(image).suffix.lstrip(".").lower() or "png"
                return f"data:image/{suffix};base64,{b64}"
            # Already a PIL Image and small enough — encode as-is (in-memory
            # "file" via BytesIO, rather than needing to save to disk first).
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        # Oversized — downscale, preserving aspect ratio, before encoding.
        scale = self.max_image_side / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        # .convert("RGB") first: resizing a palette/CMYK/RGBA image directly
        # can produce wrong colors or an unexpected channel count, and RGB is
        # what every VLM here expects anyway.
        resized = pil_image.convert("RGB").resize(new_size, PILImage.LANCZOS)
        buffer = BytesIO()
        # JPEG, not PNG: PNG is lossless and can be dramatically larger for
        # photographic content, bloating both the base64 payload sent to
        # vLLM and its own image preprocessing; quality=90 is visually
        # near-lossless for a VLM's own patch embedding while keeping the
        # request small.
        resized.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _build_messages(
        self,
        prompt: str,
        image: ImageInput,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build one "conversation" in the OpenAI chat-style format vLLM
        expects: an optional system message, then a single user message
        (containing the image + text if an image was given, or just text)."""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image is not None:
            # When an image is attached, the "content" field is a LIST of
            # typed content blocks (text block + image block) — this is the
            # standard multi-modal chat message shape.
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(image)}},
            ]
        else:
            # No image — content can just be the plain prompt string.
            content = prompt

        messages.append({"role": "user", "content": content})
        return messages

    def _parse_response(self, text: str, output_mode: str) -> Union[str, Dict[str, Any], None]:
        """Turn the model's raw output text into the shape callers expect:
        a parsed dict for structured mode (or None if it somehow didn't parse
        as valid JSON despite guided decoding), or the plain text otherwise."""
        if output_mode == "structured":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        return text

    def _make_sampling_params(self, temperature, max_new_tokens, output_mode, schema, **kwargs):
        """Build vLLM's SamplingParams object — the settings controlling HOW
        the model generates text (temperature, max length, and — crucially —
        the guided-decoding constraint if a schema was requested)."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        gd = None
        if output_mode == "structured" and schema is not None:
            # `schema.model_json_schema()` converts a pydantic BaseModel class
            # into the plain JSON Schema dict format vLLM's guided decoding
            # actually needs (if `schema` is already a plain dict, use it as-is).
            js = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            gd = StructuredOutputsParams(json=js)
        return SamplingParams(temperature=temperature, max_tokens=max_new_tokens,
                            structured_outputs=gd, **kwargs)

    def generate(
        self,
        prompt: str,
        image: ImageInput = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        output_mode: str = "free_form",
        schema: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:
        """Single (prompt, image) generation via vLLM's chat API."""
        from vllm import SamplingParams

        messages = self._build_messages(prompt, image, system_prompt)

        sampling_params = self._make_sampling_params(
            temperature=temperature, max_new_tokens=max_new_tokens,
            output_mode=output_mode, schema=schema, **kwargs
        )
        # `self.llm.chat([messages], ...)` takes a LIST of conversations (here
        # just one) and returns a list of results — we pull out the single
        # generated text from the first (and only) result.
        outputs = self.llm.chat([messages], sampling_params=sampling_params, use_tqdm=False)
        text = outputs[0].outputs[0].text or ""
        return self._parse_response(text, output_mode)

    def generate_batch(
        self,
        prompts: List[str],
        images: Optional[List[ImageInput]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        output_mode: str = "free_form",
        schema: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Union[str, Dict[str, Any], None]]:
        """TRUE batched generation: builds every conversation up front and
        hands them ALL to vLLM in a single `.chat(...)` call, letting vLLM's
        own internal scheduler decide how to efficiently run them together on
        the GPU (this is much faster than looping generate() one at a time)."""
        from vllm import SamplingParams

        if images is None:
            images = [None] * len(prompts)
        if len(images) != len(prompts):
            raise ValueError("`images` and `prompts` must be the same length.")

        conversations = [self._build_messages(p, img, system_prompt) for p, img in zip(prompts, images)]

        sampling_params = self._make_sampling_params(
            temperature=temperature, max_new_tokens=max_new_tokens,
            output_mode=output_mode, schema=schema, **kwargs
        )
        outputs = self.llm.chat(conversations, sampling_params=sampling_params, use_tqdm=len(prompts) > 1)
        return [self._parse_response(o.outputs[0].text or "", output_mode) for o in outputs]


# =============================================================================
# Factory
# =============================================================================
# `MODEL_REGISTRY` maps a short, human-typed "family" name (what you pass on
# the command line via --model_family) to the actual Python class that knows
# how to run that family. `create_vlm()` below is the ONE place any other
# file in this project should use to actually construct a VLM instance —
# nobody else needs to know which concrete class backs a given family name.
# Every registered family is vLLM-served (see the module docstring — the
# HuggingFace-served BLIP family was removed).

MODEL_REGISTRY: Dict[str, type] = {
    "qwen": VLLMBackedVLM,
    "mistral": VLLMBackedVLM,
    "llava": VLLMBackedVLM,
    "internvl": VLLMBackedVLM,
    "gemma": VLLMBackedVLM,
}

def create_vlm(family: str, model_name: str, **kwargs: Any) -> BaseVLM:
    """Construct the right VLM subclass for the given family name, e.g.
    create_vlm("qwen", "Qwen/Qwen3.5-0.8B", dtype="auto", ...)."""
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family '{family}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[family](model_name, **kwargs)


# =============================================================================
# Memory release — needed for a single-process end-to-end run (VLM inference
# followed by CLIP scoring in the SAME process). Neither vLLM nor plain torch
# release CUDA memory back to the driver on Python garbage collection alone:
# vLLM keeps a distributed process group + KV-cache allocator alive. Skipping
# this step is why the pipeline used to require two separate process
# invocations (--stage infer, then --stage score).
# =============================================================================
def unload_vlm(vlm: BaseVLM) -> None:
    """Tear down a VLM's GPU-resident state so a CLIP/metric model can be
    loaded afterward in the SAME process without contending for VRAM. Every
    registered family is VLLMBackedVLM (see MODEL_REGISTRY), so `.llm` is the
    only GPU-holding attribute there is to clear.

    WHY IS THIS SO INVOLVED? Simply doing `del vlm` in Python only removes
    OUR reference to the object — if vLLM itself is still holding onto GPU
    memory behind the scenes (e.g. a distributed process group, or cached
    CUDA memory blocks), that memory stays allocated regardless. This function
    explicitly asks each layer to let go of its own resources, in order,
    before finally asking Python's garbage collector and PyTorch's CUDA
    allocator to actually reclaim everything.
    """
    import gc

    llm = getattr(vlm, "llm", None)
    if llm is not None:
        try:
            # vLLM sets up its own "distributed" machinery internally (even
            # when running on just a single GPU) for its tensor-parallel
            # execution model — these calls tear that down explicitly.
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass  # older/newer vLLM versions relocate these; best-effort only
        try:
            # The actual model weights/executor object living deep inside
            # vLLM's internal engine — dropping this reference is what lets
            # its CUDA memory actually become reclaimable.
            del llm.llm_engine.model_executor
        except Exception:
            pass
        try:
            delattr(vlm, "llm")
        except Exception:
            vlm.llm = None

    # Force Python's garbage collector to run NOW (rather than whenever it
    # would normally get around to it) — makes sure the just-dropped
    # references are actually cleaned up before we move on.
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            # PyTorch keeps its own internal cache of freed CUDA memory
            # blocks (to speed up future allocations) rather than immediately
            # returning them to the OS/driver — empty_cache() forces that
            # memory to actually be released so another process/library
            # (like CLIPScorer loading afterward) can use it.
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # In case vLLM's own distributed cleanup above didn't fully tear
            # down PyTorch's underlying distributed process group, do it
            # directly here as a final safety net.
            dist.destroy_process_group()
    except Exception:
        pass
