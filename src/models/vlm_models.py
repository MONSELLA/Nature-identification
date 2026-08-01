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

# Chat-template variable names different model families use to gate their
# "thinking"/reasoning mode. Checked against the template TEXT so only names a
# given template actually reads are ever sent (see
# VLLMBackedVLM._chat_kwargs). Qwen uses `enable_thinking`; the rest are the
# other spellings seen in the wild. Add to this list rather than special-casing
# a family — a name that isn't in a template is simply never sent.
THINKING_SWITCH_NAMES = (
    "enable_thinking",
    "add_thinking",
    "thinking",
    "enable_reasoning",
    "reasoning",
)


def find_thinking_switches(template: str) -> List[str]:
    """Which THINKING_SWITCH_NAMES a chat template actually references.

    Matched on WORD BOUNDARIES, not as plain substrings: "thinking" occurs
    inside "enable_thinking", so a naive `in` test reports both for Qwen and
    sends a redundant second kwarg. Only whole identifiers count."""
    import re
    return [n for n in THINKING_SWITCH_NAMES
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", template or "")]


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

    def _is_recoverable_bad_image(self, exc: Exception) -> bool:
        """Hook for subclasses: does `exc` mean "one IMAGE in this batch could
        not be turned into vision tokens" (a corrupt/degenerate file), as
        opposed to a genuine engine failure? Base default is False, same
        safe no-op rationale as `_is_recoverable_overflow` above."""
        return False

    def generate_batch_safe(
        self,
        prompts: List[str],
        images: Optional[List[ImageInput]] = None,
        label: str = "batch",
        item_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Union[str, Dict[str, Any], None]]:
        """Like `generate_batch`, but tolerant of a single BAD ITEM in the
        batch WITHOUT sacrificing batching for the rest of it. A batched call
        (e.g. vLLM's single `.chat()` covering every conversation) fails
        ENTIRELY if even one item is bad, with no indication of which one —
        so on a recoverable failure we BISECT the batch in half and recurse,
        each half still generated as one batched call. This isolates the
        offending item(s) down to single-item granularity (returned as None,
        aligned with input order) while every other sample stays batched
        together, instead of degrading the whole batch to one-at-a-time
        generation.

        TWO failure kinds are recoverable, both isolated the same way:
          * `_is_recoverable_overflow` — a prompt longer than max_model_len.
          * `_is_recoverable_bad_image` — an image vLLM's multimodal
            processor turned into zero vision tokens (corrupt/degenerate
            file). Raised during prompt RENDERING, before the GPU is
            involved, so the engine survives and the rest of the batch is
            still fine.
        Anything else (notably a CUDA OOM, which kills EngineCore outright
        and leaves nothing to recover into) is re-raised untouched.

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
            overflow = self._is_recoverable_overflow(e)
            bad_image = self._is_recoverable_bad_image(e)
            if not (overflow or bad_image):
                raise
            reason = "prompt too long for max_model_len" if overflow else "unusable image (no vision tokens produced)"
            if n == 1:
                name = item_labels[0] if item_labels else "item"
                print(f"⚠️ Skipping {name} ({label}): {reason} ({e}).")
                return [None]
            mid = n // 2
            print(f"⚠️ {label}: {reason} — bisecting "
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

    def __init__(self, model_name: str, max_image_side: Optional[int] = 1024,
                 enable_thinking: bool = False, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        # `enable_thinking` — like max_image_side, an EXPLICIT named parameter
        # so it is never forwarded to vLLM's LLM(...) constructor. Controls the
        # model's own chat-template "thinking"/reasoning mode via
        # chat_template_kwargs on every .chat() call.
        #
        # DEFAULT IS FALSE, and that is a deliberate fix, not a preference.
        # Qwen3.5's chat template turns thinking ON by default and renders a
        # prompt that ENDS with an already-open "<think>\n" block:
        #     '<|im_start|>assistant\n<think>\n'
        # That interacts badly with this pipeline in two separate ways:
        #   1. STRUCTURED calls (extraction/labeling) run under guided
        #      decoding, which constrains generation to the JSON schema from
        #      the very first generated token — so the model can emit neither
        #      its reasoning nor the closing </think>. It is forced straight
        #      into JSON from inside an unterminated think block, a state it
        #      never saw in training. Measured effect: Qwen3.5-9B hit a ~10%
        #      extraction parse-failure rate where Ministral-3-8B (no thinking
        #      mode) stayed under 1% on the identical prompt, schema, dataset
        #      and token budget.
        #   2. FREE-FORM calls (the baseline caption) have no grammar to
        #      constrain them, so the model DOES emit chain-of-thought — and
        #      _parse_response returns free-form text verbatim, with no
        #      <think> stripping. The "caption" then contains reasoning text,
        #      or is consumed entirely by it before the real description
        #      starts, silently degrading everything downstream (extraction
        #      context, ClipMatch summary, CLIPScore).
        # The pipeline's schemas already carry their OWN explicit reasoning
        # fields (ObjectExtractionResponse.reasoning, TaxonomyResponse's
        # two-step nature_reasoning/sub_axes_reasoning), so template-level
        # thinking is redundant here even where it works.
        self.enable_thinking = enable_thinking
        # Only send chat_template_kwargs to templates that actually declare
        # the switch — passing an unknown kwarg to a template that doesn't
        # use it is at best ignored and at worst an error, and models without
        # a thinking mode (Mistral, LLaVA) must stay byte-identical to before.
        self._supports_thinking_toggle = False
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
        # Does THIS model's chat template actually understand the thinking
        # switch? Read the template once here rather than per call. Wrapped
        # defensively: get_tokenizer() is not part of vLLM's guaranteed public
        # surface, and a missing/odd tokenizer must degrade to "no toggle"
        # (i.e. exactly the previous behavior) rather than break startup.
        try:
            template = getattr(self.llm.get_tokenizer(), "chat_template", None) or ""
        except Exception:
            template = ""
        self._thinking_switches = find_thinking_switches(template)
        self._supports_thinking_toggle = bool(self._thinking_switches)
        emits_think = "<think>" in template
        if self._thinking_switches:
            print(f"🧠 {model_name}: chat template has thinking switch(es) "
                  f"{self._thinking_switches} — set to {self.enable_thinking}")
        elif emits_think:
            # The dangerous case for a cross-family comparison: the template
            # emits a think block but exposes NO switch to turn it off, so
            # this model silently gets extra inference-time reasoning the
            # others don't. Loud, because it invalidates the comparison rather
            # than just degrading one number.
            print(f"⚠️ {model_name}: chat template emits '<think>' but exposes NO known "
                  f"switch to disable it (looked for {THINKING_SWITCH_NAMES}). This model "
                  f"will reason where non-thinking models cannot — NOT a fair cross-family "
                  f"comparison. Inspect its template before trusting the results.")

    def _chat_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs for `self.llm.chat(...)`. Sends the thinking toggle
        ONLY to templates that declare it, so every other model's rendered
        prompt is byte-identical to before this existed (and stays
        prefix-cache compatible with artifacts produced then).

        Every switch name the template actually mentions is set, because
        families don't agree on one: Qwen uses `enable_thinking`, others use
        `thinking`/`add_thinking`/`enable_reasoning`. Passing a name the
        template never reads would be a silently ignored no-op at best, so
        only names found in the template text are sent."""
        if not self._thinking_switches:
            return {}
        return {"chat_template_kwargs": {name: self.enable_thinking
                                          for name in self._thinking_switches}}

    def _is_recoverable_overflow(self, exc: Exception) -> bool:
        """vLLM's `.chat()` raises a ValueError with this exact substring
        when a conversation's prompt (text + image tokens) exceeds
        max_model_len — the specific, bisectable overflow case
        `generate_batch_safe` knows how to recover from."""
        return isinstance(exc, ValueError) and "longer than the maximum model length" in str(exc)

    def _is_recoverable_bad_image(self, exc: Exception) -> bool:
        """vLLM raises this RuntimeError when its multimodal processor turned
        an image into ZERO vision tokens, so the chat template's `<image>`
        placeholder was never substituted:

            "Expected there to be 1 prompt placeholders corresponding to
             1 image items, but instead found 0 prompt placeholders!"

        In practice that means ONE unusable image in the batch — corrupt or
        truncated pixel data, or a degenerate size that yields no patches.

        CRITICAL detail that makes this recoverable at all: it is raised
        during vLLM's PROMPT RENDERING (`_render_and_add_requests` ->
        `_process_multimodal`), i.e. before anything is submitted to the GPU,
        so the engine is still perfectly healthy — unlike a CUDA OOM, which
        kills EngineCore and leaves nothing to recover into. Bisecting the
        batch therefore isolates the offending image and lets every other
        image in it proceed normally.

        Worth the specificity: this exact error killed two multi-hour ImageNet
        runs outright (~batch 149 of 391, then again on the same image after a
        --resume), because a RuntimeError isn't the ValueError the overflow
        path catches, so nothing recovered from it."""
        return isinstance(exc, RuntimeError) and "prompt placeholders" in str(exc)

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
            # The raw-bytes fast path is only safe for images ALREADY in a
            # mode the vision encoder handles. ImageNet's ILSVRC2012 val set
            # famously contains a handful of CMYK and grayscale JPEGs; passed
            # through untouched, one of those made vLLM's Pixtral multimodal
            # processor emit ZERO image patches, so the chat template's single
            # <image> placeholder was never filled and the whole batch died
            # with "Expected there to be 1 prompt placeholders corresponding
            # to 1 image items, but instead found 0". That is a RuntimeError,
            # not the ValueError generate_batch_safe's bisection recovers
            # from, so it killed the engine outright ~140 batches into a run.
            # The oversized branch below already had this covered (it calls
            # .convert("RGB") before resizing, for the same channel-count
            # reason) — only the fast path was exposed, which is precisely the
            # path essentially every ImageNet image takes.
            # `pil_image.mode` is read from the header alone (no full decode),
            # so the RGB check stays as cheap as the original fast path; the
            # decode/re-encode cost is paid ONLY for the rare non-RGB image.
            if pil_image.mode == "RGB":
                if isinstance(image, str):
                    with open(image, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    # Preserve the original file extension (jpg/png/etc.) in the
                    # data URL's MIME type, defaulting to png if there's none.
                    suffix = Path(image).suffix.lstrip(".").lower() or "png"
                    return f"data:image/{suffix};base64,{b64}"
            else:
                # Non-RGB (CMYK / grayscale / palette / RGBA) and small enough
                # to skip resizing — normalize the colour mode only.
                buffer = BytesIO()
                pil_image.convert("RGB").save(buffer, format="JPEG", quality=90)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
            # Reached only when the image is ALREADY RGB and was passed as a
            # PIL Image rather than a path (the str case returned above) —
            # encode as-is (in-memory "file" via BytesIO, rather than needing
            # to save to disk first).
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
        outputs = self.llm.chat([messages], sampling_params=sampling_params, use_tqdm=False,
                                 **self._chat_kwargs())
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
        outputs = self.llm.chat(conversations, sampling_params=sampling_params,
                                 use_tqdm=len(prompts) > 1, **self._chat_kwargs())
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
