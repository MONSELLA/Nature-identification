"""
src/models/vlm_models.py

Unified interface for running VLMs across different serving backends (vLLM /
HuggingFace transformers) and model families, for the BIG-5 VLM pipeline.

NO PROMPTS LIVE IN THIS FILE.

WHAT IS A "VLM" AND WHY DOES THIS FILE NEED TO BE SO ABSTRACTED?
"VLM" = Vision-Language Model: a neural network that can take BOTH an image
and text as input and produce text as output (e.g. "describe this image",
"is there a dog in this picture?"). This project needs to run and compare
SEVERAL different VLMs (Qwen, Mistral, LLaVA, the BLIP family...), and each
one is served/loaded completely differently under the hood:
  - Some are run through vLLM, a high-performance serving engine (talks to
    the model via a chat-style API, handles batching/GPU memory internally).
  - Some are run directly through HuggingFace's `transformers` library (we
    load the model weights ourselves and call `.generate()` on them).

Rather than making every OTHER file in this project (dataset loaders, the
pipeline, the scripts) know about these differences, this file defines ONE
common interface (`BaseVLM.generate` / `generate_batch`) that every model
family implements, so calling code just does
`vlm.generate_batch(prompts=..., images=..., ...)` and gets consistent
results back no matter which underlying model is actually running.

TWO KEY CONCEPTS USED THROUGHOUT THIS FILE:
  - "structured output" / "guided decoding": normally a language model can
    generate ANY text it wants. When we pass `output_mode="structured"` and a
    `schema` (a pydantic class — see src/models/prompts.py), the serving
    backend constrains generation so the model is FORCED to produce valid
    JSON matching that schema — it becomes literally impossible for the model
    to output something that doesn't parse. vLLM does this via
    `StructuredOutputsParams`; the HuggingFace backend does it via the `outlines`
    library's `JSONPrefixAllowedTokens`.
  - "batch": processing many (prompt, image) pairs in a SINGLE model call
    instead of looping one at a time. This matters a lot for speed — GPUs are
    much more efficient when given a big batch of work at once.
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
        # on the GPU. Subclasses that CAN do real batched inference (vLLM, and
        # the BLIP family below) override this method with a faster version;
        # this fallback only kicks in for a family that hasn't bothered to.
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

class VLLMBackedVLM(BaseVLM):
    """VLM family served through vLLM (used for qwen/mistral/llava — see
    MODEL_REGISTRY at the bottom of this file). vLLM exposes an OpenAI-style
    chat API (`self.llm.chat(...)`) that internally handles GPU scheduling,
    KV-cache management, and batching for us."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
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

    @staticmethod
    def _encode_image(image: ImageInput) -> str:
        """Convert an image (file path or PIL Image) into a "data URL" string
        (base64-encoded bytes embedded directly in the string) — the format
        vLLM's chat API expects for the `image_url` message field."""
        if isinstance(image, str):
            if image.startswith("data:image"):
                # Already a data URL — nothing to do.
                return image
            with open(image, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            # Preserve the original file extension (jpg/png/etc.) in the data
            # URL's MIME type, defaulting to png if there's no extension.
            suffix = Path(image).suffix.lstrip(".").lower() or "png"
            return f"data:image/{suffix};base64,{b64}"

        # Not a string — assume it's an already-loaded PIL Image object.
        # Re-encode it as PNG bytes in memory (BytesIO acts like a temporary
        # in-memory file) rather than needing to save it to disk first.
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _build_messages(
        self,
        prompt: str,
        image: ImageInput,
        system_prompt: Optional[str],
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


class HuggingFaceBackedVLM(BaseVLM):
    """Common base for VLM families served directly through HuggingFace's
    `transformers` library (the BLIP family below), rather than vLLM. Unlike
    vLLM (which handles model loading internally), here WE are responsible
    for loading the model weights ourselves — each concrete subclass
    implements `_load_model()` to do so with its own specific model classes."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        import torch

        self.device = device
        # `dtype` may arrive as a string from the CLI (e.g. "auto", "bfloat16")
        # rather than an actual torch.dtype. `from_pretrained(torch_dtype=...)`
        # accepts the string "auto" directly, but tensor `.to(device, dtype)`
        # calls elsewhere do not — they need a real torch.dtype. Resolve here,
        # once, so every downstream use gets a concrete dtype: "auto"/None
        # fall back to the bfloat16 default, other strings (e.g. "float16")
        # are looked up on the `torch` module.
        if isinstance(dtype, str):
            dtype = None if dtype == "auto" else getattr(torch, dtype, None)
        # Default to bfloat16 precision (a reduced-precision float format
        # commonly used for faster/lighter-weight inference) if the caller
        # didn't specify one.
        self.dtype = dtype or torch.bfloat16
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Each concrete subclass must implement this to actually load its
        specific model/processor classes into self.model / self.processor
        (or self.tokenizer / self.image_processor for Blip3VLM)."""
        raise NotImplementedError

    def _get_prefix_allowed_tokens_fn(
        self, output_mode: str, schema: Optional[Any], model_inputs: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Build the guided-decoding function for HuggingFace's `.generate()`
        (the HF equivalent of vLLM's StructuredOutputsParams above) — this is
        what forces the model's token-by-token output to stay valid JSON
        matching `schema`, via the third-party `outlines` library. Returns
        None (no constraint) in free_form mode.

        All four BLIP-family classes here (BLIP, BLIP-2, InstructBLIP, BLIP-3)
        feed the full text prompt into `input_ids` before generation, so
        `.generate()` hands `prefix_allowed_tokens_fn` the PROMPT tokens too,
        not just the newly generated ones. Outlines' FSM would then try to
        parse the prompt itself as JSON and immediately reject it. We slice
        off the prompt length (read from `model_inputs["input_ids"]`, which
        is already padded to a uniform width across the batch) so the FSM
        only ever sees the generated continuation."""
        if output_mode == "structured" and schema is not None:
            from outlines.integrations.transformers import JSONPrefixAllowedTokens
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None and hasattr(self, "processor"):
                # Most of these models don't have a standalone `.tokenizer`
                # attribute — their tokenizer lives inside their `.processor`
                # object instead (or the processor even acts as the tokenizer
                # directly), so fall back to that.
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
            base_fn = JSONPrefixAllowedTokens(schema, tokenizer)

            prompt_length = 0
            if model_inputs is not None and "input_ids" in model_inputs:
                prompt_length = model_inputs["input_ids"].shape[1]

            def vision_prefix_allowed_tokens_fn(batch_id: int, current_input_ids: Any) -> List[int]:
                generated_ids = current_input_ids[prompt_length:]
                return base_fn(batch_id, generated_ids)

            return vision_prefix_allowed_tokens_fn
        return None

    def _parse_response(self, text: str, output_mode: str) -> Union[str, Dict[str, Any], None]:
        if output_mode == "structured":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        return text


# =============================================================================
# Concrete model-family classes
# =============================================================================
# NOTE ON THE FOLLOWING CLASSES: BLIP, BLIP-2, InstructBLIP, and BLIP-3 are all
# different VLM architectures with slightly different loading/input
# requirements, but they share the same overall inference PATTERN: build a
# text prompt, preprocess the image, call `.generate()` with the guided-
# decoding hook, then decode the output token ids back into text. Each class
# below implements that pattern with its own model-specific details.

class BlipFamilyVLM(HuggingFaceBackedVLM):
    """Shared generate()/generate_batch() implementation for BLIP and BLIP-2
    (both use the same `self.processor(images=..., text=...)` calling
    convention). InstructBLIP and BLIP-3 below have their own slightly
    different versions since they have extra requirements (image is
    mandatory, or a special stop-token setup)."""

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
        import torch

        # BLIP has no separate "system" concept — we just prepend the system
        # prompt text directly onto the user prompt, separated by a blank line.
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if image is not None:
            from PIL import Image
            # `image` might already be a loaded PIL Image, or a file path
            # string we still need to open ourselves.
            pil_image = image if not isinstance(image, str) else Image.open(image).convert("RGB")
            inputs = self.processor(images=pil_image, text=full_prompt, return_tensors="pt").to(
                self.device, self.dtype
            )
        else:
            # WARNING: this branch is unreachable in practice. BlipForConditionalGeneration.generate()
            # (and Blip2ForConditionalGeneration.generate()) both require pixel_values as a
            # non-optional positional argument -- confirmed against transformers source
            # (BlipForConditionalGeneration even sets main_input_name = "pixel_values").
            # Calling model.generate(**inputs) below with no pixel_values in `inputs` will raise
            # TypeError: missing required positional argument. Do not rely on this path for
            # text-only evaluation; see evaluate_taxonomy_labeling.py's IMAGE_REQUIRED_FAMILIES guard.
            inputs = self.processor(text=full_prompt, return_tensors="pt").to(self.device)

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema, inputs)

        with torch.no_grad():
            # temperature > 0 means "sample randomly" (do_sample=True);
            # temperature == 0 means "always pick the single most likely next
            # token" (greedy decoding, do_sample=False) — passing
            # temperature=0.0 to do_sample=True would actually error in some
            # HF versions, hence only passing it through when sampling.
            do_sample = temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **kwargs,
            )
        # batch_decode turns the model's output token IDs back into a string;
        # [0] because we only generated for a single (batch-of-one) input here.
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
        """True batched inference for BLIP models via DataLoaders."""
        import torch

        # HuggingFace padding requires a pad_token. Fallback to eos_token if missing.
        # When batching prompts of DIFFERENT lengths together, shorter ones
        # need to be "padded" (filled with a placeholder token) to match the
        # longest one in the batch — some tokenizers don't define a dedicated
        # padding token by default, so we reuse the end-of-sequence token instead.
        if getattr(self.processor.tokenizer, "pad_token", None) is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        full_prompts = [f"{system_prompt}\n\n{p}" if system_prompt else p for p in prompts]

        if images is not None and any(img is not None for img in images):
            from PIL import Image
            pil_images = [img if not isinstance(img, str) else Image.open(img).convert("RGB") for img in images]
            inputs = self.processor(images=pil_images, text=full_prompts, return_tensors="pt", padding=True).to(
                self.device, self.dtype
            )
        else:
            # WARNING: unreachable in practice -- see matching comment in generate() above.
            # model.generate(**inputs) below requires pixel_values; this branch omits it.
            inputs = self.processor(text=full_prompts, return_tensors="pt", padding=True).to(self.device)

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema, inputs)

        with torch.no_grad():
            do_sample = temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **kwargs,
            )

        # Now decode EVERY sequence in the batch back into its own string.
        texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return [self._parse_response(t.strip(), output_mode) for t in texts]


class BlipVLM(BlipFamilyVLM):
    """Original BLIP model family."""
    def _load_model(self) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = (
            BlipForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()  # .eval() disables training-only behavior like dropout
        )


class Blip2VLM(BlipFamilyVLM):
    """BLIP-2 model family — same generate()/generate_batch() logic as BLIP
    (inherited from BlipFamilyVLM), just different model/processor classes."""
    def _load_model(self) -> None:
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )


class InstructBlipVLM(HuggingFaceBackedVLM):
    """InstructBLIP — unlike BlipFamilyVLM, this one REQUIRES an image on
    every call (no text-only fallback attempt at all), so it implements its
    own generate()/generate_batch() rather than sharing BlipFamilyVLM's."""

    def _load_model(self) -> None:
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        # NOTE: deliberately InstructBlipProcessor, not AutoProcessor —
        # AutoProcessor.from_pretrained() resolves "instructblip-vicuna-7b"
        # to InstructBlipVideoProcessor on some transformers versions (an
        # auto-class resolution quirk from InstructBlipVideo being added
        # later). That video processor treats a batch of still images as
        # frames of one video and np.stacks them, which requires identical
        # shapes and crashes on a batch of differently-sized images.
        self.processor = InstructBlipProcessor.from_pretrained(self.model_name)
        self.model = (
            InstructBlipForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

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
        import torch
        from PIL import Image

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if image is None:
            # Unlike BlipFamilyVLM's (never-actually-taken) text-only branch,
            # InstructBLIP explicitly refuses to run without an image at all.
            raise ValueError("InstructBlipVLM requires an image.")

        pil_image = image if not isinstance(image, str) else Image.open(image).convert("RGB")
        inputs = self.processor(images=pil_image, text=full_prompt, return_tensors="pt").to(
            self.device, self.dtype
        )

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema, inputs)

        with torch.no_grad():
            do_sample = temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **kwargs,
            )
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
        """True batched inference for InstructBLIP."""
        import torch
        from PIL import Image

        if getattr(self.processor.tokenizer, "pad_token", None) is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        full_prompts = [f"{system_prompt}\n\n{p}" if system_prompt else p for p in prompts]

        if images is None or not any(img is not None for img in images):
            raise ValueError("InstructBlipVLM requires images for batching.")

        pil_images = [img if not isinstance(img, str) else Image.open(img).convert("RGB") for img in images]
        inputs = self.processor(images=pil_images, text=full_prompts, return_tensors="pt", padding=True).to(
            self.device, self.dtype
        )

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema, inputs)

        with torch.no_grad():
            do_sample = temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **kwargs,
            )

        texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return [self._parse_response(t.strip(), output_mode) for t in texts]


class Blip3VLM(HuggingFaceBackedVLM):
    """BLIP-3 — architecturally different enough from the other BLIP variants
    that it needs its own tokenizer/image-processor setup (rather than one
    combined `processor`), a custom chat-style prompt format
    (`<|system|>...<|user|>...<|assistant|>`), and a custom stopping rule."""

    # BLIP-3's own special "end of turn" token id — generation should stop
    # once the model produces this token, even if max_new_tokens hasn't been
    # reached yet.
    _EOS_TOKEN_ID = 32007

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False, legacy=False
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        # BLIP-3 ships custom special tokens (e.g. <image>, <|system|>) that
        # need to be registered on the tokenizer via the model's own helper
        # method, rather than being standard/built-in tokens.
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

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
        import torch
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList

        if image is None:
            raise ValueError("Blip3VLM requires an image.")

        # A small custom rule telling `.generate()` when to stop producing
        # more tokens: as soon as the most recently generated tokens match
        # this specific "end of turn" sequence, generation halts (even before
        # max_new_tokens is reached).
        class _EosListStoppingCriteria(StoppingCriteria):
            def __init__(self, eos_sequence):
                self.eos_sequence = eos_sequence
            def __call__(self, input_ids, scores, **kw) -> bool:
                last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
                return self.eos_sequence in last_ids

        # BLIP-3 expects its OWN specific chat-style markup rather than a
        # plain string — special tokens delimiting system/user/assistant
        # turns, and an explicit `<image>` placeholder marking where the
        # image should be "inserted" into the token sequence.
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"

        pil_image = image if not isinstance(image, str) else Image.open(image).convert("RGB")
        # Image and text are processed SEPARATELY (unlike the BLIP family's
        # single combined `self.processor(...)` call) and then merged into
        # one `inputs` dict.
        inputs = self.image_processor([pil_image], return_tensors="pt", image_aspect_ratio="anyres")
        language_inputs = self.tokenizer([full_prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema, inputs)

        with torch.no_grad():
            do_sample = temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                stopping_criteria=StoppingCriteriaList([_EosListStoppingCriteria([self._EOS_TOKEN_ID])]),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **kwargs,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
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
        """True batched inference for BLIP-3."""
        import torch
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList

        if images is None or not any(img is not None for img in images):
            raise ValueError("Blip3VLM requires images for batching.")

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        class _EosListStoppingCriteria(StoppingCriteria):
            def __init__(self, eos_sequence):
                self.eos_sequence = eos_sequence
            def __call__(self, input_ids, scores, **kw) -> bool:
                last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
                return self.eos_sequence in last_ids

        full_prompts = []
        for p in prompts:
            if system_prompt:
                full_prompts.append(f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n<image>\n{p}<|end|>\n<|assistant|>\n")
            else:
                full_prompts.append(f"<|user|>\n<image>\n{p}<|end|>\n<|assistant|>\n")

        pil_images = [img if not isinstance(img, str) else Image.open(img).convert("RGB") for img in images]

        inputs = self.image_processor(pil_images, return_tensors="pt", image_aspect_ratio="anyres")
        language_inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True)
        inputs.update(language_inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema, inputs)

        with torch.no_grad():
            do_sample = temperature > 0.0
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                stopping_criteria=StoppingCriteriaList([_EosListStoppingCriteria([self._EOS_TOKEN_ID])]),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **kwargs,
            )

        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [self._parse_response(t.strip(), output_mode) for t in texts]


# =============================================================================
# Factory
# =============================================================================
# `MODEL_REGISTRY` maps a short, human-typed "family" name (what you pass on
# the command line via --model_family) to the actual Python class that knows
# how to run that family. `create_vlm()` below is the ONE place any other
# file in this project should use to actually construct a VLM instance —
# nobody else needs to know which concrete class backs a given family name.

MODEL_REGISTRY: Dict[str, type] = {
    "qwen": VLLMBackedVLM,
    "mistral": VLLMBackedVLM,
    "llava": VLLMBackedVLM,
    "internvl": VLLMBackedVLM,
    "gemma": VLLMBackedVLM,
    "blip": BlipVLM,
    "blip2": Blip2VLM,
    "instructblip": InstructBlipVLM,
    "blip3": Blip3VLM,
}

# Which family names are served via vLLM (as opposed to HuggingFace directly)
# — used by the calling scripts to decide which set of constructor keyword
# arguments (vLLM-specific vs HuggingFace-specific) to pass to create_vlm().
VLLM_FAMILIES = ("qwen", "mistral", "llava", "internvl", "gemma")

def create_vlm(family: str, model_name: str, **kwargs: Any) -> BaseVLM:
    """Construct the right VLM subclass for the given family name, e.g.
    create_vlm("qwen", "Qwen/Qwen3.5-0.8B", dtype="auto", ...)."""
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family '{family}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[family](model_name, **kwargs)


# =============================================================================
# Memory release — needed for a single-process end-to-end run (VLM inference
# followed by CLIP scoring in the SAME process). Neither vLLM nor plain
# torch/transformers release CUDA memory back to the driver on Python garbage
# collection alone: vLLM keeps a distributed process group + KV-cache
# allocator alive, and HF model tensors stay pinned until their CUDA blocks are
# explicitly freed. Skipping this step is why the pipeline used to require two
# separate process invocations (--stage infer, then --stage score).
# =============================================================================
def unload_vlm(vlm: BaseVLM) -> None:
    """Tear down a VLM's GPU-resident state so a CLIP/metric model can be
    loaded afterward in the SAME process without contending for VRAM. Safe to
    call on any BaseVLM subclass; each backend's own GPU-holding attributes
    (`.llm` for vLLM, `.model`/`.processor`/`.tokenizer`/`.image_processor` for
    the HuggingFace families) are cleared before the general GC/cache pass.

    WHY IS THIS SO INVOLVED? Simply doing `del vlm` in Python only removes
    OUR reference to the object — if the underlying library (vLLM, or torch
    itself) is still holding onto GPU memory behind the scenes (e.g. a
    distributed process group, or cached CUDA memory blocks), that memory
    stays allocated regardless. This function explicitly asks each layer to
    let go of its own resources, in order, before finally asking Python's
    garbage collector and PyTorch's CUDA allocator to actually reclaim
    everything.
    """
    import gc

    llm = getattr(vlm, "llm", None)  # VLLMBackedVLM
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

    # Whichever of these attributes this particular VLM subclass actually
    # set (vLLM-backed classes only have `.llm`; HuggingFace-backed classes
    # have some subset of the others), drop the reference to release the
    # underlying GPU tensors.
    for attr in ("llm", "model", "processor", "tokenizer", "image_processor"):
        if hasattr(vlm, attr):
            try:
                delattr(vlm, attr)
            except Exception:
                setattr(vlm, attr, None)

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
