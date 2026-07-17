"""
lib/vlm.py

Unified interface for running VLMs across different serving backends (vLLM /
HuggingFace transformers) and model families, for the BIG-5 VLM pipeline.

NO PROMPTS LIVE IN THIS FILE.
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ImageInput = Union[str, "PIL.Image.Image", None]  # noqa: F821


# =============================================================================
# Abstract base
# =============================================================================

class BaseVLM(ABC):
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


# =============================================================================
# Backend base classes
# =============================================================================

class VLLMBackedVLM(BaseVLM):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("VLLMBackedVLM requires the `vllm` package.") from e
        self.llm = LLM(model=model_name, **kwargs)

    @staticmethod
    def _encode_image(image: ImageInput) -> str:
        if isinstance(image, str):
            if image.startswith("data:image"):
                return image
            with open(image, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            suffix = Path(image).suffix.lstrip(".").lower() or "png"
            return f"data:image/{suffix};base64,{b64}"

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
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(image)}},
            ]
        else:
            content = prompt

        messages.append({"role": "user", "content": content})
        return messages
        
    def _parse_response(self, text: str, output_mode: str) -> Union[str, Dict[str, Any], None]:
        if output_mode == "structured":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        return text
    
    def _make_sampling_params(self, temperature, max_new_tokens, output_mode, schema, **kwargs):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        gd = None
        if output_mode == "structured" and schema is not None:
            js = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            gd = GuidedDecodingParams(json=js)
        return SamplingParams(temperature=temperature, max_tokens=max_new_tokens,
                            guided_decoding=gd, **kwargs)

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
        
        from vllm import SamplingParams

        messages = self._build_messages(prompt, image, system_prompt)
        
        sampling_params = self._make_sampling_params(
            temperature=temperature, max_new_tokens=max_new_tokens, 
            output_mode=output_mode, schema=schema, **kwargs
        )
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
        self.dtype = dtype or torch.bfloat16
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError
        
    def _get_prefix_allowed_tokens_fn(self, output_mode: str, schema: Optional[Any]) -> Optional[Any]:
        if output_mode == "structured" and schema is not None:
            from outlines.integrations.transformers import JSONPrefixAllowedTokens
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None and hasattr(self, "processor"):
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
            return JSONPrefixAllowedTokens(schema, tokenizer)
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

class BlipFamilyVLM(HuggingFaceBackedVLM):
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

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if image is not None:
            from PIL import Image
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

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema)

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
        """True batched inference for BLIP models via DataLoaders."""
        import torch
        
        # HuggingFace padding requires a pad_token. Fallback to eos_token if missing.
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

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema)

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


class BlipVLM(BlipFamilyVLM):
    def _load_model(self) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = (
            BlipForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )


class Blip2VLM(BlipFamilyVLM):
    def _load_model(self) -> None:
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )


class InstructBlipVLM(HuggingFaceBackedVLM):
    def _load_model(self) -> None:
        from transformers import AutoProcessor, InstructBlipForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_name)
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
            raise ValueError("InstructBlipVLM requires an image.")
            
        pil_image = image if not isinstance(image, str) else Image.open(image).convert("RGB")
        inputs = self.processor(images=pil_image, text=full_prompt, return_tensors="pt").to(
            self.device, self.dtype
        )

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema)

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

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema)

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

        class _EosListStoppingCriteria(StoppingCriteria):
            def __init__(self, eos_sequence):
                self.eos_sequence = eos_sequence
            def __call__(self, input_ids, scores, **kw) -> bool:
                last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
                return self.eos_sequence in last_ids

        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"

        pil_image = image if not isinstance(image, str) else Image.open(image).convert("RGB")
        inputs = self.image_processor([pil_image], return_tensors="pt", image_aspect_ratio="anyres")
        language_inputs = self.tokenizer([full_prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema)

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

        prefix_allowed_tokens_fn = self._get_prefix_allowed_tokens_fn(output_mode, schema)

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

MODEL_REGISTRY: Dict[str, type] = {
    "qwen": VLLMBackedVLM,
    "mistral": VLLMBackedVLM,
    "llava": VLLMBackedVLM,
    "blip": BlipVLM,
    "blip2": Blip2VLM,
    "instructblip": InstructBlipVLM,
    "blip3": Blip3VLM,
}

VLLM_FAMILIES = ("qwen", "mistral", "llava")

def create_vlm(family: str, model_name: str, **kwargs: Any) -> BaseVLM:
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family '{family}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[family](model_name, **kwargs)