"""
lib/prompts.py

Single home for ALL prompts and structured-output schemas used by the BIG-5
VLM pipeline. Keeping them here (rather than inline in each script) guarantees
that the taxonomy-labeling prompt used by the pipeline's VLM-fallback path is
byte-for-byte identical to the one used by evaluate_taxonomy_labeling.py's
calibration eval — the two cannot drift, because they import the same objects.

Contents:
  - CAPTION_PROMPT               : baseline open-ended caption (neutral, no
                                   nature-priming) — verbatim from CLAUDE.md.
  - EXTRACTION_PROMPT            : structured object-extraction instruction.
  - ObjectExtractionResponse     : pydantic schema for the extraction call.
  - TaxonomyResponse             : pydantic schema for per-object labeling.
  - _AXIS_INSTRUCTIONS           : per-axis rule strings.
  - build_classification_prompt(): the per-object taxonomy prompt (the exact
                                   VLM-fallback prompt).
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Stage 1 — Captioning (baseline, two-pass; neutral, NO nature-priming)
# =============================================================================
# Verbatim from CLAUDE.md's "hard conventions". Do NOT add "pay attention to
# nature" here unless running the nature-priming ablation explicitly.
CAPTION_PROMPT = (
    "Describe this image in detail, including the main subject, the "
    "background, the setting, and any secondary elements present."
)


# =============================================================================
# Stage 2 — Object extraction (structured)
# =============================================================================
# The image is re-sent on this call (recap §5a "second look"): the model gets
# another chance to surface objects omitted from the free-form caption. The
# instruction explicitly asks for part-objects / sub-elements (e.g. a flower
# printed on a dress), because a nature representation may be only a PART of a
# larger object.
EXTRACTION_PROMPT = (
    "Below is a description of the image:\n\n"
    "\"{caption}\"\n\n"
    "Using BOTH the image and the description, list every distinct physical "
    "object, element, or entity that appears in the image. Follow these rules:\n"
    "  - Return each object as a short noun phrase (e.g. \"wooden bench\", "
    "\"golden retriever\", \"snow-capped mountain\").\n"
    "  - Include secondary and background elements, not just the main subject.\n"
    "  - Include an element even when it is only PART of a larger object or is "
    "depicted on its surface (e.g. a flower printed on a dress -> list "
    "\"flower\"; a bird on a logo -> list \"bird\").\n"
    "  - Do not repeat the same object twice. Do not invent objects that are "
    "not supported by the image."
)


class ObjectExtractionResponse(BaseModel):
    """Structured schema for the extraction call: a flat list of object phrases."""

    objects: List[str] = Field(
        description=(
            "The distinct physical objects, elements, or entities present in the "
            "image, each as a short noun phrase. Include part-objects and "
            "background elements."
        )
    )


# =============================================================================
# Stage 3 — Per-object taxonomy labeling (the VLM-fallback prompt)
# =============================================================================
# IMPORTANT: this schema and build_classification_prompt() are the SHARED,
# canonical taxonomy-labeling prompt. evaluate_taxonomy_labeling.py imports
# them from here so the calibration eval and the pipeline's fallback are
# identical. Any change here changes BOTH — that is intentional.

class TaxonomyResponse(BaseModel):
    """
    Pydantic schema driving `outlines` / guided-decoding structured output.
    By defining `reasoning` first, we force the model into a chain-of-thought
    generation process before it commits to the final taxonomic labels.
    """

    reasoning: str = Field(
        description="One concise sentence justifying your classification based on the visual evidence."
    )
    nature: Literal["yes", "no"]
    biotic: Literal["biotic", "abiotic", "n/a"]
    material: Literal["material", "immaterial", "n/a"]


_AXIS_INSTRUCTIONS = {
    "nature": '"nature": either "yes" or "no" — whether this instance counts as nature under the definition above',
    "biotic": '"biotic": either "biotic", "abiotic", or "n/a" — only answer "biotic"/"abiotic" if "nature" is "yes"; use "n/a" if "nature" is "no"',
    "material": '"material": either "material", "immaterial", or "n/a" — only answer "material"/"immaterial" if "nature" is "yes"; use "n/a" if "nature" is "no"',
}


def build_classification_prompt(class_name, axes):
    """
    Constructs the contextualized per-object taxonomy prompt. The model is
    forced to evaluate the taxonomic labels based on the specific visual
    instance depicted in the image (the image is always attached to this call,
    including on the pipeline's unmapped/VLM-fallback path).
    """
    unknown_axes = set(axes) - set(_AXIS_INSTRUCTIONS)
    if unknown_axes:
        raise ValueError(f"Unknown axis/axes requested: {unknown_axes}")

    field_lines = "\n".join(f"  - {_AXIS_INSTRUCTIONS[axis]}" for axis in axes)

    return f"""You are analyzing a specific object identified in the provided image.
The object is classified as: {class_name}

Based on the visual evidence in the image and the definitions provided, classify this specific instance of the object.

Provide your reasoning first, followed by the specific labels according to these rules:
{field_lines}
"""
