"""
src/models/prompts.py

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

WHAT IS A "PYDANTIC SCHEMA" AND WHY DO WE NEED ONE?
When we ask the VLM a free-form question ("describe this image"), it can reply
with whatever text it wants. But when we need a MACHINE-READABLE answer (e.g.
"is this object nature or not?"), we want the model's raw text output to come
back as valid, predictable JSON that our Python code can parse without any
guesswork. A pydantic `BaseModel` class describes exactly which fields the JSON
must contain and what values are legal for each one. We hand this schema to the
VLM backend (see lib/vlm.py's `output_mode="structured"` path), which uses it
to constrain generation so the model literally cannot produce a token sequence
that violates the schema (this is called "guided decoding" / "constrained
decoding"). The result: `vlm.generate_batch(..., schema=TaxonomyResponse)`
returns a Python dict we can safely read keys from, instead of a string we'd
have to regex-parse and hope for the best.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Stage 1 — Captioning (baseline, two-pass; neutral, NO nature-priming)
# =============================================================================
# Verbatim from CLAUDE.md's "hard conventions". Do NOT add "pay attention to
# nature" here unless running the nature-priming ablation explicitly.
#
# This is the very FIRST thing we ask the VLM about an image: a plain, open-
# ended description. No JSON schema, no taxonomy jargon — just "describe what
# you see". We deliberately ask for background/setting/secondary elements
# explicitly (not just "describe this image") because a bare prompt tends to
# only describe the single most obvious subject and skip everything else
# (called "salience bias"), and tends to be too short to mention secondary
# details ("brevity bias"). This neutral caption becomes the input to Stage 2.
CAPTION_PROMPT = (
    "Describe this image in 3-4 sentences, covering the main subject, the "
    "background, the setting, and any secondary elements present. Be "
    "specific but concise."
)

# =============================================================================
# Stage 2 — Object extraction (structured)
# =============================================================================
# The image is re-sent on this call (recap §5a "second look"): the model gets
# another chance to surface objects omitted from the free-form caption. The
# instruction explicitly asks for part-objects / sub-elements (e.g. a flower
# printed on a dress), because a nature representation may be only a PART of a
# larger object.
#
# This prompt takes the Stage-1 caption as input (via the `{caption}`
# placeholder, filled in with Python's `.format(caption=...)`) and asks the
# model to turn that description PLUS a fresh look at the image into a clean
# list of individual objects/elements. This list is what gets fed into Stage 3
# (one taxonomy-labeling call per object) and into the CLIP-based metrics
# (each object becomes its own "a photo of a {object}" text embedding).
EXTRACTION_PROMPT = (
    "Below is a description of the image:\n\n"
    "\"{caption}\"\n\n"
    "Using BOTH the image and the description, list every distinct physical "
    "object, element, or entity that appears in the image. Follow these rules:\n"
    "  - Return each object as a short noun phrase (e.g. \"wooden bench\", "
    "\"golden retriever\", \"mountain\").\n"
    "  - Include secondary and background elements, not just the main subject.\n"
    "  - Include an element even when it is only PART of a larger object or is "
    "depicted on its surface (e.g. a flower printed on a dress -> list "
    "\"flower\"; a bird on a logo -> list \"bird\").\n"
    "  - Do not repeat the same object twice. Do not invent objects that are "
    "not supported by the image."
)


class ObjectExtractionResponse(BaseModel):
    """Structured schema for the extraction call: a flat list of object phrases.

    When we pass this class as `schema=` to the VLM, the model's JSON output is
    forced to look like: {"objects": ["dog", "grass", "fence", ...]}. Our code
    then just reads `result["objects"]` to get a plain Python list of strings —
    no manual text-parsing needed.
    """

    # `List[str]` tells pydantic (and the guided-decoding machinery) that this
    # field must be a JSON array of strings. `Field(description=...)` is not
    # just documentation — some structured-output backends surface this text
    # to the model itself as part of the schema, so it doubles as an
    # instruction to the model about what belongs in this field.
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

    Field order matters here! Because the model generates tokens left-to-right,
    putting `reasoning` BEFORE `nature`/`biotic`/`material` in the class means
    the model has to "think out loud" and write its justification first, and
    only afterward commit to yes/no-style answers. This tends to produce more
    reliable classifications than asking for the labels with no explanation.
    """

    # A short free-text justification, generated FIRST (see docstring above).
    reasoning: str = Field(
        description="One concise sentence justifying your classification based on the visual evidence."
    )
    # `Literal[...]` restricts this field to EXACTLY these string values — the
    # model is not allowed to answer anything else (e.g. "sort of", "maybe").
    nature: Literal["yes", "no"]
    # Only meaningful when nature == "yes"; the prompt below explains that the
    # model must answer "n/a" whenever nature == "no" (see _AXIS_INSTRUCTIONS).
    biotic: Literal["biotic", "abiotic", "n/a"]
    material: Literal["material", "immaterial", "n/a"]


class MaterialResponse(BaseModel):
    """
    Material-only schema for the MAPPED-nature fast path (see
    src/vlm_pipeline.py's label_objects_batch). When an extracted object already
    resolves to a labeled nature node via WordNet, its nature/biotic axes are
    fixed by the mapping and only material/immaterial still needs the VLM — so we
    ask ONLY that axis, with a schema that omits nature/biotic entirely (rather
    than reusing TaxonomyResponse and forcing the model to also emit two answers
    we would throw away). `reasoning` stays first for the same think-first
    reason as TaxonomyResponse. No "n/a" option: the object is known to be
    nature, so material always applies.
    """

    reasoning: str = Field(
        description="One concise sentence justifying the material/immaterial classification based on the visual evidence."
    )
    material: Literal["material", "immaterial"]


# One line of plain-English instructions per taxonomy axis, injected into the
# prompt text below. Kept as a dict (rather than hardcoded into one long
# prompt string) so build_classification_prompt() can ask for a SUBSET of axes
# if a caller only cares about e.g. nature+biotic and not material.
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

    Args:
        class_name: the object's name as a plain string, e.g. "oak tree". This
            gets dropped straight into the prompt text so the model knows
            EXACTLY which object (among possibly many in the image) it must
            classify right now.
        axes: which of "nature"/"biotic"/"material" to ask about, e.g.
            ["nature", "biotic", "material"] for the full three-axis question,
            or just ["nature"] if that's all a caller needs.

    Returns:
        The full prompt string ready to send to the VLM alongside the image.
    """
    # Guard against typos: if someone passes an axis name we don't recognize
    # (e.g. "biotic_abiotic" instead of "biotic"), fail loudly right away
    # rather than silently building a prompt missing that axis.
    unknown_axes = set(axes) - set(_AXIS_INSTRUCTIONS)
    if unknown_axes:
        raise ValueError(f"Unknown axis/axes requested: {unknown_axes}")

    # Build one "- instruction" bullet line per requested axis and join them
    # with newlines, e.g.:
    #   - "nature": either "yes" or "no" - ...
    #   - "biotic": either "biotic", "abiotic", or "n/a" - ...
    field_lines = "\n".join(f"  - {_AXIS_INSTRUCTIONS[axis]}" for axis in axes)

    # The final prompt: names the specific object, reminds the model to use
    # the visual evidence (not just the word "oak tree" in isolation), and
    # lists exactly which fields/labels it must produce and how.
    return f"""You are analyzing a specific object identified in the provided image.
The object is classified as: {class_name}

Based on the visual evidence in the image and the definitions provided, classify this specific instance of the object.

Provide your reasoning first, followed by the specific labels according to these rules:
{field_lines}
"""
