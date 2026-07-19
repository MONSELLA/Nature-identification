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

from pathlib import Path
from typing import List, Literal, Tuple

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
    "  - Return each object as a noun or a compound noun (e.g. \"bench\", "
    "\"golden retriever\", \"mountain\", \"hen of the woods\").\n"
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
            "image, each as a noun or a compound noun. Include part-objects and "
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
    Pydantic schema with Interleaved Chain-of-Thought.
    The model is forced to conclude the top-level nature gate BEFORE 
    it is allowed to evaluate the downstream sub-axes.
    """

    # 1. First, isolate the nature reasoning.
    nature_reasoning: str = Field(
        description=(
            "Step 1: Concisely describe the specific target entity in the image. "
            "Evaluate strictly whether it meets the criteria for 'nature'. "
            "Do not discuss biotic or material properties yet."
        )
    )
    
    # 2. Force the model to lock in the yes/no decision based ONLY on Step 1.
    nature: Literal["yes", "no"] = Field(
        description="The top-level classification. 'yes' if it is nature, 'no' otherwise."
    )
    
    # 3. Now, initiate a second reasoning block conditioned on the decision just made.
    sub_axes_reasoning: str = Field(
        description=(
            "Step 2: If nature is 'yes', apply the definitions to determine the 'biotic' and 'material' axes. "
            "If nature is 'no', explicitly state 'Not applicable since the entity is not nature'."
        )
    )
    
    # 4. Apply the strict mutual exclusivity rule to the final labels.
    life_category: Literal["biotic", "abiotic", "none"] = Field(
        description="ALL nature entities MUST be classified as either 'biotic' or 'abiotic'. Non-nature entities MUST be 'none'."
    )
    
    tangibility: Literal["material", "immaterial", "none"] = Field(
        description="ALL nature entities MUST be classified as either 'material' or 'immaterial'. Non-nature entities MUST be 'none'."
    )


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
    tangibility: Literal["material", "immaterial"]


# One line of plain-English instructions per taxonomy axis, injected into the
# prompt text below. Kept as a dict (rather than hardcoded into one long
# prompt string) so build_classification_prompt() can ask for a SUBSET of axes
# if a caller only cares about e.g. nature+biotic and not material.
_AXIS_INSTRUCTIONS = {
    "nature": '"nature": either "yes" or "no" — whether this instance counts as nature under the provided definition.',
    "life_category": '"biotic": either "biotic", "abiotic", or "none" — only answer "biotic"/"abiotic" if "nature" is "yes"; use "none" if "nature" is "no"',
    "tangibility": '"tangibility": either "material", "immaterial", or "none" — only answer "material"/"immaterial" if "nature" is "yes"; use "none" if "nature" is "no"',
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
        axes: which of "nature"/"life_category"/"tangibility" to ask about, e.g.
            ["nature", "life_category", "tangibility"] for the full three-axis question,
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
    #   - "biotic": either "biotic", "abiotic", or "none" - ...
    field_lines = "\n".join(f"  - {_AXIS_INSTRUCTIONS[axis]}" for axis in axes)

    # The final prompt: names the specific object, reminds the model to use
    # the visual evidence (not just the word "oak tree" in isolation), and
    # lists exactly which fields/labels it must produce and how.
    return f"""You are analyzing a specific target entity identified in the provided image.
TARGET ENTITY TO CLASSIFY: "{class_name}"

Based on the visual evidence in the image and the strict definitions provided, classify this specific "{class_name}" instance.
Follow the interleaved reasoning structure: evaluate nature first, lock in the decision, and only then evaluate the sub-axes according to these rules:
{field_lines}

FIRST EXAMPLE OUTPUT FOR TARGET "chair":
{{
  "nature_reasoning": "The target is a chair. The visual evidence shows it is made of wood with visible grain. Manufactured objects where the natural material of origin is visually identifiable count as nature-based artefacts, fulfilling the criteria for nature.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. Wood is a derivative of flora, making it biotic. The chair is a physical object that occupies space, making it material.",
  "life_category": "biotic",
  "tangibility": "material"
}}

SECOND EXAMPLE OUTPUT FOR TARGET "fan":
{{
  "nature_reasoning": "The target is a manufactured electric fan made of plastic and metal. It is a fully artificial object with no unaltered natural elements or identifiable natural textures, so it fails the criteria for nature.",
  "nature": "no",
  "sub_axes_reasoning": "Not applicable since the entity is not nature.",
  "life_category": "none",
  "tangibility": "none"
}}

THIRD EXAMPLE OUTPUT FOR TARGET "river":
{{
  "nature_reasoning": "The target is a flowing body of water. Hydrological components are explicitly classified as natural elements, fulfilling the criteria for nature.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I must evaluate the sub-axes. Water is a non-living hydrological component, making it abiotic. It has a physical presence in the landscape, making it material.",
  "life_category": "abiotic",
  "tangibility": "material"
}}

FOURTH EXAMPLE OUTPUT FOR TARGET "dog":
{{
  "nature_reasoning": "The target is a dog. The image shows this is a stylized, cartoon depiction rather than a photograph of a real animal. Fictional or stylized depictions of non-human living entities still count as nature representations, fulfilling the criteria.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. A dog is an animal (fauna), making it biotic. Because the image shows a digital or animated representation rather than a physical organism, it is immaterial.",
  "life_category": "biotic",
  "tangibility": "immaterial"
}}

FIFTH EXAMPLE OUTPUT FOR TARGET "sunset":
{{
  "nature_reasoning": "The target is a sunset. The visual evidence reveals this is an artistic painting of a sunset rather than a photograph of the sky. Artistic representations of astrophysical processes explicitly count as nature-relevant, fulfilling the criteria.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. A sunset is an astrophysical process, making it abiotic. Because this specific instance is an artistic medium (a painting) rather than the literal physical event, it is immaterial.",
  "life_category": "abiotic",
  "tangibility": "immaterial"
}}
"""


# =============================================================================
# System prompts (built from the data/big5_taxonomy/ definition files)
# =============================================================================
# SINGLE HOME for this composition logic. Previously run_vlm_pipeline.py
# (build_system_prompts) and evaluate_taxonomy_labeling.py (load_system_prompt)
# each built their own copy of the "all three definitions" system prompt string
# — same content, two separate implementations that could silently drift apart.
# Both now import this one function, guaranteeing the pipeline's UNMAPPED-object
# labeling call and the calibration eval's system prompt stay byte-identical
# (the whole point of that eval is to measure the exact same fallback prompt
# the pipeline uses — see evaluate_taxonomy_labeling.py's module docstring).
def build_system_prompts(nature_path: str, biotic_path: str, material_path: str) -> Tuple[str, str, str]:
    """Build the three system prompts the pipeline needs, reading each
    definition file once:

      - caption_system         : NATURE definition only (no axis-priming, per
                                 the recap) — used for captioning + extraction.
      - label_system_full      : ALL THREE axis definitions — used for
                                 UNMAPPED objects (where the VLM must decide
                                 nature/biotic/material from scratch) AND for
                                 evaluate_taxonomy_labeling.py's calibration
                                 eval, which measures that exact fallback path.
      - label_system_material  : MATERIAL definition only — used for MAPPED-
                                 nature objects, where WordNet already fixed
                                 nature and biotic and only material/immaterial
                                 remains. Showing the model only the relevant
                                 definition (not all three) keeps the
                                 material-only call focused and its prefix
                                 cache-friendly.

    Returns (caption_system, label_system_full, label_system_material).
    """
    nature = Path(nature_path).read_text()
    biotic = Path(biotic_path).read_text()
    material = Path(material_path).read_text()
    caption_system = f"# NATURE DEFINITION\n{nature}"
    label_system_full = (
        "# 1. NATURE DEFINITION\n" f"{nature}\n\n"
        "# 2. LIFE CATEGORY AXIS\n" f"{biotic}\n\n"
        "# 3. TANGIBILITY AXIS\n" f"{material}"
    )
    label_system_material = f"# TANGIBILITY AXIS\n{material}"
    return caption_system, label_system_full, label_system_material