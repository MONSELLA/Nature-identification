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
# you see". The explicit "including any text" clause covers BIG-5's
# text-heavy/meme/screenshot images, where a bare "describe this image" would
# otherwise skip on-image text entirely. This neutral caption becomes the
# input to Stage 2.
CAPTION_PROMPT = "Describe this image, including any text."

# =============================================================================
# Stage 1b — ClipMatch summary caption (ImageNet/Places DEFAULT for ClipMatch)
# =============================================================================
# ClipMatch's own CLIP checkpoint is typically a short-context model (vanilla
# CLIP, 77 tokens), which silently truncates the baseline caption above (run
# through LongCLIP, ~248 tokens). Rather than truncate blindly at the
# tokenizer, this asks the VLM itself to compress its OWN caption into a short
# (<=~20-word) summary, grounded in BOTH the image and that caption — so the
# model decides what to keep instead of a mid-sentence cutoff. Measured 0.41
# vs. 0.34 top-1 against the raw caption on the spot-check run, with far less
# truncation — this is now the PRIMARY text ClipMatch scores against on
# ImageNet/Places (see src/vlm_pipeline.summarize_caption_batch,
# run_vlm_pipeline.py's --no_summarize_clipmatch_caption to opt out). Still
# NOT part of the baseline two-pass pipeline itself — only feeds ClipMatch.
#
# NOW FORKED PER DATASET (superseding the earlier single shared prompt).
# ClipMatch scores this summary against a FIXED candidate vocabulary, and the
# two datasets' vocabularies are different KINDS of label:
#   - ImageNet's classes are OBJECTS — a discrete figure in a scene
#     ("golden retriever", "sea lion"), so the summary should spend its words
#     on the objects/entities present and what identifies them.
#   - Places365's classes ARE the scene ("airport terminal", "kitchen"), so
#     the summary should describe the setting as a whole; a summary that
#     fixates on an incidental foreground object buries the one thing
#     ClipMatch has to match.
# The previous shared prompt asked for BOTH subject and setting on every
# image, so each dataset paid for the half it does not want.
#
# Both cap at 30 words (down from the shared prompt's 50 — the point is to
# stay well inside a 77-token CLIP context, and a tighter budget forces the
# model to lead with the identifying content) and both end with "Output ONLY
# the summary text." to stop the model prefixing a conversational preamble
# ("Sure! Here's a summary: ...") that would be embedded as if it were image
# content. Both keep the "{caption}" placeholder and the grounded-in-the-image
# framing, and both stay neutral on nature — this feeds ClipMatch only, never
# the taxonomy axes.
SUMMARY_CAPTION_PROMPT_IMAGENET = (
    "Here is a detailed description of this image:\n\n"
    "\"{caption}\"\n\n"
    "Using both the image and this description, write a short summary (at most 20 words) "
    "that exclusively captures the most salient object or objects in the scene. "
    "Output ONLY the summary text."
)

SUMMARY_CAPTION_PROMPT_PLACES = (
    "Here is a detailed description of this image:\n\n"
    "\"{caption}\"\n\n"
    "Using both the image and this description, write a short summary (at most 20 words) "
    "that exclusively captures the overall scene, environment, or setting. "
    "Output ONLY the summary text."
)

# -----------------------------------------------------------------------------
# --no_caption variant (ImageNet only)
# -----------------------------------------------------------------------------
# With Stage 1 ablated away there is no caption to summarize, but ClipMatch
# still needs SOME caption-derived text — so this call survives and changes
# JOB: it stops being a re-summarization and becomes the run's one and only
# short, direct image description. Framing it as "summarize" would be asking
# the model to compress a description it was never given.
#
# WHAT CHANGES vs. SUMMARY_CAPTION_PROMPT_IMAGENET, and why:
#   - the "{caption}" block and its "Using both the image and this
#     description" clause are gone (nothing to reference).
#   - a NAMING-SPECIFICITY clause is added. In the two-pass baseline the
#     248-token caption was where fine-grained detail surfaced, and the
#     summary only had to preserve it. Without it, no stage in the pipeline
#     produces a specific name — and ImageNet-1k is largely fine-grained
#     (dozens of breeds/species, each with its own candidate embedding), so
#     "a dog on grass" is near-equidistant from thirty dog classes and the
#     argmax between them is close to noise. This is where ImageNet top-1 is
#     actually won.
#   - background/setting suppression is made EXPLICIT rather than resting on
#     "exclusively" alone. This is the opposite kind of specificity and it
#     hurts: the candidate side is a mean over short "a photo of a {}."
#     templates, and the measured 0.34 -> 0.41 gain came precisely from
#     stripping the long caption's context.
#   - the plural stays ("or subjects"), as in the incumbent prompt that
#     measured 0.41, but is made CONDITIONAL ("if more than one is equally
#     prominent"). An open invitation to enumerate walks back toward the
#     concatenated-object-list ClipMatch variant this project rejected twice
#     for semantic dilution, and every word spent on a second entity is a word
#     not spent on "golden retriever" instead of "dog".
# The 20-word cap and the "Output ONLY ..." guard are unchanged.
#
# NOT BENCHMARK-SHAPED ON PURPOSE: it never mentions ImageNet, a class list,
# or that the text will be matched against candidates. That would likely lift
# top-1 while measuring how well each VLM guesses the label format rather than
# how well it sees.
#
# NOTE this makes the ImageNet arm a "long caption + summary" vs "short
# caption" comparison, NOT "caption vs no caption" — two things move at once.
# The clean caption ablation lives on BIG-5 (Twitter/Weibo), where --no_caption
# genuinely removes the stage. Do not report the ImageNet delta as evidence
# about captioning.
SUMMARY_CAPTION_PROMPT_IMAGENET_NO_CAPTION = (
    "Write a short description of this image (at most 20 words) that exclusively "
    "captures the main subject, or subjects if more than one is equally prominent. "
    "Name each subject as specifically as you can — the precise breed, species, or "
    "type of object rather than a general category. "
    "Do not describe the background, the setting, or anything else. "
    "Output ONLY the description text."
)

# Keyed by the same dataset names as clip_metrics.CLIPMATCH_DATASETS — those
# are the only two datasets that run ClipMatch, so they are the only two that
# ever request a summary caption.
SUMMARY_CAPTION_PROMPTS = {
    "imagenet": SUMMARY_CAPTION_PROMPT_IMAGENET,
    "places365": SUMMARY_CAPTION_PROMPT_PLACES,
}

# The --no_caption counterparts. DELIBERATELY has no "places365" entry: the
# scene-centric variant has never been written or validated, and silently
# substituting the object-centric one (or the with-caption one, which would
# leave a literal "{caption}" placeholder in the prompt) is exactly the kind
# of quiet mismatch get_summary_caption_prompt exists to prevent. Places365 +
# --no_caption therefore raises; add a validated prompt here if that run is
# ever wanted.
SUMMARY_CAPTION_PROMPTS_NO_CAPTION = {
    "imagenet": SUMMARY_CAPTION_PROMPT_IMAGENET_NO_CAPTION,
}


def summary_caption_prompt_name(dataset: str, no_caption: bool = False) -> str:
    """The identifier recorded in the artifact header's
    `summary_caption_prompt` for this (dataset, no_caption) pair. Kept beside
    the lookup below so the recorded name can never drift from the prompt
    actually used — comparing ClipMatch numbers across artifacts written with
    different variants would be comparing different setups."""
    return f"summary_caption_prompt_{dataset}" + ("_no_caption" if no_caption else "")


def get_summary_caption_prompt(dataset: str, no_caption: bool = False) -> str:
    """The ClipMatch summary-caption prompt for `dataset` (see
    SUMMARY_CAPTION_PROMPTS). Raises on an unknown dataset rather than
    silently falling back to one of the two — a silent fallback would mean
    scoring Places images with the object-centric prompt (or vice versa) with
    nothing in the artifact to show it happened.

    `no_caption=True` returns the --no_caption variant
    (SUMMARY_CAPTION_PROMPTS_NO_CAPTION), which takes NO "{caption}"
    placeholder and must be used verbatim. Only ImageNet has one; every other
    dataset raises here rather than being handed a prompt whose "{caption}"
    slot nothing will ever fill."""
    table = SUMMARY_CAPTION_PROMPTS_NO_CAPTION if no_caption else SUMMARY_CAPTION_PROMPTS
    try:
        return table[dataset]
    except KeyError:
        if no_caption:
            raise ValueError(
                f"No --no_caption summary-caption prompt for dataset {dataset!r}; "
                f"expected one of {sorted(SUMMARY_CAPTION_PROMPTS_NO_CAPTION)}. The "
                f"caption-free variant has only been written and validated for ImageNet "
                f"(see SUMMARY_CAPTION_PROMPTS_NO_CAPTION). Either add a validated prompt "
                f"there, or pass --no_summarize_clipmatch_caption — noting that ClipMatch "
                f"then has no text at all on a --no_caption run and cannot be scored."
            ) from None
        raise ValueError(
            f"No summary-caption prompt for dataset {dataset!r}; expected one of "
            f"{sorted(SUMMARY_CAPTION_PROMPTS)}. Only these datasets run ClipMatch "
            f"(clip_metrics.CLIPMATCH_DATASETS), so only these need a summary caption."
        ) from None

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
EXTRACTION_PROMPT = """You are an expert computer vision annotator. Below is a baseline description of the image:

"{caption}"

Your task is to extract visual entities from the image, keeping the total to a maximum of 12 items to avoid noise.

RULES:
 - Macro Elements (Objective): Extract the countable Things (salient objects like dog, guitar, desk, tree), uncountable Stuff (amorphous regions like sky, grass, ocean, sand, road), and the overarching Scene (settings like forest, office, restaurant), objectively.
 - Micro Elements (Nature-Filtered): For tiny details, non-salient items, background objects, or depicted entities (e.g., a distant cat in the background, a little dog figurine, or a flower printed on a dress), ONLY extract them if they represent nature according to your system instructions. Ignore all other minor details.
 - Use the 'reasoning' field to explicitly state your two-step plan: list the macro elements (things, stuff, and scenes), then identify any valid nature-related micro elements.
 - Format all extracted entities as concise, singular nouns or compound nouns.
 - Place all chosen entities into the single 'objects' list.
 - Do not hallucinate entities not visually present.
 - Do not repeat the same extracted entity twice.

EXAMPLE 1 (Mixed Environment):
Image: A person sitting in an indoor office chair at a desk with a laptop. On the desk is a little dog figurine and an orange. The person is wearing a shirt with a geometric triangle pattern. In the background, there is a potted plant near a window.
{{
  "reasoning": "Step 1: Macro elements include things (desk, chair, person, laptop, window), and the scene (office). Step 2: For micro elements, the little dog figurine, the orange, and the potted plant represent nature, so they will be extracted. I will ignore the geometric pattern on the shirt as it is not a nature representation.",
  "objects": ["desk", "chair", "person", "laptop", "window", "office", "dog figurine", "orange", "potted plant"]
}}

EXAMPLE 2 (Pure Nature Space):
Image: A sunny beach with crashing ocean waves and sand. A surfer carrying a surfboard with a bird logo walks near the water. A small crab is resting on the sand.
{{
  "reasoning": "Step 1: Macro elements include things (surfer, surfboard), stuff (ocean, sand), and the scene (beach). Step 2: For micro elements, the small crab and the bird logo represents nature and will be extracted.",
  "objects": ["surfer", "surfboard", "ocean", "sand", "beach", "crab", "bird logo"]
}}

EXAMPLE 3 (Pure No-nature Space):
Image: A photograph of a brightly lit convenience store in an urban city. Shelves are stocked with snacks and soda bottles. A cashier stands behind the counter.
{{
  "reasoning": "Step 1: Macro elements include things (shelf, cashier, counter) and the scene (store, city). Step 2: For micro elements, there are junk food snacks and soda bottles, but since none of these represent nature, they will be ignored. I will only extract the macro items.",
  "objects": ["shelf", "cashier", "counter", "store", "city"]
}}

EXAMPLE 4 (Social Media Text & Depiction):
Image: A messy indoor bedroom with a desk, a chair and a bed. A person wearing a shirt with a flower print is taking a selfie in the mirror. A teddy bear is laying on the bed. Overlaid on the image is a text banner that says "Save the trees!".
{{
  "reasoning": "Step 1: Macro elements include things (person, mirror, desk, chair, bed) and the scene (bedroom). Step 2: For micro elements, the 'flower' print on the shirt, the 'teddy bear' (depicting an animal), and the word 'trees' from the text overlay all represent nature-related concepts and will be extracted. The rest of the messy room clutter will be ignored.",
  "objects": ["person", "mirror", "desk", "chair", "bed", "bedroom", "flower", "teddy bear", "trees"]
}}

EXAMPLE 5 (Social Media Selfie):
Image: Two people taking a selfie outdoors. The man on the left is wearing a plain black T-shirt. The woman on the right is wearing a red T-shirt with a hockey helmet logo. The background shows an outdoor setting with a paved walkway with a few trees and grass in the distance.
{{
  "reasoning": "Step 1: Macro elements include things (person, trees), stuff (grass) and the scene (urban walk). Step 2: For micro elements, the 'hockey helmet' logo on the shirt is not a nature representation, so it will be ignored.",
  "objects": ["person", "trees", "urban walk", "grass"]
}}"""


# -----------------------------------------------------------------------------
# ABLATION (--no_caption): extraction WITHOUT the Stage-1 caption
# -----------------------------------------------------------------------------
# Identical to EXTRACTION_PROMPT above except that the "{caption}" preamble is
# gone — the model looks at the image directly instead of at the image plus its
# own free-form description of it. This is the ONLY prompt difference between
# the baseline and the caption-ablation run, so any measured delta is
# attributable to the caption stage itself and not to rewritten instructions.
# NOTE: this string has no format placeholder, so it is used verbatim (never
# .format()-ed) — the {{ }} escapes inside its JSON examples are kept for
# consistency with EXTRACTION_PROMPT and are stripped by
# get_extraction_prompt() below, which is the only supported way to fetch it.
EXTRACTION_PROMPT_NO_CAPTION = """You are an expert computer vision annotator. Your task is to extract visual entities from the image, keeping the total to a maximum of 12 items to avoid noise.

RULES:
 - Macro Elements (Objective): Extract the countable Things (salient objects like dog, guitar, desk, tree), uncountable Stuff (amorphous regions like sky, grass, ocean, sand, road), and the overarching Scene (settings like forest, office, restaurant), objectively.
 - Micro Elements (Nature-Filtered): For tiny details, non-salient items, background objects, or depicted entities (e.g., a distant cat in the background, a little dog figurine, or a flower printed on a dress), ONLY extract them if they represent nature according to your system instructions. Ignore all other minor details.
 - Use the 'reasoning' field to explicitly state your two-step plan: list the macro elements (things, stuff, and scenes), then identify any valid nature-related micro elements.
 - Format all extracted entities as concise, singular nouns or compound nouns.
 - Place all chosen entities into the single 'objects' list.
 - Do not hallucinate entities not visually present.
 - Do not repeat the same extracted entity twice.

EXAMPLE 1 (Mixed Environment):
Image: A person sitting in an indoor office chair at a desk with a laptop. On the desk is a little dog figurine and an orange. The person is wearing a shirt with a geometric triangle pattern. In the background, there is a potted plant near a window.
{{
  "reasoning": "Step 1: Macro elements include things (desk, chair, person, laptop, window), and the scene (office). Step 2: For micro elements, the little dog figurine, the orange, and the potted plant represent nature, so they will be extracted. I will ignore the geometric pattern on the shirt as it is not a nature representation.",
  "objects": ["desk", "chair", "person", "laptop", "window", "office", "dog figurine", "orange", "potted plant"]
}}

EXAMPLE 2 (Pure Nature Space):
Image: A sunny beach with crashing ocean waves and sand. A surfer carrying a surfboard with a bird logo walks near the water. A small crab is resting on the sand.
{{
  "reasoning": "Step 1: Macro elements include things (surfer, surfboard), stuff (ocean, sand), and the scene (beach). Step 2: For micro elements, the small crab and the bird logo represents nature and will be extracted.",
  "objects": ["surfer", "surfboard", "ocean", "sand", "beach", "crab", "bird logo"]
}}

EXAMPLE 3 (Pure No-nature Space):
Image: A photograph of a brightly lit convenience store in an urban city. Shelves are stocked with snacks and soda bottles. A cashier stands behind the counter.
{{
  "reasoning": "Step 1: Macro elements include things (shelf, cashier, counter) and the scene (store, city). Step 2: For micro elements, there are junk food snacks and soda bottles, but since none of these represent nature, they will be ignored. I will only extract the macro items.",
  "objects": ["shelf", "cashier", "counter", "store", "city"]
}}

EXAMPLE 4 (Social Media Text & Depiction):
Image: A messy indoor bedroom with a desk, a chair and a bed. A person wearing a shirt with a flower print is taking a selfie in the mirror. A teddy bear is laying on the bed. Overlaid on the image is a text banner that says "Save the trees!".
{{
  "reasoning": "Step 1: Macro elements include things (person, mirror, desk, chair, bed) and the scene (bedroom). Step 2: For micro elements, the 'flower' print on the shirt, the 'teddy bear' (depicting an animal), and the word 'trees' from the text overlay all represent nature-related concepts and will be extracted. The rest of the messy room clutter will be ignored.",
  "objects": ["person", "mirror", "desk", "chair", "bed", "bedroom", "flower", "teddy bear", "trees"]
}}

EXAMPLE 5 (Social Media Selfie):
Image: Two people taking a selfie outdoors. The man on the left is wearing a plain black T-shirt. The woman on the right is wearing a red T-shirt with a hockey helmet logo. The background shows an outdoor setting with a paved walkway with a few trees and grass in the distance.
{{
  "reasoning": "Step 1: Macro elements include things (person, trees), stuff (grass) and the scene (urban walk). Step 2: For micro elements, the 'hockey helmet' logo on the shirt is not a nature representation, so it will be ignored.",
  "objects": ["person", "trees", "urban walk", "grass"]
}}"""

def get_extraction_prompt(no_caption: bool = False) -> str:
    """Return the Stage-2 extraction prompt for this run.

    no_caption=False (default) -> EXTRACTION_PROMPT, a TEMPLATE that still
    needs .format(caption=...) applied by the caller.
    no_caption=True  -> the caption-ablation prompt, already fully rendered
    (its JSON-example braces un-escaped), to be used verbatim.
    """
    if no_caption:
        # The literal prompt carries doubled braces so it stays byte-comparable
        # with EXTRACTION_PROMPT; undo that here since nothing will .format() it.
        return EXTRACTION_PROMPT_NO_CAPTION.replace("{{", "{").replace("}}", "}")
    return EXTRACTION_PROMPT


class ObjectExtractionResponse(BaseModel):
    """Structured schema for baseline and nature-filtered entity extraction."""
    
    reasoning: str = Field(
        description=(
            "Briefly analyze the image using a two-step process. First, identify the macro elements "
            "(countable things/objects, uncountable amorphous stuff, and overarching scene or scenes) objectively. "
            "Second, scan for micro elements (tiny details, non-salient items, depictions). "
            "ONLY extract these micro elements if they represent nature according to the system definition. "
            "Ignore all other minor details. Keep the final list to a maximum of 12 items."
        )
    )
    objects: List[str] = Field(
        description=(
            "A unified list of all extracted entities. This includes objective macro elements AND "
            "nature-filtered micro elements. Format purely as concise, singular nouns or compound nouns. Return [] if none."
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
            "Evaluate strictly whether it meets the criteria for 'nature' applying the definition. "
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
            "Step 2: If nature is 'yes', apply the definitions to determine the classification for the "
            "life_cateogry and tangibility axes. "
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
        description="One concise sentence justifying the tangibility classification of the entity based on the visual evidence and the definition provided."
    )
    tangibility: Literal["material", "immaterial"]


# One line of plain-English instructions per taxonomy axis, injected into the
# prompt text below. Kept as a dict (rather than hardcoded into one long
# prompt string) so build_classification_prompt() can ask for a SUBSET of axes
# if a caller only cares about e.g. nature+biotic and not material.
_AXIS_INSTRUCTIONS = {
    "nature": '"nature": either "yes" or "no" — whether this instance counts as nature under the provided definition.',
    "life_category": '"life_category": either "biotic", "abiotic", or "none" — only answer "biotic"/"abiotic" if "nature" is "yes"; use "none" if "nature" is "no"',
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

EXAMPLE OUTPUT FOR TARGET "sky":
{{
  "nature_reasoning": "The target entity is sky. The visual evidence shows an artistic painting on canvas depicting a sky, rather than a literal photograph of reality. Artistic depictions of atmospheric elements explicitly count as Representations of Nature. This fulfills the inclusion criteria.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. The sky is an atmospheric element, placing it under Atmospheric and Meteorological within the abiotic category. Because the entity is an artistic interpretation of nature rather than a physical non-living element directly captured by the camera in reality, it falls under Artistic and Abstract Representations and is classified as immaterial under the Tangibility axis.",
  "life_category": "abiotic",
  "tangibility": "immaterial"
}}

EXAMPLE OUTPUT FOR TARGET "chair":
{{
  "nature_reasoning": "The target entity is a chair. The visual evidence shows a manufactured structural object made of wood with clearly visible natural grain. While homogenous artefacts are excluded, manufactured objects where the natural material of origin remains visually identifiable by its structure, grain, or texture explicitly count as Nature-Based Artefacts. This fulfills the inclusion criteria.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. Wood is a processed material derived from trees, placing it under Flora and its derivatives within the biotic category. Because the visual evidence shows a physical artefact directly captured by the camera in the real world, it is classified as material under the Tangibility axis.",
  "life_category": "biotic",
  "tangibility": "material"
}}

EXAMPLE OUTPUT FOR TARGET "fan":
{{
  "nature_reasoning": "The target entity is an electric fan. The visual evidence reveals a manufactured functional object made of smooth plastic and metal. Because the object does not have an identifiable natural texture and does not depict a natural entity, it falls strictly under the Homogenous Artefacts exclusion. It fails the criteria for nature.",
  "nature": "no",
  "sub_axes_reasoning": "Not applicable since the entity is not classified as nature.",
  "life_category": "none",
  "tangibility": "none"
}}

EXAMPLE OUTPUT FOR TARGET "river":
{{
  "nature_reasoning": "The target entity is a river. The visual evidence captures a real-world flowing body of water. Hydrological components (rivers, ponds, lakes, streams) are explicitly classified under Non-Living Natural Elements. This fulfills the inclusion criteria for nature.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. A river is a fluid flow, placing it under Hydrological within the abiotic category. Because it is an observable hydrological element directly captured by the camera in reality, it is classified as material under the Tangibility axis.",
  "life_category": "abiotic",
  "tangibility": "material"
}}

EXAMPLE OUTPUT FOR TARGET "cartoon dog":
{{
  "nature_reasoning": "The target entity is a dog. The visual evidence indicates this is a stylized animated cartoon depiction rather than a physical real-world organism. Fictional, stylized, or artistic depictions explicitly referencing more-than-human living entities (fauna) count as Representations of Nature. This fulfills the inclusion criteria.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. A dog is a non-human animal (fauna), placing its representational subject under the biotic category. Because this specific instance is a depiction that reimagines or fabricates nature in a non-literal form, it falls under Fictional Representations and is classified as immaterial under the Tangibility axis.",
  "life_category": "biotic",
  "tangibility": "immaterial"
}}

EXAMPLE OUTPUT FOR TARGET "policeman":
{{
  "nature_reasoning": "The target entity is a policeman. The visual evidence shows a human being wearing a uniform. Human individuals, human body parts, or groups of people fall under the explicit Exclusion Scope and are NOT classified as nature under this taxonomy.",
  "nature": "no",
  "sub_axes_reasoning": "Not applicable since human beings are excluded from the nature taxonomy.",
  "life_category": "none",
  "tangibility": "none"
}}

EXAMPLE OUTPUT FOR TARGET "canyon":
{{
  "nature_reasoning": "The target entity is a canyon. The visual evidence shows a massive, deep valley with layered rock cliffs. Canyons and rocks fall under Geological and Topographical components within the Non-Living Natural Elements inclusion scope. This fulfills the criteria for nature.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. A canyon is a large-scale landform and earth structure, placing it under Geological and Topographical within the abiotic category. Because this is a physical non-living element directly captured by the camera in reality, it is classified as material under the Tangibility axis.",
  "life_category": "abiotic",
  "tangibility": "material"
}}

EXAMPLE OUTPUT FOR TARGET "elephant figurine":
{{
  "nature_reasoning": "The target entity is an elephant figurine. The visual evidence shows a three-dimensional decorative sculpture shaped like an animal. While homogenous artefacts are excluded, explicit animal depictions such as figurines or statues of fauna explicitly count as Representations of Nature. This fulfills the inclusion criteria.",
  "nature": "yes",
  "sub_axes_reasoning": "Since nature is 'yes', I evaluate the sub-axes. The object represents an elephant (fauna), placing its conceptual subject under the biotic category. Because the figurine is a three-dimensional manufactured object explicitly designed to depict fauna rather than being the natural entity itself, it falls under Physical Representational Objects and is classified as immaterial under the Tangibility axis.",
  "life_category": "biotic",
  "tangibility": "immaterial"
}}
"""


# =============================================================================
# Stage 3b — Material-only prompt (the MAPPED-nature fast path)
# =============================================================================
# Companion to MaterialResponse (above): the MAPPED-nature fast path in
# src/vlm_pipeline.py's label_objects_batch already knows nature=True and, when
# the mapped node carries one, the biotic/abiotic value too (both come from
# WordNet, not this call) — only material/immaterial still needs the VLM. This
# used to reuse build_classification_prompt(obj, axes=["tangibility"]), which
# was WRONG on two counts: (1) its few-shot examples are all full
# TaxonomyResponse-shaped JSON (nature_reasoning/nature/sub_axes_reasoning/
# life_category/tangibility), mismatched against the MaterialResponse schema
# (reasoning/tangibility only) actually requested on this call; and (2) it
# never told the model the nature/biotic verdict was already settled by
# mapping, so the model had to (redundantly, and possibly inconsistently)
# re-derive "is this nature" and "biotic or abiotic" from scratch even though
# those answers are thrown away. This prompt instead STATES the already-known
# nature/biotic verdict up front and asks for material/immaterial only, with
# examples that match MaterialResponse's actual shape.

def build_material_classification_prompt(class_name, biotic):
    """
    Build the per-object prompt for the MAPPED-nature material-only labeling
    call (paired with schema=MaterialResponse).

    Args:
        class_name: the object's name as a plain string, e.g. "oak tree" — same
            role as in build_classification_prompt.
        biotic: the mapped node's biotic/abiotic verdict — True ("biotic"),
            False ("abiotic"), or None when the mapped node is nature but
            carries no biotic/abiotic label (stated only as "nature" in that
            case, since we don't actually know which).

    Returns:
        The full prompt string ready to send to the VLM alongside the image.
    """
    
    return f"""You are analyzing a specific target entity identified in the provided image.
TARGET ENTITY TO CLASSIFY: "{class_name}"

Your task is to classify this specific "{class_name}" instance's tangibility, based on the visual evidence in the image and the strict definitions provided. 
First, provide a one-sentence reasoning step explaining what you see. Then, output your final tangibility classification as either "material" or "immaterial".

EXAMPLE OUTPUT FOR TARGET "river":
{{
  "reasoning": "The visual evidence captures a real-world flowing body of water. Because it is an observable hydrological element directly captured by the camera in reality, it possesses physical presence and is classified as material.",
  "tangibility": "material"
}}

EXAMPLE OUTPUT FOR TARGET "dog":
{{
  "reasoning": "The visual evidence shows a stylized animated cartoon depiction of a dog rather than a physical real-world organism directly captured by the camera. Because it is a depiction that reimagines nature in a non-literal form, it falls under Fictional Representations and is classified as immaterial.",
  "tangibility": "immaterial"
}}

EXAMPLE OUTPUT FOR TARGET "chair":
{{
  "reasoning": "The visual evidence shows a manufactured wooden chair with identifiable natural grain. Because it is a physical artefact directly captured by the camera in the real world that possesses physical presence and mass, it is classified as material.",
  "tangibility": "material"
}}

EXAMPLE OUTPUT FOR TARGET "sunset":
{{
  "reasoning": "The visual evidence reveals an artistic painting on canvas depicting a sky at dusk rather than a physical non-living element directly captured by the camera in reality. Because it is an expressive interpretation of nature, it falls under Artistic and Abstract Representations and is classified as immaterial.",
  "tangibility": "immaterial"
}}

EXAMPLE OUTPUT FOR TARGET "canyon":
{{
  "reasoning": "The visual evidence captures a real-world geological landform occupying physical space. Because it is a physical non-living element directly captured by the camera in reality, it is classified as material.",
  "tangibility": "material"
}}

EXAMPLE OUTPUT FOR TARGET "elephant":
{{
  "reasoning": "The visual evidence shows a three-dimensional manufactured object explicitly designed to depict fauna, rather than being the natural entity itself. Because its primary purpose is representational, it falls under Physical Representational Objects and is classified as immaterial.",
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
                                 the recap) — used for EXTRACTION only. The
                                 caption call itself (src/vlm_pipeline.py's
                                 caption_batch, via run_inference) deliberately
                                 does NOT receive this prompt, so the very
                                 first free-form look at the image stays
                                 completely free of nature-related context;
                                 this string is threaded through run_inference
                                 only to seed extraction_system_prompt's
                                 default.
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
    caption_system = nature
    label_system_full = (
        f"{nature}\n\n"
        f"{biotic}\n\n"
        f"{material}"
    )
    label_system_material = material
    return caption_system, label_system_full, label_system_material