"""
src/vlm_pipeline.py

Baseline BIG-5 VLM pipeline (language-based), Phase-1 inference + the hybrid
taxonomy-label resolver used in Phase-2 scoring.

Two-pass baseline per image:
  1. caption_batch        — open-ended, neutral caption (no nature-priming).
  2. extract_objects_batch— structured object list, image re-sent (a "second
                            look"); captures part-objects (e.g. a flower on a
                            dress).
  3. label_objects_batch  — ONE VLM call per extracted object, image ALWAYS
                            attached, using the exact shared taxonomy prompt +
                            TaxonomyResponse schema (identical to
                            evaluate_taxonomy_labeling.py's calibration/fallback
                            prompt). Fanned through vlm.generate_batch so vLLM
                            prefix-caches the shared image + system tokens.

Hybrid labeling (resolved in Phase 2, needs the TaxonomyGraph):
  - nature, biotic : WordNet mapping when the object resolves to a labeled node;
                     VLM fallback otherwise.
  - material       : ALWAYS the VLM (never mapping) — depends on whether THIS
                     instance is a real object vs a representation, which a
                     class name cannot encode.

Diagnostics tracked: WordNet-mapping-rate vs VLM-fallback-rate, and total
objects extracted per image.

NO metric computation lives here (see src/evaluation/clip_metrics.py,
src/evaluation/taxonomy_metrics.py). NO prompts live here (see
src/models/prompts.py).

BIG-PICTURE OVERVIEW OF THIS FILE
This is the heart of the "VLM pipeline" (as opposed to the not-yet-built
"Grounding pipeline"). Given one image, we ask the VLM three separate
questions, one after another:
  1. "Describe this image" (free text) -> the CAPTION.
  2. "List every object in this image" (structured JSON list) -> the OBJECTS.
  3. For EACH object found in step 2: "is this specific thing nature? biotic
     or abiotic? material or immaterial?" -> the LABELS.
Steps 1-3 all happen during "Phase 1" (pure VLM inference — see
scripts/run_vlm_pipeline.py). Afterward, in "Phase 2", we combine the VLM's
own answers with a WordNet lookup (whenever possible) to get a final label per
object — that combining logic lives in the second half of this file
(map_object_to_taxonomy / resolve_hybrid_label).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models.prompts import (
    CAPTION_PROMPT,
    EXTRACTION_PROMPT,
    ObjectExtractionResponse,
    TaxonomyResponse,
    build_classification_prompt,
)

# Which taxonomy axes get asked about on every per-object labeling call. All
# three axes are always requested together in one prompt/schema (see
# label_objects_batch below) rather than issuing three separate VLM calls per
# object — cheaper, and lets the model reason about all three at once.
_TAXONOMY_AXES = ["nature", "biotic", "material"]


# =============================================================================
# Label parsing
# =============================================================================
def label_to_bool(value: Optional[str], axis: str) -> Optional[bool]:
    """Standardize a TaxonomyResponse string answer into boolean logic.
    nature: yes->True/no->False; biotic: biotic->True/abiotic->False/n_a->None;
    material: material->True/immaterial->False/n_a->None. Unknown -> None.

    Why convert to bool/None instead of just keeping the string? Because every
    downstream metric (accuracy/precision/recall/F1 in
    scripts/run_vlm_pipeline.py) needs plain True/False/None values to compare
    against ground truth — this function is the single place that translates
    the VLM's exact wording into that simpler representation.
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    if axis == "nature":
        if v == "yes":
            return True
        if v == "no":
            return False
    elif axis == "biotic":
        if v == "biotic":
            return True
        if v == "abiotic":
            return False
        if v == "n/a":
            # "n/a" is a valid/expected answer (nature was "no", so biotic
            # doesn't apply) — distinguish it from an unrecognized value by
            # returning None either way, but note it's not an ERROR case.
            return None
    elif axis == "material":
        if v == "material":
            return True
        if v == "immaterial":
            return False
        if v == "n/a":
            return None
    # Axis not recognized, or value didn't match any expected string for this
    # axis (e.g. a stray typo the model produced) — treat as "no answer".
    return None


# =============================================================================
# Object normalization
# =============================================================================
def normalize_objects(raw_objects: List[str]) -> List[str]:
    """Strip, drop empties, and de-duplicate object phrases case-insensitively
    while preserving first-seen order."""
    seen = set()
    out = []
    for o in raw_objects or []:
        if not isinstance(o, str):
            # Defensive: the VLM's JSON should only ever put strings in this
            # list per the schema, but if it somehow produced a number/null,
            # skip it rather than crashing everything downstream.
            continue
        s = o.strip()
        if not s:
            continue
        # De-duplicate case-insensitively (so "Dog" and "dog" count as the
        # same object) but keep the FIRST-seen original casing/spelling in
        # the output list, using a `seen` set of lowercase keys to detect repeats.
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# =============================================================================
# Stage 1 — captioning
# =============================================================================
def caption_batch(
    vlm,
    image_paths: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """Ask the VLM the same open-ended caption question for a whole batch of
    images at once (one API/model call handling many images together, which
    is much faster than looping one image at a time)."""
    # The exact same CAPTION_PROMPT is repeated once per image — we're asking
    # every image the identical question, just pairing it with a different
    # picture each time.
    prompts = [CAPTION_PROMPT] * len(image_paths)
    outs = vlm.generate_batch(
        prompts=prompts,
        images=image_paths,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_mode="free_form",  # no JSON schema — just get back plain text
    )
    # Guard against a non-string response (e.g. None on total generation
    # failure) by substituting an empty string, so callers never have to
    # special-case a missing caption.
    return [(o if isinstance(o, str) else "") for o in outs]


# =============================================================================
# Stage 2 — object extraction (structured)
# =============================================================================
def extract_objects_batch(
    vlm,
    image_paths: List[str],
    captions: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[List[str]]:
    """Ask the VLM to list every object in each image, using the Stage-1
    caption as extra context alongside a fresh look at the image itself."""
    # Fill in each image's own caption into the shared EXTRACTION_PROMPT
    # template (see src/models/prompts.py) — so each per-image prompt is
    # slightly different this time, referencing that image's own description.
    prompts = [EXTRACTION_PROMPT.format(caption=c) for c in captions]
    outs = vlm.generate_batch(
        prompts=prompts,
        images=image_paths,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_mode="structured",       # this time we DO want JSON back...
        schema=ObjectExtractionResponse,  # ...shaped exactly like this schema
    )
    results = []
    for o in outs:
        if isinstance(o, dict):
            # Successful structured output: pull out the "objects" list (or
            # an empty list if that key is somehow missing) and clean it up.
            results.append(normalize_objects(o.get("objects", [])))
        else:  # parse failure / None
            # The model's output didn't parse into valid JSON matching our
            # schema — treat this image as having zero extracted objects
            # rather than crashing the whole batch.
            results.append([])
    return results


# =============================================================================
# Stage 3 — per-object taxonomy labeling (one call per object, batched)
# =============================================================================
def label_objects_batch(
    vlm,
    image_paths: List[str],
    objects_per_image: List[List[str]],
    system_prompt: Optional[str],
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> List[List[Dict[str, Any]]]:
    """
    One structured VLM call per (image, object) pair, image always attached.
    Returns, per image, a list of raw label dicts aligned with that image's
    object list. A parse failure for an object yields {"parse_failed": True}.
    """
    # We need to ask the VLM about MANY (image, object) pairs — e.g. if image
    # 0 has 3 objects and image 1 has 2 objects, that's 5 separate questions
    # total. Rather than looping with 5 individual model calls, we "flatten"
    # everything into 3 parallel lists (one prompt/image/owner-index per
    # question) so ALL 5 questions can be sent to `vlm.generate_batch` as ONE
    # big batched call — much more efficient, especially with vLLM's ability
    # to reuse cached prompt tokens across the batch.
    flat_prompts: List[str] = []
    flat_images: List[str] = []
    owner_image_idx: List[int] = []  # which image each flat pair belongs to

    for i, objs in enumerate(objects_per_image):
        for obj in objs:
            flat_prompts.append(build_classification_prompt(obj, axes=_TAXONOMY_AXES))
            flat_images.append(image_paths[i])
            owner_image_idx.append(i)  # remember "this question was about image i"

    flat_outs: List[Any]
    if flat_prompts:
        flat_outs = vlm.generate_batch(
            prompts=flat_prompts,
            images=flat_images,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_mode="structured",
            schema=TaxonomyResponse,
        )
    else:
        # No objects were extracted for ANY image in this batch — nothing to
        # ask, so skip the (otherwise pointless) model call entirely.
        flat_outs = []

    # Regroup flat outputs back per image.
    # Now we need to "unflatten" the results: `flat_outs[k]` is the answer for
    # whichever object `owner_image_idx[k]` says it belongs to. We build one
    # empty list per image up front, then append each answer into the correct
    # image's slot as we walk through the flat results in order.
    per_image: List[List[Dict[str, Any]]] = [[] for _ in image_paths]
    for owner, out in zip(owner_image_idx, flat_outs):
        if isinstance(out, dict):
            rec = dict(out)  # copy, so we don't mutate the VLM's own dict
            rec["parse_failed"] = False
        else:
            # This particular object's classification failed to parse as
            # valid JSON — record it as an explicit failure (all fields None)
            # rather than silently dropping it, so it's still counted in the
            # object list and downstream failure-rate diagnostics.
            rec = {"reasoning": None, "nature": None, "biotic": None, "material": None, "parse_failed": True}
        per_image[owner].append(rec)
    return per_image


# =============================================================================
# Phase-1 inference driver (VLM only; no CLIP / no metric math)
# =============================================================================
def run_inference(
    vlm,
    dataset_instances: List[Dict[str, Any]],
    caption_system_prompt: Optional[str],
    label_system_prompt: Optional[str],
    extraction_system_prompt: Optional[str] = None,
    batch_size: int = 16,
    caption_max_new_tokens: int = 256,
    label_max_new_tokens: int = 300,
    temperature: float = 0.0,
    verbose: bool = False,
):
    """
    Run caption -> extraction -> labeling over every image and yield one raw
    record per image (no WordNet mapping, no metrics — that is Phase 2):

        {
          "image_path": str,
          "targets": [...],                 # GT carried through from the loader
          "caption": str,
          "objects": [str, ...],
          "object_labels": [ {reasoning,nature,biotic,material,parse_failed}, ... ]
        }

    `object_labels[k]` aligns with `objects[k]`.

    This is a GENERATOR function (it uses `yield`, not `return`) — it
    processes and produces results one BATCH at a time rather than holding
    every image's results in memory at once, which matters when running over
    thousands of images. The caller (scripts/run_vlm_pipeline.py's
    phase_infer) writes each yielded record straight to disk as it arrives.
    """
    if extraction_system_prompt is None:
        # By default, the extraction call uses the SAME system prompt as the
        # caption call (the nature-definition context) unless a caller
        # explicitly wants something different for that stage.
        extraction_system_prompt = caption_system_prompt

    n = len(dataset_instances)
    # Standard "ceiling division" trick: computes how many batches of size
    # `batch_size` are needed to cover all `n` images, rounding UP so a
    # partial final batch still gets its own iteration (e.g. 10 images with
    # batch_size=3 needs 4 batches: 3+3+3+1).
    num_batches = (n + batch_size - 1) // batch_size

    for b in range(num_batches):
        # Slice out this batch's images. Python slicing handles the last,
        # possibly-shorter batch automatically (it just returns however many
        # elements remain, even if fewer than batch_size).
        chunk = dataset_instances[b * batch_size : (b + 1) * batch_size]
        image_paths = [inst["image_path"] for inst in chunk]

        # Run the three stages in order for this batch of images.
        captions = caption_batch(
            vlm, image_paths, caption_system_prompt,
            max_new_tokens=caption_max_new_tokens, temperature=temperature,
        )
        objects_per_image = extract_objects_batch(
            vlm, image_paths, captions, extraction_system_prompt,
            max_new_tokens=caption_max_new_tokens, temperature=temperature,
        )
        labels_per_image = label_objects_batch(
            vlm, image_paths, objects_per_image, label_system_prompt,
            max_new_tokens=label_max_new_tokens, temperature=temperature,
        )

        # `zip(...)` walks all four lists together in lockstep — for each
        # image in this batch, we have its original dataset entry (`inst`,
        # which carries the ground truth `targets`), its caption, its
        # extracted objects, and its per-object labels, all aligned by
        # position. Yield one combined dict per image.
        for inst, cap, objs, labels in zip(chunk, captions, objects_per_image, labels_per_image):
            yield {
                "image_path": inst["image_path"],
                "targets": inst.get("targets", []),
                "caption": cap,
                "objects": objs,
                "object_labels": labels,
            }

        if verbose:
            done = min((b + 1) * batch_size, n)
            print(f"[infer] {done}/{n} images processed", flush=True)


# =============================================================================
# Hybrid taxonomy resolution (Phase 2 — needs the TaxonomyGraph)
# =============================================================================
# Words we strip off the FRONT of an object phrase before trying to match it
# against our known-class vocabulary, e.g. "a dog" and "dog" should both match
# the same vocabulary entry "dog".
_ARTICLES = ("a ", "an ", "the ")


def _normalize_object(object_str: str) -> str:
    """Lowercase an object phrase and strip a single leading article, so
    "A Golden Retriever" and "golden retriever" normalize to the same string."""
    s = object_str.strip().lower()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
            break  # only strip ONE article, even if (implausibly) more than one matched
    return s.strip()


def map_object_to_taxonomy(object_str: str, tax_graph, mapping_vocab: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Map a free-text object phrase onto a WordNet synset using the AUTHORITATIVE
    dataset class->synset table (`mapping_vocab`, from
    dataset_loader.build_mapping_vocab) — NOT word-sense guessing. This avoids
    picking the wrong WordNet sense (e.g. "tiger" -> "a fierce person"): the
    synset comes from the dataset's own class mapping (tiger -> tiger.n.02).

    Returns {"synset", "is_nature", "biotic"} when the object's (normalized)
    phrase — or its trailing head noun — is a known class whose synset resolves
    to a labeled node; else None (→ image-supported VLM fallback). `biotic` is
    True/False/None (None when nature-True but the node carries no biotic label).
    """
    if not mapping_vocab:
        # This dataset has no closed class vocabulary at all (e.g. BIG-5) —
        # every object automatically falls back to the VLM's own judgment.
        return None

    norm = _normalize_object(object_str)
    candidates = [norm]
    if " " in norm:
        # Also try just the LAST word of a multi-word phrase — e.g. if the
        # model said "large polar bear" but our vocabulary only has the entry
        # "polar bear" or just "bear", this gives us another chance to match
        # by stripping down to the head noun. (Note: this simple heuristic
        # only tries the single trailing word, not every possible sub-phrase.)
        candidates.append(norm.split()[-1])  # trailing head noun (e.g. "polar bear" -> "bear")

    for cand in candidates:
        synset = mapping_vocab.get(cand)
        if synset is None:
            # This candidate string isn't a recognized class name — try the
            # next candidate (if any) instead of giving up immediately.
            continue
        # Even if the WORD matched a known class, we still need to check
        # whether that class's synset actually resolves to a LABELED taxonomy
        # node (see excel_loader.py's resolve_labels) — some classes in the
        # dataset's vocabulary might still be unmapped in the Excel.
        labels = tax_graph.resolve_labels(synset)
        if labels is not None:
            is_nature = labels["is_nature"]
            biotic = None
            if is_nature and labels.get("biotic_abiotic"):
                biotic = labels["biotic_abiotic"] == "biotic"
            return {"synset": synset, "is_nature": is_nature, "biotic": biotic}
    # Nothing matched, or matched but didn't resolve to a label.
    return None


def resolve_hybrid_label(object_str: str, vlm_label: Dict[str, Any], tax_graph, mapping_vocab: Dict[str, str]) -> Dict[str, Any]:
    """
    Combine WordNet mapping with the VLM's TaxonomyResponse into final per-object
    labels, recording the source of each axis (diagnostics):

      - nature  : WordNet when the object maps, else VLM.
      - biotic  : WordNet when the object maps AND the node carries a biotic
                  label; else VLM (still None/n-a when final nature is False).
      - material: ALWAYS the VLM.

    Returns final booleans (nature/biotic/material, None = n/a) plus
    *_source ∈ {"wordnet","vlm"} and the mapped synset (if any).

    This is the function that actually implements the project's "hybrid"
    labeling strategy described in the module docstring: prefer the
    deterministic, authoritative WordNet answer whenever we have one, and only
    fall back to asking the VLM's own opinion when WordNet can't tell us.
    """
    # First, decode whatever the VLM itself said for this object (regardless
    # of whether we'll end up using it) — we need `vlm_material` either way,
    # since material ALWAYS comes from the VLM.
    vlm_nature = label_to_bool(vlm_label.get("nature"), "nature")
    vlm_biotic = label_to_bool(vlm_label.get("biotic"), "biotic")
    vlm_material = label_to_bool(vlm_label.get("material"), "material")

    mapping = map_object_to_taxonomy(object_str, tax_graph, mapping_vocab)

    if mapping is not None:
        # This object's phrase matched a known class AND that class resolves
        # to a labeled taxonomy node — use WordNet's answer for nature.
        final_nature = mapping["is_nature"]
        nature_source = "wordnet"
        if mapping["biotic"] is not None:
            # The resolved node also carries an explicit biotic/abiotic label
            # — use it.
            final_biotic = mapping["biotic"]
            biotic_source = "wordnet"
        else:
            # mapped & nature but node has no biotic label -> VLM decides biotic
            final_biotic = vlm_biotic if final_nature else None
            biotic_source = "vlm" if final_nature else "wordnet"
        mapped_synset = mapping["synset"]
    else:
        # No usable WordNet mapping at all — fall back entirely to the VLM's
        # own judgment for both nature and biotic.
        final_nature = vlm_nature
        nature_source = "vlm"
        final_biotic = vlm_biotic
        biotic_source = "vlm"
        mapped_synset = None

    # Enforce the schema rule: no downstream labels when the instance is not nature.
    # If the final decision is "this is not nature" (whether that came from
    # WordNet or the VLM), biotic/material simply don't apply — force both to
    # None regardless of what the VLM might have said for them, so we never
    # accidentally report a biotic/material label for a non-nature object.
    if final_nature is False:
        final_biotic = None
        final_material = None
    else:
        final_material = vlm_material  # always VLM on the nature branch

    return {
        "object": object_str,
        "mapped": mapping is not None,
        "mapped_synset": mapped_synset,
        "final_nature": final_nature,
        "final_biotic": final_biotic,
        "final_material": final_material,
        "nature_source": nature_source,
        "biotic_source": biotic_source,
        "material_source": "vlm",
    }
