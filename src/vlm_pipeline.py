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

_TAXONOMY_AXES = ["nature", "biotic", "material"]


# =============================================================================
# Label parsing
# =============================================================================
def label_to_bool(value: Optional[str], axis: str) -> Optional[bool]:
    """Standardize a TaxonomyResponse string answer into boolean logic.
    nature: yes->True/no->False; biotic: biotic->True/abiotic->False/n_a->None;
    material: material->True/immaterial->False/n_a->None. Unknown -> None."""
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
            return None
    elif axis == "material":
        if v == "material":
            return True
        if v == "immaterial":
            return False
        if v == "n/a":
            return None
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
            continue
        s = o.strip()
        if not s:
            continue
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
    prompts = [CAPTION_PROMPT] * len(image_paths)
    outs = vlm.generate_batch(
        prompts=prompts,
        images=image_paths,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_mode="free_form",
    )
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
    prompts = [EXTRACTION_PROMPT.format(caption=c) for c in captions]
    outs = vlm.generate_batch(
        prompts=prompts,
        images=image_paths,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_mode="structured",
        schema=ObjectExtractionResponse,
    )
    results = []
    for o in outs:
        if isinstance(o, dict):
            results.append(normalize_objects(o.get("objects", [])))
        else:  # parse failure / None
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
    flat_prompts: List[str] = []
    flat_images: List[str] = []
    owner_image_idx: List[int] = []  # which image each flat pair belongs to

    for i, objs in enumerate(objects_per_image):
        for obj in objs:
            flat_prompts.append(build_classification_prompt(obj, axes=_TAXONOMY_AXES))
            flat_images.append(image_paths[i])
            owner_image_idx.append(i)

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
        flat_outs = []

    # Regroup flat outputs back per image.
    per_image: List[List[Dict[str, Any]]] = [[] for _ in image_paths]
    for owner, out in zip(owner_image_idx, flat_outs):
        if isinstance(out, dict):
            rec = dict(out)
            rec["parse_failed"] = False
        else:
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
    """
    if extraction_system_prompt is None:
        extraction_system_prompt = caption_system_prompt

    n = len(dataset_instances)
    num_batches = (n + batch_size - 1) // batch_size

    for b in range(num_batches):
        chunk = dataset_instances[b * batch_size : (b + 1) * batch_size]
        image_paths = [inst["image_path"] for inst in chunk]

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
_ARTICLES = ("a ", "an ", "the ")


def _normalize_object(object_str: str) -> str:
    s = object_str.strip().lower()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
            break
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
        return None

    norm = _normalize_object(object_str)
    candidates = [norm]
    if " " in norm:
        candidates.append(norm.split()[-1])  # trailing head noun (e.g. "polar bear" -> "bear")

    for cand in candidates:
        synset = mapping_vocab.get(cand)
        if synset is None:
            continue
        labels = tax_graph.resolve_labels(synset)
        if labels is not None:
            is_nature = labels["is_nature"]
            biotic = None
            if is_nature and labels.get("biotic_abiotic"):
                biotic = labels["biotic_abiotic"] == "biotic"
            return {"synset": synset, "is_nature": is_nature, "biotic": biotic}
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
    """
    vlm_nature = label_to_bool(vlm_label.get("nature"), "nature")
    vlm_biotic = label_to_bool(vlm_label.get("biotic"), "biotic")
    vlm_material = label_to_bool(vlm_label.get("material"), "material")

    mapping = map_object_to_taxonomy(object_str, tax_graph, mapping_vocab)

    if mapping is not None:
        final_nature = mapping["is_nature"]
        nature_source = "wordnet"
        if mapping["biotic"] is not None:
            final_biotic = mapping["biotic"]
            biotic_source = "wordnet"
        else:
            # mapped & nature but node has no biotic label -> VLM decides biotic
            final_biotic = vlm_biotic if final_nature else None
            biotic_source = "vlm" if final_nature else "wordnet"
        mapped_synset = mapping["synset"]
    else:
        final_nature = vlm_nature
        nature_source = "vlm"
        final_biotic = vlm_biotic
        biotic_source = "vlm"
        mapped_synset = None

    # Enforce the schema rule: no downstream labels when the instance is not nature.
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
