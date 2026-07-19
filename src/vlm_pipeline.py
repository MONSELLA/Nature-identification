"""
src/vlm_pipeline.py

Baseline BIG-5 VLM pipeline (language-based), Phase-1 inference including the
hybrid taxonomy-label resolution (mapping now happens here, not in Phase-2).

Two-pass baseline per image:
  1. caption_batch        — open-ended, neutral caption (no nature-priming).
  2. extract_objects_batch— structured object list, image re-sent (a "second
                            look"); captures part-objects (e.g. a flower on a
                            dress).
  2b. map each object      — WordNet mapping FIRST (map_object_to_taxonomy), so
                            the labeling call is routed to the minimum the VLM
                            still has to answer.
  3. label_objects_batch  — mapping-aware, image ALWAYS attached: UNMAPPED
                            objects get a full nature/biotic/material call
                            (TaxonomyResponse, three-definition system prompt);
                            MAPPED-nature objects get a material-only call
                            (MaterialResponse, material-definition system
                            prompt); MAPPED non-nature objects get NO VLM call.
                            Each group is fanned through vlm.generate_batch so
                            vLLM prefix-caches the shared image + system tokens.

Hybrid labeling (resolved in Phase 1 now, needs the TaxonomyGraph):
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
    MaterialResponse,
    ObjectExtractionResponse,
    TaxonomyResponse,
    build_classification_prompt,
)
from src.utils import BatchProgress

# Axis sets for the two per-object labeling paths (see label_objects_batch):
#   - FULL  : an UNMAPPED object — WordNet told us nothing, so the VLM must
#             decide all three axes at once (one prompt/schema, cheaper than
#             three separate calls, and lets the model reason jointly).
#   - MATERIAL: a MAPPED-nature object — nature/biotic are already fixed by the
#             WordNet mapping, so the VLM is only asked material/immaterial.
_FULL_AXES = ["nature", "life_category", "tangibility"]
_MATERIAL_AXES = ["tangibility"]

# Sentinel distinguishing "mapping not provided, compute it" from "mapping is
# genuinely None (unmapped)" in resolve_hybrid_label — a plain None default
# could not tell those two apart.
_UNSET = object()


# =============================================================================
# Label parsing
# =============================================================================
def label_to_bool(value: Optional[str], axis: str) -> Optional[bool]:
    """Standardize a TaxonomyResponse string answer into boolean logic.
    nature: yes->True/no->False; biotic: biotic->True/abiotic->False/
    n_a-or-none->None; material: material->True/immaterial->False/
    n_a-or-none->None. Unknown -> None.

    Both "n/a" and "none" are accepted as the not-applicable sentinel: the
    current TaxonomyResponse schema (src/models/prompts.py) uses "none", but
    "n/a" is accepted too so this still reads older stored artifacts written
    before that schema change without silently misparsing them.

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
        if v in ("none", "n/a"):
            # A valid/expected answer (nature was "no", so biotic doesn't
            # apply) — distinguish it from an unrecognized value by returning
            # None either way, but note it's not an ERROR case.
            return None
    elif axis == "material":
        if v == "material":
            return True
        if v == "immaterial":
            return False
        if v in ("none", "n/a"):
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
    outs = vlm.generate_batch_safe(
        prompts, image_paths,
        label="caption_batch", item_labels=image_paths,
        system_prompt=system_prompt, max_new_tokens=max_new_tokens,
        temperature=temperature, output_mode="free_form",  # no JSON schema — just get back plain text
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
    outs = vlm.generate_batch_safe(
        prompts, image_paths,
        label="extract_objects_batch", item_labels=image_paths,
        system_prompt=system_prompt, max_new_tokens=max_new_tokens, temperature=temperature,
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
# Stage 3 — per-object taxonomy labeling (mapping-aware, batched)
# =============================================================================
def _combine_taxonomy_reasoning(out: Dict[str, Any]) -> Optional[str]:
    """TaxonomyResponse (src/models/prompts.py) splits the model's
    justification into TWO fields — `nature_reasoning` (Step 1, the nature
    gate) then `sub_axes_reasoning` (Step 2, conditioned on that decision) —
    rather than one `reasoning` field. Combine them back into a single string
    here so the rest of the pipeline's object_labels shape (CSV output,
    diagnostics, etc.) keeps working against one stable `reasoning` key
    regardless of how many reasoning steps the schema asks for."""
    parts = [out.get("nature_reasoning"), out.get("sub_axes_reasoning")]
    parts = [p for p in parts if p]
    return " ".join(parts) if parts else None


def _empty_label(parse_failed: bool, vlm_called: bool) -> Dict[str, Any]:
    """A raw-label dict with every axis absent — used for a parse failure
    (vlm_called=True) or a mapped non-nature object the VLM was never asked
    about (vlm_called=False)."""
    return {"reasoning": None, "nature": None, "biotic": None, "material": None,
            "parse_failed": parse_failed, "vlm_called": vlm_called}


def label_objects_batch(
    vlm,
    image_paths: List[str],
    objects_per_image: List[List[str]],
    mappings_per_image: List[List[Optional[Dict[str, Any]]]],
    label_system_full: Optional[str],
    label_system_material: Optional[str],
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> List[List[Dict[str, Any]]]:
    """
    Per-object taxonomy labeling, routed by the object's WordNet mapping so the
    VLM is only asked what mapping could not already answer (recap §6; saves
    compute vs. always asking all three axes):

      - UNMAPPED object (mapping is None) -> FULL VLM call: nature/biotic/material
        in one TaxonomyResponse, under the full three-definition system prompt.
      - MAPPED-nature object              -> MATERIAL-only VLM call: nature/biotic
        come from WordNet, so only material/immaterial is asked (MaterialResponse,
        material-definition-only system prompt).
      - MAPPED non-nature object          -> NO VLM call at all: nature=False from
        WordNet, biotic/material are n/a. A synthetic empty label is recorded.

    The image is always attached on every VLM call (material judgment needs the
    pixels). Returns, per image, a list of raw label dicts aligned with that
    image's object list; each carries `parse_failed` and `vlm_called` flags.
    Two batched `generate_batch` calls are issued (one per system prompt) so
    vLLM prefix-caches each stable system+image prefix across the batch.
    """
    # Pre-fill every slot; the two groups below overwrite their own entries and
    # mapped-non-nature objects keep the synthetic label set here.
    per_image: List[List[Dict[str, Any]]] = [
        [_empty_label(parse_failed=False, vlm_called=False) for _ in objs]
        for objs in objects_per_image
    ]

    # Flatten each group into parallel prompt/image/owner lists so an entire
    # group goes to the GPU in ONE batched call. `owner` is (image_idx, obj_idx)
    # so results can be scattered straight back into `per_image`.
    full_prompts: List[str] = []
    full_images: List[str] = []
    full_owner: List[tuple] = []
    mat_prompts: List[str] = []
    mat_images: List[str] = []
    mat_owner: List[tuple] = []

    for i, (objs, maps) in enumerate(zip(objects_per_image, mappings_per_image)):
        for j, (obj, mp) in enumerate(zip(objs, maps)):
            if mp is None:
                full_prompts.append(build_classification_prompt(obj, axes=_FULL_AXES))
                full_images.append(image_paths[i])
                full_owner.append((i, j))
            elif mp["is_nature"]:
                mat_prompts.append(build_classification_prompt(obj, axes=_MATERIAL_AXES))
                mat_images.append(image_paths[i])
                mat_owner.append((i, j))
            # else: mapped & not nature -> keep the synthetic (vlm_called=False) label.

    # --- FULL group (unmapped objects): all three axes ---
    if full_prompts:
        outs = vlm.generate_batch_safe(
            full_prompts, full_images,
            label="label_objects_batch/full",
            item_labels=[f"{image_paths[i]}#{j}" for i, j in full_owner],
            system_prompt=label_system_full, max_new_tokens=max_new_tokens,
            temperature=temperature, output_mode="structured", schema=TaxonomyResponse,
        )
        for (i, j), out in zip(full_owner, outs):
            if isinstance(out, dict):
                per_image[i][j] = {
                    "reasoning": _combine_taxonomy_reasoning(out), "nature": out.get("nature"),
                    "biotic": out.get("life_category"), "material": out.get("tangibility"),
                    "parse_failed": False, "vlm_called": True,
                }
            else:
                per_image[i][j] = _empty_label(parse_failed=True, vlm_called=True)

    # --- MATERIAL group (mapped-nature objects): material axis only ---
    if mat_prompts:
        outs = vlm.generate_batch_safe(
            mat_prompts, mat_images,
            label="label_objects_batch/material",
            item_labels=[f"{image_paths[i]}#{j}" for i, j in mat_owner],
            system_prompt=label_system_material, max_new_tokens=max_new_tokens,
            temperature=temperature, output_mode="structured", schema=MaterialResponse,
        )
        for (i, j), out in zip(mat_owner, outs):
            if isinstance(out, dict):
                # nature/biotic intentionally left None here — they come from the
                # WordNet mapping in resolve_hybrid_label, not from this call.
                rec = _empty_label(parse_failed=False, vlm_called=True)
                rec["reasoning"] = out.get("reasoning")
                rec["material"] = out.get("tangibility")
                per_image[i][j] = rec
            else:
                per_image[i][j] = _empty_label(parse_failed=True, vlm_called=True)

    return per_image


# =============================================================================
# Phase-1 inference driver (VLM only; no CLIP / no metric math)
# =============================================================================
def run_inference(
    vlm,
    dataset_instances: List[Dict[str, Any]],
    caption_system_prompt: Optional[str],
    label_system_full: Optional[str],
    label_system_material: Optional[str],
    tax_graph,
    mapping_vocab: Dict[str, str],
    extraction_system_prompt: Optional[str] = None,
    max_hops: int = 3,
    batch_size: int = 16,
    caption_max_new_tokens: int = 256,
    label_max_new_tokens: int = 300,
    temperature: float = 0.0,
    verbose: bool = False,
):
    """
    Run caption -> extraction -> (mapping + labeling) over every image and yield
    one record per image. Mapping now happens HERE (Phase 1), not in Phase-2
    scoring, so the VLM is only queried for what WordNet could not resolve
    (recap §6) — mapped non-nature objects skip the VLM entirely and
    mapped-nature objects are asked material only. Each record:

        {
          "image_path": str,
          "targets": [...],                 # GT carried through from the loader
          "caption": str,
          "objects": [str, ...],
          "object_labels": [ {reasoning,nature,biotic,material,parse_failed,vlm_called}, ... ],
          "object_finals": [ resolve_hybrid_label(...) dicts, ... ],
        }

    `object_labels[k]` and `object_finals[k]` both align with `objects[k]`.
    `object_finals` are the fully resolved hybrid labels (WordNet + VLM), so
    Phase-2 scoring can read them directly without re-mapping.

    `max_hops` bounds how far the WordNet search may wander when mapping an
    extracted object onto a labeled taxonomy node (0 = only directly-annotated
    nodes count; see map_object_to_taxonomy).

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
    progress = BatchProgress(num_batches, label="[infer] batch", verbose=verbose)

    for b in range(num_batches):
        # Slice out this batch's images. Python slicing handles the last,
        # possibly-shorter batch automatically (it just returns however many
        # elements remain, even if fewer than batch_size).
        chunk = dataset_instances[b * batch_size : (b + 1) * batch_size]
        image_paths = [inst["image_path"] for inst in chunk]

        # Stages 1-2: caption, then structured object extraction.
        captions = caption_batch(
            vlm, image_paths, caption_system_prompt,
            max_new_tokens=caption_max_new_tokens, temperature=temperature,
        )
        objects_per_image = extract_objects_batch(
            vlm, image_paths, captions, extraction_system_prompt,
            max_new_tokens=caption_max_new_tokens, temperature=temperature,
        )

        # Map every extracted object to the taxonomy FIRST (cheap dict/graph
        # lookup) so Stage-3 labeling can route each object to the minimal VLM
        # call it actually needs (or none). Computed once here and reused for
        # both the routing and the final hybrid resolution below.
        mappings_per_image = [
            [map_object_to_taxonomy(obj, tax_graph, mapping_vocab, max_hops=max_hops) for obj in objs]
            for objs in objects_per_image
        ]

        # Stage 3: mapping-aware labeling (full for unmapped, material-only for
        # mapped-nature, skipped for mapped non-nature).
        labels_per_image = label_objects_batch(
            vlm, image_paths, objects_per_image, mappings_per_image,
            label_system_full, label_system_material,
            max_new_tokens=label_max_new_tokens, temperature=temperature,
        )

        # `zip(...)` walks all lists together in lockstep — for each image in
        # this batch we have its original dataset entry (`inst`, carrying the
        # GT `targets`), caption, extracted objects, per-object mappings, and
        # per-object raw labels, all aligned by position. Resolve the final
        # hybrid label per object (reusing the precomputed mapping) and yield.
        for inst, cap, objs, maps, labels in zip(
            chunk, captions, objects_per_image, mappings_per_image, labels_per_image
        ):
            finals = [
                resolve_hybrid_label(obj, lab, tax_graph, mapping_vocab, mapping=mp)
                for obj, lab, mp in zip(objs, labels, maps)
            ]
            yield {
                "image_path": inst["image_path"],
                "targets": inst.get("targets", []),
                "caption": cap,
                "objects": objs,
                "object_labels": labels,
                "object_finals": finals,
            }

        if verbose:
            done = min((b + 1) * batch_size, n)
            n_objs_batch = sum(len(o) for o in objects_per_image)
            n_mapped_batch = sum(1 for maps in mappings_per_image for m in maps if m is not None)
            extra = (f"objects {n_objs_batch} (mapped {n_mapped_batch}/{n_objs_batch})"
                     if n_objs_batch else "objects 0")
            progress.tick(b, n_done=done, n_total=n, extra=extra)


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


def map_object_to_taxonomy(object_str: str, tax_graph, mapping_vocab: Dict[str, str], max_hops: int = 3) -> Optional[Dict[str, Any]]:
    """
    Map a free-text object phrase onto a WordNet synset using the AUTHORITATIVE
    dataset class->synset table (`mapping_vocab`, from
    dataset_loader.build_mapping_vocab) — NOT word-sense guessing. This avoids
    picking the wrong WordNet sense (e.g. "tiger" -> "a fierce person"): the
    synset comes from the dataset's own class mapping (tiger -> tiger.n.02).

    `max_hops` bounds how far tax_graph.resolve_labels may search outward from
    the matched synset for a labeled node: 0 accepts ONLY a directly-annotated
    node (an annotator labeled that exact synset), 1 also accepts a node one
    hop away (label inherited across a single hypernym/hyponym edge), etc.

    Returns {"synset", "is_nature", "biotic"} when the object's (normalized)
    phrase — or its trailing head noun — is a known class whose synset resolves
    to a labeled node within `max_hops`; else None (→ image-supported VLM
    fallback). `biotic` is True/False/None (None when nature-True but the node
    carries no biotic label).
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
        labels = tax_graph.resolve_labels(synset, max_hops=max_hops)
        if labels is not None:
            is_nature = labels["is_nature"]
            biotic = None
            if is_nature and labels.get("biotic_abiotic"):
                biotic = labels["biotic_abiotic"] == "biotic"
            return {"synset": synset, "is_nature": is_nature, "biotic": biotic}
    # Nothing matched, or matched but didn't resolve to a label.
    return None


def resolve_hybrid_label(object_str: str, vlm_label: Dict[str, Any], tax_graph, mapping_vocab: Dict[str, str], mapping: Any = _UNSET, max_hops: int = 3) -> Dict[str, Any]:
    """
    Combine WordNet mapping with the VLM's TaxonomyResponse into final per-object
    labels, recording the source of each axis (diagnostics):

      - nature  : WordNet when the object maps, else VLM.
      - biotic  : WordNet when the object maps AND the node carries a biotic
                  label; else VLM (still None/n-a when final nature is False).
      - material: ALWAYS the VLM.

    Returns final booleans (nature/biotic/material, None = n/a) plus
    *_source ∈ {"wordnet","vlm"} and the mapped synset (if any).

    `mapping` may be supplied precomputed (as run_inference does, having already
    mapped every object to route the VLM calls) to avoid mapping twice; if left
    unset it is computed here via map_object_to_taxonomy with `max_hops`.

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

    if mapping is _UNSET:
        mapping = map_object_to_taxonomy(object_str, tax_graph, mapping_vocab, max_hops=max_hops)

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

    # Enforce the schema rule: no downstream labels when the instance is not
    # CONFIRMED nature. Force biotic/material to None whenever final_nature is
    # anything other than True — covers both an explicit "no" AND an
    # unresolved/unparseable nature answer (final_nature is None) — regardless
    # of what the VLM might have said for biotic/material. This matters
    # because the schema does not mechanically ENFORCE "sub-axes must be
    # 'none' when nature is 'no'" — a model can violate that constraint (e.g.
    # answer nature="no" but life_category="biotic") — so we cannot trust a
    # stray biotic/material answer whenever nature itself isn't a confirmed
    # "yes"; scoring it as-is could let that answer accidentally match GT by
    # luck instead of being penalized like every other unconfirmed prediction.
    if final_nature is not True:
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
