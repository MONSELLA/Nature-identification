"""
Shared building blocks for REJECTION-SAMPLING FINE-TUNING (RFT) of the VLM
pipeline's language decoder on the BIG-5 datasets.

WHAT THIS IS FOR
================
BIG-5 carries only IMAGE-LEVEL annotations (one nature/biotic/material verdict
per image), so there is no per-entity supervision to fine-tune the pipeline's
extraction/labeling calls against directly. Rejection sampling turns the
image-level annotation into per-call supervision the cheap way: run the
pipeline, keep only the images whose FINAL image-level prediction agrees with
the human annotation, and treat every VLM call that produced that image's
answer as a correct demonstration to train on.

The model's own greedy outputs are already on disk — one JSON-Lines artifact
per (dataset, model) under results/vlm_pipeline/... — so no new inference is
needed to build a training set. This module reads those artifacts back and
reconstructs, for every accepted image, the EXACT prompt/response pairs that
inference produced.

WHY RECONSTRUCTION IS EXACT AND NOT APPROXIMATE
==============================================
Every VLM call the pipeline makes is a pure function of data the artifact
already stores, so nothing has to be guessed:

  - extraction : prompts.EXTRACTION_PROMPT.format(caption=<record caption>),
                 under the nature-definition system prompt. The response is
                 ObjectExtractionResponse(reasoning=<extraction_reasoning>,
                 objects=<objects>).
  - label_full : prompts.build_classification_prompt(<object>, _FULL_AXES),
                 under the three-definition system prompt. The response is a
                 TaxonomyResponse.
  - label_material : prompts.build_material_classification_prompt(<object>,
                 biotic=<the mapped node's own biotic verdict>), under the
                 material-definition system prompt. The response is a
                 MaterialResponse.

Both prompt builders are IMPORTED from src.models.prompts rather than copied,
so a prompt change can never silently desynchronize the training data from
what inference actually sends. The material call's `biotic` argument is read
off `object_finals[k]["final_biotic"]`, which on the `mapped_nature_material`
route IS the mapping's own verdict verbatim (see vlm_pipeline.resolve_hybrid_
label — that branch assigns final_biotic = mapping["biotic"] and the VLM is
never asked about biotic there).

THE ONE RECONSTRUCTION SUBTLETY: TaxonomyResponse's SPLIT REASONING
==================================================================
TaxonomyResponse asks for TWO reasoning fields (`nature_reasoning`, then
`sub_axes_reasoning`), but vlm_pipeline._combine_taxonomy_reasoning joins them
with a single space before storing, so `object_labels[k]["reasoning"]` is one
flat string. Training needs them separated again to rebuild the assistant's
literal JSON.

`split_taxonomy_reasoning` below recovers the boundary from the model's own
highly regular phrasing, and — this is the part that makes it safe rather than
a guess — it NEVER guesses: it either finds a known Step-2 opener or returns
None, and the dataset builder DROPS an example it cannot split rather than
inventing a plausible boundary. Measured on the gemma-4-12B-it BIG-5 artifacts
this recovers 27213/27213 full-call labels (100%): every nature="no" answer
opens Step 2 with "Not applicable…" and every nature="yes" answer with "Since
nature is 'yes'…". A different model, or a re-run after a prompt change, may
split at a lower rate — which the builder reports rather than hiding.

Runs of vlm_pipeline.py made AFTER this module existed also store
`nature_reasoning`/`sub_axes_reasoning` as their own artifact fields, so this
recovery step is a back-compatibility path for the artifacts that already
exist, not a permanent dependency.

DISTILLATION (FUTURE, ALREADY SUPPORTED)
========================================
Nothing here assumes the artifact was produced by the model being trained.
Point the builder at a heavier model's artifact and the same code emits the
same example shape — the prompts are rebuilt from the artifact's own caption
and object list, so a teacher's caption correctly conditions the teacher's
extraction target. The student model id is a separate flag on the trainer.
The only thing that changes is that the examples become off-policy, which is
recorded per example in `source_model`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Make `src` importable when this file is run directly as a script from
# anywhere (fine_tuning/ is one level below the repo root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.prompts import (  # noqa: E402
    EXTRACTION_PROMPT,
    build_classification_prompt,
    build_material_classification_prompt,
)
from src.vlm_pipeline import _FULL_AXES  # noqa: E402


# =============================================================================
# Artifact reading
# =============================================================================
# A vlm_responses_<model>.jsonl artifact is not homogeneous: its FIRST line is
# a header record (dataset/model/max_hops provenance) and its LAST is a footer
# (inference_time_seconds). Both are identified by carrying `record_type` and
# no `image_path`. Filtering on the presence of "image_path" rather than on
# record_type == "header" specifically is deliberate — it stays correct if a
# future run adds another non-image record type.
def read_artifact(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (header, [image records]) for one vlm_responses_*.jsonl file."""
    header: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "image_path" in rec:
                records.append(rec)
            elif rec.get("record_type") == "header":
                header = rec
    return header, records


# Every BIG-5 image on disk is named "<platform_id>_<slot>.<ext>" — one flat
# folder per platform, several images per social-media POST (see
# dataset_loader.load_big5). The post id is what splits must be grouped by:
# images from one post are near-duplicates of each other far more often than
# two random images are (same photoshoot, same meme template, same event), so
# splitting at image level leaks training content into the test set.
_IMAGE_STEM_RE = re.compile(r"^(?P<post>.+)_(?P<slot>\d+)$")


def post_id_of(image_path: str) -> str:
    """The social-media post an image belongs to, derived from its filename.

    Falls back to the whole stem when the name doesn't match the expected
    "<post>_<slot>" shape, which degrades to image-level grouping for that one
    file rather than silently grouping unrelated images together.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    m = _IMAGE_STEM_RE.match(stem)
    return m.group("post") if m else stem


def group_key(image_path: str, platform: str) -> str:
    """Split-grouping key. Namespaced by platform because Twitter and Weibo id
    spaces are unrelated and could in principle collide."""
    return f"{platform}/{post_id_of(image_path)}"


# =============================================================================
# The acceptance rule (what "a correct prediction" means for rejection sampling)
# =============================================================================
# Two rules, both scoring the SAME thing the evaluation scores — an image-level
# verdict against an image-level annotation — and differing only in whether the
# biotic and material hits have to come from the SAME entity.
#
#   "strict"  (default): at least one predicted nature entity carries BOTH a
#             life-category and a tangibility matching the GT. One entity is
#             right about the whole scene.
#   "lenient" : the rule scripts/run_vlm_pipeline.py's BIG-5 branch already
#             uses for reporting — each axis is satisfied independently, so a
#             "dog" entity can supply the biotic hit while a separate "painting
#             of a mountain" entity supplies the immaterial one.
#
# Measured on the gemma-4-12B-it BIG-5 artifacts the two barely differ (3397 vs
# 3366 of 3634 nature images accepted). "strict" is the default because a
# training demonstration should be one coherent judgment, not two half-correct
# ones that happen to add up; "lenient" exists so the training filter can be
# made to agree exactly with the reported metric when that is what you want.
#
# NON-NATURE images are scored identically under both rules: accepted iff the
# pipeline predicted NO nature entity at all. There is no "found an explicit
# non-nature entity" signal to require, exactly as in the evaluation.
#
# CODER DISAGREEMENT: load_big5 stores gt_biotic/gt_material as LISTS, which
# hold BOTH values when the human coders split (e.g. "material; immaterial").
# Per the project convention such an image genuinely counts as both labels, so
# an entity matching EITHER direction satisfies that axis here.
ACCEPT_RULES = ("strict", "lenient")


@dataclass
class Verdict:
    """Why one image was accepted or rejected — kept as a value rather than a
    bare bool so the builder can report a rejection breakdown instead of only
    a total."""
    accepted: bool
    reason: str
    gt_nature: Optional[bool]
    n_nature_entities: int


def image_verdict(record: Dict[str, Any], rule: str = "strict") -> Verdict:
    """Apply the acceptance rule to one artifact image record."""
    if rule not in ACCEPT_RULES:
        raise ValueError(f"Unknown acceptance rule {rule!r}; expected one of {ACCEPT_RULES}")

    targets = record.get("targets") or []
    target = targets[0] if targets else {}
    gt_nature = target.get("gt_nature")
    finals = record.get("object_finals") or []
    nature_entities = [e for e in finals if e.get("final_nature") is True]
    n_ent = len(nature_entities)

    if gt_nature is None:
        return Verdict(False, "gt_nature_missing", None, n_ent)

    if gt_nature is False:
        ok = n_ent == 0
        return Verdict(ok, "ok" if ok else "predicted_nature_on_non_nature_image", False, n_ent)

    # GT is nature from here on.
    if n_ent == 0:
        return Verdict(False, "no_nature_entity_on_nature_image", True, 0)

    gt_biotic = [bool(v) for v in (target.get("gt_biotic") or [])]
    gt_material = [bool(v) for v in (target.get("gt_material") or [])]

    if rule == "strict":
        for e in nature_entities:
            bio_ok = (not gt_biotic) or (e.get("final_biotic") in gt_biotic)
            mat_ok = (not gt_material) or (e.get("final_material") in gt_material)
            if bio_ok and mat_ok:
                return Verdict(True, "ok", True, n_ent)
        return Verdict(False, "no_entity_matches_both_axes", True, n_ent)

    # lenient: each GT direction must be covered by SOME nature entity.
    def has(axis: str, value: bool) -> bool:
        return any(e.get(axis) is value for e in nature_entities)

    if not all(has("final_biotic", b) for b in gt_biotic):
        return Verdict(False, "life_category_axis_unmatched", True, n_ent)
    if not all(has("final_material", m) for m in gt_material):
        return Verdict(False, "tangibility_axis_unmatched", True, n_ent)
    return Verdict(True, "ok", True, n_ent)


# =============================================================================
# TaxonomyResponse reasoning recovery (see the module docstring)
# =============================================================================
# Openers the model uses to begin `sub_axes_reasoning`. Anchored to the start
# of the recovered Step-2 text and matched case-sensitively where the model is
# consistent; the list is ordered most-specific-first. A label whose combined
# reasoning matches NONE of these is reported unsplittable and dropped — never
# split at a guessed midpoint.
_SUB_AXES_OPENERS = (
    # nature="no": the schema literally instructs this wording.
    r"Not applicable\b",
    # nature="yes": observed verbatim across every accepted sample.
    r"Since nature is ['\"]?yes['\"]?[,.]?",
    r"Since the entity is nature\b",
    r"Because nature is ['\"]?yes['\"]?[,.]?",
    r"Step 2[:.]",
)
_SUB_AXES_RE = re.compile(r"(?P<opener>" + "|".join(_SUB_AXES_OPENERS) + r")")


def split_taxonomy_reasoning(combined: Optional[str]) -> Optional[Tuple[str, str]]:
    """Recover (nature_reasoning, sub_axes_reasoning) from the space-joined
    string vlm_pipeline._combine_taxonomy_reasoning produced.

    Returns None when no known Step-2 opener is present, so the caller can drop
    the example instead of training on a fabricated split. Uses the LAST match
    rather than the first: the Step-1 text can legitimately mention a phrase
    like "not applicable" mid-sentence, whereas the real boundary is the final
    place Step 2 could begin.
    """
    if not combined:
        return None
    matches = list(_SUB_AXES_RE.finditer(combined))
    if not matches:
        return None
    cut = matches[-1].start()
    nature_part = combined[:cut].strip()
    sub_part = combined[cut:].strip()
    if not nature_part or not sub_part:
        return None
    return nature_part, sub_part


def taxonomy_reasoning_fields(label: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The two reasoning fields for one full-call label, preferring the raw
    fields when the artifact stores them (runs made after this module landed)
    and falling back to recovering them from the joined string otherwise."""
    raw_nature = label.get("nature_reasoning")
    raw_sub = label.get("sub_axes_reasoning")
    if raw_nature and raw_sub:
        return str(raw_nature), str(raw_sub)
    return split_taxonomy_reasoning(label.get("reasoning"))


# =============================================================================
# Training examples
# =============================================================================
# Which system prompt an example needs, stored as a KEY rather than the prompt
# text itself. The definition files run to several thousand characters and
# would be repeated in every one of ~20k examples, bloating the dataset file
# for no benefit; the trainer resolves the key once via
# prompts.build_system_prompts and reuses one interned string, which also keeps
# the tokenized prefix identical across examples.
SYSTEM_KEYS = {
    "extraction": "nature",       # nature definition only
    "label_full": "full",         # all three definitions
    "label_material": "material", # material definition only
}

STAGES = tuple(SYSTEM_KEYS)


@dataclass
class Example:
    """One (system, user, image) -> assistant training pair."""
    image_path: str
    platform: str
    group: str
    stage: str
    system_key: str
    prompt: str
    target: str
    gt_nature: Optional[bool]
    source_model: str
    object: Optional[str] = None
    object_index: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class BuildStats:
    """Per-image and per-example counters the builder reports at the end.
    Deliberately granular: an example that silently vanishes (unparsed
    extraction, unsplittable reasoning) is otherwise invisible."""
    counts: Dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


def examples_for_record(
    record: Dict[str, Any],
    platform: str,
    source_model: str,
    stages: Tuple[str, ...],
    stats: BuildStats,
) -> List[Example]:
    """Rebuild every training example one ACCEPTED image contributes.

    The caller is responsible for having applied `image_verdict` — this
    function does not re-check acceptance, it only reconstructs calls.
    """
    out: List[Example] = []
    image_path = record["image_path"]
    grp = group_key(image_path, platform)
    targets = record.get("targets") or []
    gt_nature = (targets[0] if targets else {}).get("gt_nature")

    common = dict(image_path=image_path, platform=platform, group=grp,
                  gt_nature=gt_nature, source_model=source_model)

    # --- Stage 2: object extraction -----------------------------------------
    if "extraction" in stages:
        if record.get("extraction_parse_failed"):
            # The structured output never parsed, so `objects` is an empty
            # fallback rather than a real answer — there is no correct
            # demonstration here to train on, even on an accepted image.
            stats.bump("extraction_skipped_parse_failed")
        elif record.get("extraction_reasoning") is None:
            # Artifact predates the reasoning field, or the schema had none.
            # Training on a target missing a schema-required field would teach
            # the model to omit it.
            stats.bump("extraction_skipped_no_reasoning")
        else:
            caption = record.get("caption") or ""
            target = json.dumps(
                {"reasoning": record["extraction_reasoning"],
                 "objects": record.get("objects") or []},
                ensure_ascii=False,
            )
            out.append(Example(
                stage="extraction", system_key=SYSTEM_KEYS["extraction"],
                prompt=EXTRACTION_PROMPT.format(caption=caption), target=target,
                **common,
            ))
            stats.bump("examples_extraction")

    # --- Stage 3: per-object taxonomy labeling ------------------------------
    want_full = "label_full" in stages
    want_material = "label_material" in stages
    if want_full or want_material:
        objects = record.get("objects") or []
        labels = record.get("object_labels") or []
        finals = record.get("object_finals") or []
        for idx, obj in enumerate(objects):
            label = labels[idx] if idx < len(labels) else {}
            final = finals[idx] if idx < len(finals) else {}
            if not label.get("vlm_called"):
                # Human-term exclusion — no call was made, so there is no
                # response to imitate.
                stats.bump("label_skipped_no_vlm_call")
                continue
            if label.get("parse_failed"):
                stats.bump("label_skipped_parse_failed")
                continue

            # Which of the two labeling calls this object took. `label_route`
            # is authoritative; fall back to the shape of the stored label for
            # artifacts written before that field existed (a material-only
            # call leaves nature/biotic None and fills material).
            route = final.get("label_route")
            if route is None:
                route = ("mapped_nature_material"
                         if label.get("nature") is None and label.get("material") is not None
                         else "vlm_full")

            if route == "mapped_nature_material":
                if not want_material:
                    continue
                if label.get("reasoning") is None or label.get("material") is None:
                    stats.bump("label_material_skipped_incomplete")
                    continue
                target = json.dumps(
                    {"reasoning": label["reasoning"], "tangibility": label["material"]},
                    ensure_ascii=False,
                )
                # `final_biotic` on this route IS the mapping's own verdict —
                # see resolve_hybrid_label's mapped-nature branch — which is
                # exactly what inference passed to this prompt builder.
                prompt = build_material_classification_prompt(obj, biotic=final.get("final_biotic"))
                out.append(Example(stage="label_material", system_key=SYSTEM_KEYS["label_material"],
                                   prompt=prompt, target=target, object=obj, object_index=idx,
                                   **common))
                stats.bump("examples_label_material")
            else:
                if not want_full:
                    continue
                if label.get("nature") is None:
                    stats.bump("label_full_skipped_incomplete")
                    continue
                # TaxonomyResponse's own schema instruction is "ALL nature
                # entities MUST be classified as either 'biotic' or 'abiotic'
                # [...]. Non-nature entities MUST be 'none'" — but the schema
                # does not MECHANICALLY enforce that (see
                # vlm_pipeline.resolve_hybrid_label's docstring, which already
                # forces final_biotic/final_material to None whenever
                # final_nature isn't True for exactly this reason). A model
                # CAN violate it — measured on the real gemma-4-12B-it BIG-5
                # artifacts: 16/27213 full-call labels say nature="yes" but
                # leave life_category/tangibility as "none" (0 occurrences of
                # the mirror case, nature="no" with sub-axes set, but checked
                # for anyway). Training on that verbatim would teach the
                # model to reproduce a self-contradictory answer — drop it,
                # same "never fabricate" policy as the unsplittable-reasoning
                # case below, rather than silently accepting whatever the
                # model happened to say.
                biotic = label.get("biotic") or "none"
                material = label.get("material") or "none"
                nature_says_yes = label["nature"] == "yes"
                subaxes_are_set = biotic != "none" or material != "none"
                if nature_says_yes != subaxes_are_set:
                    stats.bump("label_full_skipped_inconsistent_subaxes")
                    continue
                parts = taxonomy_reasoning_fields(label)
                if parts is None:
                    # Cannot recover the two-field split — drop rather than
                    # fabricate a boundary (see the module docstring).
                    stats.bump("label_full_skipped_unsplittable_reasoning")
                    continue
                nature_reasoning, sub_axes_reasoning = parts
                # Key order MUST match TaxonomyResponse's field order: the
                # schema is a chain of thought, and training the model to emit
                # the verdict before its justification would defeat it.
                target = json.dumps({
                    "nature_reasoning": nature_reasoning,
                    "nature": label["nature"],
                    "sub_axes_reasoning": sub_axes_reasoning,
                    "life_category": biotic,
                    "tangibility": material,
                }, ensure_ascii=False)
                out.append(Example(stage="label_full", system_key=SYSTEM_KEYS["label_full"],
                                   prompt=build_classification_prompt(obj, axes=_FULL_AXES),
                                   target=target, object=obj, object_index=idx, **common))
                stats.bump("examples_label_full")

    return out


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream a JSON-Lines file (splits/dataset files written by this package)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
