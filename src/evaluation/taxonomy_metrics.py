"""
lib/taxonomy_metrics.py

Taxonomy-aware evaluation metrics for the BIG-5 VLM pipeline — hierarchical
precision/recall/F1 (hP/hR/hF1), Wu-Palmer similarity, and free-text →
WordNet-synset resolution.

Ported/adapted from the reference `evaluation.py`
(`TaxonomyEvaluationPipeline._compute_hierarchical_metrics`,
`_get_ancestral_closure`, `_compute_wup_similarity`, `_resolve_to_wordnet`),
but rewritten as free functions that operate on the maintained
`lib.excel_loader.TaxonomyGraph` (its `.graph` DiGraph and
`add_synset_and_ancestors`), so there is a single graph in the project rather
than a second copy.

Scope (per CLAUDE.md): these metrics run for ImageNet + Places ONLY (single-
label, closed candidate vocab). No spaCy noun-chunking is needed here — the
VLM pipeline already produces an explicit object list, so `resolve_to_wordnet`
consumes that list directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
from nltk.corpus import wordnet as wn


# Optional hand-curated synonyms for senses WordNet lemmatizes awkwardly.
# Mirrors the reference file's `custom_synonyms`; extend as needed.
CUSTOM_SYNONYMS = {
    "person.n.01": ["boy", "girl", "man", "woman", "kid", "child", "guy", "lady"],
}


# =============================================================================
# Wu-Palmer similarity
# =============================================================================
def compute_wup_similarity(
    synset_id_1: Optional[str] = None,
    synset_id_2: Optional[str] = None,
    perfect_match: bool = False,
) -> float:
    """Wu-Palmer similarity between two synset id strings. Returns 0.0 on any
    error (invalid string, no shared root). `perfect_match=True` short-circuits
    to 1.0 to skip needless computation on known-identical inputs."""
    if perfect_match:
        return 1.0
    try:
        s1 = wn.synset(synset_id_1)
        s2 = wn.synset(synset_id_2)
        score = s1.wup_similarity(s2)
        return score if score is not None else 0.0
    except Exception:
        return 0.0


# =============================================================================
# Ancestral closure + hierarchical precision/recall/F1
# =============================================================================
def get_ancestral_closure(tax_graph, synset_str: str) -> set:
    """
    Return the set containing `synset_str` and every directional ancestor up to
    the root, using `tax_graph.graph`. If the synset is a valid WordNet concept
    not yet in the graph, it is added in-place (caching its full ancestral path)
    before the closure is computed. Returns an empty set if it cannot be
    resolved by WordNet at all.
    """
    graph = tax_graph.graph
    if synset_str not in graph:
        try:
            wn.synset(synset_str)  # validate before inserting
            tax_graph.add_synset_and_ancestors(synset_str)
        except Exception:
            return set()
    if synset_str not in graph:
        return set()

    ancestors = nx.ancestors(graph, synset_str)
    ancestors.add(synset_str)
    return ancestors


def compute_hierarchical_metrics(
    tax_graph,
    y_true_synset: Optional[str] = None,
    y_pred_synset: Optional[str] = None,
    perfect_match: bool = False,
) -> Dict[str, float]:
    """
    Hierarchical precision/recall/F1 between a GT synset and a predicted synset,
    measured as the overlap of their ancestral closures (Kosmopoulos et al. /
    Snæbjarnarson et al. style):

        hP = |anc(true) ∩ anc(pred)| / |anc(pred)|
        hR = |anc(true) ∩ anc(pred)| / |anc(true)|
        hF1 = 2·hP·hR / (hP + hR)

    Gives graded credit for predicting a correct ancestor/relative (e.g. "canine"
    when GT is "golden retriever") that a strict exact-match check would miss.
    Returns all zeros when the prediction is None (a mapping failure) or when
    either closure is empty. `perfect_match=True` short-circuits to all-ones.
    """
    if perfect_match:
        return {"hp": 1.0, "hr": 1.0, "hf1": 1.0}
    if y_pred_synset is None:
        return {"hp": 0.0, "hr": 0.0, "hf1": 0.0}

    closure_true = get_ancestral_closure(tax_graph, y_true_synset)
    closure_pred = get_ancestral_closure(tax_graph, y_pred_synset)
    if not closure_true or not closure_pred:
        return {"hp": 0.0, "hr": 0.0, "hf1": 0.0}

    intersection = closure_true & closure_pred
    hp = len(intersection) / len(closure_pred)
    hr = len(intersection) / len(closure_true)
    hf1 = (2 * hp * hr) / (hp + hr) if (hp + hr) > 0 else 0.0
    return {"hp": hp, "hr": hr, "hf1": hf1}


# =============================================================================
# Free-text object → WordNet synset resolution
# =============================================================================
def resolve_to_wordnet(
    candidate_scores: List[float],
    pred_synset_id: str,
    candidate_strings: List[str],
    threshold: float = 0.0,
) -> Optional[str]:
    """
    Map a list of free-text object phrases to a single canonical WordNet synset
    id, used to turn the VLM's extracted-object list into the predicted node for
    hP/hR. Candidates are considered in descending `candidate_scores` order
    (e.g. each object's CLIP similarity to the ClipMatch-predicted class), only
    those above `threshold`. The first candidate that maps to WordNet wins;
    polysemous candidates are disambiguated by maximizing Wu-Palmer similarity
    against `pred_synset_id` (the ClipMatch-predicted class). Returns None if no
    candidate maps to the WordNet vocabulary.

    Object phrases are expected space-joined (e.g. "polar bear"); they are
    converted to WordNet's underscore form ("polar_bear") for lookup.
    """
    sorted_candidates = [
        string
        for score, string in sorted(zip(candidate_scores, candidate_strings), key=lambda p: p[0], reverse=True)
        if score > threshold
    ]

    for candidate in sorted_candidates:
        key = candidate.strip().replace(" ", "_")
        # Custom synonym expansion: if this string is a known alias of the
        # predicted class, accept the predicted class directly.
        if pred_synset_id in CUSTOM_SYNONYMS and candidate.strip().lower() in CUSTOM_SYNONYMS[pred_synset_id]:
            return pred_synset_id
        try:
            synsets = wn.synsets(key, pos="n")
        except Exception:
            continue
        if not synsets:
            continue
        if len(synsets) == 1:
            return synsets[0].name()
        # Polysemy: pick the sense closest to the predicted class by Wu-Palmer.
        best = max(synsets, key=lambda s: compute_wup_similarity(s.name(), pred_synset_id))
        return best.name()

    return None
