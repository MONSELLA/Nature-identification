"""
src/evaluation/taxonomy_metrics.py

Taxonomy-aware evaluation metrics for the BIG-5 VLM pipeline — hierarchical
precision/recall/F1 (hP/hR/hF1), Wu-Palmer similarity, and free-text →
WordNet-synset resolution.

Ported/adapted from the reference `evaluation.py`
(`TaxonomyEvaluationPipeline._compute_hierarchical_metrics`,
`_get_ancestral_closure`, `_compute_wup_similarity`, `_resolve_to_wordnet`),
but rewritten as free functions that operate on the maintained
`src.loaders.excel_loader.TaxonomyGraph` (its `.graph` DiGraph and
`add_synset_and_ancestors`), so there is a single graph in the project rather
than a second copy.

Scope (per CLAUDE.md): these metrics run for ImageNet + Places ONLY (single-
label, closed candidate vocab). No spaCy noun-chunking is needed here — the
VLM pipeline already produces an explicit object list, so `resolve_to_wordnet`
consumes that list directly.

BACKGROUND — WHAT IS WORDNET, AND WHY DO WE CARE ABOUT "HIERARCHICAL" METRICS?
WordNet is a big lexical database of English nouns/verbs/etc. organized into a
tree-like hierarchy of concepts called "synsets" (e.g. "golden_retriever.n.01"
is a specific dog breed synset). Each synset has "hypernyms" — more general
parent concepts one level up (golden_retriever -> dog -> canine -> carnivore
-> ... -> entity). Walking upward through hypernyms from any synset eventually
reaches a single root concept ("entity.n.01").

A plain "did the model predict the EXACT right class" check (called an
exact-match / Contains check elsewhere in this project) is unforgiving: if the
ground truth is "golden_retriever" and the model predicts "dog", that's
technically wrong, but it's not a MEANINGLESS wrong answer — the model got the
right general idea, just at a coarser level of detail. Hierarchical
precision/recall/F1 (hP/hR/hF1, defined below) reward that kind of "close but
not exact" prediction with partial credit, by comparing how much of each
synset's full ancestor chain overlaps with the other's.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
from nltk.corpus import wordnet as wn


# Optional hand-curated synonyms for senses WordNet lemmatizes awkwardly.
# Mirrors the reference file's `custom_synonyms`; extend as needed.
# Example: WordNet's literal synsets for "boy"/"girl"/etc. don't map cleanly
# onto "person.n.01" the way we'd want for this project, so we hardcode the
# mapping here instead of relying on WordNet's own synonym lookup for these.
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
    to 1.0 to skip needless computation on known-identical inputs.

    Wu-Palmer similarity is a classic WordNet-based measure of "how close are
    these two concepts", based on how deep their common ancestor is in the
    WordNet tree relative to how deep each synset itself is. It returns a
    number between 0.0 (totally unrelated) and 1.0 (identical concept).
    `wn.synset(...).wup_similarity(...)` is NLTK's built-in implementation —
    we're just wrapping it with error-handling so a bad/unknown synset string
    never crashes the caller, it just scores as "not similar at all" (0.0).
    """
    if perfect_match:
        return 1.0
    try:
        # wn.synset(...) looks up a synset object from its string id, e.g.
        # "dog.n.01" -> the actual WordNet Synset object for that concept.
        s1 = wn.synset(synset_id_1)
        s2 = wn.synset(synset_id_2)
        score = s1.wup_similarity(s2)
        # wup_similarity can return None (e.g. if the two synsets share no
        # common ancestor at all) — treat that the same as zero similarity.
        return score if score is not None else 0.0
    except Exception:
        # Covers: either string isn't a valid WordNet synset id, or any other
        # unexpected NLTK error. Either way, we can't say they're similar.
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

    "Ancestral closure" just means: this synset itself, PLUS its parent, PLUS
    its parent's parent, ... all the way up to the WordNet root. For example
    the closure of "golden_retriever.n.01" includes golden_retriever, dog,
    canine, carnivore, placental, mammal, ... entity. We use this set (rather
    than just comparing two single synset strings) so that comparing two
    DIFFERENT-but-related synsets can still show a lot of overlap.
    """
    graph = tax_graph.graph
    if synset_str not in graph:
        # This synset hasn't been added to our taxonomy graph yet (maybe the
        # VLM predicted some object we've never seen before). Before adding
        # it, first confirm WordNet actually recognizes this string as a real
        # synset — if `wn.synset(...)` raises, we bail out with an empty set
        # rather than polluting the graph with garbage.
        try:
            wn.synset(synset_str)  # validate before inserting
            tax_graph.add_synset_and_ancestors(synset_str)
        except Exception:
            return set()
    if synset_str not in graph:
        # Defensive double-check: even after trying to add it above, if for
        # some reason it's still not in the graph, there's nothing more we can
        # do — return an empty closure so downstream code treats this as "no
        # overlap possible" instead of crashing.
        return set()

    # networkx's `ancestors()` walks every directed edge that points INTO this
    # node, transitively, and returns the full set of nodes reachable that
    # way. Because our graph's edges point parent -> child (see
    # excel_loader.py's add_synset_and_ancestors), "ancestors of X" correctly
    # means "every more-general concept above X in the hierarchy".
    ancestors = nx.ancestors(graph, synset_str)
    ancestors.add(synset_str)  # the closure includes the synset itself, too
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

    Intuition for the formulas above: think of each closure (the GT's full
    ancestor chain, and the prediction's full ancestor chain) as a "bag of
    concepts". `intersection` is how many of those concepts they share.
      - hP ("hierarchical precision") asks: out of everything the PREDICTION
        claims (its whole ancestor chain), how much of that is also true of
        the GT? If you predict something very generic (e.g. "entity"), your
        closure is huge but mostly irrelevant, so hP drops.
      - hR ("hierarchical recall") asks the reverse: out of everything that's
        actually true about the GT (its whole ancestor chain), how much did
        your prediction also capture? If you predict something too SPECIFIC
        or in a totally different branch, you miss most of the GT's ancestor
        chain, so hR drops.
      - hF1 is just the standard harmonic mean of hP and hR (same shape as the
        familiar precision/recall F1 you'd compute in `sklearn`), giving one
        combined number that penalizes being either too vague or too narrow.
    """
    if perfect_match:
        # Caller already knows this is an exact match (e.g. predicted synset
        # string == GT synset string) — skip the set-math and return 1.0/1.0/1.0.
        return {"hp": 1.0, "hr": 1.0, "hf1": 1.0}
    if y_pred_synset is None:
        # No prediction at all (e.g. the model extracted zero objects, or
        # nothing mapped to any candidate class) — this is scored as a total
        # miss, not skipped/excluded (see CLAUDE.md's "prediction-unmapped
        # penalized as wrong" convention).
        return {"hp": 0.0, "hr": 0.0, "hf1": 0.0}

    closure_true = get_ancestral_closure(tax_graph, y_true_synset)
    closure_pred = get_ancestral_closure(tax_graph, y_pred_synset)
    if not closure_true or not closure_pred:
        # Either synset string couldn't be resolved by WordNet at all —
        # can't compute any meaningful overlap.
        return {"hp": 0.0, "hr": 0.0, "hf1": 0.0}

    # Set intersection (`&`) — the concepts common to BOTH ancestor chains.
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

    WHY DOES THIS FUNCTION EXIST? ClipMatch (see clip_metrics.py) tells us
    WHICH candidate class the model's extracted objects best match overall,
    but for hP/hR we need an actual WordNet synset to compare against the GT's
    ancestor chain. This function picks the single BEST extracted-object
    phrase to represent that match and converts it into a synset id.
    """
    # Pair up each object phrase with its score, sort by score descending
    # (best/most-confident match first), and drop anything below `threshold`.
    # `zip(candidate_scores, candidate_strings)` pairs them positionally, e.g.
    # scores=[0.9, 0.2], strings=["dog","car"] -> [(0.9,"dog"), (0.2,"car")].
    sorted_candidates = [
        string
        for score, string in sorted(zip(candidate_scores, candidate_strings), key=lambda p: p[0], reverse=True)
        if score > threshold
    ]

    # Walk candidates from most to least confident, and take the FIRST one
    # that we can successfully turn into a WordNet synset. Less confident
    # candidates are only reached if earlier ones fail to resolve at all.
    for candidate in sorted_candidates:
        # WordNet's dictionary keys use underscores instead of spaces, e.g.
        # the phrase "polar bear" is looked up as "polar_bear".
        key = candidate.strip().replace(" ", "_")
        # Custom synonym expansion: if this string is a known alias of the
        # predicted class, accept the predicted class directly.
        if pred_synset_id in CUSTOM_SYNONYMS and candidate.strip().lower() in CUSTOM_SYNONYMS[pred_synset_id]:
            return pred_synset_id
        try:
            # `wn.synsets(key, pos="n")` returns EVERY noun sense WordNet knows
            # for this word/phrase — a word like "bank" has multiple unrelated
            # meanings (river bank vs. financial bank), each its own synset.
            synsets = wn.synsets(key, pos="n")
        except Exception:
            continue
        if not synsets:
            # This phrase isn't a recognized WordNet noun at all — try the
            # next-best candidate instead of giving up entirely.
            continue
        if len(synsets) == 1:
            # No ambiguity — there's only one possible sense, so use it.
            return synsets[0].name()
        # Polysemy: pick the sense closest to the predicted class by Wu-Palmer.
        # `max(..., key=...)` scores every candidate sense's similarity to the
        # already-known predicted class and keeps whichever sense scores
        # highest — i.e. whichever meaning of this ambiguous word makes most
        # sense given what we already believe the image is about.
        best = max(synsets, key=lambda s: compute_wup_similarity(s.name(), pred_synset_id))
        return best.name()

    # None of the candidate phrases mapped to any WordNet noun sense at all.
    return None
