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

    KNOWN NLTK QUIRK (confirmed empirically, not a bug in this wrapper): a
    synset compared against ITSELF is not always exactly 1.0. NLTK's
    wup_similarity relies on `min_depth()`, which is not internally
    consistent across the two call sides when a synset has MULTIPLE distinct
    hypernym paths of different lengths (multiple inheritance — common in
    WordNet). Example: `wn.synset("dog.n.01").wup_similarity(wn.synset("dog.n.01"))`
    returns ~0.929, not 1.0, because "dog.n.01" has both a 9-edge path (via
    domestic_animal) and a 14-edge path (via canine/carnivore/mammal) up to
    entity.n.01. This is expected, well-known NLTK behavior for
    multiply-inheriting synsets, not a project bug — don't "fix" a
    less-than-1.0 self-similarity you see in results without checking for
    this first. hP/hR/hF1 (compute_hierarchical_metrics, below) are unaffected
    since they're plain ancestral-closure set overlap, not this NLTK formula.
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
def _synsets_for_phrase(phrase: str) -> List["wn.synset"]:
    """WordNet noun synsets for one space-joined phrase, or [] if unrecognized
    (WordNet keys use underscores, e.g. "polar bear" -> "polar_bear")."""
    key = phrase.strip().replace(" ", "_")
    if not key:
        return []
    try:
        return wn.synsets(key, pos="n")
    except Exception:
        return []


def _best_sense(synsets: List["wn.synset"], pred_synset_id: str) -> "wn.synset":
    """Disambiguate polysemous synsets by maximizing Wu-Palmer similarity
    against `pred_synset_id` (the ClipMatch-predicted class)."""
    if len(synsets) == 1:
        return synsets[0]
    return max(synsets, key=lambda s: compute_wup_similarity(s.name(), pred_synset_id))


def resolve_to_wordnet(
    candidate_scores: List[float],
    pred_synset_id: str,
    candidate_strings: List[str],
    threshold: float = 0.0,
) -> Optional[str]:
    """
    Map the SINGLE best-scoring free-text object phrase to a canonical WordNet
    synset id, used to turn the VLM's extracted-object list into the predicted
    node for hP/hR. Only the highest-scoring candidate above `threshold` is
    tried directly (no falling through to the next-best candidate on failure —
    the caller passes one candidate in practice; see run_vlm_pipeline.py).

    If that top candidate is a COMPOUND NOUN (multiple space-separated words)
    and does not map to WordNet as a whole phrase (e.g. "bear rug" has no
    synset), each of its constituent words is tried individually as a fallback
    (e.g. "bear", "rug") — compound nouns are frequently absent from WordNet
    even when their parts are common nouns. Among whichever constituent words
    DO resolve, the one whose (polysemy-disambiguated) synset maximizes
    Wu-Palmer similarity to `pred_synset_id` is kept. Returns None if neither
    the full phrase nor any constituent word maps to the WordNet vocabulary.

    WHY DOES THIS FUNCTION EXIST? ClipMatch (see clip_metrics.py) tells us
    WHICH candidate class the model's extracted objects best match overall,
    but for hP/hR we need an actual WordNet synset to compare against the GT's
    ancestor chain. This function picks the single BEST extracted-object
    phrase to represent that match and converts it into a synset id.
    """
    if not candidate_strings:
        return None
    # Take only the single highest-scoring candidate (ties broken by whichever
    # max() finds first). `zip` pairs scores/strings positionally.
    best_score, top_candidate = max(zip(candidate_scores, candidate_strings), key=lambda p: p[0])
    if best_score <= threshold:
        return None

    # Custom synonym expansion: if this string is a known alias of the
    # predicted class, accept the predicted class directly.
    if pred_synset_id in CUSTOM_SYNONYMS and top_candidate.strip().lower() in CUSTOM_SYNONYMS[pred_synset_id]:
        return pred_synset_id

    synsets = _synsets_for_phrase(top_candidate)
    if synsets:
        return _best_sense(synsets, pred_synset_id).name()

    # The full phrase didn't resolve. Only fall back to per-word truncations
    # if it's actually a compound (multi-word) — a single unmapped word has
    # no smaller unit to try.
    words = top_candidate.strip().split()
    if len(words) <= 1:
        return None

    resolved: List["wn.synset"] = []
    for word in words:
        word_synsets = _synsets_for_phrase(word)
        if word_synsets:
            resolved.append(_best_sense(word_synsets, pred_synset_id))
    if not resolved:
        return None
    # Among the constituent words that DID map, keep whichever synset
    # maximizes Wu-Palmer similarity to the predicted class.
    return max(resolved, key=lambda s: compute_wup_similarity(s.name(), pred_synset_id)).name()
