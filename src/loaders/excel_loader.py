"""
src/loaders/excel_loader.py

Loads the BIG-5 WordNet-based taxonomy Excel file and builds a queryable
taxonomy graph. The graph-construction logic (add_synset_and_ancestors /
loading annotations row-by-row) is ported directly from
`TaxonomyEvaluationPipeline` in evaluation.py, so both scripts resolve
synsets the same way and never silently drift apart.

What this module ADDS on top of evaluation.py's graph builder:
  - `resolve_labels()`: nearest-labeled-ancestor lookup. The Excel only
    labels certain "anchor" nodes (broad parent concepts); most target
    classes (e.g. a specific ImageNet leaf synset) are NOT themselves
    labeled and need to inherit their nature/biotic/abiotic label from the
    nearest labeled ancestor above them. This is the automatic-propagation
    step described in the project's SOTA writeup.
  - `get_mapped_classes()`: filters a dataset's (class_name, synset_id)
    list down to only the classes that resolve to a definitive label — the
    "mapped" subset. Per project convention, unmapped classes are dropped
    entirely for this kind of evaluation, not treated as negatives.

USAGE:
    from src.loaders.excel_loader import TaxonomyGraph

    graph = TaxonomyGraph()
    graph.load_excel("/home/pmonserrat/code/flat_wordnet_tree_fixed.xlsx")
    mapped = graph.get_mapped_classes(class_synset_pairs)
    # mapped = [{"class_name": ..., "synset_id": ..., "is_nature": ...,
    #            "biotic_abiotic": ..., "resolved_from_node": ..., "hops": ...}, ...]

NOTE ON DEFAULTS: bio_col="Biotic/abiotic", mat_col="Material/immaterial",
sheet_name="data corrected" — taken directly from evaluate_imagenet.py's
existing call to load_custom_excel_annotations, so this stays consistent
with the closed-set scripts. Override via load_excel()'s arguments if your
copy of the file differs.

BACKGROUND — WHAT ARE WE ACTUALLY BUILDING HERE?
The BIG-5 researchers hand-annotated an Excel file that says, for certain
broad WordNet concepts (e.g. "canine.n.02"), whether that concept counts as
"nature" and whether it's biotic/abiotic. But they obviously couldn't
hand-label every single one of WordNet's ~80,000+ noun synsets one by one —
only a manageable number of "anchor" concepts got a direct label.

This module's job is two-fold:
  1. Build a directed graph (a tree-like structure, technically a DAG —
     Directed Acyclic Graph) out of WordNet's built-in parent/child
     relationships ("hypernyms": more general concepts above a given one),
     so we can walk from any specific synset UP toward more general ones.
  2. Given ANY WordNet synset (even one that was never directly labeled in
     the Excel), walk upward through that graph until we hit the nearest
     ancestor that DOES have a label, and "inherit" that label. E.g. if
     "golden_retriever.n.01" itself isn't labeled but its ancestor
     "canine.n.02" is labeled nature=True/biotic, then golden retrievers
     automatically count as nature/biotic too.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

# Make sure NLTK's WordNet corpus data is actually downloaded on this machine
# before we try to use it. `wn.synsets("dog")` is just a cheap "does this
# work" probe; if it raises LookupError (data missing), download it now
# rather than crashing later mid-pipeline.
try:
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# The single topmost concept in WordNet's entire noun hierarchy — every noun
# synset eventually has "entity.n.01" as an ancestor if you walk up far enough.
ROOT_NODE = "entity.n.01"
# Matches a WordNet synset id string like "golden_retriever.n.01" or
# "hen-of-the-woods.n.01": word characters/hyphens/apostrophes, a dot, a
# single part-of-speech letter (n=noun, v=verb, a=adjective, s=adjective
# satellite, r=adverb), another dot, then a number (the WordNet "sense" index).
_SYNSET_PATTERN = re.compile(r"([\w\-']+\.[nvasr]\.[0-9]+)")


class TaxonomyGraph:
    """
    Builds and queries the WordNet-based BIG-5 taxonomy graph from the
    curated Excel annotation file.

    Node attributes set on labeled (anchor) nodes only:
      - is_nature: bool
      - biotic_abiotic: "biotic" | "abiotic" (only meaningful if is_nature)
      - material_immaterial: "material" | "immaterial" (loaded but unused
        by this evaluation — material/immaterial isn't well-defined at the
        class-name level; see resolve_labels() docstring)

    Unlabeled descendant nodes inherit attributes from their nearest
    labeled ancestor via `resolve_labels()`.
    """

    def __init__(self) -> None:
        # A DiGraph is a graph where edges have a DIRECTION (as opposed to a
        # plain undirected Graph). We use directed edges pointing
        # PARENT -> CHILD (see add_synset_and_ancestors below), so that later
        # we can ask "what are ALL of this node's ancestors" by following
        # edges backwards, or "what's the nearest labeled ancestor" the same way.
        self.graph = nx.DiGraph()

    # -------------------------------------------------------------------
    # Graph construction (ported from evaluation.py's
    # TaxonomyEvaluationPipeline.add_synset_and_ancestors /
    # load_custom_excel_annotations — kept behaviorally identical)
    # -------------------------------------------------------------------

    def add_synset_and_ancestors(self, synset_str: str) -> None:
        """
        Recursively parses a WordNet synset string, fetches its hypernyms,
        and populates the DAG with directional edges flowing from parent to
        child, traversing upward until hitting the ultimate root entity.
        """
        try:
            # Look up the actual WordNet Synset object from its id string.
            synset = wn.synset(synset_str)
        except Exception:
            # Not a real/recognized WordNet synset — nothing we can add.
            return

        current_node = synset.name()

        if current_node == ROOT_NODE:
            # Base case of the recursion: we've reached the very top of the
            # hierarchy. Just make sure it exists as a node (it has no
            # parent, so no edge to add), then stop recursing.
            self.graph.add_node(current_node)
            return

        # "Hypernyms" are WordNet's term for more general parent concepts —
        # e.g. the hypernym of "golden_retriever.n.01" is "retriever.n.01" (a
        # broader dog category). Most synsets have exactly one hypernym, but
        # some have more than one (a concept can belong to multiple broader
        # categories at once).
        hypernyms = synset.hypernyms()
        if not hypernyms:
            # Rare edge case: WordNet gave us no hypernym for this synset, but
            # it's also not the official root. To keep our graph fully
            # connected (so every node can eventually be traced back to
            # ROOT_NODE), we force-link it directly to the root ourselves.
            self.graph.add_edge(ROOT_NODE, current_node)
            return

        for hypernym in hypernyms:
            parent_node = hypernym.name()
            # Add a directed edge PARENT -> CHILD (this is what lets us later
            # walk "upward" from a child by looking at its predecessors).
            self.graph.add_edge(parent_node, current_node)
            # Recursively make sure the parent's OWN ancestors are in the
            # graph too, all the way up to the root. Because networkx graphs
            # silently ignore adding an edge/node that's already there, this
            # recursion is safe to call repeatedly on synsets we've already
            # processed — it just won't do any extra work for nodes already
            # fully wired up.
            self.add_synset_and_ancestors(parent_node)

    def load_excel(
        self,
        excel_path: str,
        bio_col: str = "Biotic/abiotic",
        mat_col: str = "Material/immaterial",
        sheet_name: object = "data corrected",
    ) -> None:
        """
        Reads the Excel file and loads its annotations into the graph.
        Defaults match evaluate_imagenet.py's existing
        `load_custom_excel_annotations(df_taxonomy, "Biotic/abiotic",
        "Material/immaterial")` call against sheet "data corrected" — override
        if your copy of the file differs.
        """
        # pandas reads the whole named Excel sheet into a DataFrame (a table),
        # using the first row as column headers by default.
        df_excel = pd.read_excel(excel_path, sheet_name=sheet_name)
        self._load_annotations(df_excel, bio_col, mat_col)

    def _load_annotations(self, df_excel: pd.DataFrame, bio_col: str, mat_col: str) -> None:
        """
        Ported from evaluation.py's load_custom_excel_annotations. Each row
        represents a hierarchy path (columns other than bio_col/mat_col,
        left-to-right, shallow-to-deep); the DEEPEST non-empty cell is taken
        as the synset the row annotates. `is_nature` is inferred True iff
        the material/immaterial column is non-empty for that row.
        """
        # `.iterrows()` walks the spreadsheet one row at a time; `index` is
        # the pandas row number (0-based) and `row` is that row's data as a
        # pandas Series (dict-like, keyed by column name).
        for index, row in df_excel.iterrows():
            mat_val = row[mat_col]
            bio_val = row[bio_col]

            # Normalize each annotation cell into either a clean lowercase
            # string, or None if the cell was blank/NaN. `pd.notna(x)` is
            # pandas' way of checking "is this NOT a missing value" (Excel
            # blank cells show up as NaN in pandas).
            incoming_bio = (
                str(bio_val).strip().lower() if pd.notna(bio_val) and str(bio_val).strip() != "" else None
            )
            incoming_mat = (
                str(mat_val).strip().lower() if pd.notna(mat_val) and str(mat_val).strip() != "" else None
            )
            # A row counts as describing a "nature" concept exactly when its
            # material/immaterial cell was filled in at all — the presence of
            # that annotation is itself the nature/no-nature signal.
            is_nature = incoming_mat is not None

            # This spreadsheet stores a HIERARCHY PATH across several columns
            # per row (shallow concept in an early column, progressively more
            # specific concepts in later columns), with the material/biotic
            # columns removed from consideration here (`.drop(...)`). We scan
            # left-to-right and keep overwriting `raw_synset` with each
            # non-empty cell we see — so by the time we hit the FIRST empty
            # cell (and `break`), `raw_synset` holds the DEEPEST (most
            # specific) synset string this row actually specifies.
            raw_synset = None
            hierarchy_data = row.drop(labels=[bio_col, mat_col])
            for val in hierarchy_data:
                if pd.isna(val) or str(val).strip() == "":
                    break
                raw_synset = str(val).strip()

            if not raw_synset:
                # This row had no hierarchy path at all (e.g. a fully blank
                # row) — nothing to annotate.
                continue

            # The hierarchy cell might contain extra text around the actual
            # synset id (e.g. a human-readable label plus the id in
            # parentheses) — pull out just the part that matches WordNet's
            # "word.pos.number" synset id pattern.
            match = _SYNSET_PATTERN.search(raw_synset)
            if not match:
                continue
            synset_str = match.group(1)

            # Was this exact synset already added to the graph by an earlier
            # row? If so, we're about to potentially OVERWRITE its label —
            # worth checking below whether the two rows actually agree.
            is_duplicate_entry = synset_str in self.graph
            self.add_synset_and_ancestors(synset_str)

            if synset_str not in self.graph:
                # add_synset_and_ancestors silently does nothing if the
                # string isn't a real WordNet synset — catch that here and
                # warn instead of silently losing this row's annotation.
                print(
                    f"WORDNET ERROR (Excel Row {index + 2}): could not load "
                    f"synset '{synset_str}' into the graph. Skipping annotation."
                )
                continue

            if is_duplicate_entry:
                # This synset was already labeled by a PREVIOUS row in the
                # spreadsheet. If the new row disagrees with what's already
                # recorded, that's a data-quality problem worth flagging
                # loudly (rather than silently picking one value) — print a
                # warning so a human can go check the spreadsheet.
                # (`index + 2` converts pandas' 0-based row index into the
                # 1-based Excel row number a human would actually see, plus 1
                # more for the header row.)
                existing_bio = self.graph.nodes[synset_str].get("biotic_abiotic")
                existing_mat = self.graph.nodes[synset_str].get("material_immaterial")
                excel_row = index + 2
                if existing_bio is not None and incoming_bio is not None and existing_bio != incoming_bio:
                    print(
                        f"CONFLICT WARNING (Excel Row {excel_row}): '{synset_str}' "
                        f"biotic mismatch. Graph has '{existing_bio}', row says '{incoming_bio}'."
                    )
                if existing_mat is not None and incoming_mat is not None and existing_mat != incoming_mat:
                    print(
                        f"CONFLICT WARNING (Excel Row {excel_row}): '{synset_str}' "
                        f"material mismatch. Graph has '{existing_mat}', row says '{incoming_mat}'."
                    )

            # Actually record the annotation as attributes ON the graph node
            # (networkx lets you attach arbitrary key/value data to any node —
            # `self.graph.nodes[synset_str]` is a dict-like view of that
            # node's attributes). `is_nature` is always (re)written; the
            # biotic/material fields are only written when this row actually
            # provided a value, so we never overwrite a real label with a blank.
            self.graph.nodes[synset_str]["is_nature"] = is_nature
            if incoming_bio:
                self.graph.nodes[synset_str]["biotic_abiotic"] = incoming_bio
            if incoming_mat:
                self.graph.nodes[synset_str]["material_immaterial"] = incoming_mat

    # -------------------------------------------------------------------
    # Querying / resolution (new in this module)
    # -------------------------------------------------------------------

    def resolve_labels(self, synset_str: str) -> Optional[Dict[str, object]]:
        """
        Returns the resolved taxonomy labels for ANY synset by walking
        upward (via predecessors, i.e. parent edges) to the NEAREST node
        that has `is_nature` set — a labeled anchor. This is the automatic
        propagation step: most target classes aren't directly labeled in
        the Excel, only certain broad parent concepts are.

        NOTE ON MATERIAL/IMMATERIAL: intentionally NOT returned here. That
        axis depends on whether a specific image instance is a real object
        or a representation of one — a property of a photographed instance,
        not of the WordNet concept. A class name alone (with no image) has
        no principled answer for this axis, so it's out of scope for a
        text-only, class-name-level evaluation.

        Returns None if the synset can't be resolved to ANY labeled
        ancestor (including itself) — treat this as UNMAPPED and exclude
        it, per project convention.
        """
        if synset_str not in self.graph:
            # We might be asked about a synset that was never mentioned in
            # the Excel at all (e.g. a specific ImageNet leaf class). As long
            # as it's a real WordNet synset, add it (and its ancestor chain)
            # to the graph on the fly so we can still search upward from it.
            try:
                wn.synset(synset_str)
                self.add_synset_and_ancestors(synset_str)
            except Exception:
                return None
        if synset_str not in self.graph:
            return None

        # Breadth-first search (BFS) walking UPWARD through the hierarchy:
        # start at `synset_str` itself (0 hops away from itself), then visit
        # its parents (1 hop), then grandparents (2 hops), etc., stopping as
        # soon as we find a node that carries an explicit "is_nature" label.
        # BFS (rather than depth-first) guarantees we find the CLOSEST
        # labeled ancestor first, since we always process all nodes at
        # distance N before moving on to distance N+1.
        visited = {synset_str}
        queue = deque([(synset_str, 0)])
        while queue:
            node, hops = queue.popleft()
            attrs = self.graph.nodes[node]
            if "is_nature" in attrs:
                # Found the nearest labeled ancestor (or the node itself, if
                # it happened to be directly labeled) — return its label,
                # tagged with which node it actually came from and how many
                # hops away that was (useful for diagnostics/debugging).
                return {
                    "resolved_from_node": node,
                    "hops": hops,
                    "is_nature": attrs["is_nature"],
                    "biotic_abiotic": attrs.get("biotic_abiotic"),
                }
            # `predecessors(node)` gives the nodes with an edge POINTING INTO
            # `node` — since our edges run parent->child, a node's
            # predecessors are exactly its parents (more general concepts).
            for parent in self.graph.predecessors(node):
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, hops + 1))
        # Walked all the way up without ever finding a labeled node — this
        # synset has no connection at all to anything the Excel annotated.
        return None

    def get_mapped_classes(
        self,
        class_synset_pairs: List[Tuple[str, str]],
    ) -> List[Dict[str, object]]:
        """
        Given (class_name, synset_id) pairs for a dataset, returns only the
        entries that resolve to a definitive taxonomy label — the MAPPED
        subset. Unmapped classes are dropped, per project convention:
        "we just calculate accuracy for the target classes that are
        mapped — those not mapped are not necessary."
        """
        mapped = []
        for class_name, synset_id in class_synset_pairs:
            labels = self.resolve_labels(synset_id)
            if labels is None:
                # This class couldn't be traced to any labeled ancestor at
                # all — skip it entirely rather than guessing a label.
                continue
            mapped.append(
                {
                    "class_name": class_name,
                    "synset_id": synset_id,
                    "is_nature": labels["is_nature"],
                    "biotic_abiotic": labels["biotic_abiotic"],
                    "resolved_from_node": labels["resolved_from_node"],
                    "hops": labels["hops"],
                }
            )
        return mapped
