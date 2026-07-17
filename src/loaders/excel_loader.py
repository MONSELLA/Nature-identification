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
"""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

try:
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

ROOT_NODE = "entity.n.01"
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
            synset = wn.synset(synset_str)
        except Exception:
            return

        current_node = synset.name()

        if current_node == ROOT_NODE:
            self.graph.add_node(current_node)
            return

        hypernyms = synset.hypernyms()
        if not hypernyms:
            self.graph.add_edge(ROOT_NODE, current_node)
            return

        for hypernym in hypernyms:
            parent_node = hypernym.name()
            self.graph.add_edge(parent_node, current_node)
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
        for index, row in df_excel.iterrows():
            mat_val = row[mat_col]
            bio_val = row[bio_col]

            incoming_bio = (
                str(bio_val).strip().lower() if pd.notna(bio_val) and str(bio_val).strip() != "" else None
            )
            incoming_mat = (
                str(mat_val).strip().lower() if pd.notna(mat_val) and str(mat_val).strip() != "" else None
            )
            is_nature = incoming_mat is not None

            raw_synset = None
            hierarchy_data = row.drop(labels=[bio_col, mat_col])
            for val in hierarchy_data:
                if pd.isna(val) or str(val).strip() == "":
                    break
                raw_synset = str(val).strip()

            if not raw_synset:
                continue

            match = _SYNSET_PATTERN.search(raw_synset)
            if not match:
                continue
            synset_str = match.group(1)

            is_duplicate_entry = synset_str in self.graph
            self.add_synset_and_ancestors(synset_str)

            if synset_str not in self.graph:
                print(
                    f"WORDNET ERROR (Excel Row {index + 2}): could not load "
                    f"synset '{synset_str}' into the graph. Skipping annotation."
                )
                continue

            if is_duplicate_entry:
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
            try:
                wn.synset(synset_str)
                self.add_synset_and_ancestors(synset_str)
            except Exception:
                return None
        if synset_str not in self.graph:
            return None

        visited = {synset_str}
        queue = deque([(synset_str, 0)])
        while queue:
            node, hops = queue.popleft()
            attrs = self.graph.nodes[node]
            if "is_nature" in attrs:
                return {
                    "resolved_from_node": node,
                    "hops": hops,
                    "is_nature": attrs["is_nature"],
                    "biotic_abiotic": attrs.get("biotic_abiotic"),
                }
            for parent in self.graph.predecessors(node):
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, hops + 1))
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