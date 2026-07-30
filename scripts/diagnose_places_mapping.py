#!/usr/bin/env python3
"""
diagnose_places_mapping.py

One-off diagnostic: reproduces get_candidate_vocab's/load_places365's exact
Places365 category -> synset -> taxonomy-label resolution, PER CATEGORY
(not just the aggregate counts phase_infer prints), so a discrepancy between
the taxonomy Excel's own "mapped" count and get_candidate_vocab's actual
output size can be tracked down to the specific categories responsible.

Context: the Excel's own accounting says N categories are mapped, but
get_candidate_vocab() surfaces fewer — meaning some categories the Excel
already considers correctly mapped are being lost inside THIS repo's
resolution heuristic (_resolve_places_name_to_synset / get_gt_from_graph),
not because they're genuinely unmapped. This script prints one row per
Places365 category with every stage's verdict, so you can diff it against
the Excel's own list and find exactly which ones fall through the gap.

Usage:
    python scripts/diagnose_places_mapping.py \
        --places_categories_txt /path/to/categories_places365.txt \
        --excel_path /path/to/flat_wordnet_tree_fixed.xlsx \
        --sheet_name "data corrected"

Prints a table to stdout; pass --csv_out to also write it to a CSV for
easier diffing in a spreadsheet against the Excel's own "mapped" list.
"""
import argparse
import csv
import sys

from src.loaders.dataset_loader import (
    _load_places_categories,
    _load_places_taxonomy_synsets,
    _resolve_places_name_to_synset,
    get_gt_from_graph,
)
from src.loaders.excel_loader import TaxonomyGraph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--places_categories_txt", required=True)
    p.add_argument("--excel_path", required=True)
    p.add_argument("--sheet_name", default="data corrected")
    p.add_argument("--csv_out", default=None, help="Optional path to also write the table as CSV.")
    args = p.parse_args()

    graph = TaxonomyGraph()
    sheet = args.sheet_name if not str(args.sheet_name).isdigit() else int(args.sheet_name)
    graph.load_excel(args.excel_path, sheet_name=sheet)

    id_to_name = _load_places_categories(args.places_categories_txt)
    exclusion, taxonomy_synsets = _load_places_taxonomy_synsets(args.excel_path)

    rows = []
    counts = {"excluded": 0, "unresolved": 0, "resolved_but_unlabeled": 0, "mapped": 0}
    for cid, name in sorted(id_to_name.items()):
        if name in exclusion:
            rows.append({"places_id": cid, "category": name, "status": "excluded",
                         "synset": None, "gt_nature": None, "gt_biotic": None})
            counts["excluded"] += 1
            continue

        synset = _resolve_places_name_to_synset(name, taxonomy_synsets)
        if synset is None:
            rows.append({"places_id": cid, "category": name, "status": "unresolved",
                         "synset": None, "gt_nature": None, "gt_biotic": None})
            counts["unresolved"] += 1
            continue

        gt = get_gt_from_graph(synset, graph)
        if gt is None:
            rows.append({"places_id": cid, "category": name, "status": "resolved_but_unlabeled",
                         "synset": synset, "gt_nature": None, "gt_biotic": None})
            counts["resolved_but_unlabeled"] += 1
            continue

        rows.append({"places_id": cid, "category": name, "status": "mapped",
                      "synset": synset, "gt_nature": gt["gt_nature"], "gt_biotic": gt["gt_biotic"]})
        counts["mapped"] += 1

    # Print sorted by status so the interesting "resolved_but_unlabeled" and
    # "unresolved" rows (the ones most likely responsible for a mapped-count
    # mismatch against the Excel's own accounting) are easy to scan together.
    for status in ("resolved_but_unlabeled", "unresolved", "excluded", "mapped"):
        group = [r for r in rows if r["status"] == status]
        if not group:
            continue
        print(f"\n=== {status} ({len(group)}) ===")
        for r in group:
            extra = f" -> {r['synset']}" if r["synset"] else ""
            print(f"  [{r['places_id']:>3}] {r['category']}{extra}")

    print("\n" + "=" * 50)
    print(f"Totals: {counts} (sum={sum(counts.values())}, expected 365)")
    print("=" * 50)
    print("\nCompare the 'resolved_but_unlabeled' and 'unresolved' lists above against "
          "the Excel's own 'mapped' list — any category that appears in BOTH your Excel's "
          "mapped list AND one of these two groups here is exactly the kind of category "
          "responsible for the count mismatch: this repo's heuristic is failing on a "
          "category the Excel itself already resolved correctly.")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["places_id", "category", "status", "synset", "gt_nature", "gt_biotic"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    sys.exit(main())
