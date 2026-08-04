#!/usr/bin/env python3
"""
run_grounding_pipeline.py

Driver for the GROUNDING stage only (SAM3 segmentation -> nature relevance
score). It reads the JSON-Lines artifact the VLM pipeline wrote
(`run_vlm_pipeline.py --stage infer`, default
`<results_dir>/<run_name>/responses/vlm_responses_<model>.jsonl`) and ENRICHES each image
record with its own fields, so ONE artifact ends up holding everything for a
given image — caption, entities, taxonomy labels, masks, relevance score —
rather than the grounding stage writing a second, separate output file that
would have to be re-joined later.

    caption/extraction/labeling  (run_vlm_pipeline.py --stage infer)
        |   vlm_responses_<model>.jsonl
        v
    SAM3 grounding + relevance   (THIS SCRIPT)
        |   grounded_vlm_responses_<model>.jsonl
        v
    (analysis / future scoring)

Fields added per image record (see src/grounding_pipeline.py's module docstring
for the full shape):
    "object_groundings"                        — aligned index-for-index with
                                "objects", one entry per extracted entity,
                                carrying {grounded, mask_rle, pixel_count, ...}.
                                Entities SAM3 could not confirm are MARKED
                                (`grounded: false`), never deleted, so the full
                                VLM output stays auditable.
    "nature_relevance_score_coverage_ratio"     — float in [0, 1], nature px /
                                total px, over the UNION of surviving nature
                                masks (overlapping pixels counted once).
    "nature_relevance_score_center_weighted"    — float in [0, 1], same union
                                mask but pixels near the image center count
                                more (Gaussian falloff, --center_sigma).
                                BOTH scores are always computed and stored —
                                not a selectable either/or.

Only entities whose HYBRID resolved label is nature (`final_nature is True`) are
grounded — everything else gets `grounded: null` ("never attempted"), which is
deliberately distinct from `false` ("SAM3 looked and found nothing").

WRITE SAFETY: the enriched artifact goes to a NEW file by default
(`grounded_<input stem>.jsonl` next to the input), so a crash mid-run can never
corrupt the VLM artifact that took GPU-hours to produce. `--in_place` rewrites
the input instead, and still does so via a temporary file that is only moved
over the original once the whole run has finished successfully.

HOW TO READ THIS FILE
Like run_vlm_pipeline.py, this script contains no deep logic — all of it lives
in src/grounding_pipeline.py. Start at `main()` at the bottom.
"""

import os

# Same rationale as run_vlm_pipeline.py: quiet two low-value third-party startup
# notices before torch/transformers are imported, via setdefault so either can
# still be overridden from the shell.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from src.grounding_pipeline import (
    DEFAULT_CENTER_SIGMA,
    DEFAULT_INSTANCE_SCORE_THRESHOLD,
    DEFAULT_MASK_THRESHOLD,
    SAM3_MODEL_ID,
    SAM3Grounder,
    grounding_diagnostics,
    ground_records,
    stream_artifact,
)
from src.utils import format_duration

# Must match run_vlm_pipeline.RESPONSES_SUBDIR exactly — duplicated here
# (rather than imported) so this script stays independently runnable without
# pulling in run_vlm_pipeline.py's heavier VLM-serving imports (vLLM, etc.)
# just to ground an existing artifact.
RESPONSES_SUBDIR = "responses"


# =============================================================================
# The grounding phase
# =============================================================================
def phase_ground(args):
    """Stream the VLM artifact through SAM3 and write the enriched artifact.

    Streaming (read a line -> ground a batch -> write a line) rather than
    loading everything into memory mirrors how phase_infer wrote the file, and
    keeps memory flat regardless of dataset size — the masks are the bulky part
    and each one is RLE-encoded and written out immediately.
    """
    in_path = Path(args.responses_file)
    if not in_path.exists():
        print(f"Artifact not found: {in_path} — run run_vlm_pipeline.py --stage infer first.")
        sys.exit(1)
    out_path = _resolve_output_path(args, in_path)

    print(f"🚀 [grounding] {in_path} -> {out_path} (model='{args.sam3_model}')")
    phase_t0 = time.time()

    header, footer_holder, records = stream_artifact(in_path)
    if args.max_samples is not None:
        records = _take(records, args.max_samples)

    # Instance grounding (SAM3's per-instance masks/boxes, on top of the
    # concept-level semantic map) is what COCO's box-IoU evaluation needs and
    # nothing else currently consumes — so "auto" turns it on for COCO
    # artifacts and leaves every other dataset exactly as it was. Resolved
    # from the artifact's OWN header rather than a CLI --dataset, so grounding
    # can't disagree with the run it is enriching. It reads off the same
    # forward pass (see SAM3Grounder.segment_pairs_full), so the cost is
    # post-processing, not a second pass.
    if args.instance_grounding == "auto":
        instance_grounding = str(header.get("dataset", "")).lower() == "coco"
    else:
        instance_grounding = args.instance_grounding == "on"

    grounder = SAM3Grounder(
        model_name=args.sam3_model, device=args.device, dtype=args.dtype,
        mask_threshold=args.mask_threshold,
        instance_score_threshold=args.instance_score_threshold,
        debug_semantic_range=args.debug_semantic_range,
        hf_token=args.hf_token,
    )

    # Record the grounding configuration in the artifact header, next to the
    # VLM's own run metadata, so an enriched artifact is self-describing — you
    # can tell which SAM3 checkpoint/threshold produced these masks without
    # having to find the command line that made them. Both relevance-score
    # methods are always computed (see src/grounding_pipeline.py), so
    # center_sigma is always relevant, not gated on a method choice.
    header = dict(header)
    header["grounding"] = {
        "sam3_model": args.sam3_model,
        "mask_threshold": args.mask_threshold,
        "center_sigma": args.center_sigma,
        "instance_grounding": instance_grounding,
        # Only meaningful when instance_grounding is on, but recorded either
        # way so two artifacts are always comparable field-for-field.
        "instance_score_threshold": args.instance_score_threshold,
        "source_artifact": str(in_path),
    }

    counts = {"objects_total": 0, "nature_entities": 0, "grounded_entities": 0,
              "instances_total": 0}
    n_images = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary sibling first and move it into place at the end, so a
    # crash never leaves a half-written artifact that looks complete (and so
    # --in_place cannot destroy its own input mid-run).
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with open(tmp_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for rec in ground_records(
            grounder, records,
            batch_size=args.batch_size,
            max_pairs_per_forward=args.max_pairs_per_forward,
            center_sigma=args.center_sigma,
            instance_grounding=instance_grounding,
            n_total=args.max_samples,
            verbose=args.verbose,
        ):
            n_images += 1
            groundings = rec.get("object_groundings", [])
            counts["objects_total"] += len(groundings)
            counts["nature_entities"] += sum(1 for g in groundings if g["is_nature"])
            counts["grounded_entities"] += sum(1 for g in groundings if g["grounded"])
            counts["instances_total"] += sum(len(e["instances"])
                                             for e in rec.get("object_instances", []))
            rec["record_type"] = "image"
            f.write(json.dumps(rec) + "\n")

        # Carry the VLM footer's timings forward (so the enriched artifact keeps
        # the inference cost it inherited) and add this stage's own.
        footer = dict(footer_holder[0] or {})
        footer["record_type"] = "footer"
        footer["grounding_time_seconds"] = time.time() - phase_t0
        f.write(json.dumps(footer) + "\n")
    tmp_path.replace(out_path)

    diagnostics = grounding_diagnostics(counts, n_images)
    _print_summary(args, header, diagnostics, footer["grounding_time_seconds"], out_path)
    if args.wandb:
        _log_wandb(args, header, diagnostics)


def _take(iterator, n):
    """Yield at most `n` items — `itertools.islice` on a generator, spelled out
    so --max_samples reads the same way it does in run_vlm_pipeline.py."""
    for i, item in enumerate(iterator):
        if i >= n:
            return
        yield item


def _print_summary(args, header, d, elapsed, out_path):
    """Pretty-print the grounding run's diagnostics to the console."""
    print("\n" + "=" * 60)
    print(f"🛰️  GROUNDING PIPELINE: {header.get('model')} on "
          f"{str(header.get('dataset', '?')).upper()}  ({d['n_images']} images)")
    print("=" * 60)
    print(f"SAM3: {args.sam3_model} | mask threshold {args.mask_threshold} "
          f"| relevance: coverage_ratio + center_weighted (sigma {args.center_sigma})")
    print(f"Entities: {d['objects_total']} total | {d['nature_entities']} nature "
          f"({d['nature_entities_per_image']:.2f}/image)")
    # The headline diagnostic: how often an independent segmenter could actually
    # find, in pixels, the nature the VLM claimed was there (recap §9 — this is
    # AGREEMENT WITH AN INDEPENDENT MODEL, not ground truth).
    print(f"Grounding-confirmation rate: {d['grounding_confirmation_rate']:.1%} "
          f"({d['grounded_entities']}/{d['nature_entities']} nature entities confirmed)")
    if header.get("grounding", {}).get("instance_grounding"):
        print(f"Instances (SAM3 instance head, score > {args.instance_score_threshold}): "
              f"{d['instances_total']} "
              f"({d['instances_per_grounded_entity']:.2f} per grounded entity) — "
              f"scored against COCO GT boxes by run_vlm_pipeline.py --stage score")
    print(f"Elapsed (D-HH:MM:SS): {format_duration(elapsed)}")
    print(f"💾 wrote {out_path}")
    print("=" * 60)


def _log_wandb(args, header, d):
    """Push grounding diagnostics to W&B, resuming the same run the VLM stage
    logged into when --wandb_run_id was threaded through (see run_pipeline.py)."""
    import wandb
    run_id = getattr(args, "wandb_run_id", None)
    name = f"grounding_pipeline_{header.get('dataset')}_{header.get('model')}".replace("/", "_")
    wandb.init(entity="paumonserrat03-universitat-aut-noma-de-barcelona", project="TFM_VLM",
               config=vars(args), name=name, id=run_id, resume="allow" if run_id else None)
    wandb.log({
        "Grounding/NatureEntities": d["nature_entities"],
        "Grounding/NatureEntitiesPerImage": d["nature_entities_per_image"],
        "Grounding/GroundedEntities": d["grounded_entities"],
        "Grounding/ConfirmationRate": d["grounding_confirmation_rate"],
    })
    wandb.finish()


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser():
    """Define every flag this script accepts.

    Split out from parse_args() (which just calls this + .parse_args()) for the
    same reason run_vlm_pipeline.py does it: scripts/run_pipeline.py builds this
    parser purely to INTROSPECT it, so it can split a combined command line
    across the two stages and rebuild each stage's own CLI."""
    p = argparse.ArgumentParser(
        description="Grounding pipeline: SAM3 segmentation + nature relevance score, "
                    "enriching the VLM pipeline's JSON-Lines artifact in place.")
    p.add_argument("--responses_file", type=str, default=None,
                   help="VLM-pipeline artifact to read and enrich (written by "
                        "run_vlm_pipeline.py --stage infer). Default: the same path that "
                        "script would have used, i.e. '<results_dir>/<run_name>/"
                        f"{RESPONSES_SUBDIR}/vlm_responses_<model_slug>.jsonl', which requires "
                        "--model_name so the slug can be rebuilt.")
    p.add_argument("--grounded_file", type=str, default=None,
                   help="Where to write the ENRICHED artifact. Default: "
                        "'grounded_<input stem>.jsonl' next to --responses_file. Ignored "
                        "when --in_place is passed.")
    p.add_argument("--in_place", action="store_true",
                   help="Rewrite --responses_file itself instead of writing a separate "
                        "enriched file, so a single artifact accumulates both stages' "
                        "output. Still written via a temporary file that only replaces the "
                        "original once the run finishes, so a crash cannot corrupt the "
                        "input.")

    # Identify the input artifact the same way run_vlm_pipeline.py names it.
    p.add_argument("--model_family", type=str, default=None,
                   help="NOT used to build the default --responses_file name (that's "
                        "--model_name alone, matching run_vlm_pipeline.py's _model_slug). "
                        "Accepted here only so the same flags used for the VLM run can be "
                        "passed straight through without editing.")
    p.add_argument("--model_name", type=str, default=None,
                   help="Used to reconstruct the default --responses_file name (suffixed with "
                        "this model's slug so each model's artifact is distinct). Unnecessary "
                        "if --responses_file is given.")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Base directory the VLM artifact lives under (matches "
                        "run_vlm_pipeline.py's flag of the same name).")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional --results_dir subfolder, matching run_vlm_pipeline.py's "
                        "flag of the same name.")

    # SAM3
    p.add_argument("--sam3_model", type=str, default=SAM3_MODEL_ID,
                   help=f"HuggingFace repo id for SAM3 (default {SAM3_MODEL_ID}). One model "
                        "handles both discrete objects and amorphous regions — there is no "
                        "thing/stuff routing and no separate detector in this pipeline.")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace token for gated repos (facebook/sam3 requires accepting "
                        "its license). Default: the HF_TOKEN env var, or the token cached by "
                        "`huggingface-cli login` if neither is set. Prefer those over this "
                        "flag so the token never ends up in shell history or a Slurm script.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="auto",
                   help="'auto' (checkpoint default) or a torch dtype name, e.g. float16 / "
                        "bfloat16 / float32.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of IMAGES processed per batch (same meaning as in the VLM "
                        "pipeline). Note SAM3 resolves ONE text prompt per forward pass, so "
                        "each image contributes one (image, entity) pair per nature entity; "
                        "--max_pairs_per_forward is what actually bounds GPU work.")
    p.add_argument("--max_pairs_per_forward", type=int, default=16,
                   help="Maximum (image, entity) pairs sent to SAM3 in a single forward "
                        "pass. Caps VRAM independently of --batch_size, since a busy image "
                        "can carry many nature entities.")
    p.add_argument("--mask_threshold", type=float, default=DEFAULT_MASK_THRESHOLD,
                   help=f"Probability threshold binarizing SAM3's semantic map (default "
                        f"{DEFAULT_MASK_THRESHOLD}, the model's own stated default). "
                        "semantic_seg is raw LOGITS; the sigmoid is applied before this "
                        "threshold — see src/grounding_pipeline.py's module docstring.")
    p.add_argument("--instance_grounding", type=str, default="auto",
                   choices=["auto", "on", "off"],
                   help="Also read SAM3's INSTANCE head (per-instance masks + boxes) "
                        "into 'object_instances', which is what COCO's box-IoU "
                        "detection evaluation scores against. 'auto' (default) turns "
                        "it on only for COCO artifacts, read from the artifact's own "
                        "header. It comes off the SAME forward pass as the semantic "
                        "map, so it costs post-processing only — no second pass — and "
                        "leaves object_groundings and both relevance scores unchanged.")
    p.add_argument("--instance_score_threshold", type=float,
                   default=DEFAULT_INSTANCE_SCORE_THRESHOLD,
                   help="Confidence gate for SAM3's instance head (SAM3's own default: "
                        f"{DEFAULT_INSTANCE_SCORE_THRESHOLD}). Distinct from "
                        "--mask_threshold, which binarizes PIXELS; this one decides "
                        "which instance queries count as detections at all. Lower it "
                        "to trade detection precision for recall.")
    p.add_argument("--debug_semantic_range", action="store_true",
                   help="Print the RAW (pre-sigmoid) semantic_seg min/max for the first two "
                        "forward passes, to re-confirm on real data that the map is "
                        "unbounded logits rather than an already-normalized probability map.")

    # Relevance score — BOTH methods are always computed and stored (not a
    # choice between them): "nature_relevance_score_coverage_ratio" (nature
    # pixels / total pixels) and "nature_relevance_score_center_weighted"
    # (same union mask, but weighted by a Gaussian in distance from the image
    # center). Both land in [0, 1] and are computed over the UNION of the
    # surviving nature masks.
    p.add_argument("--center_sigma", type=float, default=DEFAULT_CENTER_SIGMA,
                   help="Width of the center_weighted Gaussian, in units where the image "
                        "center is 0 and each edge midpoint is 1 (aspect-ratio "
                        f"independent). Default {DEFAULT_CENTER_SIGMA}. Only affects "
                        "nature_relevance_score_center_weighted.")

    # shared
    p.add_argument("--max_samples", type=int, default=None,
                   help="Ground only the first N image records of the artifact (quick "
                        "smoke test). Note this is the artifact's own order — pair it with "
                        "run_vlm_pipeline.py's --max_samples to control which images those "
                        "are in the first place.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_run_id", type=str, default=None, help=argparse.SUPPRESS)
    # ^ Internal plumbing (same as run_vlm_pipeline.py): run_pipeline.py
    # generates one shared W&B run id and threads it into both stages'
    # subprocess CLIs so they log into a single run. Hidden from --help.
    return p


def parse_args():
    p = build_arg_parser()
    args = p.parse_args()
    if args.responses_file is None and not args.model_name:
        # argparse can't express "required unless another flag is set", so this
        # raises the same kind of clean CLI error it would have.
        p.error("--responses_file is required unless --model_name is given (it's what the "
                "default artifact filename is built from).")
    return args


def _resolve_responses_file(args):
    """Fill in --responses_file's default by rebuilding exactly the path
    run_vlm_pipeline.py's _resolve_responses_file would have written to for the
    same --results_dir/--run_name/model (including the RESPONSES_SUBDIR
    grouping), so the two scripts agree on where the artifact lives without
    the path having to be passed by hand. Mutates and returns `args`."""
    if args.responses_file is None:
        out_dir = Path(args.results_dir)
        if args.run_name:
            out_dir = out_dir / args.run_name
        slug = args.model_name.replace("/", "_")
        args.responses_file = str(out_dir / RESPONSES_SUBDIR / f"vlm_responses_{slug}.jsonl")
    return args


def _resolve_output_path(args, in_path: Path) -> Path:
    """Where the enriched artifact goes: the input itself under --in_place, an
    explicit --grounded_file if given, else 'grounded_<input stem>.jsonl'
    alongside the input (so it inherits --results_dir/--run_name for free)."""
    if args.in_place:
        return in_path
    if args.grounded_file:
        return Path(args.grounded_file)
    return in_path.with_name(f"grounded_{in_path.name}")


def main():
    args = parse_args()
    args = _resolve_responses_file(args)
    phase_ground(args)


if __name__ == "__main__":
    main()
