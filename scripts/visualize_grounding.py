#!/usr/bin/env python3
"""
visualize_grounding.py

Render ONE image's SAM3 grounding as a figure — the picture with the pipeline's
nature masks drawn on it, which is what the Grounding pipeline actually outputs
for a single image. Built for making method/qualitative figures for the thesis,
not for producing metrics.

WHERE THE ENTITIES COME FROM. This script never invents them. It reads the
image's own record out of a `vlm_responses_<model>.jsonl` artifact — the same
raw prediction record `--stage infer` wrote — and grounds exactly the entities
that record's HYBRID labels marked as nature (`final_nature is True`), which is
precisely the subset `src/grounding_pipeline.py` would ground in a real run.
So the figure shows what the pipeline predicted for that model, not a fresh
guess. `--objects` overrides this for a quick what-if, and is labelled as such
in the output filename so an ad-hoc render is never mistaken for a pipeline one.

WHERE THE MASKS COME FROM. If the artifact was already enriched by the
Grounding pipeline (it carries `object_groundings` with `mask_rle`), those
stored masks are drawn as-is and SAM3 is NEVER loaded — the figure is then
bit-identical to the run being written about. Only when the artifact has not
been grounded does this script run SAM3 itself, through the pipeline's own
`SAM3Grounder` with the same defaults (`--mask_threshold`, prompts built by
`grounding_pipeline.entity_prompt`), so even the recomputed case follows the
shipped code path rather than a parallel implementation. Which of the two
happened is printed, and recorded in the sidecar JSON.

OUTPUTS (into `--out_dir`, default `results/figures/grounding/`):
  <stem>_<model>_overlay.png    the figure: masks over the image, one colour
                                per entity, with a legend
  <stem>_<model>_panels.png     one panel per entity (only with --panels)
  <stem>_<model>_grounding.json sidecar: entities, prompts, pixel counts, both
                                relevance scores, provenance (artifact path,
                                model, mask source, thresholds)

The sidecar exists so a figure in the thesis can always be traced back to the
run that produced it — a PNG alone cannot tell you which artifact, model or
threshold made it.

USAGE
  # from an artifact (the normal case)
  python scripts/visualize_grounding.py \\
      --image /home/pmonserrat/datasets/big_5/twitter/1703862454143304161_0.jpg \\
      --responses_file results/vlm_pipeline/ablation_no_caption/big5_twitter/responses/\\
vlm_responses_google_gemma-4-26B-A4B-it.jsonl

  # ad-hoc entities, no artifact
  python scripts/visualize_grounding.py --image path/to.jpg \\
      --objects "dog,mushroom plush" --model_label gemma-4-26B-A4B

Needs the SAM3 checkpoint (gated) unless the artifact is already grounded:
  export HF_TOKEN=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import grounding_pipeline as gp  # noqa: E402

# Distinct, print-safe colours. Ordered so the first few stay distinguishable
# in greyscale too (a thesis may well be printed in black and white).
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 212), (0, 128, 128), (170, 110, 40), (128, 0, 0),
]


def _load_record(responses_file: str, image_path: str) -> Dict[str, Any]:
    """Find one image's record in an artifact, matched on BASENAME.

    Basename, not the full path, because `image_path` inside an artifact is
    whatever the machine that ran inference saw — the same BIG-5 image appears
    as an absolute cluster path in one run and a relative one in another (both
    exist in this project's own artifacts). The basename is
    `<platform_id>_<slot>.<ext>` and is stable across both.
    """
    want = os.path.basename(image_path)
    header, _footer, records = gp.stream_artifact(responses_file)
    for rec in records:
        if os.path.basename(rec.get("image_path", "")) == want:
            rec["_header"] = header
            return rec
    raise SystemExit(
        f"{want!r} is not in {responses_file}.\n"
        f"Check the model/run — a BIG-5 image only appears in the artifacts of runs "
        f"whose dataset covered it (e.g. a Twitter image is not in a Weibo artifact)."
    )


def _nature_entities(rec: Dict[str, Any]) -> List[str]:
    """The entities the Grounding pipeline would ground for this record: those
    whose HYBRID label came out nature (`final_nature is True`).

    Deliberately reads `object_finals`, not the raw per-object VLM answer — the
    hybrid label is what the pipeline acts on (WordNet decides when it can, the
    VLM when it cannot), so anything else would draw a different set of masks
    than a real grounding run.
    """
    finals = rec.get("object_finals") or []
    return [f["object"] for f in finals if f.get("final_nature") is True]


def _masks_from_artifact(rec: Dict[str, Any], entities: List[str]):
    """Reuse the stored masks when this artifact was already grounded.

    Returns None when it wasn't, or when the stored groundings don't cover the
    requested entities (e.g. --objects asked for something the run never
    grounded) — the caller then falls back to running SAM3.
    """
    groundings = rec.get("object_groundings")
    if not groundings:
        return None
    by_obj = {g.get("object"): g for g in groundings}
    out = []
    for ent in entities:
        g = by_obj.get(ent)
        if not g or not g.get("mask_rle"):
            return None                       # incomplete -> recompute all
        out.append(gp.decode_rle(g["mask_rle"]).astype(bool))
    return out


def _run_sam3(image, entities: List[str], args) -> List[np.ndarray]:
    """Ground `entities` with the pipeline's own SAM3Grounder.

    One (image, prompt) pair per entity: SAM3 resolves exactly one text prompt
    per forward pass (`Sam3Model.forward` shares one batch dim between
    pixel_values and input_ids), so the image is repeated once per entity —
    the same flattening `grounding_pipeline._ground_batch` does.
    """
    grounder = gp.SAM3Grounder(
        model_name=args.sam3_model, device=args.device, dtype=args.dtype,
        mask_threshold=args.mask_threshold, hf_token=args.hf_token,
    )
    pairs = [(image, gp.entity_prompt(e)) for e in entities]
    return grounder.segment_pairs(pairs)


def _overlay(image, masks: List[np.ndarray], labels: List[str], alpha: float,
             outline: bool = True):
    """Composite masks over the image, one colour each, plus a legend.

    Drawn back-to-front in the given order, so a later entity's colour wins on
    overlapping pixels. That is a DISPLAY choice only and does not mirror how
    the relevance score treats overlap — there, a pixel counts as nature if ANY
    entity covers it, which is why the score is computed from the RLE union
    rather than from this rendering.
    """
    from PIL import Image, ImageDraw, ImageFont

    base = image.convert("RGB")
    overlay = np.array(base).astype(np.float32)
    for i, mask in enumerate(masks):
        colour = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        sel = mask.astype(bool)
        if not sel.any():
            continue
        overlay[sel] = (1.0 - alpha) * overlay[sel] + alpha * colour
    out = Image.fromarray(overlay.astype(np.uint8))

    if outline:
        # A 1px boundary makes each region readable at print size, where a
        # translucent fill alone can wash out against a busy photograph.
        draw = ImageDraw.Draw(out)
        for i, mask in enumerate(masks):
            sel = mask.astype(bool)
            if not sel.any():
                continue
            edge = sel & ~(
                np.roll(sel, 1, 0) & np.roll(sel, -1, 0)
                & np.roll(sel, 1, 1) & np.roll(sel, -1, 1)
            )
            ys, xs = np.nonzero(edge)
            colour = PALETTE[i % len(PALETTE)]
            for x, y in zip(xs, ys):
                draw.point((int(x), int(y)), fill=colour)

    # Legend: a solid swatch + the entity name, on a translucent strip so it
    # stays legible over any photograph. Sized RELATIVE to the image, because
    # BIG-5 images run from small screenshots to full phone-camera frames and a
    # fixed 16px legend is either unreadable or overwhelming at the extremes.
    scale = max(1.0, min(base.size) / 400.0)
    swatch, pad = int(18 * scale), int(8 * scale)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(16 * scale))
    except OSError:
        font = ImageFont.load_default()
    rows = [(labels[i], PALETTE[i % len(PALETTE)], masks[i].any())
            for i in range(len(labels))]
    if rows:
        draw = ImageDraw.Draw(out, "RGBA")
        height = pad + len(rows) * (swatch + pad)
        width = max(int(draw.textlength(f"{n}  (not found)", font=font))
                    for n, _, _ in rows) + swatch + 3 * pad
        draw.rectangle([0, 0, width, height], fill=(0, 0, 0, 140))
        y = pad
        for name, colour, found in rows:
            draw.rectangle([pad, y, pad + swatch, y + swatch], fill=colour)
            text = name if found else f"{name}  (not found)"
            draw.text((pad * 2 + swatch, y + 1), text, fill=(255, 255, 255), font=font)
            y += swatch + pad
    return out


def _panels(image, masks: List[np.ndarray], labels: List[str], alpha: float):
    """One panel per entity, side by side, for showing what each prompt found
    individually — the overlay alone cannot show that two entities overlap."""
    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        tile = _overlay(image, [mask], [label], alpha, outline=False)
        tiles.append((tile, label))
    if not tiles:
        return None
    w, h = tiles[0][0].size
    scale = max(1.0, min(w, h) / 400.0)
    band = int(28 * scale)
    strip = Image.new("RGB", (w * len(tiles), h + band), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(16 * scale))
    except OSError:
        font = ImageFont.load_default()
    for i, (tile, label) in enumerate(tiles):
        strip.paste(tile, (i * w, band))
        draw.text((i * w + int(6 * scale), int(6 * scale)), label,
                  fill=(0, 0, 0), font=font)
    return strip


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render one image's SAM3 nature masks as a figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to the image to render.")
    src = ap.add_argument_group("where the entities come from")
    src.add_argument("--responses_file", help="vlm_responses_<model>.jsonl to read this "
                                              "image's predicted entities from.")
    src.add_argument("--objects", help="Comma-separated entities, INSTEAD of an artifact. "
                                       "Ad-hoc: the output is tagged 'adhoc' so it is never "
                                       "confused with a pipeline prediction.")
    src.add_argument("--model_label", help="Name used in output filenames. Read from the "
                                           "artifact header when not given.")
    ap.add_argument("--out_dir", default="results/figures/grounding",
                    help="Where the figure and its sidecar are written.")
    ap.add_argument("--alpha", type=float, default=0.45, help="Mask opacity.")
    ap.add_argument("--panels", action="store_true",
                    help="Also write a per-entity panel strip.")
    ap.add_argument("--force_sam3", action="store_true",
                    help="Re-run SAM3 even if the artifact already has masks.")
    sam = ap.add_argument_group("SAM3 (only used when masks must be computed)")
    sam.add_argument("--sam3_model", default=gp.SAM3_MODEL_ID)
    sam.add_argument("--device", default="cuda")
    sam.add_argument("--dtype", default="auto")
    sam.add_argument("--mask_threshold", type=float, default=gp.DEFAULT_MASK_THRESHOLD)
    sam.add_argument("--center_sigma", type=float, default=gp.DEFAULT_CENTER_SIGMA)
    sam.add_argument("--hf_token", default=None,
                     help="Defaults to $HF_TOKEN. facebook/sam3 is a gated repo.")
    args = ap.parse_args()

    if not args.responses_file and not args.objects:
        ap.error("give --responses_file (normal) or --objects (ad-hoc).")

    from PIL import Image
    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    rec: Dict[str, Any] = {}
    model_label = args.model_label
    if args.responses_file:
        rec = _load_record(args.responses_file, args.image)
        if not model_label:
            # The header's own model_name, not the filename — the filename is a
            # slug of it, and the header is what every other output in this
            # project is keyed by.
            header = rec.get("_header") or {}
            model_label = (header.get("model_name")
                           or os.path.basename(args.responses_file)
                              .replace("vlm_responses_", "").replace(".jsonl", ""))

    if args.objects:
        entities = [o.strip() for o in args.objects.split(",") if o.strip()]
        model_label = (model_label or "adhoc") + "_adhoc"
    else:
        entities = _nature_entities(rec)

    print(f"image     : {args.image}  ({width}x{height})")
    if rec:
        print(f"extracted : {rec.get('objects')}")
    print(f"grounding : {entities or '(no nature entities — nothing to draw)'}")
    if not entities:
        # A real, meaningful prediction: this model found no nature here. Say so
        # and stop, rather than writing a figure identical to the input.
        raise SystemExit("No nature entities for this image — no masks to render.")

    masks = None if args.force_sam3 else _masks_from_artifact(rec, entities)
    mask_source = "artifact (already grounded)"
    if masks is None:
        print(f"masks     : running SAM3 ({args.sam3_model}) on {len(entities)} entities...")
        masks = _run_sam3(image, entities, args)
        mask_source = f"sam3 ({args.sam3_model}, mask_threshold={args.mask_threshold})"
    else:
        print("masks     : reusing the masks stored in the artifact (SAM3 not loaded)")

    rles = [gp.encode_rle(m) for m in masks if m.any()]
    scores = gp.nature_relevance_scores(rles, height, width, center_sigma=args.center_sigma)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.image).stem}_{(model_label or 'unknown').replace('/', '_')}"

    overlay = _overlay(image, masks, entities, args.alpha)
    overlay_path = out_dir / f"{stem}_overlay.png"
    overlay.save(overlay_path)

    panels_path = None
    if args.panels:
        strip = _panels(image, masks, entities, args.alpha)
        if strip is not None:
            panels_path = out_dir / f"{stem}_panels.png"
            strip.save(panels_path)

    sidecar = {
        "image_path": args.image,
        "image_size": {"width": width, "height": height},
        "model": model_label,
        "responses_file": args.responses_file,
        "extracted_objects": rec.get("objects"),
        "grounded_entities": [
            {"object": e,
             "prompt": gp.entity_prompt(e),
             "grounded": bool(m.any()),
             "pixel_count": int(m.sum()),
             "pixel_fraction": float(m.sum()) / float(width * height)}
            for e, m in zip(entities, masks)
        ],
        "nature_relevance_score_coverage_ratio": scores["coverage_ratio"],
        "nature_relevance_score_center_weighted": scores["center_weighted"],
        "mask_source": mask_source,
        "mask_threshold": args.mask_threshold,
        "center_sigma": args.center_sigma,
    }
    sidecar_path = out_dir / f"{stem}_grounding.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print()
    for e in sidecar["grounded_entities"]:
        state = f"{e['pixel_fraction']:.1%} of frame" if e["grounded"] else "NOT FOUND by SAM3"
        print(f"  {e['object']:<24} {state}")
    print(f"\nnature relevance: coverage {scores['coverage_ratio']:.3f} | "
          f"center-weighted {scores['center_weighted']:.3f}")
    print(f"\nwrote {overlay_path}")
    if panels_path:
        print(f"wrote {panels_path}")
    print(f"wrote {sidecar_path}")


if __name__ == "__main__":
    main()
