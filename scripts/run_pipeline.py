#!/usr/bin/env python3
"""
run_pipeline.py

End-to-end driver for BOTH halves of the BIG-5 nature-detection pipeline, on
one shared per-image artifact:

    VLM pipeline (caption -> entity extraction -> entity labeling)
        |   run_vlm_pipeline.py --stage infer
        |   -> vlm_responses_<model>.jsonl
        v
    Grounding pipeline (SAM3 segmentation -> nature relevance score)
        |   run_grounding_pipeline.py --in_place
        v
    ONE artifact per run, holding everything for every image: caption,
    entities, taxonomy labels, masks and the relevance score.

This runs the VLM's INFERENCE stage only, not its CLIP scoring stage — grounding
does not consume any scoring output, and scoring stays a separate, deliberate
step (`run_vlm_pipeline.py --stage score`, which reads the very same artifact
and simply ignores the grounding fields added to it).

PROCESS ISOLATION — the reason this script exists rather than one big process:
each stage runs as its OWN genuine OS subprocess (`subprocess.run([sys.executable,
<script>, ...])`), exactly as run_vlm_pipeline.py's `--stage all` does. When the
VLM subprocess exits, the OS reclaims 100% of its VRAM before SAM3 is ever
loaded, so the two models never hold GPU memory at the same time and we don't
depend on in-process CUDA cleanup (vLLM/torch release it unreliably on GC).
Deliberately `subprocess.run` and NOT `multiprocessing.Process` (even with
'spawn'): vLLM's V1 engine spawns its own internal engine-core subprocess, and
nesting that inside another spawned process tree can silently deadlock at
vLLM's async-engine IPC handshake — see run_vlm_pipeline.py's module docstring
for the confirmed reproduction.

COMMAND LINE: this script accepts the UNION of the two stages' own flags — it
introspects both parsers rather than redeclaring anything, so flags stay
correct as either stage's CLI evolves. Flags shared by both (--results_dir,
--run_name, --device, --verbose, --wandb, --max_samples, ...) are passed to
both. For the few where one shared value would be wrong, grounding-specific
overrides are provided (--grounding_batch_size, --grounding_dtype,
--grounding_device) — see GROUNDING_OVERRIDES below.

    python scripts/run_pipeline.py --dataset big5 \
        --model_family qwen2_5_vl --model_name Qwen/Qwen3.5-0.8B \
        --run_name baseline --grounding_batch_size 4 --verbose

HOW TO READ THIS FILE
Start at `main()` at the bottom. There is no pipeline logic here at all — this
script only splits a command line, resolves the shared artifact path, and runs
two subprocesses in order.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import run_grounding_pipeline
import run_vlm_pipeline
from run_vlm_pipeline import _args_to_cli


# Grounding flags that CANNOT simply inherit the shared value of their
# same-named VLM flag, exposed here under a distinct name.
#   batch_size — means images per batch in both, but a VLM batch size (16 by
#     default) and a SAM3 batch size are tuned against completely different
#     memory profiles.
#   dtype/device — the two models may want different precision, and (on a
#     multi-GPU box) different devices.
# Maps: this script's flag -> (grounding parser dest, help text suffix).
GROUNDING_OVERRIDES = {
    "grounding_batch_size": "batch_size",
    "grounding_dtype": "dtype",
    "grounding_device": "device",
}


def build_arg_parser():
    """A parser for THIS script's own additions only (the stage-specific
    overrides above, plus --skip_* switches).

    Everything else is deliberately NOT redeclared here: the two stage parsers
    are the single source of truth for their own flags, and `split_argv` below
    routes the command line through them directly. Redeclaring them would mean
    two definitions drifting apart the moment either stage gains a flag."""
    p = argparse.ArgumentParser(
        description="Run the VLM pipeline then the Grounding pipeline end-to-end over one "
                    "shared per-image artifact, each stage in its own subprocess.",
        # This parser only ever sees its own flags (split_argv strips them out
        # before the stage parsers run), so an unrecognized abbreviation must
        # not silently swallow a stage flag.
        allow_abbrev=False, add_help=False)
    p.add_argument("--grounding_batch_size", type=int, default=None,
                   help="Override the grounding stage's --batch_size (images per SAM3 "
                        "batch). Without this, the shared --batch_size is used for both "
                        "stages, which is rarely what you want — the VLM's and SAM3's "
                        "memory profiles are unrelated.")
    p.add_argument("--grounding_dtype", type=str, default=None,
                   help="Override the grounding stage's --dtype (e.g. run the VLM in "
                        "bfloat16 but SAM3 in float32).")
    p.add_argument("--grounding_device", type=str, default=None,
                   help="Override the grounding stage's --device.")
    p.add_argument("--skip_vlm", action="store_true",
                   help="Skip the VLM inference stage and ground an artifact that already "
                        "exists (e.g. re-run grounding with a different --mask_threshold or "
                        "--center_sigma without paying for inference again).")
    p.add_argument("--skip_grounding", action="store_true",
                   help="Run only the VLM inference stage. Equivalent to calling "
                        "run_vlm_pipeline.py --stage infer directly; provided so a script "
                        "or job file can toggle the grounding half off without changing "
                        "which command it invokes.")
    return p


def _print_combined_help():
    """`--help` has to be handled here rather than left to argparse: this
    script's command line is the UNION of three parsers, and whichever one saw
    -h first would otherwise print only its own third of the flags and exit."""
    print(__doc__)
    print("\n" + "=" * 70 + "\nrun_pipeline.py's own flags\n" + "=" * 70)
    build_arg_parser().print_help()
    print("\n" + "=" * 70 + "\nVLM stage flags (run_vlm_pipeline.py; --stage is forced to 'infer')\n"
          + "=" * 70)
    run_vlm_pipeline.build_arg_parser().print_help()
    print("\n" + "=" * 70 + "\nGrounding stage flags (run_grounding_pipeline.py; --responses_file "
          "and --in_place are set automatically)\n" + "=" * 70)
    run_grounding_pipeline.build_arg_parser().print_help()


def split_argv(argv):
    """Route one combined command line to the two stage parsers.

    Each stage parser gets the whole (post-override-stripping) command line via
    `parse_known_args`, so a flag both stages declare — --results_dir, --device,
    --verbose, --wandb, --max_samples — is seen and honored by both, while a
    flag only one declares is simply left over by the other. A token BOTH
    parsers reject is a genuine typo and is reported as such, rather than being
    silently dropped (which is what plain parse_known_args would do).

    Returns (own_args, vlm_args, vlm_parser, ground_args, ground_parser).
    """
    own_parser = build_arg_parser()
    own_args, rest = own_parser.parse_known_args(argv)

    vlm_parser = run_vlm_pipeline.build_arg_parser()
    ground_parser = run_grounding_pipeline.build_arg_parser()

    if own_args.skip_vlm:
        # Nothing from the VLM stage is going to run, so don't make the user
        # supply its required flags (--dataset) just to get past the parser on
        # the way to a grounding-only invocation.
        for action in vlm_parser._actions:
            action.required = False

    vlm_args, vlm_left = vlm_parser.parse_known_args(rest)
    ground_args, ground_left = ground_parser.parse_known_args(rest)

    # A flag-looking token neither parser recognized belongs to nobody.
    unknown = [t for t in vlm_left if t.startswith("--") and t in ground_left]
    if unknown:
        own_parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    for own_dest, ground_dest in GROUNDING_OVERRIDES.items():
        value = getattr(own_args, own_dest, None)
        if value is not None:
            setattr(ground_args, ground_dest, value)

    return own_args, vlm_args, vlm_parser, ground_args, ground_parser


def _run_subprocess(script_name, argv, stage_label):
    """Run one stage as a genuine OS subprocess and wait for it, raising if it
    failed. stdout/stderr/env/cwd are inherited, so logs interleave live exactly
    as they would running that stage by hand."""
    script = Path(__file__).resolve().parent / script_name
    cmd = [sys.executable, str(script)] + argv
    print(f"▶️  [pipeline] {stage_label}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{stage_label} subprocess failed (exit code {result.returncode}).")


def main():
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        _print_combined_help()
        return
    own_args, vlm_args, vlm_parser, ground_args, ground_parser = split_argv(sys.argv[1:])

    if own_args.skip_vlm and own_args.skip_grounding:
        print("Both --skip_vlm and --skip_grounding given — nothing to run.")
        sys.exit(1)

    # Both stages must agree on ONE artifact path — that shared file is the
    # whole storage convention (grounding enriches the VLM's own records rather
    # than writing a parallel output). Resolve it once with the VLM script's own
    # resolver (so the default name/location stays defined in exactly one place)
    # and pin it explicitly on both Namespaces, rather than trusting each stage
    # to independently rederive the same default.
    vlm_args.stage = "infer"
    vlm_args = run_vlm_pipeline._resolve_responses_file(vlm_args)
    ground_args.responses_file = vlm_args.responses_file
    # --in_place: grounding writes back into that same artifact (safely — via a
    # temp file swapped in only on success), which is the point of running the
    # two stages together.
    ground_args.in_place = True

    # One shared W&B run id across both subprocesses, so the VLM stage's metrics
    # and the grounding stage's diagnostics land in a single run instead of two
    # (same mechanism as run_vlm_pipeline.py's --stage all).
    if getattr(vlm_args, "wandb", False) or getattr(ground_args, "wandb", False):
        import wandb
        run_id = wandb.util.generate_id()
        vlm_args.wandb_run_id = run_id
        ground_args.wandb_run_id = run_id

    if not own_args.skip_vlm:
        _run_subprocess("run_vlm_pipeline.py",
                        _args_to_cli(vlm_args, vlm_parser, stage="infer"),
                        "VLM inference")
    if not own_args.skip_grounding:
        # The VLM subprocess has exited by now, so its VRAM is fully reclaimed
        # by the OS before SAM3 loads — see the module docstring.
        _run_subprocess("run_grounding_pipeline.py",
                        _args_to_cli(ground_args, ground_parser),
                        "Grounding")

    print(f"✅ [pipeline] done — enriched artifact: {vlm_args.responses_file}")


if __name__ == "__main__":
    main()
