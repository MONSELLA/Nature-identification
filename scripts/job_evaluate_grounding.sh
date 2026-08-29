#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --account=acct_gen
#SBATCH --job-name=eval_grounding_gemma
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/dev/null
#
# Evaluate the GROUNDING pipeline for the Gemma model on BOTH ground truths:
#
#   1. COCO         — run_vlm_pipeline.py --stage score, whose COCO block does
#                     the mask-IoU detection evaluation (IoU sweep, size split,
#                     label + axis agreement) against COCO's own instance
#                     segmentation.
#   2. BIG-5 manual — score_grounding_gt.py against Pau's hand-drawn grounding
#                     annotations (170 nature + 170 no-nature images).
#
# Both read the SAME kind of artifact (a grounded vlm_responses_*.jsonl) and
# report the same families of number, deliberately: the two evaluations are
# built on shared primitives (src/evaluation/detection_metrics.py) so their
# tables can be read side by side.
#
# ONE NUMBER THAT IS NOT COMPARABLE BETWEEN THEM — PRECISION. COCO annotates
# only 80 curated classes, so a correctly-segmented tree is exempted rather
# than charged (measured: 76% of predictions on the gemma COCO run were
# exempted, which is why COCO precision must always be quoted next to
# `excluded_predictions`). The BIG-5 annotations are EXHAUSTIVE — every nature
# entity in the image was drawn — so nothing is exempted and every unmatched
# prediction is a false positive. BIG-5 precision is therefore the stricter,
# more meaningful number of the two. Recall carries no such caveat on either.
#
# USAGE — all four are POSITIONAL arguments, deliberately not env vars (see
# below for why), so they always take effect:
#   sbatch job_evaluate_grounding.sh                                       # both, default model
#   sbatch job_evaluate_grounding.sh both  google/gemma-4-26B-A4B-it
#   sbatch job_evaluate_grounding.sh coco  google/gemma-4-26B-A4B-it       # COCO only
#   sbatch job_evaluate_grounding.sh big5  google/gemma-4-26B-A4B-it       # BIG-5 manual GT only
#
# $3 RESPONSES_ROOT (default: results/vlm_pipeline/baseline) and $4 LABEL
# (default: MODEL_SLUG) exist for evaluating a FINE-TUNED ADAPTER rather than
# a base model. A LoRA adapter's artifact still carries the BASE model's own
# slug (e.g. "google_gemma-4-12B-it" whether or not an adapter was applied —
# the model name doesn't change), so MODEL_SLUG alone cannot distinguish a
# base-model run from an adapter's. $3 points artifact lookup at wherever the
# adapter's own infer+ground run actually wrote its artifacts (NOT the
# baseline tree, which holds only base-model artifacts and would otherwise be
# silently read instead of the adapter's own); $4 keeps this run's OUTPUT
# (results JSON run_name/output_file, big5_grounding_<label> files, the log
# filename) from colliding with the base model's own results, which are also
# keyed on MODEL_SLUG alone. Example, evaluating a self-distilled adapter
# whose infer+ground artifacts were written under
# results/vlm_pipeline/rft_grounding/:
#   sbatch job_evaluate_grounding.sh both google/gemma-4-12B-it \
#     /home/pmonserrat/code/results/vlm_pipeline/rft_grounding  rft_selfdistill
#
# The BIG-5 half RUNS SAM3 ITSELF when needed: it subsets the artifacts to the
# 340 annotated images and grounds that subset if it is not already grounded.
# It never runs VLM inference — the artifacts must already exist (for an
# adapter, scripts/job_evaluate_grounding_infer.sh produces them).
#
# Run it from this scripts/ directory, like every other job_*.sh here.

# facebook/sam3 is a GATED HuggingFace repo, so grounding needs authentication.
# Either run `hf auth login` once (the token is then cached in
# ~/.cache/huggingface/token and picked up automatically), or export HF_TOKEN in
# your shell before `sbatch`. Never hardcode it here — this file is tracked.
#
# UNSET IT IF EMPTY, deliberately: an empty HF_TOKEN is WORSE than none at all.
# huggingface_hub would send a literal "Authorization: Bearer " header (httpx
# then raises `Illegal header value b'Bearer '`) instead of falling back to the
# cached login token. So only export it when it actually has a value.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tfm

MODE="${1:-both}"

# MODEL_NAME is a POSITIONAL argument ($2), NOT read from an env var set
# before `sbatch` (e.g. `MODEL_NAME=... sbatch job.sh`). That pattern set the
# variable in the SUBMITTING shell, but `sbatch` only registers a request with
# the controller — the job itself runs later, in a separate shell spawned by
# `slurmd` on (possibly) a different node, and whether that shell inherits the
# submitting environment depends on the cluster's export policy and any
# profile/module scripts the batch shell re-sources. When it silently doesn't
# propagate, MODEL_NAME falls back to the default below without ANY error —
# and if that default is a model whose evaluation already completed, the run
# reports "already done" for a model you never actually asked to evaluate.
# CONFIRMED CAUSE of exactly this symptom on a real run. A positional argument
# has no such ambiguity: `sbatch script.sh arg1 arg2` always passes `$1 $2`
# into the job's own `$1 $2`, unaffected by any environment propagation policy.
MODEL_NAME="${2:-google/gemma-4-12B-it}"
# MODEL_SLUG is DERIVED, exactly matching how the pipeline names its own
# artifact files (scripts/run_vlm_pipeline.py's _model_slug:
# args.model_name.replace("/", "_") — nothing else, not model_family). This
# is what the ARTIFACT is named regardless of whether a LoRA adapter was
# applied — see $3/$4 above for why that ambiguity needs separate handling.
MODEL_SLUG="${MODEL_NAME//\//_}"
# LABEL ($4) is what every OUTPUT of this run is named after. LABEL_GIVEN
# tracks whether $4 was ACTUALLY passed (not just whether it happens to equal
# MODEL_SLUG) — the base-model default path (COCO's shared
# "vlm_pipeline/baseline/coco/" results file, "big5_grounding_<slug>") stays
# BYTE-IDENTICAL to the original hardcoded behavior whenever $4 is omitted,
# so existing base-model re-runs keep merging into the same results file
# they always have. Only an EXPLICIT $4 switches output onto a separate,
# adapter-specific path — see run_coco/run_big5 below for where this branches.
LABEL_GIVEN="${4:-}"
LABEL="${LABEL_GIVEN:-$MODEL_SLUG}"

mkdir -p ../logs
# LABEL in the log filename (not bare MODEL_SLUG), so a base-model run and
# any of its fine-tuned adapters' runs never overwrite each other's log.
exec > "../logs/out_eval_grounding_${LABEL}_${MODE}.log" 2>&1

# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR="/home/pmonserrat/code/results/"
# Where to READ existing artifacts from. Default: the baseline tree (base
# models only). Override via $3 to point at wherever an adapter's own
# infer+ground run wrote its artifacts instead.
RESPONSES_ROOT="${3:-/home/pmonserrat/code/results/vlm_pipeline/baseline}"

# COCO
COCO_ARTIFACT="${RESPONSES_ROOT}/coco/responses/vlm_responses_${MODEL_SLUG}.jsonl"
COCO_INSTANCES_JSON="/home/pmonserrat/datasets/coco/annotations/instances_val2017.json"

# BIG-5 manual grounding GT — the folder holding nature/, no_nature/,
# manifest.json and coco_instances.json (output of
# scripts/convert_grounding_annotations.py, copied to the cluster).
#
# Accepts either layout, because what gets copied to the cluster has varied:
# the converter writes these under a `processed/` subfolder, but the tree has
# also been copied with that level flattened away. Whichever actually contains
# `nature/` wins, so the script does not need editing when the copy changes.
BIG5_GROUNDING_ROOT="/home/pmonserrat/datasets/big_5/grounding"
if [ -d "${BIG5_GROUNDING_ROOT}/processed/nature" ]; then
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}/processed"
else
    BIG5_GT_DIR="${BIG5_GROUNDING_ROOT}"
fi

# The manual annotations span BOTH platforms (84 twitter / 86 weibo images),
# but the pipeline writes ONE ARTIFACT PER DATASET NAME. Both must be passed:
# an absent platform contributes no predictions, so every one of its GT regions
# would score as a false negative and recall would be silently ~halved. The
# scorer merges them by image basename.
BIG5_ARTIFACTS=()
for ds in big5_twitter big5_weibo big5; do
    f="${RESPONSES_ROOT}/${ds}/responses/vlm_responses_${MODEL_SLUG}.jsonl"
    [ -e "$f" ] && BIG5_ARTIFACTS+=("$f")
done

# SAM3 grounding knobs, used only when the subset still needs grounding.
# Deliberately modest: BIG-5 images are raw social-media resolutions with no
# upper bound, and a vision encoder's attention memory grows with patch count,
# so what is comfortable on pre-resized COCO/ImageNet images is not here
# (recap v18/v19). Raise them if the GPU has headroom.
GROUND_BATCH="${GROUND_BATCH:-4}"
GROUND_MAX_PAIRS="${GROUND_MAX_PAIRS:-8}"

# facebook/sam3 is a GATED HuggingFace repo. run_grounding_pipeline.py picks
# this up automatically when --hf_token isn't passed. Left as a passthrough
# rather than hardcoded here — export it before sbatch, or copy the line from
# job_grounding_pipeline.sh.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi

OUT_DIR="/home/pmonserrat/code/results/grounding_eval"
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo "Grounding evaluation"
echo "  MODEL_NAME     = $MODEL_NAME    (from \$2; default if you didn't pass one)"
echo "  MODEL_SLUG     = $MODEL_SLUG    (artifact filenames are keyed on this)"
echo "  RESPONSES_ROOT = $RESPONSES_ROOT    (from \$3; where artifacts are READ from)"
echo "  LABEL          = $LABEL    (from \$4; every OUTPUT is named after this)"
echo "  mode           = $MODE"
echo "=============================================================="

# =============================================================================
# 1. COCO
# =============================================================================
run_coco () {
    echo
    echo "################## COCO detection evaluation ##################"
    if [ ! -e "$COCO_ARTIFACT" ]; then
        echo "SKIP: no COCO artifact at $COCO_ARTIFACT"
        return 1
    fi
    echo "artifact: $COCO_ARTIFACT"

    # --instances_json is REQUIRED, not optional: COCO's per-instance
    # segmentation is deliberately never stored in the artifact (it would bloat
    # every record for a scoring-only use), and without it the detection block
    # is skipped outright rather than silently degrading to box matching.
    #
    # --detection_iou_threshold stays at its 0.5 default; the IoU SWEEP
    # (0.50...0.95) is computed regardless and is the headline number, since it
    # reads out mask tightness rather than a single operating point.
    # Byte-identical to the original hardcoded path when $4 wasn't given (a
    # base-model run merges into the SAME shared results file it always has);
    # only an explicit LABEL diverts output onto its own adapter-specific path.
    if [ -n "$LABEL_GIVEN" ]; then
        coco_run_name="vlm_pipeline/grounding_eval_${LABEL}/coco/"
        coco_output_file="vlm_pipeline_coco_${LABEL}_results.json"
    else
        coco_run_name="vlm_pipeline/baseline/coco/"
        coco_output_file="vlm_pipeline_coco_results.json"
    fi
    python run_vlm_pipeline.py --stage score \
        --dataset coco \
        --responses_file "$COCO_ARTIFACT" \
        --instances_json "$COCO_INSTANCES_JSON" \
        --clipscore_model longclip \
        --results_dir "$RESULTS_DIR" \
        --run_name "$coco_run_name" \
        --output_file "$coco_output_file" \
        --verbose \
        --resume
}

# =============================================================================
# 2. BIG-5 manual grounding GT
# =============================================================================
run_big5 () {
    echo
    echo "############# BIG-5 manual grounding evaluation ###############"
    if [ ${#BIG5_ARTIFACTS[@]} -eq 0 ]; then
        echo "SKIP: no BIG-5 artifact found under ${RESPONSES_ROOT}/{big5_twitter,big5_weibo,big5}/responses/"
        echo "      expected filename: vlm_responses_${MODEL_SLUG}.jsonl"
        return 1
    fi
    if [ ! -d "$BIG5_GT_DIR/nature" ]; then
        echo "SKIP: no nature/ directory found under either"
        echo "        ${BIG5_GROUNDING_ROOT}/processed/  or  ${BIG5_GROUNDING_ROOT}/"
        echo "      Copy the converter's processed/ output to the cluster first."
        return 1
    fi
    echo "artifacts: ${BIG5_ARTIFACTS[*]}"
    echo "GT dir   : $BIG5_GT_DIR"

    # --- 1. Cut the artifacts down to the annotated images -------------------
    # The BIG-5 artifacts hold every image of their platform (~6663 records)
    # while the GT covers 340. Grounding the full set would cost ~20x the SAM3
    # time for no extra measurement, on exactly the raw social-media
    # resolutions that make the vision encoder OOM-prone. The subset is a
    # SEPARATE file: the production artifacts are never modified, so a later
    # full BIG-5 grounding run stays the single source of truth for them.
    echo
    echo "--- subsetting artifacts to the annotated images ---"
    # LABEL, not MODEL_SLUG: this file is a PERSISTENT CACHE reused across
    # invocations (SUBSET_SOURCES below), not just a one-off output. A
    # base-model run and any fine-tuned adapter's run sharing MODEL_SLUG
    # would silently merge their grounded predictions for the SAME 340
    # images into ONE file, each invocation's "last occurrence wins" merge
    # nondeterministically overwriting the other's results. LABEL keeps
    # them in genuinely separate caches; the default (LABEL == MODEL_SLUG
    # when $4 is omitted) reproduces the original base-model-only behavior
    # exactly.
    SUBSET="${OUT_DIR}/big5_grounding_subset_${LABEL}.jsonl"

    # BUG FIXED HERE: the original BIG5_ARTIFACTS (twitter/weibo) are NEVER
    # grounded — SAM3 only ever runs --in_place on THIS subset file, never on
    # them. Rebuilding the subset from only those originals on every run
    # therefore always produced a fresh, ungrounded copy and silently
    # OVERWROTE the previous run's already-grounded subset, so
    # SUBSET_GROUNDED read 0/340 and SAM3 re-ran from scratch every single
    # invocation — including ones whose only intent was to re-score after a
    # metrics/display change. Fix: if a subset from a PREVIOUS run exists,
    # pass it too, LAST in the list, so its already-grounded records win the
    # merge (subset_artifact_for_gt.py keeps the LAST occurrence of each image
    # name) instead of being discarded. Written to a temp path and moved into
    # place afterward — subset_artifact_for_gt.py reads every input fully
    # before it opens the output, so reading and overwriting the same path
    # would technically work, but going through a temp file matches this
    # project's own crash-safety convention (run_grounding_pipeline.py
    # --in_place) and costs nothing.
    SUBSET_SOURCES=("${BIG5_ARTIFACTS[@]}")
    [ -e "$SUBSET" ] && SUBSET_SOURCES+=("$SUBSET")
    SUBSET_TMP="${SUBSET}.tmp"
    SUBSET_INFO=$(python subset_artifact_for_gt.py \
        --artifact "${SUBSET_SOURCES[@]}" \
        --gt_dir "$BIG5_GT_DIR" \
        --out "$SUBSET_TMP") || return 1
    mv "$SUBSET_TMP" "$SUBSET"
    echo "$SUBSET_INFO"

    # "<grounded>/<total>" — how many of the subset's records already carry
    # SAM3 masks.
    local counts n_grounded n_total
    counts=$(echo "$SUBSET_INFO" | grep -o 'SUBSET_GROUNDED=[0-9]*/[0-9]*' | cut -d= -f2)
    n_grounded="${counts%%/*}"
    n_total="${counts##*/}"

    # --- 2. Ground it, if it isn't already -----------------------------------
    if [ "$n_grounded" -lt "$n_total" ]; then
        echo
        echo "--- grounding $((n_total - n_grounded)) of $n_total records (SAM3) ---"
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "NOTE: HF_TOKEN is unset and facebook/sam3 is a GATED repo."
            echo "      Either 'export HF_TOKEN=...' before sbatch, or copy the"
            echo "      export line from job_grounding_pipeline.sh into this file."
        fi
        # --in_place on the SUBSET (not the production artifact): the subset is
        # this evaluation's own working copy, and enriching it in place keeps
        # one file carrying both stages, per the project's one-artifact rule.
        # A crash cannot corrupt it — run_grounding_pipeline.py writes via a
        # temp file and only swaps it in on success.
        #
        # --instance_grounding stays "auto", which is OFF for a non-COCO
        # artifact: detection here reads the SEMANTIC head only, so the
        # instance head would be pure cost.
        #
        # --batch_size/--max_pairs_per_forward kept modest because BIG-5 images
        # are uncapped phone-camera/screenshot resolutions, unlike the
        # pre-resized COCO/ImageNet images (recap v18/v19).
        python run_grounding_pipeline.py \
            --responses_file "$SUBSET" \
            --in_place \
            --batch_size "$GROUND_BATCH" \
            --max_pairs_per_forward "$GROUND_MAX_PAIRS" \
            --verbose \
        || { echo "GROUNDING FAILED — not scoring an ungrounded artifact"; return 1; }
    else
        echo "  all $n_total subset records already grounded — skipping SAM3"
    fi

    # Void handling is ON by default and should stay on: the annotation draws
    # `cloud` on top of `sky` (pairwise GT IoU up to 0.747), and voiding the
    # contested pixels is what makes the score independent of whether SAM3's
    # "sky" includes or excludes the clouds in front of it. Measured on the
    # worst such image: a perfect-but-other-convention sky scores 0.436 without
    # voiding (FAILING the 0.50 threshold) and 1.000 with it.
    # Pass --no_void to produce the comparison run.
    echo
    echo "--- scoring ---"
    python score_grounding_gt.py \
        --artifact "$SUBSET" \
        --gt_dir "$BIG5_GT_DIR" \
        --out "${OUT_DIR}/big5_grounding_${LABEL}"
}

# =============================================================================
# Dispatch
# =============================================================================
STATUS=0
case "$MODE" in
    coco) run_coco || STATUS=1 ;;
    big5) run_big5 || STATUS=1 ;;
    both)
        # Deliberately independent: a failure in one must not stop the other,
        # since they read different artifacts and answer different questions.
        run_coco || { echo "COCO evaluation FAILED — continuing to BIG-5"; STATUS=1; }
        run_big5 || { echo "BIG-5 evaluation FAILED"; STATUS=1; }
        ;;
    *)
        echo "Unknown mode '$MODE' — expected one of: both, coco, big5"
        exit 2
        ;;
esac

if [ -n "$LABEL_GIVEN" ]; then
    coco_results_path="${RESULTS_DIR}vlm_pipeline/grounding_eval_${LABEL}/coco/vlm_pipeline_coco_${LABEL}_results.json"
else
    coco_results_path="${RESULTS_DIR}vlm_pipeline/baseline/coco/vlm_pipeline_coco_results.json"
fi
echo
echo "=============================================================="
echo "Done (exit status $STATUS)."
echo "  COCO   -> ${coco_results_path}"
echo "            (detection, detection_iou_sweep, detection_by_size,"
echo "             detection_labels, detection_axis_agreement)"
echo "  BIG-5  -> ${OUT_DIR}/big5_grounding_${LABEL}_results.json"
echo "            + _per_image.csv (one browsable row per image)"
echo "=============================================================="
exit $STATUS
